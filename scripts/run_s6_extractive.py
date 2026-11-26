"""Chạy cấu hình S6: dùng XLM-R đã fine-tune làm Extractive Reader trên top-k chunks.

Ví dụ chạy trên Colab:
    PYTHONPATH=. python scripts/run_s6_extractive.py \
        --model_dir /content/drive/MyDrive/QA_RAG_Project/checkpoint/xlmr_run2_lr5e5_ep3_len384/best_model \
        --retriever dense --top_k 3 --null_thresh 0.3477
"""
import argparse
import collections
import glob
import json
import re
import string
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.postprocess import apply_null_threshold, postprocess_qa_predictions  # noqa: E402

REFUSAL_TEXT = "Không tìm thấy thông tin trong tài liệu"


def load_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_vi(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    punct = string.punctuation + "“”‘’…–—«»"
    text = re.sub(f"[{re.escape(punct)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fallback_token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_vi(pred).split()
    g_toks = normalize_vi(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def get_tv3_scorer():
    """Thử nạp hàm score_row chung của TV3 từ rag/generation/generate.py."""
    try:
        from rag.generation.generate import score_row
        return score_row
    except Exception as e:
        print(f"[WARN] Không import được rag.generation.generate.score_row ({e}); dùng hàm Token-F1 nội bộ.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Chạy cấu hình S6 (XLM-R Extractive Reader cho RAG)")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/content/drive/MyDrive/QA_RAG_Project/checkpoint/xlmr_run2_lr5e5_ep3_len384/best_model",
    )
    parser.add_argument("--model_id", type=str, default="xlmr_run2_lr5e5_ep3_len384")
    parser.add_argument("--config_name", type=str, default="S6")
    parser.add_argument("--retrieval_file", type=str, default="results/retrieval/retrieval_top10.jsonl")
    parser.add_argument("--chunks_file", type=str, default="data/corpus/chunks.jsonl")
    parser.add_argument("--questions_glob", type=str, default="data/rag_test/*.jsonl")
    parser.add_argument("--retriever", type=str, default="dense", choices=["dense", "bm25"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--null_thresh", type=float, default=0.3477)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results/rag")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device} | Model: {args.model_dir}")
    print(f"[INFO] Config: {args.config_name} | Retriever: {args.retriever} | top_k={args.top_k} | null_thresh={args.null_thresh}")

    # 1. Nạp dữ liệu
    chunks_map = {row["id"]: unicodedata.normalize("NFC", row["text"]) for row in load_jsonl(args.chunks_file)}

    q_paths = sorted(glob.glob(args.questions_glob))
    if not q_paths:
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi theo mẫu: {args.questions_glob}")
    questions = []
    for p in q_paths:
        questions.extend(load_jsonl(p))
    print(f"[INFO] Đã nạp {len(questions)} câu hỏi từ {q_paths}")

    ret_map = {
        str(r["qid"]): r
        for r in load_jsonl(args.retrieval_file)
        if r["retriever"] == args.retriever
    }

    # 2. Nạp model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir).to(device)
    model.eval()
    cls_id = tokenizer.cls_token_id

    # 3. Chuẩn bị examples & sliding-window features cho từng cặp (câu hỏi, chunk trong top-k)
    examples = []
    features = []
    all_input_ids = []
    all_attention_mask = []
    question_chunk_ex_ids = {}  # qid -> list of (rank, chunk_id, ex_id)

    for q in questions:
        qid = str(q["id"])
        q_text = unicodedata.normalize("NFC", q["question"])
        ret_item = ret_map.get(qid)
        if ret_item is None:
            print(f"[WARN] Thiếu kết quả truy hồi ({args.retriever}) cho {qid}, bỏ qua.")
            continue

        top_chunk_ids = ret_item["chunk_ids"][: args.top_k]
        question_chunk_ex_ids[qid] = []

        for rank, cid in enumerate(top_chunk_ids, start=1):
            ctx = chunks_map.get(cid, "")
            if not ctx:
                continue
            ex_id = f"{qid}__c{rank}"
            examples.append({"id": ex_id, "context": ctx})
            question_chunk_ex_ids[qid].append((rank, cid, ex_id))

            enc = tokenizer(
                q_text,
                ctx,
                truncation="only_second",
                max_length=args.max_len,
                stride=args.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            for w_idx in range(len(enc["input_ids"])):
                input_ids = enc["input_ids"][w_idx]
                attn_mask = enc["attention_mask"][w_idx]
                seq_ids = enc.sequence_ids(w_idx)
                raw_offsets = enc["offset_mapping"][w_idx]
                # Gán (-1, -1) cho token không thuộc context để khớp hoàn toàn với qa/postprocess.py
                offsets = [
                    (int(o[0]), int(o[1])) if seq_ids[k] == 1 else (-1, -1)
                    for k, o in enumerate(raw_offsets)
                ]
                cls_index = input_ids.index(cls_id) if cls_id in input_ids else 0

                features.append(
                    {
                        "example_id": ex_id,
                        "offset_mapping": offsets,
                        "cls_index": cls_index,
                    }
                )
                all_input_ids.append(input_ids)
                all_attention_mask.append(attn_mask)

    print(f"[INFO] Tổng số cặp (câu hỏi, chunk): {len(examples)} | Tổng cửa sổ trượt: {len(features)}")

    # 4. Chạy suy luận theo batch
    start_logits_list = []
    end_logits_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_input_ids), args.batch_size), desc="S6 Inference"):
            batch_ids = torch.tensor(all_input_ids[i : i + args.batch_size], dtype=torch.long, device=device)
            batch_mask = torch.tensor(all_attention_mask[i : i + args.batch_size], dtype=torch.long, device=device)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            start_logits_list.append(outputs.start_logits.cpu().numpy())
            end_logits_list.append(outputs.end_logits.cpu().numpy())

    raw_predictions = (
        np.concatenate(start_logits_list, axis=0),
        np.concatenate(end_logits_list, axis=0),
    )

    # 5. Hậu xử lý bằng đúng hàm của TV1 (qa/postprocess.py)
    best_non_null_preds, null_score_diffs, _ = postprocess_qa_predictions(
        examples=examples,
        features=features,
        raw_predictions=raw_predictions,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )

    # 6. Chọn ứng viên tốt nhất giữa top-k chunks cho từng câu hỏi và áp ngưỡng null
    tv3_score_row = get_tv3_scorer()
    tv3_scorer_ok = tv3_score_row is not None
    rows_out = []

    for q in questions:
        qid = str(q["id"])
        if qid not in question_chunk_ex_ids:
            continue

        top_chunk_ids = ret_map[qid]["chunk_ids"][: args.top_k]
        gold_answer = (q.get("answer") or "").strip()
        gold_chunks = q.get("gold_chunks") or []
        answerable = bool(q.get("answerable", True))
        q_type = q.get("type") or ("out_of_corpus" if not answerable else "unknown")
        retrieval_hit = bool(set(top_chunk_ids) & set(gold_chunks)) if answerable else None

        # Chọn chunk có null_score_diff nhỏ nhất (mô hình tin tưởng có đáp án nhất)
        best_rank, best_cid, best_text, best_diff = None, None, "", float("inf")
        chunk_candidates = []
        for rank, cid, ex_id in question_chunk_ex_ids[qid]:
            cand_text = best_non_null_preds.get(ex_id, "")
            cand_diff = float(null_score_diffs.get(ex_id, float("inf")))
            chunk_candidates.append(
                {
                    "rank": rank,
                    "chunk_id": cid,
                    "span": cand_text,
                    "null_score_diff": round(cand_diff, 4),
                }
            )
            if cand_text and cand_diff < best_diff:
                best_rank, best_cid, best_text, best_diff = rank, cid, cand_text, cand_diff

        # Áp ngưỡng bằng đúng hàm apply_null_threshold
        thresholded = apply_null_threshold({qid: best_text}, {qid: best_diff}, args.null_thresh)[qid]
        if not thresholded:
            is_refusal = True
            final_answer = REFUSAL_TEXT
            pred_chunk_id = None
        else:
            is_refusal = False
            final_answer = thresholded
            pred_chunk_id = best_cid

        # Tạo bản ghi đầu ra tương thích với hallucination_breakdown.py & generate.py
        gen_row = {
            "qid": qid,
            "config": args.config_name,
            "model_id": args.model_id,
            "retriever": args.retriever,
            "k": args.top_k,
            "top_k": args.top_k,
            "chunk_ids": top_chunk_ids,
            "answer": final_answer,
            "pred_chunk_id": pred_chunk_id,
            "pred_chunk_rank": best_rank if not is_refusal else None,
            "null_score_diff": round(best_diff, 4) if best_diff < float("inf") else None,
            "is_refusal": is_refusal,
            "question": q["question"],
            "type": q_type,
            "answerable": answerable,
            "gold_answer": gold_answer,
            "gold_chunks": gold_chunks,
            "retrieval_hit": retrieval_hit,
            "chunk_candidates": chunk_candidates,
        }

        # Chấm điểm tự động (ưu tiên dùng score_row của TV3 để đồng bộ với S2)
        if tv3_scorer_ok:
            try:
                sample = {"qid": qid, "answer": gold_answer, "in_corpus": answerable}
                s = tv3_score_row(gen_row, sample)
                gen_row["token_f1"] = round(float(s["token_f1"]), 4)
                gen_row["contains_gold"] = bool(s.get("contains_gold", False))
                gen_row["refused_by_tv3"] = bool(s.get("refused", is_refusal))
            except Exception as e:
                print(f"[WARN] score_row của TV3 gặp lỗi ({e}); chuyển sang hàm Token-F1 nội bộ.")
                tv3_scorer_ok = False

        if not tv3_scorer_ok:
            eff_gold = gold_answer if answerable else (gold_answer or REFUSAL_TEXT)
            gen_row["token_f1"] = round(fallback_token_f1(final_answer, eff_gold), 4)
            gen_row["contains_gold"] = bool(
                answerable and gold_answer and normalize_vi(gold_answer) in normalize_vi(final_answer)
            )
            gen_row["refused_by_tv3"] = is_refusal

        rows_out.append(gen_row)

    # 7. Tổng hợp kết quả theo hit/miss, answerable/unanswerable và loại câu hỏi
    ans_rows = [r for r in rows_out if r["answerable"]]
    ooc_rows = [r for r in rows_out if not r["answerable"]]
    hit_rows = [r for r in ans_rows if r["retrieval_hit"] is True]
    miss_rows = [r for r in ans_rows if r["retrieval_hit"] is False]

    def avg_f1(rs):
        return round(100.0 * float(np.mean([r["token_f1"] for r in rs])), 2) if rs else 0.0

    def rate_pct(num, den):
        return round(100.0 * num / den, 2) if den else 0.0

    by_type = {}
    for t in sorted({r["type"] for r in rows_out}):
        sub = [r for r in rows_out if r["type"] == t]
        by_type[t] = {
            "n": len(sub),
            "token_f1": avg_f1(sub),
            "refusal_rate": rate_pct(sum(r["is_refusal"] for r in sub), len(sub)),
        }

    summary = {
        "config": args.config_name,
        "model_id": args.model_id,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "null_thresh": args.null_thresh,
        "scorer": "rag.generation.generate.score_row" if tv3_scorer_ok else "fallback_token_f1",
        "total": len(rows_out),
        "overall_token_f1": avg_f1(rows_out),
        "answerable": {
            "n": len(ans_rows),
            "token_f1": avg_f1(ans_rows),
            "contains_gold_rate": rate_pct(sum(r["contains_gold"] for r in ans_rows), len(ans_rows)),
            "false_refusal_rate": rate_pct(sum(r["is_refusal"] for r in ans_rows), len(ans_rows)),
        },
        "hit_miss": {
            "hit": {
                "n": len(hit_rows),
                "token_f1": avg_f1(hit_rows),
                "false_refusal_rate": rate_pct(sum(r["is_refusal"] for r in hit_rows), len(hit_rows)),
            },
            "miss": {
                "n": len(miss_rows),
                "token_f1": avg_f1(miss_rows),
                "false_refusal_rate": rate_pct(sum(r["is_refusal"] for r in miss_rows), len(miss_rows)),
            },
        },
        "out_of_corpus": {
            "n": len(ooc_rows),
            "refusal_accuracy": rate_pct(sum(r["is_refusal"] for r in ooc_rows), len(ooc_rows)),
        },
        "by_type": by_type,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"s6_{args.retriever}_k{args.top_k}.jsonl"
    metric_path = out_dir / f"s6_{args.retriever}_k{args.top_k}_metrics.json"

    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== TỔNG HỢP KẾT QUẢ S6 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[DONE] Predictions: {pred_path}")
    print(f"[DONE] Metrics:     {metric_path}")


if __name__ == "__main__":
    main()