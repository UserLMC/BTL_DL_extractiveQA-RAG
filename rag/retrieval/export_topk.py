"""Xuất top-10 kết quả truy hồi của MỌI câu hỏi ra một file chung cho TV3, kèm Recall theo k.

Đầu ra (mặc định trong results/retrieval/):
- retrieval_top10.jsonl : mỗi dòng {"qid", "retriever", "chunk_ids", "scores", "first_hit"}
    đúng định dạng mà rag/generation/generate.py đọc qua --retrieval.
    retriever = "bm25" (S1) hoặc "dense" (= dense tốt nhất, dùng cho S2, S4, S5, S6).
    Có cả câu ngoài kho (first_hit = null) vì LLM vẫn phải trả lời các câu này.
- recall_by_k.csv       : Recall@k với k = 1..10 của từng retriever (ablation k, vẽ hình).

Ví dụ:
    python -m rag.retrieval.export_topk                      # BM25 + dense e5-base
    python -m rag.retrieval.export_topk --dense-model bkai   # đổi mô hình dense
    python -m rag.retrieval.export_topk --no-dense           # chỉ BM25, không cần GPU
"""
import argparse
import csv
import glob
import json
from pathlib import Path

from rag.corpus.chunk import load_chunks
from rag.retrieval.eval_retrieval import DEFAULT_QUESTIONS, first_hit_rank, recall_at_k
from rag.retrieval.gold import load_questions, relevance

DEPTH = 10
OUT_DIR = Path("results/retrieval")
# Dense tốt nhất theo Bảng 1 (MRR@10) trong results/retrieval/TABLES.md
BEST_DENSE = "e5-base"


def export_rows(name, retriever, chunks, questions, depth=DEPTH):
    """Một dòng cho mỗi câu hỏi: top-`depth` chunk id + điểm + hạng của chunk đúng đầu tiên."""
    results = retriever.batch_search([q["question"] for q in questions], k=depth)
    rows, recalls = [], []
    for q, res in zip(questions, results):
        ranked = [i for i, _ in res]
        row = {"qid": q["id"], "retriever": name,
               "chunk_ids": [chunks[i]["id"] for i in ranked],
               "scores": [round(s, 4) for _, s in res],
               "first_hit": None}
        if q.get("answerable", True):
            rel = relevance(chunks, q)
            row["first_hit"] = first_hit_rank(ranked, rel)
            recalls.append([recall_at_k(ranked, rel, k) for k in range(1, depth + 1)])
        rows.append(row)
    curve = [sum(r[k] for r in recalls) / len(recalls) for k in range(depth)] if recalls else []
    return rows, curve, len(recalls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", nargs="+", default=None)
    ap.add_argument("--chunks", default="data/corpus/chunks.jsonl")
    ap.add_argument("--dense-model", default=BEST_DENSE)
    ap.add_argument("--extra-dense", nargs="*", default=["bkai"],
                    help="mô hình dense chỉ đưa vào recall_by_k.csv (không xuất top-10)")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    paths = args.questions or sorted(glob.glob(DEFAULT_QUESTIONS))
    questions = load_questions(paths)
    ids = [q["id"] for q in questions]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"Trùng id câu hỏi giữa các file: {dup}")
    chunks = load_chunks(args.chunks)

    from rag.retrieval.bm25 import BM25Retriever

    retrievers = [("bm25", "BM25", lambda: BM25Retriever(chunks), True)]
    if not args.no_dense:
        from rag.retrieval.dense import DenseRetriever

        retrievers.append(("dense", f"Dense {args.dense_model}",
                           lambda: DenseRetriever(chunks, model_key=args.dense_model), True))
        for key in args.extra_dense:
            if key != args.dense_model:
                retrievers.append((f"dense_{key}", f"Dense {key}",
                                   lambda key=key: DenseRetriever(chunks, model_key=key), False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, curves = [], {}
    n_ans = 0
    for name, label, build, export in retrievers:
        rows, curve, n_ans = export_rows(name, build(), chunks, questions)
        curves[label] = curve
        if export:
            all_rows += rows
        print(f"{label}: xong {len(rows)} câu", flush=True)

    with open(out_dir / "retrieval_top10.jsonl", "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "recall_by_k.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["retriever", "k", "recall", "n"])
        for label, curve in curves.items():
            for k, v in enumerate(curve, start=1):
                w.writerow([label, k, f"{v:.4f}", n_ans])

    print(f"\n{len(questions)} câu ({n_ans} có đáp án) từ {len(paths)} file, {len(chunks)} chunk")
    print("| Retriever | R@1 | R@3 | R@5 |\n|---|---|---|---|")
    for label, c in curves.items():
        print(f"| {label} | {c[0]:.3f} | {c[2]:.3f} | {c[4]:.3f} |")
    print(f"Đã ghi {out_dir / 'retrieval_top10.jsonl'} và {out_dir / 'recall_by_k.csv'}")


if __name__ == "__main__":
    main()
