"""
generate.py – Pipeline sinh câu trả lời RAG (TV3).

    câu hỏi + top-k chunk (BM25 / dense / oracle / none) -> prompt -> LLM -> jsonl

Định dạng đầu vào (thống nhất với TV2):
  test   (rag_test_v1.jsonl): {"qid", "question", "answer", "gold_chunk_ids": [...],
                               "qtype": "fact|number|condition|multi", "in_corpus": true|false}
  chunks (chunks.jsonl)     : {"chunk_id", "doc_id", "doc_name", "article", "clause"?, "text"}
  retrieval (retrieval_top10.jsonl): {"qid", "retriever": "bm25"|"dense", "chunk_ids": [top-10 theo thứ hạng]}

Đầu ra (1 dòng / (qid, config)):
  {"qid", "config", "retriever", "k", "prompt_version", "chunk_ids", "hit", "model_id",
   "answer", "latency_s", "n_new_tokens"}

Ví dụ trên Kaggle (T4):
  python -m rag.generation.generate --configs S0 S1 S2 S3 S4 S5 \
      --test data/rag_test_v1.jsonl --chunks data/chunks.jsonl \
      --retrieval data/retrieval_top10.jsonl --out outputs/generations.jsonl \
      --backend hf --model Qwen/Qwen2.5-7B-Instruct
  python -m rag.generation.generate --score outputs/generations.jsonl --test data/rag_test_v1.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from rag.generation.prompts import (
    REFUSAL_TEXT,
    answer_for_scoring,
    build_messages,
    extract_citations,
    _fold,
    is_refusal,
    refusal_kind,
)

# Dùng lại metric của TV1: repo hiện đặt ở qa/qa_metrics.py (tên cũ: eval/qa_metrics.py).
# tv3/ nằm trong repo nên thư mục gốc repo (chứa qa/) là cha của tv3/.
_TV3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_DIR = os.path.dirname(_TV3_DIR)
for _p in (_REPO_DIR, _TV3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from qa.qa_metrics import compute_f1, get_tokens, normalize_answer
except ImportError:  # pragma: no cover - tên cũ
    from eval.qa_metrics import compute_f1, get_tokens, normalize_answer


# ---------------------------------------------------------------------------
# Ma trận cấu hình (mục 7 kế hoạch). S6 là reader XLM-R của TV1, không chạy ở đây.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenConfig:
    name: str
    retriever: str        # "none" | "bm25" | "dense" | "oracle"
    k: int
    prompt_version: str   # "v1" | "v2"
    style: str = "full"   # "full" (câu hoàn chỉnh, từ 24/11) | "short" (bản cũ 22/11)


CONFIGS: Dict[str, GenConfig] = {
    "S0": GenConfig("S0", "none", 0, "v2"),
    "S1": GenConfig("S1", "bm25", 3, "v2"),
    "S2": GenConfig("S2", "dense", 3, "v2"),
    "S3": GenConfig("S3", "oracle", 0, "v2"),   # k = số chunk gold của câu đó
    "S4": GenConfig("S4", "dense", 5, "v2"),
    "S5": GenConfig("S5", "dense", 3, "v1"),
}
# Cấu hình phụ cho thử nghiệm prompt (A/B trên ~10 câu, 24/11); KHÔNG thuộc ma trận S0-S6.
AB_CONFIGS: Dict[str, GenConfig] = {
    "S1_v1": GenConfig("S1_v1", "bm25", 3, "v1"),                 # prompt v1 (không từ chối/trích dẫn)
    "S1_short": GenConfig("S1_short", "bm25", 3, "v2", "short"),  # v2 kiểu cũ: đáp án trơ trụi
}
ALL_CONFIGS: Dict[str, GenConfig] = {**CONFIGS, **AB_CONFIGS}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def token_recall(gold: str, pred: str) -> float:
    """
    Tỉ lệ âm tiết của đáp án chuẩn xuất hiện trong câu trả lời (không phạt câu trả lời dài).
    Bổ sung cho token-F1: gold dạng span ngắn ("14 tín chỉ") còn LLM trả lời cả câu thì F1 thấp
    dù câu trả lời đúng. Gold rỗng -> 1.0 nếu pred cũng rỗng, ngược lại 0.0 (giống compute_f1).
    """
    g, p = get_tokens(gold), get_tokens(pred)
    if not g or not p:
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    return sum(common.values()) / len(g)


def contains_gold(gold: str, pred: str) -> bool:
    """Câu trả lời (đã chuẩn hóa) chứa nguyên chuỗi đáp án chuẩn (đã chuẩn hóa)."""
    g, p = normalize_answer(gold), normalize_answer(pred)
    return bool(g) and f" {g} " in f" {p} "


def log_wandb(project: Optional[str], name: str, config: Dict, metrics: Dict) -> None:
    """Log tóm tắt lên Weights & Biases nếu có --wandb-project; lỗi/thiếu thư viện thì chỉ cảnh báo."""
    if not project:
        return
    try:
        import wandb
    except ImportError:  # pragma: no cover
        print("[cảnh báo] chưa cài wandb, bỏ qua log W&B", file=sys.stderr)
        return
    run = wandb.init(project=project, name=name, config=config, reinit=True)
    for cfg_name, m in metrics.items():
        run.log({f"{cfg_name}/{k}": v for k, v in m.items() if v is not None})
    run.summary.update({f"{c}/{k}": v for c, m in metrics.items() for k, v in m.items() if v is not None})
    run.finish()


def load_retrieval(path: Optional[str]) -> Dict[Tuple[str, str], List[str]]:
    """{(qid, retriever): [chunk_id theo thứ hạng]}"""
    if not path:
        return {}
    return {(str(r["qid"]), r["retriever"]): list(r["chunk_ids"]) for r in read_jsonl(path)}


def select_chunks(
    cfg: GenConfig,
    sample: Dict,
    chunk_index: Dict[str, Dict],
    retrieval: Dict[Tuple[str, str], List[str]],
) -> List[Dict]:
    qid = str(sample["qid"])
    if cfg.retriever == "none":
        return []
    if cfg.retriever == "oracle":
        ids = list(sample.get("gold_chunk_ids") or [])
    else:
        key = (qid, cfg.retriever)
        if key not in retrieval:
            raise KeyError(f"Thiếu kết quả truy hồi cho qid={qid}, retriever={cfg.retriever}")
        ids = retrieval[key][: cfg.k]
    missing = [c for c in ids if c not in chunk_index]
    if missing:
        raise KeyError(f"chunk_id không có trong chunks.jsonl: {missing}")
    return [chunk_index[c] for c in ids]


# ---------------------------------------------------------------------------
# Backend LLM
# ---------------------------------------------------------------------------
class HFBackend:
    """Qwen2.5-Instruct lượng tử hóa 4-bit NF4 (bitsandbytes), giải mã greedy."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", max_new_tokens: int = 200,
                 load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,   # T4 không hỗ trợ bf16
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,             # greedy -> tái lập
                temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0,
                pad_token_id=self.tok.eos_token_id,
            )
        new = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip(), int(new.shape[0])


def _syllables(s: str) -> List[str]:
    s = unicodedata.normalize("NFC", s or "").lower()
    return re.findall(r"\w+", s)


_STOP = set("là của và các có được cho trong theo với không những này đó thì khi nào gì bao nhiêu "
            "sinh viên ai như thế phải".split())


class MockBackend:
    """
    Backend giả lập, KHÔNG dùng LLM – để kiểm thử pipeline offline (CI, máy không GPU).
    Chọn câu trong ngữ cảnh trùng âm tiết với câu hỏi nhiều nhất, gắn trích dẫn [i];
    nếu độ trùng < min_overlap hoặc không có ngữ cảnh -> trả câu từ chối.
    """

    model_id = "mock-extractive"

    def __init__(self, min_overlap: float = 0.5):
        self.min_overlap = min_overlap

    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        user = messages[-1]["content"]
        q = user.split("Câu hỏi:")[-1].split("Trả lời:")[0]
        qtok = {t for t in _syllables(q) if t not in _STOP}
        blocks = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[\d+\]|\n\nCâu hỏi:)", user, flags=re.S)
        best, best_score = None, 0.0
        for idx, text in blocks:
            for sent in re.split(r"(?<=[.;])\s+", text.strip()):
                stoks = set(_syllables(sent))
                score = len(qtok & stoks) / max(1, len(qtok))
                if score > best_score:
                    best, best_score = (idx, sent.strip()), score
        if best is None or best_score < self.min_overlap:
            ans = REFUSAL_TEXT + "."
        else:
            ans = f"{best[1].rstrip('.')} [{best[0]}]."
        return ans, len(ans.split())


def make_backend(name: str, model_id: str, max_new_tokens: int):
    if name == "hf":
        return HFBackend(model_id, max_new_tokens)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"backend không hỗ trợ: {name}")


# ---------------------------------------------------------------------------
# Chạy sinh
# ---------------------------------------------------------------------------
def done_keys(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    return {(str(r["qid"]), r["config"]) for r in read_jsonl(out_path)}


def run_generation(
    test: List[Dict],
    chunks: List[Dict],
    retrieval: Dict[Tuple[str, str], List[str]],
    config_names: Iterable[str],
    backend,
    out_path: str,
    limit: Optional[int] = None,
    log_every: int = 10,
) -> int:
    """Sinh cho mọi (qid, config) chưa có trong out_path. Trả về số dòng mới ghi."""
    # chunks.jsonl của TV2 dùng khóa "id"; chấp nhận cả "chunk_id".
    chunks = [{**c, "chunk_id": c.get("chunk_id") or c["id"]} for c in chunks]
    chunk_index = {c["chunk_id"]: c for c in chunks}
    # Không trộn kết quả của 2 mô hình trong cùng một file (vd. đổi 7B -> 3B giữa chừng).
    if os.path.exists(out_path):
        prev = {r.get("model_id") for r in read_jsonl(out_path)}
        cur = getattr(backend, "model_id", "?")
        if prev and prev != {cur}:
            raise ValueError(
                f"{out_path} đã có kết quả của mô hình {sorted(map(str, prev))}, "
                f"nhưng backend hiện tại là '{cur}'. Dùng --out khác cho mô hình mới."
            )
    done = done_keys(out_path)
    samples = test[:limit] if limit else test
    n_new = 0
    for name in config_names:
        cfg = ALL_CONFIGS[name]
        for s in samples:
            qid = str(s["qid"])
            if (qid, name) in done:
                continue
            ctx = select_chunks(cfg, s, chunk_index, retrieval)
            msgs = build_messages(s["question"], ctx, version=cfg.prompt_version,
                                  closed_book=(cfg.retriever == "none"), style=cfg.style)
            t0 = time.time()
            answer, n_tok = backend.generate(msgs)
            gold = set(s.get("gold_chunk_ids") or [])
            ids = [c["chunk_id"] for c in ctx]
            append_jsonl(out_path, {
                "qid": qid,
                "config": name,
                "retriever": cfg.retriever,
                "k": len(ctx),
                "prompt_version": cfg.prompt_version,
                "chunk_ids": ids,
                "hit": (bool(gold & set(ids)) if gold and cfg.retriever != "none" else None),
                "model_id": getattr(backend, "model_id", "?"),
                "answer": answer,
                "latency_s": round(time.time() - t0, 3),
                "n_new_tokens": n_tok,
            })
            n_new += 1
            if log_every and n_new % log_every == 0:
                print(f"[{name}] {n_new} câu mới ...", flush=True)
    return n_new


# ---------------------------------------------------------------------------
# Chấm tự động: token-F1, từ chối, trích dẫn
# ---------------------------------------------------------------------------
# Cụm từ chối đứng ở BẤT KỲ vị trí nào của câu (kể cả "Chủ đề ... không liên quan ... không có thông tin").
# Chỉ là chỉ báo CẬN TRÊN: is_refusal() (đứng đầu câu, ngắn) mới là định nghĩa chính; câu trong kho có thể
# chứa "không có thông tin ..." ở giữa câu mà vẫn trả lời thật. Dùng để đọc S5 (prompt không có lệnh từ chối).
_ANY_REFUSAL_PHRASES = ("khong tim thay thong tin", "khong co thong tin", "khong de cap",
                        "khong lien quan", "khong duoc cung cap")


def refusal_anywhere(text: str) -> bool:
    f = _fold(text or "")
    return any(p in f for p in _ANY_REFUSAL_PHRASES)


def score_row(row: Dict, sample: Dict) -> Dict:
    in_corpus = bool(sample.get("in_corpus", bool(sample.get("answer"))))
    gold = sample.get("answer", "") if in_corpus else ""
    ans = row["answer"]
    refused = is_refusal(ans)
    cites = extract_citations(ans)
    scored = answer_for_scoring(ans)
    return {
        "token_f1": compute_f1(gold, scored),
        "token_recall": token_recall(gold, scored),
        "contains_gold": contains_gold(gold, scored) if in_corpus else False,
        "partial_refusal": refusal_kind(ans) == "partial",
        "refused": refused,
        "refusal_strict": is_refusal(ans, strict=True),
        "refusal_any": refusal_anywhere(ans),
        "in_corpus": in_corpus,
        # trích dẫn ngoài phạm vi [1..k] = chắc chắn sai (cờ "trích dẫn sai" cấp tự động)
        "bad_citation": any(c < 1 or c > row["k"] for c in cites) if row["k"] else bool(cites),
        "has_citation": bool(cites),
    }


def summarize(rows: List[Dict], test: List[Dict]) -> Dict[str, Dict]:
    by_qid = {str(s["qid"]): s for s in test}
    agg: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        agg[r["config"]].append({**score_row(r, by_qid[str(r["qid"])]), "hit": r.get("hit"),
                                 "qtype": by_qid[str(r["qid"])].get("qtype", "?")})
    out = {}
    for cfg, items in sorted(agg.items()):
        inc = [x for x in items if x["in_corpus"]]
        ooc = [x for x in items if not x["in_corpus"]]
        hits = [x for x in inc if x["hit"] is True]
        miss = [x for x in inc if x["hit"] is False]
        avg = lambda xs, k: round(100 * sum(float(x[k]) for x in xs) / len(xs), 2) if xs else None
        out[cfg] = {
            "n": len(items),
            "token_F1": avg(items, "token_f1"),
            "token_F1_in_corpus": avg(inc, "token_f1"),
            "token_recall_in_corpus": avg(inc, "token_recall"),
            "contains_gold_in_corpus": avg(inc, "contains_gold"),
            "partial_refusal_rate": avg(items, "partial_refusal"),
            "refusal_acc_out_of_corpus": avg(ooc, "refused"),      # từ chối đúng
            "false_refusal_in_corpus": avg(inc, "refused"),        # từ chối sai
            # cận trên: có cụm từ chối ở bất kỳ vị trí nào (xem refusal_anywhere); S5 cần đọc cột này
            "refusal_any_phrase_out_of_corpus": avg(ooc, "refusal_any"),
            "refusal_any_phrase_in_corpus": avg(inc, "refusal_any"),
            "refusal_format_strict": avg([x for x in items if x["refused"]], "refusal_strict"),
            "citation_rate": avg([x for x in inc if not x["refused"]], "has_citation"),
            "bad_citation_rate": avg([x for x in items if not x["refused"]], "bad_citation"),
            "token_F1_hit": avg(hits, "token_f1"),
            "token_F1_miss": avg(miss, "token_f1"),
            # F1 theo loại câu (chỉ câu trong kho): fact / number / condition / multi
            "token_F1_by_type": {t: avg([x for x in inc if x["qtype"] == t], "token_f1")
                                 for t in sorted({x["qtype"] for x in inc})},
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Sinh câu trả lời RAG (S0–S5) và chấm token-F1.")
    ap.add_argument("--test", required=True)
    ap.add_argument("--chunks")
    ap.add_argument("--retrieval")
    ap.add_argument("--out", default="outputs/generations.jsonl")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(ALL_CONFIGS))
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qids", nargs="+", help="chỉ chạy/chấm các qid này (vd. thử A/B prompt trên 10 câu)")
    ap.add_argument("--score", metavar="GEN_JSONL", help="chỉ chấm file sinh có sẵn")
    ap.add_argument("--wandb-project", help="nếu đặt, log bảng tóm tắt lên Weights & Biases")
    a = ap.parse_args(argv)

    test = read_jsonl(a.test)
    if a.qids:
        test = [t for t in test if str(t["qid"]) in set(a.qids)]
        if not test:
            raise SystemExit("--qids không khớp câu nào trong --test")
    if a.score:
        res = summarize(read_jsonl(a.score), test)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        log_wandb(a.wandb_project, "tv3-score", {"gen": a.score, "test": a.test}, res)
        return
    backend = make_backend(a.backend, a.model, a.max_new_tokens)
    n = run_generation(test, read_jsonl(a.chunks), load_retrieval(a.retrieval),
                       a.configs, backend, a.out, limit=a.limit)
    print(f"Đã ghi {n} dòng mới vào {a.out}")
    res = summarize(read_jsonl(a.out), test)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    log_wandb(a.wandb_project, "tv3-generation",
              {"model": getattr(backend, "model_id", a.model), "configs": a.configs,
               "max_new_tokens": a.max_new_tokens, "test": a.test}, res)


if __name__ == "__main__":
    main()
