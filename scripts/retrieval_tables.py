"""Tái tạo các bảng kết quả truy hồi trong báo cáo -> results/retrieval/TABLES.md

    python scripts/retrieval_tables.py                 # mọi bảng
    python scripts/retrieval_tables.py --no-dense      # chỉ BM25 (không cần GPU)

Bảng 1: so sánh retriever (BM25 / dense) trên chunk chia theo Điều.
Bảng 2: ablation cách chia đoạn (Điều vs cửa sổ 150/300 từ) với BM25 và dense tốt nhất.
Bảng 3: BM25 tách từ (pyvi) vs tách âm tiết.
Bảng 4: độ nhạy của BM25 với k1, b (tham khảo — không dùng để chọn cấu hình).
Bảng 5: kết quả theo loại câu hỏi của retriever tốt nhất.
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.corpus.chunk import load_chunks  # noqa: E402
from rag.retrieval.bm25 import BM25Retriever  # noqa: E402
from rag.retrieval.eval_retrieval import DEFAULT_QUESTIONS, HEADER, evaluate, fmt_row  # noqa: E402
from rag.retrieval.gold import load_questions  # noqa: E402

CHUNKINGS = {"Điều (≤300 từ)": "data/corpus/chunks.jsonl",
             "Cửa sổ 150 từ": "data/corpus/chunks_w150.jsonl",
             "Cửa sổ 300 từ": "data/corpus/chunks_w300.jsonl"}
DENSE_MODELS = ["e5-base", "bkai"]


def dense(chunks, key):
    from rag.retrieval.dense import DenseRetriever

    return DenseRetriever(chunks, model_key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out", default="results/retrieval/TABLES.md")
    args = ap.parse_args()

    questions = load_questions(sorted(glob.glob(DEFAULT_QUESTIONS)))
    n_ans = sum(1 for q in questions if q.get("answerable", True))
    base = load_chunks(CHUNKINGS["Điều (≤300 từ)"])
    out = [f"# Kết quả truy hồi\n\n{len(questions)} câu hỏi ({n_ans} có đáp án trong kho), "
           f"{len(base)} chunk chia theo Điều.\n"]

    # Bảng 1
    out += ["## Bảng 1. So sánh retriever (chunk theo Điều)\n", HEADER]
    results = {"BM25": evaluate(BM25Retriever(base), base, questions)}
    if not args.no_dense:
        for key in DENSE_MODELS:
            results[f"Dense {key}"] = evaluate(dense(base, key), base, questions)
    for name, r in results.items():
        out.append(fmt_row(name, r["overall"]))
    best_dense = max((k for k in results if k.startswith("Dense")),
                     key=lambda k: results[k]["overall"]["MRR@10"], default=None)
    best = max(results, key=lambda k: results[k]["overall"]["MRR@10"])

    # Bảng 2
    out += ["\n## Bảng 2. Ablation cách chia đoạn\n", HEADER]
    for cname, path in CHUNKINGS.items():
        if not Path(path).exists():
            out.append(f"| {cname} | (chưa tạo {path}) |")
            continue
        chunks = load_chunks(path)
        out.append(fmt_row(f"BM25 – {cname}", evaluate(BM25Retriever(chunks), chunks, questions)["overall"]))
        if best_dense:
            key = best_dense.split()[-1]
            out.append(fmt_row(f"{best_dense} – {cname}",
                               evaluate(dense(chunks, key), chunks, questions)["overall"]))

    # Bảng 3
    out += ["\n## Bảng 3. BM25: tách từ vs tách âm tiết\n", HEADER]
    for mode in ("word", "syllable"):
        r = evaluate(BM25Retriever(base, token_mode=mode), base, questions)
        out.append(fmt_row(f"BM25 – {mode}", r["overall"]))

    # Bảng 4
    out += ["\n## Bảng 4. Độ nhạy BM25 theo k1, b (tham khảo)\n", HEADER]
    for k1 in (0.9, 1.2, 1.5, 2.0):
        for b in (0.3, 0.75, 0.9):
            r = evaluate(BM25Retriever(base, k1=k1, b=b), base, questions)
            out.append(fmt_row(f"k1={k1}, b={b}", r["overall"]))

    # Bảng 5
    out += [f"\n## Bảng 5. Theo loại câu hỏi – {best}\n", HEADER]
    for t, s in results[best]["by_type"].items():
        out.append(fmt_row(str(t), s))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
