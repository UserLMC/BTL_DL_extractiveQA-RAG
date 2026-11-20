"""Đánh giá truy hồi: Recall@k và MRR@10 trên bộ câu hỏi test RAG.

Chỉ tính trên câu có đáp án trong kho (answerable=true).
- Recall@k : tỉ lệ evidence của câu hỏi có ít nhất một chunk liên quan nằm trong top-k,
             lấy trung bình trên các câu (câu một evidence thì là hit@k).
- MRR@10   : 1 / hạng của chunk liên quan đầu tiên trong top-10 (0 nếu không có).

Ví dụ:
    python -m rag.retrieval.eval_retrieval --retriever bm25
    python -m rag.retrieval.eval_retrieval --retriever dense --model e5-base
    python -m rag.retrieval.eval_retrieval --retriever bm25 --chunks data/corpus/chunks_w150.jsonl
    python -m rag.retrieval.eval_retrieval --retriever bm25 --grid      # thử k1, b
Kết quả từng câu (top-10 + hạng đúng) ghi ra results/retrieval/<tên>.json.
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from rag.corpus.chunk import load_chunks
from rag.retrieval.gold import load_questions, relevance

KS = (1, 3, 5, 10)
OUT_DIR = Path("results/retrieval")
DEFAULT_QUESTIONS = "data/rag_test/*.jsonl"


# ------------------------------------------------------------ chỉ số
def recall_at_k(ranked, rel_sets, k):
    if not rel_sets:
        return 0.0
    top = set(ranked[:k])
    return sum(1 for rel in rel_sets if rel & top) / len(rel_sets)


def mrr_at_k(ranked, rel_sets, k=10):
    relevant = set().union(*rel_sets) if rel_sets else set()
    for rank, idx in enumerate(ranked[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def first_hit_rank(ranked, rel_sets):
    relevant = set().union(*rel_sets) if rel_sets else set()
    return next((r for r, i in enumerate(ranked, start=1) if i in relevant), None)


def summarize(rows):
    out = {f"R@{k}": sum(r[f"R@{k}"] for r in rows) / len(rows) for k in KS}
    out["MRR@10"] = sum(r["MRR@10"] for r in rows) / len(rows)
    out["n"] = len(rows)
    return out


# ------------------------------------------------------------ chạy đánh giá
def build_retriever(args, chunks):
    if args.retriever == "bm25":
        from rag.retrieval.bm25 import BM25Retriever

        return BM25Retriever(chunks, k1=args.k1, b=args.b, token_mode=args.token_mode)
    from rag.retrieval.dense import DenseRetriever

    return DenseRetriever(chunks, model_key=args.model)


def evaluate(retriever, chunks, questions, depth=10):
    qs = [q for q in questions if q.get("answerable", True)]
    results = retriever.batch_search([q["question"] for q in qs], k=max(depth, max(KS)))
    rows = []
    for q, res in zip(qs, results):
        ranked = [i for i, _ in res]
        rel = relevance(chunks, q)
        row = {"id": q["id"], "type": q.get("type"), "question": q["question"],
               "gold": sorted({chunks[i]["id"] for r in rel for i in r}),
               "top": [{"id": chunks[i]["id"], "score": round(s, 4)} for i, s in res],
               "first_hit": first_hit_rank(ranked, rel)}
        for k in KS:
            row[f"R@{k}"] = recall_at_k(ranked, rel, k)
        row["MRR@10"] = mrr_at_k(ranked, rel, 10)
        rows.append(row)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"overall": summarize(rows),
            "by_type": {t: summarize(rs) for t, rs in sorted(by_type.items(), key=str)},
            "per_question": rows}


def fmt_row(name, s):
    return f"| {name} | " + " | ".join(f"{s[f'R@{k}']:.3f}" for k in KS) + f" | {s['MRR@10']:.3f} | {s['n']} |"


HEADER = "| Cấu hình | " + " | ".join(f"R@{k}" for k in KS) + " | MRR@10 | n |\n|" + "---|" * (len(KS) + 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", nargs="+", default=None)
    ap.add_argument("--chunks", default="data/corpus/chunks.jsonl")
    ap.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--model", default="e5-base", help="khoá trong rag.retrieval.dense.MODELS")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--token-mode", choices=["word", "syllable"], default="word")
    ap.add_argument("--grid", action="store_true", help="BM25: thử lưới k1 x b")
    ap.add_argument("--errors", type=int, default=0, help="in N câu truy hồi sai (top-k)")
    ap.add_argument("--errors-k", type=int, default=3)
    ap.add_argument("--save-index", default=None, help="dense: thư mục lưu FAISS index")
    args = ap.parse_args()

    questions = load_questions(args.questions or sorted(glob.glob(DEFAULT_QUESTIONS)))
    chunks = load_chunks(args.chunks)
    chunk_tag = Path(args.chunks).stem

    if args.grid:
        from rag.retrieval.bm25 import BM25Retriever

        print(HEADER)
        for k1 in (0.9, 1.2, 1.5, 2.0):
            for b in (0.3, 0.5, 0.75, 0.9):
                r = BM25Retriever(chunks, k1=k1, b=b, token_mode=args.token_mode)
                print(fmt_row(f"k1={k1}, b={b}", evaluate(r, chunks, questions)["overall"]), flush=True)
        return

    retriever = build_retriever(args, chunks)
    res = evaluate(retriever, chunks, questions)
    name = f"{retriever.name}__{chunk_tag}"
    res["config"] = {"retriever": retriever.name, "chunks": args.chunks, "n_chunks": len(chunks)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(HEADER)
    print(fmt_row(name, res["overall"]))
    for t, s in res["by_type"].items():
        print(fmt_row(f"  loại={t}", s))

    if args.errors:
        miss = [r for r in res["per_question"] if r[f"R@{args.errors_k}"] < 1]
        print(f"\n{len(miss)} câu chưa lấy đủ evidence trong top-{args.errors_k}:")
        for r in miss[:args.errors]:
            print(f"- [{r['id']}|{r['type']}] {r['question']}\n    gold={r['gold']} "
                  f"hạng đúng đầu tiên={r['first_hit']}\n    top3={[t['id'] for t in r['top'][:3]]}")

    if args.save_index and args.retriever == "dense":
        retriever.save_index(args.save_index)
        print(f"Đã lưu index vào {args.save_index}")


if __name__ == "__main__":
    main()
