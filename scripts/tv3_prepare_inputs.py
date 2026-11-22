"""tv3_prepare_inputs.py - Chuyển định dạng của repo (TV2) sang định dạng đầu vào của generate.py.

Đọc  : data/rag_test/questions_tv*.jsonl   {id, question, answer, answerable, type, gold_chunks, ...}
       data/corpus/chunks.jsonl            {id, doc_id, doc_name, article, text, ...}
Ghi  : outputs/rag_inputs/test.jsonl        {qid, question, answer, gold_chunk_ids, qtype, in_corpus}
       outputs/rag_inputs/chunks.jsonl      (thêm khóa chunk_id = id)
       outputs/rag_inputs/retrieval_top10.jsonl  {qid, retriever, chunk_ids}   (BM25 của TV2 cho MỌI câu, cả ngoài kho)

Chạy từ gốc repo:  python scripts/tv3_prepare_inputs.py [--retrievers bm25] [--chunks data/corpus/chunks.jsonl]
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.getcwd())


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", nargs="+", default=sorted(glob.glob("data/rag_test/questions_tv*.jsonl")))
    ap.add_argument("--chunks", default="data/corpus/chunks.jsonl")
    ap.add_argument("--out-dir", default="outputs/rag_inputs")
    ap.add_argument("--retrievers", nargs="+", default=["bm25"], choices=["bm25", "dense"])
    ap.add_argument("--dense-model", default="e5-base")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    chunks = read_jsonl(a.chunks)
    for c in chunks:
        c["chunk_id"] = c.get("chunk_id") or c["id"]
    test, seen = [], set()
    for p in a.questions:
        for r in read_jsonl(p):
            if r["id"] in seen:
                raise SystemExit(f"id trùng: {r['id']}")
            seen.add(r["id"])
            inc = bool(r["answerable"])
            test.append({"qid": r["id"], "question": r["question"], "answer": r["answer"] if inc else "",
                         "gold_chunk_ids": list(r.get("gold_chunks") or []) if inc else [],
                         "qtype": r["type"] if inc else "out_of_corpus", "in_corpus": inc,
                         "author": r.get("author")})
    print(f"{len(test)} câu ({sum(t['in_corpus'] for t in test)} trong kho) từ {len(a.questions)} file")
    empty = [t["qid"] for t in test if t["in_corpus"] and not t["gold_chunk_ids"]]
    if empty:
        print("[WARN] câu trong kho chưa có gold_chunks (kho thiếu chunk?):", empty)

    ret_rows = []
    qs = [t["question"] for t in test]
    for name in a.retrievers:
        if name == "bm25":
            from rag.retrieval.bm25 import BM25Retriever
            r = BM25Retriever(chunks)
        else:
            from rag.retrieval.dense import DenseRetriever  # noqa
            r = DenseRetriever(chunks, a.dense_model)
        for t, hits in zip(test, r.batch_search(qs, k=a.k)):
            ret_rows.append({"qid": t["qid"], "retriever": name, "chunk_ids": [chunks[i]["chunk_id"] for i, _ in hits]})
        print("retriever xong:", name)

    def dump(name, rows):
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    dump("test.jsonl", test); dump("chunks.jsonl", chunks); dump("retrieval_top10.jsonl", ret_rows)
    print("Đã ghi vào", a.out_dir)


if __name__ == "__main__":
    main()
