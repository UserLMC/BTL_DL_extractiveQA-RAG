"""Xuất các câu truy hồi sai ra Markdown để phân loại lỗi bằng tay.

    python scripts/retrieval_errors.py results/retrieval/bm25_word_k1=1.5_b=0.75__chunks.json --k 3

Với mỗi câu chưa lấy đủ evidence trong top-k: câu hỏi, đoạn trích đúng, chunk đúng và
3 chunk được xếp đầu (trích 300 ký tự). Cột "Loại lỗi" để điền tay, gợi ý:
  đồng nghĩa / con số / nhiều Điều / nhiễu từ văn bản khác / lỗi OCR / câu hỏi mơ hồ
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.corpus.chunk import load_chunks  # noqa: E402
from rag.retrieval.gold import load_questions  # noqa: E402
import glob  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    chunks = {c["id"]: c for c in load_chunks(res["config"]["chunks"])}
    qs = {q["id"]: q for q in load_questions(sorted(glob.glob("data/rag_test/*.jsonl")))}
    misses = [r for r in res["per_question"] if r[f"R@{args.k}"] < 1]

    lines = [f"# Lỗi truy hồi – {res['config']['retriever']} (top-{args.k})\n",
             f"{len(misses)}/{len(res['per_question'])} câu chưa lấy đủ evidence.\n",
             "| id | loại câu | hạng đúng đầu tiên | Loại lỗi (điền tay) |", "|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['type']} | {r['first_hit']} | |" for r in misses]
    for r in misses:
        q = qs[r["id"]]
        lines += [f"\n## {r['id']} – {q['question']}\n",
                  f"- Đáp án: {q['answer']}",
                  *[f"- Evidence ({e['doc_id']}): «{e['text']}»" for e in q.get("evidence", [])],
                  f"- Chunk đúng: {', '.join(r['gold'])} — hạng đầu tiên: {r['first_hit']}",
                  "- Top-3 được chọn:"]
        for t in r["top"][:3]:
            c = chunks[t["id"]]
            lines.append(f"  - `{t['id']}` ({t['score']}): {c['text'][:300].replace(chr(10), ' ')}…")
    out = Path(args.out or Path(args.result_json).with_suffix(".errors.md"))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(misses)} câu sai -> {out}")


if __name__ == "__main__":
    main()
