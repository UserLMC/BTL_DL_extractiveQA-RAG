"""Đáp án vàng cho truy hồi: định vị đoạn trích (evidence) trong văn bản gốc.

Mỗi câu hỏi test ghi `evidence`: danh sách {"doc_id", "text"} với text chép NGUYÊN VĂN
từ data/corpus/text/<doc_id>.txt. Một chunk được coi là liên quan tới một evidence nếu
cùng văn bản và phủ >= MIN_COVER phần ký tự của evidence. Cách này không phụ thuộc vào
cách chia đoạn, nên cùng bộ câu hỏi dùng được cho ablation chia theo Điều / cửa sổ.

Kiểm tra bộ câu hỏi và điền gold_chunks (theo chunks.jsonl chia theo Điều):
    python -m rag.retrieval.gold data/rag_test/questions_tv2.jsonl --write
"""
import argparse
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from rag.corpus.chunk import load_chunks, load_doc_text

MIN_COVER = 0.5


@lru_cache(maxsize=None)
def _normalized_doc(doc_id):
    """Văn bản đã gộp khoảng trắng + bảng ánh xạ vị trí về văn bản gốc."""
    text, _ = load_doc_text(doc_id)
    text = unicodedata.normalize("NFC", text)
    out, idx_map, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch.lower())
        idx_map.append(i)
    return "".join(out), idx_map


def locate(doc_id, evidence_text):
    """Vị trí (start, end) của evidence trong văn bản gốc, hoặc None nếu không thấy."""
    norm, idx_map = _normalized_doc(doc_id)
    needle = re.sub(r"\s+", " ", unicodedata.normalize("NFC", evidence_text)).strip().lower()
    pos = norm.find(needle)
    if pos < 0 or not needle:
        return None
    return idx_map[pos], idx_map[pos + len(needle) - 1] + 1


def evidence_spans(question):
    spans = []
    for ev in question.get("evidence", []):
        span = locate(ev["doc_id"], ev["text"])
        if span is None:
            raise ValueError(f"{question['id']}: không tìm thấy evidence trong {ev['doc_id']}: "
                             f"{ev['text'][:80]!r}")
        spans.append((ev["doc_id"], *span))
    return spans


def covers(chunk, span):
    doc_id, s, e = span
    if chunk["doc_id"] != doc_id:
        return False
    overlap = min(chunk["end"], e) - max(chunk["start"], s)
    return overlap >= MIN_COVER * (e - s)


def relevance(chunks, question):
    """Với mỗi evidence: tập chỉ số các chunk liên quan."""
    return [{i for i, c in enumerate(chunks) if covers(c, sp)} for sp in evidence_spans(question)]


def load_questions(paths):
    qs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            qs.extend(json.loads(line) for line in f if line.strip())
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--write", action="store_true", help="ghi gold_chunks vào lại file")
    args = ap.parse_args()

    chunks = load_chunks()
    n_err = 0
    for path in args.files:
        qs = load_questions([path])
        ids = [q["id"] for q in qs]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            print(f"[LỖI] {path}: trùng id {sorted(dup)}")
            n_err += 1
        for q in qs:
            for field in ("id", "question", "answer", "answerable", "type"):
                if field not in q:
                    print(f"[LỖI] {q.get('id')}: thiếu trường {field}")
                    n_err += 1
            if not q.get("answerable", True):
                q["gold_chunks"] = []
                continue
            try:
                rel = relevance(chunks, q)
            except ValueError as err:
                print(f"[LỖI] {err}")
                n_err += 1
                continue
            if not rel:
                print(f"[LỖI] {q['id']}: câu có đáp án nhưng thiếu evidence")
                n_err += 1
            for k, r in enumerate(rel):
                if not r:
                    # đoạn trích có trong văn bản nhưng không chunk nào chứa nó:
                    # kho bị mất nội dung (lỗi chia đoạn), câu này không retriever nào trả lời được
                    print(f"[LỖI] {q['id']}: evidence {k + 1} không thuộc chunk nào trong kho")
                    n_err += 1
            q["gold_chunks"] = sorted({chunks[i]["id"] for r in rel for i in r})
        if args.write:
            with open(path, "w", encoding="utf-8") as f:
                for q in qs:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"{path}: {len(qs)} câu, "
              f"{sum(1 for q in qs if q.get('answerable', True))} có đáp án")
    raise SystemExit(1 if n_err else 0)


if __name__ == "__main__":
    main()
