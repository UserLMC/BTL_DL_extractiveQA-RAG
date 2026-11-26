"""Chia văn bản quy chế thành chunk.

Hai chế độ:
- dieu      : mỗi Điều là một chunk; Điều dài hơn --max-words thì cắt thành nhiều phần
              (~300 từ, gối ~50 từ), mỗi phần giữ dòng tiêu đề "Điều N. ..." ở đầu.
- window    : cửa sổ cố định --max-words từ, gối --overlap từ, bỏ qua cấu trúc
              (dùng cho ablation).

Mỗi chunk lưu vị trí ký tự (start, end) trong data/corpus/text/<doc_id>.txt để
đánh giá truy hồi theo đoạn trích (evidence) độc lập với cách chia.

    python -m rag.corpus.chunk                               # theo Điều -> chunks.jsonl
    python -m rag.corpus.chunk --mode window --max-words 150 # -> chunks_w150.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from rag.corpus.download import load_sources
from rag.corpus.parse_pdf import TEXT_DIR

CHUNK_DIR = Path("data/corpus")

CHAPTER_RE = re.compile(r"^(Chương|CHƯƠNG)\s+([IVXLC]+|\d+)\b\.?\s*(.*)$")
# "Điều 3." / "Điều 3:" / "Điều 3" cuối dòng / "Điều 9 Tuyển sinh" (OCR mất dấu chấm)
# — không khớp trích dẫn "Điều 3, chương II", "Điều 41 của Quy chế này".
ARTICLE_RE = re.compile(r"^(Điều|ĐIỀU)\s+(\d+)(?:\s*[.:]\s*(.*)|\s+(.*)|$)")


def match_article(p):
    """(số Điều, tiêu đề) nếu đoạn p là đầu một Điều, ngược lại None."""
    m = ARTICLE_RE.match(p)
    if not m:
        return None
    if m.group(3) is None and m.group(4) is not None:
        # thiếu dấu chấm: chỉ nhận khi chữ tiếp theo viết hoa ("Điều 9 Tuyển sinh")
        if not m.group(4)[:1].isupper():
            return None
        return int(m.group(2)), m.group(4)
    return int(m.group(2)), m.group(3) or ""
# Văn bản hướng dẫn không chia Điều mà chia mục "1. Nguyên tắc chung"
SECTION_RE = re.compile(r"^(\d+)\.\s+(\S.*)$")
TAIL_RE = re.compile(r"^(Nơi nhận|NƠI NHẬN)\b")
# Sau chữ ký: dòng "(Ban hành kèm theo Quyết định ...)" mở đầu văn bản đính kèm
ATTACH_RE = re.compile(r"ban hành kèm theo", re.IGNORECASE)


def load_doc_text(doc_id):
    """Trả về (text, danh sách (start, end, đoạn)) với text = các đoạn nối bằng '\\n'."""
    paras = (TEXT_DIR / f"{doc_id}.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
    spans, pos = [], 0
    for p in paras:
        spans.append((pos, pos + len(p), p))
        pos += len(p) + 1
    return "\n".join(paras), spans


def _word_spans(text, start, end):
    return [(m.start() + start, m.end() + start) for m in re.finditer(r"\S+", text[start:end])]


def _windows(word_spans, size, overlap):
    """Cắt danh sách vị trí từ thành các cửa sổ [i, j) gối nhau."""
    if len(word_spans) <= size:
        return [(0, len(word_spans))]
    step = size - overlap
    out, i = [], 0
    while True:
        j = min(i + size, len(word_spans))
        out.append((i, j))
        if j == len(word_spans):
            return out
        i += step


# ------------------------------------------------------------ theo Điều
def split_structure(doc_id, paras):
    """Tách văn bản thành các khối: phần mở đầu, từng Điều, phụ lục.

    Văn bản thường gồm Quyết định ban hành (Điều 1-3) rồi tới Quy chế/Quy định
    (Điều 1...). Khi số Điều quay lại nhỏ hơn thì coi là sang phần mới (part).
    """
    has_articles = any(match_article(p) for _, _, p in paras)

    def match_unit(p):
        if has_articles:
            return match_article(p)
        m = SECTION_RE.match(p)
        return (int(m.group(1)), m.group(2)) if m else None

    blocks = []
    part, chapter, chapter_title = 0, None, None
    last_article = 0
    cur = {"kind": "preamble", "part": 0, "chapter": None, "chapter_title": None,
           "article": None, "article_title": None, "start": paras[0][0], "end": paras[0][1]}
    pending_chapter_title = False

    in_appendix = False
    for start, end, p in paras:
        # trong phụ lục, "1." hay "Điều" là nội dung bảng chứ không phải đơn vị mới
        m_ch = None if in_appendix else CHAPTER_RE.match(p)
        m_art = None if in_appendix else match_unit(p)
        if TAIL_RE.match(p):
            # "Nơi nhận", chữ ký: không có nội dung, bỏ tới đơn vị kế tiếp
            blocks.append(cur)
            cur = {"kind": "tail", "start": start, "end": end}
            continue
        if cur["kind"] == "tail" and ATTACH_RE.search(p):
            # Văn bản đính kèm sau chữ ký. Thường ngay sau là "Chương I"/"Điều 1" (khối này
            # bị bỏ như phần chữ ký). Nhưng văn bản sửa đổi (uet_dc2025) chỉ có nội dung
            # "Điều 3, chương II: ..." — không phải đầu Điều — nên phải giữ lại, nếu không
            # cả phần quy định mới bị mất khỏi kho.
            blocks.append(cur)
            cur = {"kind": "attachment", "part": part, "chapter": None, "chapter_title": None,
                   "article": None, "article_title": None, "start": end + 1, "end": end + 1}
            continue
        if m_ch:
            chapter, chapter_title = m_ch.group(2), m_ch.group(3).strip() or None
            pending_chapter_title = chapter_title is None
            # Chương I sau khi đã có Điều -> bắt đầu phần quy chế
            if chapter in ("I", "1") and last_article > 0:
                part += 1
                last_article = 0
            continue
        if pending_chapter_title and not m_art:
            chapter_title = p
            pending_chapter_title = False
            continue
        if m_art:
            n, title = m_art
            title = title.strip()
            if n <= last_article:
                part += 1
                chapter, chapter_title = None, None
            last_article = n
            blocks.append(cur)
            cur = {"kind": "article" if has_articles else "section", "part": part,
                   "chapter": chapter, "chapter_title": chapter_title, "article": n,
                   # dòng tiêu đề dài là nội dung chứ không phải tên Điều
                   "article_title": title if title and len(title.split()) <= 20
                and not title.endswith((".", ";")) else None,
                   "start": start, "end": end}
            continue
        if re.match(r"^(PHỤ LỤC|Phụ lục)\b", p):
            in_appendix = True
            blocks.append(cur)
            cur = {"kind": "appendix", "part": part, "chapter": None, "chapter_title": None,
                   "article": None, "article_title": p, "start": start, "end": end}
            continue
        if cur["kind"] == "attachment":
            # có nội dung thật trước đơn vị kế tiếp -> giữ như một phụ lục
            cur["kind"] = "appendix"
            cur["start"] = start
            if len(p.split()) <= 20 and not p.endswith((".", ";")):
                cur["article_title"] = p
        cur["end"] = end
    blocks.append(cur)
    return [b for b in blocks if b["kind"] not in ("tail", "attachment")]


def chunk_by_article(src, max_words=300, overlap=50):
    text, paras = load_doc_text(src["doc_id"])
    chunks = []
    n_appendix = 0
    for b in split_structure(src["doc_id"], paras):
        if b["kind"] == "article":
            key = f"p{b['part']}_d{b['article']}"
        elif b["kind"] == "section":
            key = f"p{b['part']}_m{b['article']}"
        elif b["kind"] == "appendix":
            key = f"p{b['part']}_pl{n_appendix}"
            n_appendix += 1
        else:
            key = f"p{b['part']}_pre"
        body = text[b["start"]:b["end"]]
        first_line_end = b["start"] + (body.find("\n") if "\n" in body else len(body))
        heading = text[b["start"]:first_line_end] if b["kind"] != "preamble" else ""

        words = _word_spans(text, b["start"], b["end"])
        for sub, (i, j) in enumerate(_windows(words, max_words, overlap)):
            s, e = words[i][0], words[j - 1][1]
            piece = text[s:e]
            if sub > 0 and heading:
                piece = heading + "\n" + piece  # giữ ngữ cảnh "Điều N. ..." cho phần sau
            chunks.append({
                "id": f"{src['doc_id']}__{key}_{sub}",
                "doc_id": src["doc_id"],
                "doc_name": src["doc_name"],
                "part": b["part"],
                "chapter": b["chapter"],
                "chapter_title": b["chapter_title"],
                "article": b["article"],
                "article_title": b["article_title"],
                "sub": sub,
                "start": s,
                "end": e,
                "n_words": j - i,
                "text": piece,
            })
    return chunks


# ------------------------------------------------------------ cửa sổ cố định
def chunk_by_window(src, max_words=300, overlap=50):
    text, _ = load_doc_text(src["doc_id"])
    words = _word_spans(text, 0, len(text))
    chunks = []
    for k, (i, j) in enumerate(_windows(words, max_words, overlap)):
        s, e = words[i][0], words[j - 1][1]
        chunks.append({
            "id": f"{src['doc_id']}__w{max_words}_{k}",
            "doc_id": src["doc_id"],
            "doc_name": src["doc_name"],
            "start": s,
            "end": e,
            "n_words": j - i,
            "text": text[s:e],
        })
    return chunks


def chunk_path(mode="dieu", max_words=300):
    return CHUNK_DIR / ("chunks.jsonl" if mode == "dieu" else f"chunks_w{max_words}.jsonl")


def load_chunks(path=None):
    path = Path(path) if path else chunk_path()
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dieu", "window"], default="dieu")
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()

    fn = chunk_by_article if args.mode == "dieu" else chunk_by_window
    all_chunks = []
    for src in load_sources():
        if not (TEXT_DIR / f"{src['doc_id']}.txt").exists():
            continue
        cs = fn(src, args.max_words, args.overlap)
        print(f"{src['doc_id']}: {len(cs)} chunk")
        all_chunks.extend(cs)

    out = chunk_path(args.mode, args.max_words)
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n = [c["n_words"] for c in all_chunks]
    print(f"-> {out}: {len(all_chunks)} chunk, số từ TB {sum(n) / len(n):.0f}, max {max(n)}")


if __name__ == "__main__":
    main()
