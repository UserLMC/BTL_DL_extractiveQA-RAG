"""PDF -> text sạch cho kho quy chế.

- PDF có lớp chữ: đọc bằng PyMuPDF.
- PDF scan (needs_ocr=yes trong sources.csv): OCR bằng EasyOCR (tiếng Việt), kết quả
  OCR thô được cache ở data/cache/ocr/ để không phải chạy lại.
- Làm sạch: bỏ số trang, bỏ header/footer lặp lại, sửa lỗi OCR theo
  data/corpus/ocr_fixes.tsv, nối các dòng bị ngắt thành đoạn.

Đầu ra: data/corpus/text/<doc_id>.txt, mỗi dòng là một đoạn.
    python -m rag.corpus.parse_pdf            # mọi văn bản
    python -m rag.corpus.parse_pdf vnu_3626   # một văn bản
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from rag.corpus.download import PDF_DIR, load_sources

TEXT_DIR = Path("data/corpus/text")
OCR_CACHE = Path("data/cache/ocr")
FIXES = Path("data/corpus/ocr_fixes.tsv")
MANUAL_DIR = Path("data/corpus/manual")

# Dòng bắt đầu một đơn vị mới (không được nối vào dòng trước)
BLOCK_START = re.compile(
    r"^(Chương|CHƯƠNG|Điều|ĐIỀU|Mục|MỤC|Phụ lục|PHỤ LỤC)\s"
    r"|^\d+(\.\d+)*[.)]\s"        # 1.  1.2.  2)
    r"|^[a-zđ]\)\s"               # a)  đ)
    r"|^[-–•+*]\s"                # gạch đầu dòng
)
# Tiêu đề: "Chương II", "Điều 5." (phải có số — "Điều này" không phải tiêu đề) hoặc mục "1.2."
HEADING_UNIT = re.compile(r"^(Chương|CHƯƠNG|Điều|ĐIỀU|Mục|MỤC)\s+([IVXLC]+|\d+)\b")
HEADING_NUM = re.compile(r"^\d+(\.\d+)*\.\s")
PAGE_NO = re.compile(r"^(-\s*)?(trang\s*)?\d{1,3}(\s*-)?$|^\d{1,3}\s*/\s*\d{1,3}$", re.I)


# ---------------------------------------------------------------- đọc PDF
def text_layer_pages(pdf_path):
    import fitz

    with fitz.open(pdf_path) as doc:
        return [page.get_text().splitlines() for page in doc]


def _group_ocr_lines(results):
    """Gom các hộp chữ của EasyOCR thành dòng theo toạ độ y."""
    boxes = []
    for bbox, text, _conf in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        boxes.append((min(ys), max(ys), min(xs), text))
    if not boxes:
        return []
    heights = sorted(b[1] - b[0] for b in boxes)
    tol = heights[len(heights) // 2] * 0.5
    boxes.sort(key=lambda b: (b[0] + b[1]) / 2)
    lines, cur, cur_y = [], [], None
    for b in boxes:
        y = (b[0] + b[1]) / 2
        if cur and abs(y - cur_y) > tol:
            lines.append(cur)
            cur = []
        cur.append(b)
        cur_y = y if len(cur) == 1 else (cur_y * (len(cur) - 1) + y) / len(cur)
    lines.append(cur)
    return [" ".join(b[3] for b in sorted(line, key=lambda b: b[2])) for line in lines]


def ocr_pages(pdf_path, doc_id, dpi=200):
    import fitz

    cache = OCR_CACHE / f"{doc_id}.json"
    pages = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    with fitz.open(pdf_path) as doc:
        n_pages = len(doc)
        if len(pages) >= n_pages:
            return pages

        import easyocr
        import numpy as np

        reader = easyocr.Reader(["vi"], gpu=True, verbose=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        for i in range(len(pages), n_pages):  # chạy tiếp từ trang chưa OCR
            pix = doc[i].get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            pages.append(_group_ocr_lines(reader.readtext(img, detail=1, paragraph=False)))
            cache.write_text(json.dumps(pages, ensure_ascii=False, indent=0), encoding="utf-8")
            print(f"  OCR {doc_id} trang {i + 1}/{n_pages}", flush=True)
    return pages


# ---------------------------------------------------------------- làm sạch
def load_fixes(doc_id, is_ocr=True):
    """ocr_fixes.tsv: doc_id<TAB>sai<TAB>đúng (doc_id='*' áp dụng cho mọi văn bản OCR)."""
    if not FIXES.exists():
        return []
    fixes = []
    for line in FIXES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        scope, wrong, right = line.split("\t")
        if scope == doc_id or (scope == "*" and is_ocr):
            fixes.append((unicodedata.normalize("NFC", wrong), unicodedata.normalize("NFC", right)))
    return fixes


def apply_fixes(text, fixes):
    for wrong, right in fixes:
        variants = {(wrong, right), (wrong[:1].upper() + wrong[1:], right[:1].upper() + right[1:])}
        for w, r in variants:
            # chỉ thay nguyên từ, tránh sửa nhầm bên trong từ khác
            text = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", r, text)
    return text


def normalize_line(line, is_ocr):
    line = unicodedata.normalize("NFC", line)
    line = line.replace(" ", " ").replace("​", "")
    if is_ocr:
        # EasyOCR hay đọc chữ "l" thành "I" ở đầu từ: Iượng, Iớp, Iần
        line = re.sub(r"(?<![\w])I(?=[a-zà-ỹ])(?!n\b)", "l", line)
        # khoảng trắng thừa trước dấu câu
        line = re.sub(r"\s+([,.;:)])", r"\1", line)
        line = re.sub(r"([(])\s+", r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


def drop_headers_footers(pages):
    """Bỏ số trang và các dòng lặp ở đầu/cuối trang (xuất hiện ở >= 1/2 số trang)."""
    if len(pages) < 3:
        return [[ln for ln in p if not PAGE_NO.match(ln)] for p in pages]
    edge = Counter()
    for p in pages:
        for ln in set(p[:2] + p[-2:]):
            edge[ln] += 1
    repeated = {ln for ln, c in edge.items() if c >= len(pages) / 2 and ln}
    return [[ln for ln in p if ln not in repeated and not PAGE_NO.match(ln)] for p in pages]


def merge_paragraphs(lines):
    """Nối các dòng bị ngắt do xuống dòng trong PDF thành đoạn."""
    paras = []
    for ln in lines:
        if not ln:
            continue
        if not paras:
            paras.append(ln)
            continue
        prev = paras[-1]
        new_block = BLOCK_START.match(ln) or prev.endswith((":", ";", ".", "?", "!"))
        # Dòng tiêu đề (Chương/Điều/mục, hoặc dòng viết hoa toàn bộ) mà dòng sau bắt đầu
        # bằng chữ hoa thì dòng sau là nội dung mới, không phải phần tiêu đề bị ngắt.
        starts_upper = ln[:1].isupper()
        # dòng bị ngắt thường dài ~15-20 từ, tiêu đề thật thì ngắn hơn
        n_words = len(prev.split())
        is_heading = (HEADING_UNIT.match(prev) and n_words <= 14) or (
            HEADING_NUM.match(prev) and n_words <= 10)
        is_caps = prev.upper() == prev and any(ch.isalpha() for ch in prev)
        if starts_upper and (is_heading or is_caps):
            new_block = True
        if new_block:
            paras.append(ln)
        elif prev.endswith("-") and not prev.endswith(" -"):
            paras[-1] = prev[:-1] + ln  # từ bị ngắt bằng gạch nối
        else:
            paras[-1] = prev + " " + ln
    return paras


def manual_overrides(doc_id):
    """Trang chép tay thay cho OCR: data/corpus/manual/<doc_id>_p<a>-<b>.txt (trang a..b, đếm từ 1).

    Dùng cho bảng biểu OCR không đọc được (vd. phụ lục khung kỷ luật của vnu_32).
    Trả về {chỉ số trang 0-based: danh sách dòng}; các trang sau trang đầu để rỗng.
    """
    out = {}
    for f in sorted(MANUAL_DIR.glob(f"{doc_id}_p*.txt")):
        m = re.match(rf"{re.escape(doc_id)}_p(\d+)(?:-(\d+))?$", f.stem)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        lines = [unicodedata.normalize("NFC", ln).strip()
                 for ln in f.read_text(encoding="utf-8").splitlines()]
        out[a - 1] = [ln for ln in lines if ln]
        for i in range(a, b):
            out[i] = []
    return out


def parse_doc(src, force_ocr=False):
    pdf = PDF_DIR / f"{src['doc_id']}.pdf"
    is_ocr = src.get("needs_ocr", "").strip().lower() == "yes" or force_ocr
    pages = ocr_pages(pdf, src["doc_id"]) if is_ocr else text_layer_pages(pdf)
    pages = [[normalize_line(ln, is_ocr) for ln in p] for p in pages]
    pages = [[ln for ln in p if ln] for p in pages]
    pages = drop_headers_footers(pages)
    for i, lines in manual_overrides(src["doc_id"]).items():
        if i < len(pages):
            pages[i] = lines
    paras = merge_paragraphs([ln for p in pages for ln in p])
    fixes = load_fixes(src["doc_id"], is_ocr)
    return [apply_fixes(p, fixes) for p in paras]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", nargs="*", help="mặc định: mọi văn bản đã tải")
    args = ap.parse_args()

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for src in load_sources():
        if args.doc_ids and src["doc_id"] not in args.doc_ids:
            continue
        if not (PDF_DIR / f"{src['doc_id']}.pdf").exists():
            print(f"[bỏ qua] {src['doc_id']}: chưa có PDF (chạy rag.corpus.download)")
            continue
        paras = parse_doc(src)
        out = TEXT_DIR / f"{src['doc_id']}.txt"
        out.write_text("\n".join(paras) + "\n", encoding="utf-8")
        print(f"[xong] {out}: {len(paras)} đoạn, {sum(len(p.split()) for p in paras)} từ")


if __name__ == "__main__":
    main()
