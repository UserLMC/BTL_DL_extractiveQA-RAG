from rag.corpus.chunk import _windows, match_article, split_structure
from rag.corpus.parse_pdf import apply_fixes
from rag.retrieval.eval_retrieval import first_hit_rank, mrr_at_k, recall_at_k
from rag.retrieval.gold import covers
from rag.retrieval.text import tokenize


def test_recall_single_evidence():
    ranked = [5, 2, 9, 1]
    assert recall_at_k(ranked, [{9}], 1) == 0
    assert recall_at_k(ranked, [{9}], 3) == 1


def test_recall_multi_evidence_is_fraction():
    ranked = [5, 2, 9, 1]
    # 2 evidence, chỉ evidence đầu có chunk trong top-3
    assert recall_at_k(ranked, [{2}, {7}], 3) == 0.5
    # một evidence có thể được phủ bởi nhiều chunk (Điều dài bị cắt)
    assert recall_at_k(ranked, [{7, 5}], 1) == 1


def test_mrr_and_first_hit():
    ranked = [5, 2, 9, 1]
    assert mrr_at_k(ranked, [{9}]) == 1 / 3
    assert mrr_at_k(ranked, [{4}]) == 0
    assert mrr_at_k(ranked, [{1}], k=3) == 0
    assert first_hit_rank(ranked, [{1}, {2}]) == 2


def test_covers_needs_half_of_evidence():
    chunk = {"doc_id": "a", "start": 100, "end": 200}
    assert covers(chunk, ("a", 120, 180))
    assert covers(chunk, ("a", 150, 240))        # phủ 50/90 > 0.5
    assert not covers(chunk, ("a", 180, 260))    # phủ 20/80
    assert not covers(chunk, ("b", 120, 180))    # khác văn bản


def test_windows_overlap_and_cover_all():
    w = _windows(list(range(700)), 300, 50)
    assert w == [(0, 300), (250, 550), (500, 700)]
    assert _windows(list(range(10)), 300, 50) == [(0, 10)]


def test_article_heading_ignores_citations():
    assert match_article("Điều 12. Cảnh báo học tập") == (12, "Cảnh báo học tập")
    assert match_article("Điều 3: Danh hiệu Thủ khoa")[0] == 3
    assert match_article("Điều 7") == (7, "")
    assert match_article("Điều 9 Tuyển sinh 1 Chỉ tiêu")[0] == 9   # OCR mất dấu chấm
    assert match_article("Điều 3, chương II: Quy định về xét") is None
    assert match_article("Điều 41 của Quy chế này.") is None
    assert match_article("Điều phối sử dụng các nguồn lực") is None


def _paras(lines):
    out, pos = [], 0
    for p in lines:
        out.append((pos, pos + len(p), p))
        pos += len(p) + 1
    return out


def test_attachment_after_signature_is_kept():
    # văn bản sửa đổi: nội dung mới nằm SAU chữ ký, mở đầu bằng trích dẫn "Điều 3, chương II"
    lines = ["Điều 1. Sửa đổi Điều 3 như phụ lục kèm theo.", "Điều 2. Hiệu lực từ ngày ký.",
             "Nơi nhận:", "- Lưu VT.", "PHÓ HIỆU TRƯỞNG",
             "QUY ĐỊNH (Ban hành kèm theo Quyết định số 12/QĐ)",
             "Điều 3, chương II: Quy định về xét Thủ khoa", "Sinh viên phải đạt loại Giỏi trở lên."]
    blocks = split_structure("x", _paras(lines))
    assert [b["kind"] for b in blocks] == ["preamble", "article", "article", "appendix"]
    app = blocks[-1]
    assert app["article_title"] == "Điều 3, chương II: Quy định về xét Thủ khoa"
    text = "\n".join(lines)
    assert text[app["start"]:app["end"]].endswith("loại Giỏi trở lên.")


def test_attachment_followed_by_articles_is_dropped():
    # quyết định ban hành quy chế: sau dòng "Ban hành kèm theo" là Chương/Điều -> không sinh khối rác
    lines = ["Điều 1. Ban hành Quy chế.", "Nơi nhận:", "- Lưu VT.",
             "QUY CHẾ (Ban hành kèm theo Quyết định số 5/QĐ)", "Chương I", "QUY ĐỊNH CHUNG",
             "Điều 1. Phạm vi", "Quy chế này quy định về đào tạo."]
    kinds = [b["kind"] for b in split_structure("x", _paras(lines))]
    assert kinds == ["preamble", "article", "article"]


def test_ocr_fixes_whole_word_and_capitalized():
    fixes = [("dào", "đào"), ("dơn", "đơn")]
    assert apply_fixes("Dào tạo tại dơn vị dào tạo", fixes) == "Đào tạo tại đơn vị đào tạo"
    assert apply_fixes("Quy chế ndào", fixes) == "Quy chế ndào"   # không sửa bên trong từ


def test_tokenize_word_and_syllable():
    assert "sinh_viên" in tokenize("Sinh viên bị cảnh báo học tập.", "word")
    assert tokenize("Sinh viên, học tập!", "syllable") == ["sinh", "viên", "học", "tập"]
