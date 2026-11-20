"""
prompts.py – Prompt cho pipeline RAG (TV3).

Hai phiên bản prompt dùng trong ma trận thí nghiệm (mục 7 kế hoạch):
  - v1: chỉ yêu cầu trả lời dựa trên ngữ cảnh, KHÔNG có lệnh từ chối, KHÔNG bắt trích dẫn  -> S5
  - v2: chỉ dựa trên ngữ cảnh + trích dẫn [n] + câu từ chối cố định                         -> S0..S4
v1 và v2 sinh từ CÙNG một hàm, chỉ khác cờ, để S2 vs S5 là so sánh có kiểm soát.

Ngoài ra file cung cấp các hàm hậu xử lý dùng chung cho generate.py / nli_judge.py:
  is_refusal(), strip_citations(), extract_citations(), answer_for_scoring().
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

# Câu từ chối cố định – phải khớp từng chữ với kế hoạch để đếm tự động.
REFUSAL_TEXT = "Không tìm thấy thông tin trong tài liệu"

PROMPT_VERSIONS = ("v1", "v2")

_SYSTEM_BASE = (
    "Bạn là trợ lý hỏi đáp về quy chế đào tạo và công tác sinh viên của Đại học Quốc gia Hà Nội. "
    "Trả lời bằng tiếng Việt, ngắn gọn, chính xác: chỉ một câu nêu thẳng đáp án "
    "(con số, điều kiện, mốc thời gian), không nhắc lại câu hỏi."
)

_RULES_V1 = [
    "Chỉ sử dụng thông tin trong các đoạn ngữ cảnh được cung cấp để trả lời.",
]

_RULES_V2 = [
    "Chỉ sử dụng thông tin trong các đoạn ngữ cảnh được cung cấp; không dùng kiến thức bên ngoài.",
    "Sau mỗi ý, ghi số thứ tự đoạn làm căn cứ trong ngoặc vuông, ví dụ [1] hoặc [1][2].",
    "Giữ nguyên con số, mốc thời gian và điều kiện như trong ngữ cảnh; không suy diễn thêm.",
    f'Nếu ngữ cảnh không chứa thông tin để trả lời, chỉ trả lời đúng một câu: "{REFUSAL_TEXT}."',
]

# Closed-book (S0): không có ngữ cảnh. Giữ lệnh từ chối của v2 để S0 so được với S1..S4
# (chỉ khác yếu tố "có/không có truy hồi").
_RULES_CLOSED_BOOK = [
    "Trả lời dựa trên hiểu biết của bạn về quy chế của Đại học Quốc gia Hà Nội.",
    f'Nếu bạn không chắc chắn hoặc không biết, chỉ trả lời đúng một câu: "{REFUSAL_TEXT}."',
]


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def format_chunk_header(chunk: Dict) -> str:
    """'(Tên văn bản – Điều 12, Khoản 2)' từ metadata chunk; thiếu trường nào thì bỏ trường đó."""
    parts = []
    if chunk.get("doc_name"):
        parts.append(str(chunk["doc_name"]))
    loc = []
    if chunk.get("article"):
        art = str(chunk["article"])
        loc.append(art if art.lower().startswith("điều") else f"Điều {art}")
    if chunk.get("clause"):
        cl = str(chunk["clause"])
        loc.append(cl if cl.lower().startswith("khoản") else f"Khoản {cl}")
    if loc:
        parts.append(", ".join(loc))
    return f"({' – '.join(parts)})" if parts else ""


def format_context(chunks: List[Dict]) -> str:
    """Đánh số đoạn [1], [2], ... theo đúng thứ tự xếp hạng của retriever."""
    lines = []
    for i, ch in enumerate(chunks, start=1):
        header = format_chunk_header(ch)
        text = _nfc(ch.get("text", "")).strip()
        lines.append(f"[{i}] {header}\n{text}" if header else f"[{i}]\n{text}")
    return "\n\n".join(lines)


def build_messages(
    question: str,
    chunks: Optional[List[Dict]] = None,
    version: str = "v2",
    closed_book: bool = False,
) -> List[Dict[str, str]]:
    """
    Trả về list message theo định dạng chat (dùng cho tokenizer.apply_chat_template của Qwen).

    question    : câu hỏi tiếng Việt
    chunks      : list dict {chunk_id, doc_name, article, clause?, text}, đã xếp hạng
    version     : 'v1' | 'v2'
    closed_book : True cho S0 (bỏ qua chunks)
    """
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"version phải thuộc {PROMPT_VERSIONS}, nhận '{version}'")
    question = _nfc(question).strip()

    if closed_book:
        rules = _RULES_CLOSED_BOOK
        user = f"Câu hỏi: {question}\nTrả lời:"
    else:
        rules = _RULES_V2 if version == "v2" else _RULES_V1
        context = format_context(chunks or [])
        user = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}\nTrả lời:"

    system = _SYSTEM_BASE + "\nQuy tắc:\n" + "\n".join(f"- {r}" for r in rules)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def messages_to_text(messages: List[Dict[str, str]]) -> str:
    """Dạng phẳng để log/debug hoặc cho backend không có chat template."""
    return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)


# ---------------------------------------------------------------------------
# Hậu xử lý câu trả lời
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")


def extract_citations(text: str) -> List[int]:
    """'... [1][3] ... [2, 3]' -> [1, 3, 2] (giữ thứ tự xuất hiện, bỏ trùng)."""
    seen, out = set(), []
    for m in _CITATION_RE.finditer(text or ""):
        for n in re.split(r"[,;]", m.group(1)):
            n = int(n.strip())
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def strip_citations(text: str) -> str:
    """Xóa thẻ [n] và khoảng trắng thừa trước dấu câu."""
    t = _CITATION_RE.sub("", text or "")
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    return " ".join(t.split())


def _fold(s: str) -> str:
    """Chữ thường + bỏ dấu tiếng Việt + bỏ dấu câu: so khớp từ chối bền với biến thể."""
    s = unicodedata.normalize("NFD", _nfc(s).lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


_REFUSAL_PATTERNS = [
    _fold(REFUSAL_TEXT),                              # câu chuẩn
    "khong tim thay thong tin",                       # biến thể rút gọn
    "tai lieu khong de cap",
    "ngu canh khong de cap",
    "khong co thong tin",
]
# Lời mở đầu lịch sự được phép đứng trước cụm từ chối ("Xin lỗi, tôi ...").
_REFUSAL_PREFIX = re.compile(r"^(?:(?:xin loi|rat tiec|tiec la|toi|minh|rat xin loi)\s+)*")
# Nếu sau cụm từ chối còn chuyển ý ("nhưng", "tuy nhiên", ...) thì câu có nội dung thật -> KHÔNG phải từ chối.
_CONTRAST = re.compile(r"\b(?:nhung|tuy nhien|tuy vay|song|tuy|ma|va theo|theo)\b")
_MAX_REFUSAL_WORDS = 20


def refusal_kind(text: str) -> str:
    """
    'full'    : cả câu là lời từ chối (bắt đầu bằng cụm từ chối, ngắn, không chuyển ý).
    'partial' : bắt đầu bằng cụm từ chối nhưng còn nội dung khác ("... không đề cập X nhưng Y ...").
    'none'    : không phải từ chối (kể cả khi cụm từ chối chỉ nằm GIỮA câu,
                ví dụ "Sinh viên không có thông tin liên hệ thì ...").
    """
    f = _fold(strip_citations(text))
    if not f:
        return "none"
    core = _REFUSAL_PREFIX.sub("", f)
    if not any(core.startswith(p) for p in _REFUSAL_PATTERNS):
        return "none"
    if len(f.split()) > _MAX_REFUSAL_WORDS or _CONTRAST.search(core):
        return "partial"
    return "full"


def is_refusal(text: str, strict: bool = False) -> bool:
    """
    strict=True : chỉ chấp nhận câu từ chối chuẩn (dùng đo "tuân thủ prompt").
    strict=False: chấp nhận biến thể phổ biến nhưng CHỈ khi cả câu là lời từ chối (refusal_kind == 'full').
                  Câu trả lời có nội dung thật (kể cả câu bắt đầu bằng "Tài liệu không đề cập X nhưng ...")
                  không bị tính là từ chối; dùng refusal_kind() để gắn cờ 'partial' khi gán nhãn.
    """
    if strict:
        f = _fold(strip_citations(text))
        return bool(f) and f == _fold(REFUSAL_TEXT)
    return refusal_kind(text) == "full"


def answer_for_scoring(text: str) -> str:
    """
    Chuẩn hóa câu trả lời sinh trước khi tính token-F1 bằng qa.qa_metrics.compute_f1:
      - câu từ chối -> "" (để khớp gold rỗng của câu ngoài kho: F1 = 1)
      - xóa thẻ trích dẫn [n] (tránh token '1', '2' làm giảm precision)
    """
    if is_refusal(text):
        return ""
    return strip_citations(text)
