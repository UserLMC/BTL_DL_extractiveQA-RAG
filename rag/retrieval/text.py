"""Chuẩn hoá và tách từ tiếng Việt dùng chung cho BM25 và dense retriever."""
import re
import unicodedata
from functools import lru_cache

PUNCT = re.compile(r"^[\W_]+$")


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


@lru_cache(maxsize=None)
def _vi_tokenizer():
    from pyvi import ViTokenizer

    return ViTokenizer


def segment(text):
    """Tách từ bằng pyvi: 'sinh viên học tập' -> 'sinh_viên học_tập'."""
    return _vi_tokenizer().tokenize(normalize(text))


def tokenize(text, mode="word", lowercase=True):
    """Danh sách token cho BM25.

    mode='word'     : tách từ bằng pyvi (từ ghép nối bằng '_')
    mode='syllable' : tách theo âm tiết (khoảng trắng)
    Bỏ token chỉ gồm dấu câu.
    """
    text = normalize(text)
    if lowercase:
        text = text.lower()
    toks = segment(text).split() if mode == "word" else re.findall(r"\w+|[^\w\s]", text)
    return [t for t in toks if not PUNCT.match(t)]
