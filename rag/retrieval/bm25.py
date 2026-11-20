"""Retriever BM25 (rank_bm25, Okapi BM25).

score(q, d) = sum_t IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
- k1: mức bão hoà của tần suất từ (k1 lớn -> lặp từ càng được cộng điểm)
- b : mức chuẩn hoá theo độ dài đoạn (b=0 không chuẩn hoá, b=1 chuẩn hoá đầy đủ)
"""
import numpy as np
from rank_bm25 import BM25Okapi

from rag.retrieval.text import tokenize


def chunk_text_for_index(chunk):
    """Đưa tên văn bản và tiêu đề Điều vào nội dung được đánh chỉ mục."""
    head = [chunk.get("doc_name") or ""]
    if chunk.get("article_title"):
        head.append(chunk["article_title"])
    return " . ".join(head + [chunk["text"]])


class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75, token_mode="word"):
        self.chunks = chunks
        self.token_mode = token_mode
        self.name = f"bm25_{token_mode}_k1={k1}_b={b}"
        corpus = [tokenize(chunk_text_for_index(c), token_mode) for c in chunks]
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def search(self, query, k=10):
        scores = self.bm25.get_scores(tokenize(query, self.token_mode))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def batch_search(self, queries, k=10):
        return [self.search(q, k) for q in queries]
