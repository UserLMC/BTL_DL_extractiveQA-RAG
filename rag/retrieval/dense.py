"""Dense retriever: bi-encoder (sentence-transformers) + FAISS.

Câu hỏi và chunk được mã hoá độc lập thành vector; vector được chuẩn hoá L2 nên
inner product (IndexFlatIP) bằng cosine similarity.

Embedding của chunk được cache ở data/cache/emb/ theo (mô hình, file chunk).
"""
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from rag.retrieval.bm25 import chunk_text_for_index
from rag.retrieval.text import normalize, segment

CACHE = Path("data/cache/emb")

# Cấu hình từng mô hình: tiền tố truy vấn/đoạn và có cần tách từ trước không
MODELS = {
    "e5-base": {
        "name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ", "passage_prefix": "passage: ", "segment": False,
    },
    "bkai": {
        # PhoBERT-based: đầu vào phải tách từ bằng pyvi, tối đa 256 token
        "name": "bkai-foundation-models/vietnamese-bi-encoder",
        "query_prefix": "", "passage_prefix": "", "segment": True,
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "query_prefix": "", "passage_prefix": "", "segment": False,
    },
}


class DenseRetriever:
    def __init__(self, chunks, model_key="e5-base", device=None, batch_size=16):
        from sentence_transformers import SentenceTransformer

        self.cfg = MODELS[model_key]
        self.name = f"dense_{model_key}"
        self.chunks = chunks
        self.model = SentenceTransformer(self.cfg["name"], device=device)
        if self.model.device.type == "cuda":
            self.model.half()
        self.batch_size = batch_size
        emb = self._encode_chunks()
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def _prep(self, text, prefix):
        text = segment(text) if self.cfg["segment"] else normalize(text)
        return prefix + text

    def _encode(self, texts):
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
        return emb.astype(np.float32)

    def _encode_chunks(self):
        texts = [self._prep(chunk_text_for_index(c), self.cfg["passage_prefix"]) for c in self.chunks]
        key = hashlib.md5(json.dumps([self.cfg["name"], texts]).encode()).hexdigest()[:12]
        path = CACHE / f"{self.name}_{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self._encode(texts)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    def batch_search(self, queries, k=10):
        q = self._encode([self._prep(x, self.cfg["query_prefix"]) for x in queries])
        scores, idx = self.index.search(q, k)
        return [[(int(i), float(s)) for i, s in zip(row_i, row_s) if i >= 0]
                for row_i, row_s in zip(idx, scores)]

    def search(self, query, k=10):
        return self.batch_search([query], k)[0]

    def save_index(self, out_dir):
        """Đóng băng index: lưu FAISS index + danh sách id chunk theo đúng thứ tự."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / f"{self.name}.faiss"))
        (out / f"{self.name}.ids.json").write_text(
            json.dumps([c["id"] for c in self.chunks], ensure_ascii=False), encoding="utf-8")
