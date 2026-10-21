"""Rebuildable FAISS CPU retrieval with a safe, transient numpy fallback."""
from __future__ import annotations
import numpy as np
from src.embeddings.encoder import Encoder

class Retriever:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def search(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not query.strip() or not chunks:
            return []
        if all(c.get("embedding") for c in chunks):
            vectors = np.stack([np.frombuffer(c["embedding"], dtype="float32") for c in chunks]).astype("float32")
        else:
            vectors = self.encoder.encode([c["text"] for c in chunks])
        q = self.encoder.encode([query])
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            scores, ids = index.search(q, min(top_k, len(chunks)))
            pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        except ImportError:
            scores = (vectors @ q[0]).astype(float)
            pairs = [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:top_k]]
        return [{**chunks[i], "score": score} for i, score in pairs]
