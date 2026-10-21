"""Lazy CPU sentence-transformer adapter; never downloads until first use."""
from __future__ import annotations
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Encoder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def status(self) -> tuple[bool, str]:
        try:
            from sentence_transformers import SentenceTransformer
            return True, "Ready" if self._model is not None else "Installed; model loads on first index or search"
        except ImportError:
            return False, "sentence-transformers is not installed"

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._load().encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")
