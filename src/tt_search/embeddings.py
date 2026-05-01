from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class EmbeddingProvider(Protocol):
    model_name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""


@dataclass
class SentenceTransformerEmbeddingProvider:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            sample = self._model.encode(["query: dimension probe"], normalize_embeddings=True)
            dim = int(np.asarray(sample).shape[-1])
        self.dim = int(dim)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {text}" for text in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=np.float32).tolist()


def normalize_vector(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).astype(np.float32).tolist()
