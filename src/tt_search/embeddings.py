from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_BATCH_SIZE = 32
EMBEDDING_BACKEND = "sentence-transformers"
DeviceOption = Literal["auto", "cpu", "mps"]


class EmbeddingProvider(Protocol):
    model_name: str
    dim: int
    backend: str
    device: str
    batch_size: int
    prefix_policy: str

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""


@dataclass
class SentenceTransformerEmbeddingProvider:
    model_name: str = DEFAULT_MODEL
    device: DeviceOption = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.backend = EMBEDDING_BACKEND
        self.device = resolve_device(self.device)
        self.prefix_policy = prefix_policy_for_model(self.model_name)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=requires_trust_remote_code(self.model_name),
        )
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            sample = self._model.encode(
                [prefix_query("dimension probe", self.prefix_policy)],
                normalize_embeddings=True,
                device=self.device,
            )
            dim = int(np.asarray(sample).shape[-1])
        self.dim = int(dim)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [prefix_passage(text, self.prefix_policy) for text in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [prefix_query(text, self.prefix_policy)],
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )
        return np.asarray(vector[0], dtype=np.float32).tolist()


def resolve_device(device: DeviceOption) -> str:
    if device == "cpu":
        return "cpu"
    if device == "mps":
        if not mps_is_available():
            raise ValueError("MPS device was requested, but PyTorch MPS is not available")
        return "mps"
    if mps_is_available():
        return "mps"
    return "cpu"


def mps_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def prefix_policy_for_model(model_name: str) -> str:
    if model_name.startswith("cl-nagoya/ruri-v3-"):
        return "ruri-v3"
    if model_name == "pfnet/plamo-embedding-1b":
        return "plamo"
    return "e5"


def prefix_query(text: str, prefix_policy: str) -> str:
    if prefix_policy == "ruri-v3":
        return f"検索クエリ: {text}"
    if prefix_policy == "e5":
        return f"query: {text}"
    return text


def prefix_passage(text: str, prefix_policy: str) -> str:
    if prefix_policy == "ruri-v3":
        return f"検索文書: {text}"
    if prefix_policy == "e5":
        return f"passage: {text}"
    return text


def requires_trust_remote_code(model_name: str) -> bool:
    return model_name == "pfnet/plamo-embedding-1b"


def normalize_vector(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).astype(np.float32).tolist()
