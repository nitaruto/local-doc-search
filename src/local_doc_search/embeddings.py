from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_BATCH_SIZE = 32
EMBEDDING_BACKEND = "sentence-transformers"
PLAMO_MODEL = "pfnet/plamo-embedding-1b"
PLAMO_BACKEND = "plamo-custom"
PLAMO_RETRY_ATTEMPTS = 5
SARASHINA_V2_PREFIX = "sbintuitions/sarashina-embedding-v2-"
SARASHINA_V2_RETRIEVAL_INSTRUCTION = (
    "質問を与えるので、その質問に答えるのに役立つ関連文書を検索してください。"
)
DeviceOption = Literal["auto", "cpu", "mps"]
RuntimeDevice = Literal["cpu", "mps"]


class EmbeddingProvider(Protocol):
    model_name: str
    dim: int
    backend: str
    device: str
    batch_size: int
    prefix_policy: str

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...


@dataclass
class SentenceTransformerEmbeddingProvider:
    model_name: str = DEFAULT_MODEL
    device: str = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.backend = EMBEDDING_BACKEND
        self.device = resolve_device(cast(DeviceOption, self.device))
        self.prefix_policy = prefix_policy_for_model(self.model_name)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=requires_trust_remote_code(self.model_name),
            model_kwargs=sentence_transformer_model_kwargs(self.model_name),
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


@dataclass
class PlamoEmbeddingProvider:
    model_name: str = PLAMO_MODEL
    device: str = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.backend = PLAMO_BACKEND
        self.device = resolve_plamo_device(cast(DeviceOption, self.device))
        self.prefix_policy = "plamo"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        ensure_plamo_max_length(self._model)
        self._model = self._model.to(self.device)
        refresh_plamo_rotary_cache(self._model)
        self._model.eval()
        sample = self._encode_documents(["dimension probe"])
        self.dim = len(sample[0])

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._encode_documents(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        import torch

        def encode() -> object:
            with torch.inference_mode():
                return self._model.encode_query([text], self._tokenizer)

        return self._encode_with_retry(encode)[0]

    def _encode_documents(self, texts: list[str]) -> list[list[float]]:
        import torch

        def encode() -> object:
            with torch.inference_mode():
                return self._model.encode_document(texts, self._tokenizer)

        return self._encode_with_retry(encode)

    def _encode_with_retry(self, encode: Callable[[], object]) -> list[list[float]]:
        last_error: ValueError | None = None
        for attempt in range(1, PLAMO_RETRY_ATTEMPTS + 1):
            try:
                return tensor_to_vectors(encode())
            except ValueError as error:
                if "non-finite" not in str(error):
                    raise
                warnings.warn(
                    "PLaMo embedding returned non-finite values; "
                    f"retrying ({attempt}/{PLAMO_RETRY_ATTEMPTS}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                last_error = error
        assert last_error is not None
        raise last_error


def create_embedding_provider(
    *,
    model_name: str = DEFAULT_MODEL,
    device: DeviceOption = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingProvider:
    if model_name == PLAMO_MODEL:
        return PlamoEmbeddingProvider(model_name=model_name, device=device, batch_size=batch_size)
    return SentenceTransformerEmbeddingProvider(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )


def tensor_to_vectors(value: object) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = cast(Any, value).detach().cpu().float()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if not np.isfinite(arr).all():
        raise ValueError("Embedding model returned non-finite values")
    return [normalize_vector(row.tolist()) for row in arr]


def ensure_plamo_max_length(model: object) -> None:
    config = getattr(model, "config", None)
    if config is None or hasattr(config, "max_length"):
        return
    max_length = getattr(config, "max_position_embeddings", 4096)
    config.max_length = int(max_length)


def refresh_plamo_rotary_cache(model: object) -> None:
    layers = getattr(getattr(model, "layers", None), "layers", [])
    for layer in layers:
        rotary_emb = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        if rotary_emb is None or not hasattr(rotary_emb, "_set_cos_sin_cache"):
            continue
        seq_len = int(rotary_emb.max_position_embeddings)
        inv_freq = rotary_emb.inv_freq
        rotary_emb._set_cos_sin_cache(
            seq_len=seq_len,
            device=inv_freq.device,
            dtype=inv_freq.dtype,
        )


def resolve_device(device: DeviceOption) -> RuntimeDevice:
    if device == "cpu":
        return "cpu"
    if device == "mps":
        if not mps_is_available():
            raise ValueError("MPS device was requested, but PyTorch MPS is not available")
        return "mps"
    if mps_is_available():
        return "mps"
    return "cpu"


def resolve_plamo_device(device: DeviceOption) -> RuntimeDevice:
    return resolve_device(device)


def mps_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def prefix_policy_for_model(model_name: str) -> str:
    if model_name.startswith("cl-nagoya/ruri-v3-"):
        return "ruri-v3"
    if model_name.startswith(SARASHINA_V2_PREFIX):
        return "sarashina-v2"
    if model_name == PLAMO_MODEL:
        return "plamo"
    return "e5"


def sentence_transformer_model_kwargs(model_name: str) -> dict[str, object] | None:
    if model_name.startswith(SARASHINA_V2_PREFIX):
        import torch

        return {"torch_dtype": torch.bfloat16}
    return None


def prefix_query(text: str, prefix_policy: str) -> str:
    if prefix_policy == "ruri-v3":
        return f"検索クエリ: {text}"
    if prefix_policy == "sarashina-v2":
        return f"task: {SARASHINA_V2_RETRIEVAL_INSTRUCTION}\nquery: {text}"
    if prefix_policy == "e5":
        return f"query: {text}"
    return text


def prefix_passage(text: str, prefix_policy: str) -> str:
    if prefix_policy == "ruri-v3":
        return f"検索文書: {text}"
    if prefix_policy == "sarashina-v2":
        return f"text: {text}"
    if prefix_policy == "e5":
        return f"passage: {text}"
    return text


def requires_trust_remote_code(model_name: str) -> bool:
    return model_name == PLAMO_MODEL


def normalize_vector(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).astype(np.float32).tolist()
