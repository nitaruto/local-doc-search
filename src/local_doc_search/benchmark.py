from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Literal

from .embeddings import EmbeddingProvider

BenchmarkTask = Literal["passage", "query"]


DEFAULT_BENCHMARK_TEXTS = [
    "日本語を含む社内文書を検索するため、SQLite FTS5 とベクトル検索を組み合わせる。",
    "The search tool should support multilingual retrieval across local Markdown files.",
    "モデルロード時間と推論時間を分けて計測し、batch size ごとの違いを比較する。",
    "Hybrid search can rerank vector candidates by FTS or FTS candidates by vectors.",
    "長い文書は段落単位でchunkに分割し、Markdownのsection境界を越えないようにする。",
    "Codex session history stores user turns and final assistant answers with session metadata.",
    "PLaMo uses custom remote code while Sarashina runs through SentenceTransformer.",
    "検索クエリと文書prefixはモデルごとの推奨形式に合わせる必要がある。",
]


@dataclass(frozen=True)
class BenchmarkTiming:
    seconds: float
    vectors: int
    chars: int


def load_benchmark_texts(input_file: Path | None, *, documents: int) -> list[str]:
    if documents < 1:
        raise ValueError("documents must be >= 1")
    source = read_input_texts(input_file) if input_file is not None else DEFAULT_BENCHMARK_TEXTS
    if not source:
        raise ValueError("benchmark input is empty")
    texts: list[str] = []
    while len(texts) < documents:
        texts.extend(source[: documents - len(texts)])
    return texts


def read_input_texts(path: Path) -> list[str]:
    text = path.expanduser().read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n")]
    if len(blocks) == 1:
        blocks = [line.strip() for line in text.splitlines()]
    return [block for block in blocks if block]


def summarize_timings(timings: list[BenchmarkTiming]) -> dict[str, Any]:
    if not timings:
        raise ValueError("timings must not be empty")
    seconds = [timing.seconds for timing in timings]
    total_vectors = mean([timing.vectors for timing in timings])
    total_chars = mean([timing.chars for timing in timings])
    mean_seconds = mean(seconds)
    return {
        "repeat": len(timings),
        "seconds": seconds,
        "mean_seconds": mean_seconds,
        "median_seconds": median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "vectors_per_second": total_vectors / mean_seconds if mean_seconds > 0 else 0.0,
        "chars_per_second": total_chars / mean_seconds if mean_seconds > 0 else 0.0,
    }


def benchmark_provider(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    task: BenchmarkTask,
    warmup: int,
    repeat: int,
    clock: Callable[[], float] = perf_counter,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if task == "query":
        texts = texts[:1]

    for _ in range(warmup):
        run_embedding(provider, texts, task=task)
        if synchronize is not None:
            synchronize()

    timings: list[BenchmarkTiming] = []
    for _ in range(repeat):
        if synchronize is not None:
            synchronize()
        started_at = clock()
        vectors = run_embedding(provider, texts, task=task)
        if synchronize is not None:
            synchronize()
        elapsed = clock() - started_at
        timings.append(
            BenchmarkTiming(
                seconds=elapsed,
                vectors=len(vectors),
                chars=sum(len(text) for text in texts),
            )
        )
    return summarize_timings(timings)


def run_embedding(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    task: BenchmarkTask,
) -> list[list[float]]:
    if task == "passage":
        return provider.embed_passages(texts)
    return [provider.embed_query(texts[0])]


def synchronize_torch_device(device: str) -> None:
    if device != "mps":
        return
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "mps"):
        torch.mps.synchronize()
