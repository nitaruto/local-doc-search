from __future__ import annotations

from pathlib import Path

import pytest

from tt_search.db import connect, ensure_schema, format_info
from tt_search.embeddings import normalize_vector
from tt_search.indexer import index_paths
from tt_search.search import search


class FakeEmbedder:
    model_name = "fake"
    dim = 3

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return normalize_vector(
            [
                lower.count("検索") + lower.count("search") + lower.count("sqlite"),
                lower.count("料理") + lower.count("カレー"),
                lower.count("旅行") + lower.count("京都"),
            ]
        )


@pytest.fixture()
def sample_roots(tmp_path: Path) -> tuple[Path, Path]:
    root1 = tmp_path / "docs1"
    root2 = tmp_path / "docs2"
    root1.mkdir()
    root2.mkdir()
    (root1 / "search.md").write_text(
        "日本語の検索テストです。\nSQLite FTS5 trigram とベクトル検索を試します。\n",
        encoding="utf-8",
    )
    (root1 / "cook.txt").write_text("カレーの料理メモです。\n玉ねぎを炒めます。\n", encoding="utf-8")
    (root2 / "travel.md").write_text("京都旅行のメモです。\n寺院を巡ります。\n", encoding="utf-8")
    (root2 / "ignored.log").write_text("検索対象外です。\n", encoding="utf-8")
    return root1, root2


def build_db(db: Path, roots: list[Path], extensions: list[str] | None = None) -> None:
    with connect(db) as con:
        index_paths(con, roots=roots, extensions=extensions, embedder=FakeEmbedder())


def test_index_multiple_roots_and_extension_filter(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots), [".md"])

    with connect(db) as con:
        info = format_info(con)

    assert info["file_count"] == 2
    assert info["chunk_count"] == 2
    assert info["metadata"]["embedding_model"] == "fake"


def test_japanese_trigram_fts_search(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots))

    with connect(db) as con:
        results = search(
            con,
            query="日本語",
            mode="fts",
            limit=10,
            candidates=10,
            embedder=None,
        )

    assert [Path(result.path).name for result in results] == ["search.md"]


def test_short_query_uses_like_fallback(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots))

    with connect(db) as con:
        results = search(
            con,
            query="京都",
            mode="fts",
            limit=10,
            candidates=10,
            embedder=None,
        )

    assert [Path(result.path).name for result in results] == ["travel.md"]
    assert results[0].source == "like"


def test_hybrid_modes_return_results(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots))

    with connect(db) as con:
        fts_vec = search(
            con,
            query="検索 sqlite",
            mode="fts-vec",
            limit=3,
            candidates=10,
            embedder=FakeEmbedder(),
        )
        vec_fts = search(
            con,
            query="検索 sqlite",
            mode="vec-fts",
            limit=3,
            candidates=10,
            embedder=FakeEmbedder(),
        )

    assert fts_vec
    assert vec_fts
    assert fts_vec[0].source == "fts-vec"
    assert vec_fts[0].source == "vec-fts"


def test_reindex_updates_changed_file(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    root1, root2 = sample_roots
    build_db(db, [root1, root2], [".md"])

    (root1 / "search.md").write_text("更新後の内容です。\n料理とカレーの話です。\n", encoding="utf-8")
    build_db(db, [root1, root2], [".md"])

    with connect(db) as con:
        results = search(
            con,
            query="カレー",
            mode="fts",
            limit=10,
            candidates=10,
            embedder=None,
        )

    assert [Path(result.path).name for result in results] == ["search.md"]


def test_schema_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    with connect(db) as con:
        ensure_schema(con, embedding_dim=3, embedding_model="fake")
        with pytest.raises(ValueError):
            ensure_schema(con, embedding_dim=4, embedding_model="other")
