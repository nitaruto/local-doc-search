from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_search import cli
from tt_search.db import (
    connect,
    ensure_schema,
    fingerprint_many,
    format_info,
    list_indexed_files,
    validate_embedding_compatible,
)
from tt_search.embeddings import (
    normalize_vector,
    prefix_passage,
    prefix_policy_for_model,
    prefix_query,
    resolve_device,
)
from tt_search.indexer import index_paths
from tt_search.search import search, search_many

runner = CliRunner()


class FakeEmbedder:
    model_name = "fake"
    dim = 3
    backend = "fake"
    device = "cpu"
    batch_size = 2
    prefix_policy = "fake"

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


class RecordingProgress:
    def __init__(self) -> None:
        self.total_files: int | None = None
        self.events: list[tuple[str, str, int]] = []

    def on_scan_complete(self, total_files: int) -> None:
        self.total_files = total_files

    def on_file_done(self, *, path: Path, status: str, chunks: int = 0) -> None:
        self.events.append((status, path.name, chunks))

    def on_embedding_start(self, *, path: Path, chunks: int) -> None:
        self.events.append(("embedding", path.name, chunks))


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
    (root1 / "cook.txt").write_text(
        "カレーの料理メモです。\n玉ねぎを炒めます。\n", encoding="utf-8"
    )
    (root2 / "travel.md").write_text("京都旅行のメモです。\n寺院を巡ります。\n", encoding="utf-8")
    (root2 / "ignored.log").write_text("検索対象外です。\n", encoding="utf-8")
    return root1, root2


def build_db(db: Path, roots: list[Path], extensions: list[str] | None = None) -> None:
    with connect(db) as con:
        index_paths(con, roots=roots, extensions=extensions, embedder=FakeEmbedder())


def test_index_multiple_roots_and_extension_filter(
    tmp_path: Path, sample_roots: tuple[Path, Path]
) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots), [".md"])

    with connect(db) as con:
        info = format_info(con)
        rows = con.execute(
            "SELECT path, relative_path FROM files ORDER BY relative_path"
        ).fetchall()

    assert info["file_count"] == 2
    assert info["chunk_count"] == 2
    assert info["metadata"]["embedding_model"] == "fake"
    assert [row["relative_path"] for row in rows] == ["search.md", "travel.md"]


def test_index_excludes_relative_path_regex(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    archive = root / "archive"
    root.mkdir()
    archive.mkdir()
    (root / "keep.md").write_text("検索対象です。\n", encoding="utf-8")
    (archive / "old.md").write_text("除外対象です。\n", encoding="utf-8")
    (root / "memo.tmp.md").write_text("一時ファイルです。\n", encoding="utf-8")
    db = tmp_path / "index.sqlite"

    with connect(db) as con:
        stats = index_paths(
            con,
            roots=[root],
            extensions=[".md"],
            embedder=FakeEmbedder(),
            exclude_patterns=[r"^archive/", r"\.tmp\.md$"],
        )
        rows = con.execute("SELECT relative_path FROM files ORDER BY relative_path").fetchall()

    assert stats.scanned_files == 1
    assert stats.excluded_files == 2
    assert [row["relative_path"] for row in rows] == ["keep.md"]


def test_list_indexed_files_returns_relative_path_order(
    tmp_path: Path, sample_roots: tuple[Path, Path]
) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots), [".md"])

    with connect(db) as con:
        rows = list_indexed_files(con)

    assert [row.relative_path for row in rows] == ["search.md", "travel.md"]
    assert all(Path(row.path).is_absolute() for row in rows)
    assert all(Path(row.root_path).is_absolute() for row in rows)
    assert all(row.size > 0 for row in rows)


def test_cli_files_outputs_json(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots), [".md"])

    result = runner.invoke(cli.app, ["files", "--db", str(db), "--json"])

    assert result.exit_code == 0
    assert '"relative_path": "search.md"' in result.stdout
    assert '"relative_path": "travel.md"' in result.stdout
    assert '"content_hash":' in result.stdout


def test_cli_files_outputs_one_file_per_line(
    tmp_path: Path, sample_roots: tuple[Path, Path]
) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots), [".md"])

    result = runner.invoke(cli.app, ["files", "--db", str(db)])

    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("relative_path=") for line in lines)
    assert all(" path=" in line for line in lines)
    assert all(" root_path=" in line for line in lines)
    assert all(" content_hash=" in line for line in lines)
    assert not any("┏" in line or "│" in line for line in lines)


def test_index_exclude_applies_per_root_relative_path(tmp_path: Path) -> None:
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    (root1 / "archive").mkdir(parents=True)
    (root2 / "archive").mkdir(parents=True)
    (root1 / "archive" / "old.md").write_text("除外1\n", encoding="utf-8")
    (root2 / "archive" / "old.md").write_text("除外2\n", encoding="utf-8")
    (root1 / "keep.md").write_text("保持1\n", encoding="utf-8")
    (root2 / "keep.md").write_text("保持2\n", encoding="utf-8")
    db = tmp_path / "index.sqlite"

    with connect(db) as con:
        stats = index_paths(
            con,
            roots=[root1, root2],
            extensions=[".md"],
            embedder=FakeEmbedder(),
            exclude_patterns=[r"^archive/"],
        )

    assert stats.scanned_files == 2
    assert stats.excluded_files == 2


def test_index_exclude_invalid_regex_raises(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    db = tmp_path / "index.sqlite"

    with connect(db) as con, pytest.raises(re.error):
        index_paths(
            con,
            roots=[root],
            extensions=[".md"],
            embedder=FakeEmbedder(),
            exclude_patterns=["["],
        )


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
    assert [result.relative_path for result in results] == ["search.md"]
    assert [(result.start_line, result.end_line) for result in results] == [(1, 2)]


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

    (root1 / "search.md").write_text(
        "更新後の内容です。\n料理とカレーの話です。\n", encoding="utf-8"
    )
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


def test_chunk_line_numbers_are_indexed(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "lines.md").write_text(
        "title\n\nfirst paragraph\nsecond line\n\nlast paragraph\n",
        encoding="utf-8",
    )
    db = tmp_path / "index.sqlite"
    build_db(db, [root], [".md"])

    with connect(db) as con:
        rows = con.execute(
            """
            SELECT chunk_index, start_line, end_line, text
            FROM chunks
            ORDER BY chunk_index
            """
        ).fetchall()

    assert [(row["start_line"], row["end_line"]) for row in rows] == [(1, 1), (3, 4), (6, 6)]


def test_index_progress_reports_scan_and_file_events(
    tmp_path: Path, sample_roots: tuple[Path, Path]
) -> None:
    db = tmp_path / "index.sqlite"
    progress = RecordingProgress()

    with connect(db) as con:
        index_paths(
            con,
            roots=list(sample_roots),
            extensions=[".md"],
            embedder=FakeEmbedder(),
            progress=progress,
        )

    assert progress.total_files == 2
    assert ("embedding", "search.md", 1) in progress.events
    assert ("indexed", "search.md", 1) in progress.events
    assert ("embedding", "travel.md", 1) in progress.events
    assert ("indexed", "travel.md", 1) in progress.events


def test_schema_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    with connect(db) as con:
        ensure_schema(con, embedding_dim=3, embedding_model="fake")
        with pytest.raises(ValueError):
            ensure_schema(con, embedding_dim=4, embedding_model="other")


def test_embedding_metadata_is_saved(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots))

    with connect(db) as con:
        info = format_info(con)

    assert info["metadata"]["embedding_backend"] == "fake"
    assert info["metadata"]["embedding_device"] == "cpu"
    assert info["metadata"]["embedding_batch_size"] == "2"
    assert info["metadata"]["embedding_prefix_policy"] == "fake"


def test_device_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tt_search.embeddings.mps_is_available", lambda: True)
    assert resolve_device("auto") == "mps"
    assert resolve_device("mps") == "mps"

    monkeypatch.setattr("tt_search.embeddings.mps_is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("mps")


def test_model_prefix_policy() -> None:
    assert prefix_policy_for_model("intfloat/multilingual-e5-small") == "e5"
    assert prefix_query("検索", "e5") == "query: 検索"
    assert prefix_passage("文章", "e5") == "passage: 文章"

    assert prefix_policy_for_model("cl-nagoya/ruri-v3-70m") == "ruri-v3"
    assert prefix_query("検索", "ruri-v3") == "検索クエリ: 検索"
    assert prefix_passage("文章", "ruri-v3") == "検索文書: 文章"

    assert prefix_policy_for_model("pfnet/plamo-embedding-1b") == "plamo"
    assert prefix_query("検索", "plamo") == "検索"
    assert prefix_passage("文章", "plamo") == "文章"


def test_cli_search_uses_model_from_db_metadata(
    tmp_path: Path, sample_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots))

    created_models: list[str] = []

    class RecordingEmbedder(FakeEmbedder):
        def __init__(self, model_name: str, **_: object) -> None:
            created_models.append(model_name)

    monkeypatch.setattr(cli, "SentenceTransformerEmbeddingProvider", RecordingEmbedder)
    cli.search_cmd(
        db=[db],
        query="検索 sqlite",
        mode="vec",
        limit=1,
        candidates=3,
        no_server=True,
        explain=False,
        json_output=True,
    )

    assert created_models == ["fake"]


def test_search_many_merges_multiple_dbs(tmp_path: Path) -> None:
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    (root1 / "search.md").write_text("検索とsqliteのメモです。\n", encoding="utf-8")
    (root2 / "travel.md").write_text("京都旅行のメモです。\n", encoding="utf-8")
    db1 = tmp_path / "one.sqlite"
    db2 = tmp_path / "two.sqlite"
    build_db(db1, [root1])
    build_db(db2, [root2])

    results = search_many(
        [db1, db2],
        query="メモ",
        mode="fts",
        limit=10,
        candidates=10,
        embedder=None,
    )

    assert {Path(result.db_path or "").name for result in results} == {"one.sqlite", "two.sqlite"}
    assert {Path(result.path).name for result in results} == {"search.md", "travel.md"}


def test_embedding_compatibility_rejects_mismatched_model(
    tmp_path: Path, sample_roots: tuple[Path, Path]
) -> None:
    db1 = tmp_path / "one.sqlite"
    db2 = tmp_path / "two.sqlite"
    build_db(db1, [sample_roots[0]])
    build_db(db2, [sample_roots[1]])
    with connect(db2) as con:
        con.execute(
            """
            INSERT INTO metadata(key, value) VALUES ('embedding_model', 'other')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )

    fingerprints = fingerprint_many([db1, db2])
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_embedding_compatible(fingerprints)
