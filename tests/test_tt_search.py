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
    list_indexed_roots,
    validate_embedding_compatible,
)
from tt_search.embeddings import (
    PLAMO_BACKEND,
    PLAMO_MODEL,
    PlamoEmbeddingProvider,
    create_embedding_provider,
    ensure_plamo_max_length,
    normalize_vector,
    prefix_passage,
    prefix_policy_for_model,
    prefix_query,
    refresh_plamo_rotary_cache,
    resolve_device,
    tensor_to_vectors,
)
from tt_search.indexer import (
    MarkdownSectionStrategy,
    ParagraphPackingStrategy,
    chunk_file,
    chunk_text,
    index_paths,
    strategy_for_path,
)
from tt_search.mcp import McpSearchServer
from tt_search.search import require_vec_distance, search, search_many

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


class FakeRichProgress:
    def __init__(self) -> None:
        self.added: list[tuple[str, int]] = []
        self.updates: list[dict[str, object]] = []

    def add_task(self, description: str, *, total: int) -> int:
        self.added.append((description, total))
        return 1

    def update(self, task_id: int, **kwargs: object) -> None:
        self.updates.append({"task_id": task_id, **kwargs})


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


def test_list_indexed_roots_returns_counts(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots), [".md"])

    with connect(db) as con:
        rows = list_indexed_roots(con)

    assert [Path(row.root_path).name for row in rows] == ["docs1", "docs2"]
    assert [row.file_count for row in rows] == [1, 1]
    assert [row.chunk_count for row in rows] == [1, 1]


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


def test_mcp_server_lists_and_calls_search_tool(
    tmp_path: Path, sample_roots: tuple[Path, Path]
) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots))
    server = McpSearchServer([db], device="cpu")

    initialize = server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    tools = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    call = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "日本語", "mode": "fts", "limit": 3},
            },
        }
    )

    assert initialize is not None
    assert initialize["result"]["serverInfo"]["name"] == "tt-search"
    assert tools is not None
    assert [tool["name"] for tool in tools["result"]["tools"]] == ["search", "roots"]
    assert call is not None
    payload = call["result"]["content"][0]["text"]
    assert '"relative_path": "search.md"' in payload
    assert '"start_line": 1' in payload
    assert '"end_line": 2' in payload


def test_mcp_server_roots_tool(tmp_path: Path, sample_roots: tuple[Path, Path]) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots), [".md"])
    server = McpSearchServer([db], device="cpu")

    call = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "roots",
                "arguments": {},
            },
        }
    )

    assert call is not None
    payload = call["result"]["content"][0]["text"]
    assert '"db_path":' in payload
    assert '"root_path":' in payload
    assert '"file_count": 1' in payload
    assert '"chunk_count": 1' in payload


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

    assert [(row["start_line"], row["end_line"]) for row in rows] == [(1, 6)]
    assert rows[0]["text"] == "title\n\nfirst paragraph\nsecond line\n\nlast paragraph"


def test_chunk_text_packs_short_paragraphs_until_max_chars() -> None:
    text = "aaa\n\nbbb\n\ncccc\n"

    chunks = chunk_text(text, max_chars=10)

    assert [chunk.text for chunk in chunks] == ["aaa\n\nbbb", "bbb\n\ncccc"]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 3), (3, 5)]
    assert [chunk.index for chunk in chunks] == [0, 1]


def test_chunk_text_drops_overlap_when_it_would_exceed_max_chars() -> None:
    text = "aaaa\n\nbbbb\n\ncccc\n"

    chunks = chunk_text(text, max_chars=8)

    assert [chunk.text for chunk in chunks] == ["aaaa", "bbbb", "cccc"]


def test_chunk_text_splits_long_paragraph_with_overlap() -> None:
    text = "a" * 300

    chunks = chunk_text(text, max_chars=200)

    assert [len(chunk.text) for chunk in chunks] == [200, 200, 140]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 200),
        (80, 280),
        (160, 300),
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_chunk_strategy_is_selected_by_extension() -> None:
    assert isinstance(strategy_for_path(Path("notes.md")), MarkdownSectionStrategy)
    assert isinstance(strategy_for_path(Path("notes.markdown")), MarkdownSectionStrategy)
    assert isinstance(strategy_for_path(Path("notes.txt")), ParagraphPackingStrategy)
    assert isinstance(strategy_for_path(Path("notes.unknown")), ParagraphPackingStrategy)


def test_chunk_file_uses_selected_strategy() -> None:
    chunks = chunk_file(Path("notes.md"), "aaa\n\nbbb\n", max_chars=10)

    assert [chunk.text for chunk in chunks] == ["aaa\n\nbbb"]


def test_markdown_chunks_do_not_cross_section_boundaries() -> None:
    text = "# A\n\npara a\n\n# B\n\npara b\n"

    chunks = chunk_file(Path("notes.md"), text, max_chars=100)

    assert [chunk.text for chunk in chunks] == ["# A\n\npara a", "# B\n\npara b"]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 3), (5, 7)]


def test_markdown_paragraph_overlap_stays_inside_section() -> None:
    text = "# A\n\naaa\n\nbbb\n\ncccc\n\n# B\n\nddd\n"

    chunks = chunk_file(Path("notes.md"), text, max_chars=10)

    assert [chunk.text for chunk in chunks] == [
        "# A\n\naaa",
        "aaa\n\nbbb",
        "bbb\n\ncccc",
        "# B\n\nddd",
    ]


def test_markdown_heading_inside_fenced_code_is_not_section_boundary() -> None:
    text = "# A\n\n```\n# not heading\n```\n\npara a\n\n# B\n\npara b\n"

    chunks = chunk_file(Path("notes.md"), text, max_chars=100)

    assert len(chunks) == 2
    assert chunks[0].text == "# A\n\n```\n# not heading\n```\n\npara a"
    assert chunks[1].text == "# B\n\npara b"


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


def test_rich_index_progress_reports_chunk_rate() -> None:
    times = iter([0.0, 0.0, 1.0, 2.0])
    progress = FakeRichProgress()
    reporter = cli.RichIndexProgress(progress, clock=lambda: next(times))

    reporter.on_scan_complete(2)
    reporter.on_embedding_start(path=Path("a.md"), chunks=3)
    reporter.on_file_done(path=Path("a.md"), status="indexed", chunks=3)

    assert progress.added == [("Indexing files", 2)]
    descriptions = [str(update["description"]) for update in progress.updates]
    assert "total=0 chunks, 0.00 chunks/s" in descriptions[0]
    assert "total=3 chunks, 1.50 chunks/s" in descriptions[1]


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


def test_plamo_provider_uses_custom_encode_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from transformers import AutoModel, AutoTokenizer

    calls: list[tuple[str, object]] = []

    class FakeTokenizer:
        pass

    class FakeConfig:
        max_position_embeddings = 1234

    class FakePlamoModel:
        def __init__(self) -> None:
            self.config = FakeConfig()

        def to(self, device: str) -> FakePlamoModel:
            calls.append(("to", device))
            return self

        def eval(self) -> None:
            calls.append(("eval", None))

        def encode_document(self, texts: list[str], tokenizer: FakeTokenizer) -> np.ndarray:
            calls.append(("document", texts))
            return np.array([[3.0, 4.0, 0.0] for _ in texts], dtype=np.float32)

        def encode_query(self, text: str, tokenizer: FakeTokenizer) -> np.ndarray:
            calls.append(("query", text))
            return np.array([0.0, 5.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model_name, trust_remote_code: FakeTokenizer(),
    )
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        lambda model_name, trust_remote_code, dtype: FakePlamoModel(),
    )

    provider = create_embedding_provider(model_name=PLAMO_MODEL, device="cpu", batch_size=1)

    assert isinstance(provider, PlamoEmbeddingProvider)
    assert provider.backend == PLAMO_BACKEND
    assert provider.prefix_policy == "plamo"
    assert provider.dim == 3
    assert provider._model.config.max_length == 1234
    passage_vectors = provider.embed_passages(["a", "b"])
    assert passage_vectors[0] == pytest.approx([0.6, 0.8, 0.0])
    assert passage_vectors[1] == pytest.approx([0.6, 0.8, 0.0])
    assert provider.embed_query("q") == pytest.approx([0.0, 1.0, 0.0])
    assert ("document", ["dimension probe"]) in calls
    assert ("document", ["a"]) in calls
    assert ("document", ["b"]) in calls
    assert ("query", ["q"]) in calls


def test_plamo_auto_uses_mps_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoModel, AutoTokenizer

    calls: list[tuple[str, object]] = []

    class FakeTokenizer:
        pass

    class FakeConfig:
        max_position_embeddings = 1234

    class FakePlamoModel:
        config = FakeConfig()

        def to(self, device: str) -> FakePlamoModel:
            calls.append(("to", device))
            return self

        def eval(self) -> None:
            pass

        def encode_document(self, texts: list[str], tokenizer: FakeTokenizer) -> object:
            import numpy as np

            return np.array([[3.0, 4.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr("tt_search.embeddings.mps_is_available", lambda: True)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model_name, trust_remote_code: FakeTokenizer(),
    )
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        lambda model_name, trust_remote_code, dtype: FakePlamoModel(),
    )

    provider = create_embedding_provider(model_name=PLAMO_MODEL, device="auto", batch_size=1)

    assert provider.device == "mps"
    assert ("to", "mps") in calls


def test_plamo_retries_non_finite_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoModel, AutoTokenizer

    calls: list[str] = []

    class FakeTokenizer:
        pass

    class FakeConfig:
        max_position_embeddings = 1234

    class FakePlamoModel:
        config = FakeConfig()

        def __init__(self) -> None:
            self.document_calls = 0
            self.query_calls = 0

        def to(self, device: str) -> FakePlamoModel:
            return self

        def eval(self) -> None:
            pass

        def encode_document(self, texts: list[str], tokenizer: FakeTokenizer) -> object:
            import numpy as np

            self.document_calls += 1
            calls.append(f"document:{self.document_calls}")
            if self.document_calls == 1:
                return np.array([[float("nan"), 1.0, 0.0]], dtype=np.float32)
            return np.array([[3.0, 4.0, 0.0] for _ in texts], dtype=np.float32)

        def encode_query(self, text: list[str], tokenizer: FakeTokenizer) -> object:
            import numpy as np

            self.query_calls += 1
            calls.append(f"query:{self.query_calls}")
            if self.query_calls == 1:
                return np.array([[float("nan"), 1.0, 0.0]], dtype=np.float32)
            return np.array([[0.0, 5.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model_name, trust_remote_code: FakeTokenizer(),
    )
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        lambda model_name, trust_remote_code, dtype: FakePlamoModel(),
    )

    with pytest.warns(RuntimeWarning, match="PLaMo embedding returned non-finite values"):
        provider = create_embedding_provider(model_name=PLAMO_MODEL, device="cpu", batch_size=1)

    assert provider.dim == 3
    with pytest.warns(RuntimeWarning, match="PLaMo embedding returned non-finite values"):
        assert provider.embed_query("q") == pytest.approx([0.0, 1.0, 0.0])
    assert calls[:2] == ["document:1", "document:2"]
    assert calls[-2:] == ["query:1", "query:2"]


def test_ensure_plamo_max_length_preserves_existing_value() -> None:
    class FakeConfig:
        max_length = 2048
        max_position_embeddings = 4096

    class FakeModel:
        config = FakeConfig()

    ensure_plamo_max_length(FakeModel())

    assert FakeModel.config.max_length == 2048


def test_refresh_plamo_rotary_cache() -> None:
    calls: list[tuple[int, str, str]] = []

    class FakeInvFreq:
        device = "cpu"
        dtype = "float32"

    class FakeRotaryEmbedding:
        max_position_embeddings = 4096
        inv_freq = FakeInvFreq()

        def _set_cos_sin_cache(self, *, seq_len: int, device: str, dtype: str) -> None:
            calls.append((seq_len, device, dtype))

    class FakeAttention:
        rotary_emb = FakeRotaryEmbedding()

    class FakeLayer:
        self_attn = FakeAttention()

    class FakeLayers:
        layers = [FakeLayer(), FakeLayer()]

    class FakeModel:
        layers = FakeLayers()

    refresh_plamo_rotary_cache(FakeModel())

    assert calls == [(4096, "cpu", "float32"), (4096, "cpu", "float32")]


def test_tensor_to_vectors_accepts_bfloat16_tensor() -> None:
    import torch

    vectors = tensor_to_vectors(torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.bfloat16))

    assert vectors[0] == pytest.approx([0.6, 0.8, 0.0])


def test_tensor_to_vectors_rejects_non_finite_values() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="non-finite"):
        tensor_to_vectors(np.array([[float("nan"), 1.0]], dtype=np.float32))


def test_cli_search_uses_model_from_db_metadata(
    tmp_path: Path, sample_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "index.sqlite"
    build_db(db, list(sample_roots))

    created_models: list[str] = []

    def recording_provider(model_name: str, **_: object) -> FakeEmbedder:
        created_models.append(model_name)
        return FakeEmbedder()

    monkeypatch.setattr(cli, "create_embedding_provider", recording_provider)
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


def test_require_vec_distance_rejects_null_distance() -> None:
    with pytest.raises(ValueError, match="rebuild"):
        require_vec_distance(None)


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
