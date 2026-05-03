from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from shlex import quote
from time import perf_counter
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from .benchmark import (
    BenchmarkTask,
    benchmark_provider,
    load_benchmark_texts,
    synchronize_torch_device,
)
from .client import (
    ServerSearchError,
    find_live_server,
    find_live_servers,
    find_subset_live_servers,
    search_via_server,
)
from .codex_history import (
    CODEX_HISTORY_DB,
    CODEX_HISTORY_INDEX_KIND,
    CODEX_HISTORY_MODEL,
    CODEX_SESSIONS_ROOT,
    index_codex_sessions,
    validate_codex_roots,
)
from .db import (
    as_json,
    connect,
    fingerprint_many,
    format_info,
    get_metadata,
    list_indexed_files,
    normalize_db_paths,
    validate_embedding_compatible,
)
from .embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    PLAMO_MODEL,
    DeviceOption,
    EmbeddingProvider,
    create_embedding_provider,
)
from .indexer import index_paths
from .mcp import run_mcp_server
from .search import SearchMode, SearchResult, resolve_search, search_many
from .server import run_server

app = typer.Typer(help="Local multilingual text search with SQLite FTS5 trigram and sqlite-vec.")
console = Console()
SARASHINA_BENCHMARK_MODEL = "sbintuitions/sarashina-embedding-v2-1b"


@app.command()
def index(
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path.")],
    root: Annotated[
        list[Path],
        typer.Option("--root", help="Directory or file to index. Can be specified multiple times."),
    ],
    ext: Annotated[
        list[str] | None,
        typer.Option("--ext", help="File extension to include. Can be specified multiple times."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            help="Regex pattern matched against root-relative POSIX paths. Can be repeated.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", help="sentence-transformers model name.")] = (
        DEFAULT_MODEL
    ),
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, help="Embedding batch size for indexing.")
    ] = DEFAULT_BATCH_SIZE,
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Clear existing index first.")] = False,
) -> None:
    """Build or update a search database."""
    if not root:
        raise typer.BadParameter("At least one --root is required")
    print_index_start_summary(
        command="index",
        db=db,
        roots=root,
        model=model,
        device=device,
        batch_size=batch_size,
        rebuild=rebuild,
        extensions=ext,
        exclude_patterns=exclude,
    )
    embedder = create_embedding_provider(
        model_name=model,
        device=device,
        batch_size=batch_size,
    )
    with connect(db) as con:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            reporter = RichIndexProgress(progress)
            stats = index_paths(
                con,
                roots=root,
                extensions=ext,
                embedder=embedder,
                rebuild=rebuild,
                progress=reporter,
                exclude_patterns=exclude,
            )
    console.print(
        as_json(
            {
                "scanned_files": stats.scanned_files,
                "excluded_files": stats.excluded_files,
                "indexed_files": stats.indexed_files,
                "skipped_files": stats.skipped_files,
                "chunks": stats.chunks,
                "removed_files": stats.removed_files,
            }
        )
    )


@app.command(name="benchmark-embeddings")
def benchmark_embeddings_cmd(
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            help=(
                "Embedding model to benchmark. Can be specified multiple times. "
                "Defaults to PLaMo and Sarashina."
            ),
        ),
    ] = None,
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, help="Embedding batch size.")
    ] = DEFAULT_BATCH_SIZE,
    documents: Annotated[
        int, typer.Option("--documents", min=1, help="Number of benchmark documents.")
    ] = 32,
    repeat: Annotated[
        int, typer.Option("--repeat", min=1, help="Measured encode repetitions.")
    ] = 3,
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Warmup encode repetitions.")
    ] = 1,
    task: Annotated[
        BenchmarkTask,
        typer.Option("--task", help="Embedding task to benchmark: passage or query."),
    ] = "passage",
    input_file: Annotated[
        Path | None,
        typer.Option(
            "--input-file",
            help="UTF-8 text file. Paragraphs, or non-empty lines, are benchmark documents.",
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSON instead of key=value lines.")
    ] = False,
) -> None:
    """Benchmark embedding model load and encode speed."""
    models = model or [PLAMO_MODEL, SARASHINA_BENCHMARK_MODEL]
    texts = load_benchmark_texts(input_file, documents=documents)
    rows: list[dict[str, object]] = []
    for model_name in models:
        started_at = perf_counter()
        provider = create_embedding_provider(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        provider_device = provider.device
        synchronize_torch_device(provider_device)
        load_seconds = perf_counter() - started_at
        encode = benchmark_provider(
            provider,
            texts,
            task=task,
            warmup=warmup,
            repeat=repeat,
            synchronize=lambda device=provider_device: synchronize_torch_device(device),
        )
        rows.append(
            {
                "model": provider.model_name,
                "backend": provider.backend,
                "device": provider.device,
                "batch_size": provider.batch_size,
                "prefix_policy": provider.prefix_policy,
                "dim": provider.dim,
                "task": task,
                "documents": len(texts) if task == "passage" else 1,
                "input_chars": sum(len(text) for text in texts)
                if task == "passage"
                else len(texts[0]),
                "load_seconds": load_seconds,
                "encode": encode,
            }
        )
    if json_output:
        console.print(as_json(rows))
        return
    for row in rows:
        encode = cast(dict[str, Any], row["encode"])
        load_seconds = cast(float, row["load_seconds"])
        typer.echo(
            " ".join(
                [
                    f"model={quote(str(row['model']))}",
                    f"backend={row['backend']}",
                    f"device={row['device']}",
                    f"batch_size={row['batch_size']}",
                    f"task={row['task']}",
                    f"documents={row['documents']}",
                    f"dim={row['dim']}",
                    f"load_seconds={load_seconds:.4f}",
                    f"mean_seconds={float(encode['mean_seconds']):.4f}",
                    f"vectors_per_second={float(encode['vectors_per_second']):.2f}",
                    f"chars_per_second={float(encode['chars_per_second']):.2f}",
                ]
            )
        )


class RichIndexProgress:
    def __init__(self, progress: Progress, *, clock: Callable[[], float] = perf_counter) -> None:
        self.progress = progress
        self.task_id: TaskID | None = None
        self.clock = clock
        self.started_at = clock()
        self.processed_chunks = 0
        self.current_file_embedded_chunks = 0

    def on_scan_complete(self, total_files: int) -> None:
        self.started_at = self.clock()
        self.task_id = self.progress.add_task("Indexing files", total=total_files)

    def on_file_done(self, *, path: Path, status: str, chunks: int = 0) -> None:
        if self.task_id is None:
            return
        detail = self.chunk_rate_label()
        if chunks:
            detail = f"{chunks} chunks, {detail}"
        description = self.progress_description(status=status, path=path, detail=detail)
        self.progress.update(self.task_id, description=description, advance=1)

    def on_embedding_start(self, *, path: Path, chunks: int) -> None:
        if self.task_id is None:
            return
        self.current_file_embedded_chunks = 0
        self.progress.update(
            self.task_id,
            description=self.progress_description(
                status="embedding",
                path=path,
                detail=f"{chunks} chunks, {self.chunk_rate_label()}",
            ),
        )

    def on_embedding_batch_done(
        self, *, path: Path, embedded_chunks: int, total_chunks: int
    ) -> None:
        if self.task_id is None:
            return
        self.processed_chunks += embedded_chunks - self.current_file_embedded_chunks
        self.current_file_embedded_chunks = embedded_chunks
        self.progress.update(
            self.task_id,
            description=self.progress_description(
                status="embedding",
                path=path,
                detail=(
                    f"{embedded_chunks}/{total_chunks} chunks, "
                    f"{self.chunk_rate_label()}"
                ),
            ),
        )

    def progress_description(self, *, status: str, path: Path, detail: str) -> str:
        return f"{status}: {path.name}\n{detail}"

    def chunk_rate_label(self) -> str:
        elapsed = max(self.clock() - self.started_at, 1e-9)
        rate = self.processed_chunks / elapsed
        return f"total={self.processed_chunks} chunks, {rate:.2f} chunks/s"


def search_cmd(
    db: Annotated[
        list[Path] | None,
        typer.Option("--db", help="SQLite DB path. Can be specified multiple times."),
    ] = None,
    query: Annotated[
        str | None, typer.Option("--query", "-q", help="Semantic/vector search query.")
    ] = None,
    pattern: Annotated[
        str | None, typer.Option("--pattern", help="FTS5 MATCH pattern.")
    ] = None,
    mode: Annotated[
        SearchMode | None,
        typer.Option("--mode", help="Search mode: fts, vec, fts-vec, vec-fts."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="Number of results.")] = 10,
    candidates: Annotated[
        int, typer.Option("--candidates", min=1, help="Candidate count before rerank.")
    ] = 50,
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    explain: Annotated[bool, typer.Option("--explain", help="Show component scores.")] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSON instead of a table.")
    ] = False,
    no_server: Annotated[
        bool, typer.Option("--no-server", help="Do not use a running local-doc-search server.")
    ] = False,
) -> None:
    """Search indexed files."""
    rows = run_cli_search(
        db=db,
        query=query,
        pattern=pattern,
        mode=mode,
        limit=limit,
        candidates=candidates,
        device=device,
        no_server=no_server,
    )
    output_results(rows, json_output=json_output, explain=explain)


app.command(name="search")(search_cmd)


def print_index_start_summary(
    *,
    command: str,
    db: Path,
    roots: list[Path],
    model: str,
    device: DeviceOption,
    batch_size: int,
    rebuild: bool,
    extensions: list[str] | None,
    exclude_patterns: list[str] | None,
) -> None:
    typer.echo(f"== local-doc-search {command} start ==")
    typer.echo(f"db={db.expanduser()}")
    typer.echo(f"model={model}")
    typer.echo(f"device={device}")
    typer.echo(f"batch_size={batch_size}")
    typer.echo(f"rebuild={str(rebuild).lower()}")
    typer.echo(f"roots={','.join(str(root.expanduser()) for root in roots)}")
    typer.echo(f"extensions={','.join(extensions) if extensions else 'default'}")
    typer.echo(f"exclude={','.join(exclude_patterns) if exclude_patterns else ''}")


@app.command(name="tui-search")
def tui_search_cmd(
    db: Annotated[
        list[Path] | None,
        typer.Option("--db", help="SQLite DB path. Can be specified multiple times."),
    ] = None,
    query: Annotated[
        str | None, typer.Option("--query", "-q", help="Semantic/vector search query.")
    ] = None,
    pattern: Annotated[
        str | None, typer.Option("--pattern", help="FTS5 MATCH pattern.")
    ] = None,
    mode: Annotated[
        SearchMode | None,
        typer.Option("--mode", help="Search mode: fts, vec, fts-vec, vec-fts."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="Number of results.")] = 20,
    candidates: Annotated[
        int, typer.Option("--candidates", min=1, help="Candidate count before rerank.")
    ] = 50,
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    no_server: Annotated[
        bool, typer.Option("--no-server", help="Do not use a running local-doc-search server.")
    ] = False,
    pager: Annotated[
        str, typer.Option("--pager", help="Pager/editor command used to open the selected result.")
    ] = "less",
    preview_lines: Annotated[
        int,
        typer.Option(
            "--preview-lines",
            min=1,
            help="Number of context lines shown in the fzf preview.",
        ),
    ] = 80,
    no_preview: Annotated[
        bool, typer.Option("--no-preview", help="Disable fzf preview window.")
    ] = False,
    preview_window: Annotated[
        str,
        typer.Option("--preview-window", help="fzf preview-window layout."),
    ] = "down:60%",
) -> None:
    """Select a search result with fzf and open the document at the hit line."""
    rows = run_cli_search(
        db=db,
        query=query,
        pattern=pattern,
        mode=mode,
        limit=limit,
        candidates=candidates,
        device=device,
        no_server=no_server,
    )
    open_result_from_fzf(
        rows,
        pager=pager,
        preview_lines=preview_lines,
        no_preview=no_preview,
        preview_window=preview_window,
    )


def run_cli_search(
    *,
    db: list[Path] | None,
    query: str | None,
    pattern: str | None,
    mode: SearchMode | None,
    limit: int,
    candidates: int,
    device: DeviceOption,
    no_server: bool,
) -> list[SearchResult]:
    db_paths = normalize_db_paths(db or [])
    resolved = resolve_cli_search(query=query, pattern=pattern, mode=mode)
    if not db_paths:
        if no_server:
            raise typer.BadParameter("--db is required when --no-server is used")
        registries = find_live_servers()
        if not registries:
            raise typer.BadParameter("No live local-doc-search server found. Specify --db.")
        if len(registries) > 1:
            raise typer.BadParameter(
                "Multiple live local-doc-search servers found. Specify --db."
            )
        rows = search_via_server(
            registries[0],
            vector_query=resolved.vector_query,
            fts_query=resolved.fts_query,
            fts_is_pattern=resolved.fts_is_pattern,
            mode=resolved.mode,
            limit=limit,
            candidates=max(candidates, limit),
        )
        return rows

    if not no_server:
        registry = find_live_server(db_paths)
        if registry is None:
            subset_registries = find_subset_live_servers(db_paths)
            if len(subset_registries) > 1:
                raise typer.BadParameter(
                    "Multiple live local-doc-search servers contain the requested DBs. "
                    "Use an exact --db set or stop duplicate servers."
                )
            if subset_registries:
                registry = subset_registries[0]
        if registry is not None and server_device_matches(registry, device):
            try:
                rows = search_via_server(
                    registry,
                    db_paths=db_paths,
                    vector_query=resolved.vector_query,
                    fts_query=resolved.fts_query,
                    fts_is_pattern=resolved.fts_is_pattern,
                    mode=resolved.mode,
                    limit=limit,
                    candidates=max(candidates, limit),
                )
                return rows
            except ServerSearchError as exc:
                console.print(f"[yellow]Warning: {exc}. Falling back to local search.[/yellow]")

    embedder = build_search_embedder(db_paths, mode=resolved.mode, device=device)
    return search_many(
        db_paths,
        vector_query=resolved.vector_query,
        fts_query=resolved.fts_query,
        fts_is_pattern=resolved.fts_is_pattern,
        mode=resolved.mode,
        limit=limit,
        candidates=max(candidates, limit),
        embedder=embedder,
    )


@app.command(name="codex-index")
def codex_index_cmd(
    root: Annotated[
        list[Path] | None,
        typer.Option(
            "--root",
            help="Codex sessions root or JSONL file. Defaults to ~/.codex/sessions.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", help="sentence-transformers model name.")] = (
        CODEX_HISTORY_MODEL
    ),
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, help="Embedding batch size for indexing.")
    ] = DEFAULT_BATCH_SIZE,
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Clear existing index first.")] = False,
) -> None:
    """Build or update the fixed Codex history search database."""
    try:
        roots = validate_codex_roots(root or [CODEX_SESSIONS_ROOT])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_index_start_summary(
        command="codex-index",
        db=CODEX_HISTORY_DB,
        roots=roots,
        model=model,
        device=device,
        batch_size=batch_size,
        rebuild=rebuild,
        extensions=None,
        exclude_patterns=None,
    )
    embedder = create_embedding_provider(
        model_name=model,
        device=device,
        batch_size=batch_size,
    )
    with connect(CODEX_HISTORY_DB) as con:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            reporter = RichIndexProgress(progress)
            stats = index_codex_sessions(
                con,
                roots=roots,
                embedder=embedder,
                rebuild=rebuild,
                progress=reporter,
            )
    console.print(
        as_json(
            {
                "db": str(CODEX_HISTORY_DB),
                "scanned_files": stats.scanned_files,
                "excluded_files": stats.excluded_files,
                "indexed_files": stats.indexed_files,
                "skipped_files": stats.skipped_files,
                "chunks": stats.chunks,
                "removed_files": stats.removed_files,
            }
        )
    )


@app.command(name="codex-search")
def codex_search_cmd(
    query: Annotated[
        str | None, typer.Option("--query", "-q", help="Semantic/vector search query.")
    ] = None,
    pattern: Annotated[
        str | None, typer.Option("--pattern", help="FTS5 MATCH pattern.")
    ] = None,
    mode: Annotated[
        SearchMode | None,
        typer.Option("--mode", help="Search mode: fts, vec, fts-vec, vec-fts."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="Number of results.")] = 10,
    candidates: Annotated[
        int, typer.Option("--candidates", min=1, help="Candidate count before rerank.")
    ] = 50,
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    explain: Annotated[bool, typer.Option("--explain", help="Show component scores.")] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSON instead of a table.")
    ] = False,
    no_server: Annotated[
        bool, typer.Option("--no-server", help="Do not use a running local-doc-search server.")
    ] = False,
) -> None:
    """Search the fixed Codex history search database."""
    validate_codex_history_db(CODEX_HISTORY_DB)
    resolved = resolve_cli_search(query=query, pattern=pattern, mode=mode)
    if not no_server:
        registry = find_live_server([CODEX_HISTORY_DB])
        if registry is not None and server_device_matches(registry, device):
            try:
                rows = search_via_server(
                    registry,
                    vector_query=resolved.vector_query,
                    fts_query=resolved.fts_query,
                    fts_is_pattern=resolved.fts_is_pattern,
                    mode=resolved.mode,
                    limit=limit,
                    candidates=max(candidates, limit),
                )
                output_results(rows, json_output=json_output, explain=explain)
                return
            except ServerSearchError as exc:
                console.print(f"[yellow]Warning: {exc}. Falling back to local search.[/yellow]")

    embedder = build_search_embedder([CODEX_HISTORY_DB], mode=resolved.mode, device=device)
    rows = search_many(
        [CODEX_HISTORY_DB],
        vector_query=resolved.vector_query,
        fts_query=resolved.fts_query,
        fts_is_pattern=resolved.fts_is_pattern,
        mode=resolved.mode,
        limit=limit,
        candidates=max(candidates, limit),
        embedder=embedder,
    )
    output_results(rows, json_output=json_output, explain=explain)


@app.command(name="codex-server")
def codex_server_cmd(
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=0, help="Bind port. Use 0 for auto.")] = 0,
) -> None:
    """Run a local search server for the fixed Codex history database."""
    validate_codex_history_db(CODEX_HISTORY_DB)
    run_server([CODEX_HISTORY_DB], host=host, port=port, device=device)


@app.command()
def server(
    db: Annotated[
        list[Path],
        typer.Option("--db", help="SQLite DB path. Can be specified multiple times."),
    ],
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=0, help="Bind port. Use 0 for auto.")] = 0,
) -> None:
    """Run a local search server for one or more compatible DBs."""
    db_paths = normalize_db_paths(db)
    if not db_paths:
        raise typer.BadParameter("At least one --db is required")
    run_server(db_paths, host=host, port=port, device=device)


@app.command()
def mcp(
    db: Annotated[
        list[Path],
        typer.Option("--db", help="SQLite DB path. Can be specified multiple times."),
    ],
    device: Annotated[
        DeviceOption, typer.Option("--device", help="Embedding device: auto, cpu, or mps.")
    ] = "auto",
) -> None:
    """Run a stdio MCP server for coding agents."""
    db_paths = normalize_db_paths(db)
    if not db_paths:
        raise typer.BadParameter("At least one --db is required")
    run_mcp_server(db_paths, device=device)


def build_search_embedder(
    db_paths: list[Path],
    *,
    mode: SearchMode,
    device: DeviceOption,
) -> EmbeddingProvider | None:
    if mode == "fts":
        return None
    fingerprints = fingerprint_many(db_paths)
    metadata = validate_embedding_compatible(fingerprints)
    model = metadata.get("embedding_model")
    if model is None:
        raise typer.BadParameter(
            "DB does not contain embedding metadata. Rebuild it with `local-doc-search index`."
        )
    return create_embedding_provider(model_name=model, device=device)


def resolve_cli_search(
    *,
    query: str | None,
    pattern: str | None,
    mode: SearchMode | None,
):
    try:
        return resolve_search(query=query, pattern=pattern, mode=mode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def server_device_matches(registry: dict[str, object], device: DeviceOption) -> bool:
    if device == "auto":
        return True
    return registry.get("device") == device


def validate_codex_history_db(db_path: Path) -> None:
    if not db_path.exists():
        raise typer.BadParameter(
            f"Codex history DB does not exist: {db_path}. Run `local-doc-search codex-index` first."
        )
    with connect(db_path) as con:
        index_kind = get_metadata(con, "index_kind")
    if index_kind != CODEX_HISTORY_INDEX_KIND:
        raise typer.BadParameter(
            f"DB is not a Codex history index: {db_path}. "
            "Run `local-doc-search codex-index --rebuild`."
        )


def open_result_from_fzf(
    rows: list[SearchResult],
    *,
    pager: str,
    preview_lines: int,
    no_preview: bool,
    preview_window: str,
) -> None:
    if not rows:
        console.print("[yellow]No results.[/yellow]")
        return
    if shutil.which("fzf") is None:
        raise typer.BadParameter(
            "fzf is required for tui-search. Install fzf or use search --json."
        )

    lines = [fzf_line(row) for row in rows]
    cmd = [
        "fzf",
        "--delimiter=\t",
        "--with-nth=1",
        "--height=100%",
        "--header=Enter: open selected result",
    ]
    if no_preview:
        cmd.append("--no-preview")
    else:
        cmd.extend(
            [
                "--preview",
                preview_command(preview_lines),
                f"--preview-window={preview_window}",
            ]
        )
    proc = subprocess.run(
        cmd,
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 130 or not proc.stdout.strip():
        return
    if proc.returncode != 0:
        raise typer.Exit(proc.returncode)
    row = decode_fzf_selection(proc.stdout)
    open_result_in_pager(row, pager=pager)


def fzf_line(row: SearchResult) -> str:
    location = f"{row.relative_path}:{row.start_line}-{row.end_line}"
    label = f"{row.score:8.4f}  {row.source:<7}  {location}"
    if row.db_path:
        label = f"{row.score:8.4f}  {row.source:<7}  {Path(row.db_path).name}  {location}"
    encoded = encode_result(row)
    return f"{label}\t{encoded}"


def encode_result(row: SearchResult) -> str:
    data = json.dumps(row.__dict__, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(data).decode("ascii")


def decode_fzf_selection(selection: str) -> SearchResult:
    encoded = selection.rstrip("\n").split("\t")[-1]
    data = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    return SearchResult(**data)


def preview_command(preview_lines: int) -> str:
    script = r"""
import base64
import json
import sys

item = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode("utf-8"))
context = int(sys.argv[2])
path = item["path"]
start = max(int(item["start_line"]), 1)
end = max(int(item["end_line"]), start)
before = max(context // 6, 3)
after = max(context - before, 1)
from_line = max(start - before, 1)
to_line = max(end + after, from_line)
try:
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no < from_line:
                continue
            if line_no > to_line:
                break
            marker = ">" if start <= line_no <= end else " "
            print(f"{marker}{line_no:6d} {line}", end="")
except OSError as exc:
    print(f"preview error: {exc}", file=sys.stderr)
"""
    return f"{quote(sys.executable)} -c {quote(script)} {{2}} {preview_lines}"


def open_result_in_pager(row: SearchResult, *, pager: str) -> None:
    executable = shutil.which(pager)
    if executable is None:
        raise typer.BadParameter(f"Pager/editor command not found: {pager}")
    start_line = str(max(row.start_line, 1))
    if Path(executable).name in {"less", "vim", "nvim", "vi"}:
        cmd = [executable, f"+{start_line}", row.path]
    elif Path(executable).name == "bat":
        cmd = [executable, "--style=numbers", "--highlight-line", start_line, row.path]
    else:
        cmd = [executable, row.path]
    raise typer.Exit(subprocess.run(cmd, check=False).returncode)


def output_results(rows: Sequence[SearchResult], *, json_output: bool, explain: bool) -> None:
    if json_output:
        console.print(as_json([row.__dict__ for row in rows]))
        return
    print_results(rows, explain=explain)


@app.command()
def info(
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show database metadata."""
    with connect(db) as con:
        data = format_info(con)
    if json_output:
        console.print(as_json(data))
        return
    table = Table("key", "value")
    metadata = cast(dict[str, str], data["metadata"])
    for key, value in metadata.items():
        table.add_row(str(key), str(value))
    table.add_row("file_count", str(data["file_count"]))
    table.add_row("chunk_count", str(data["chunk_count"]))
    console.print(table)


@app.command(name="files")
def files_cmd(
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """List files indexed in a database."""
    with connect(db) as con:
        rows = list_indexed_files(con)
    if json_output:
        console.print(as_json([row.__dict__ for row in rows]))
        return
    for row in rows:
        typer.echo(
            " ".join(
                [
                    f"relative_path={quote(row.relative_path)}",
                    f"path={quote(row.path)}",
                    f"root_path={quote(row.root_path)}",
                    f"size={row.size}",
                    f"mtime_ns={row.mtime_ns}",
                    f"content_hash={row.content_hash}",
                ]
            )
        )


def print_results(rows: Sequence[SearchResult], *, explain: bool) -> None:
    show_db_path = any(row.db_path for row in rows)
    show_session = any(getattr(row, "session_id", None) for row in rows)
    columns = ["score"]
    if show_db_path:
        columns.append("db_path")
    if show_session:
        columns.extend(["session_id", "cwd", "role", "timestamp", "line_no"])
    columns.extend(["path", "relative_path", "lines", "chunk", "snippet"])
    table = Table(*columns)
    if explain:
        table.add_column("fts_rank")
        table.add_column("vec_distance")
    for row in rows:
        snippet = " ".join(row.text.split())
        if len(snippet) > 160:
            snippet = f"{snippet[:157]}..."
        values = [f"{row.score:.4f}"]
        if show_db_path:
            values.append(row.db_path or "")
        if show_session:
            values.extend(
                [
                    row.session_id or "",
                    row.cwd or "",
                    row.role or "",
                    row.timestamp or "",
                    "" if getattr(row, "line_no", None) is None else str(row.line_no),
                ]
            )
        values.extend(
            [
                row.path,
                row.relative_path,
                f"{row.start_line}-{row.end_line}",
                str(row.chunk_index),
                snippet,
            ]
        )
        if explain:
            values.extend(
                [
                    "" if row.fts_rank is None else f"{row.fts_rank:.4f}",
                    "" if row.vec_distance is None else f"{row.vec_distance:.4f}",
                ]
            )
        table.add_row(*values)
    console.print(table)
