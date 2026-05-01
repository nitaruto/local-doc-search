from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .db import as_json, connect, format_info
from .embeddings import DEFAULT_MODEL, SentenceTransformerEmbeddingProvider
from .indexer import index_paths
from .search import SearchMode, search

app = typer.Typer(help="Local Japanese text search with SQLite FTS5 trigram and sqlite-vec.")
console = Console()


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
    model: Annotated[str, typer.Option("--model", help="sentence-transformers model name.")] = (
        DEFAULT_MODEL
    ),
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Clear existing index first.")] = False,
) -> None:
    """Build or update a search database."""
    if not root:
        raise typer.BadParameter("At least one --root is required")
    embedder = SentenceTransformerEmbeddingProvider(model_name=model)
    with connect(db) as con:
        stats = index_paths(con, roots=root, extensions=ext, embedder=embedder, rebuild=rebuild)
    console.print(
        as_json(
            {
                "scanned_files": stats.scanned_files,
                "indexed_files": stats.indexed_files,
                "skipped_files": stats.skipped_files,
                "chunks": stats.chunks,
                "removed_files": stats.removed_files,
            }
        )
    )


@app.command()
def search_cmd(
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path.")],
    query: Annotated[str, typer.Option("--query", "-q", help="Search query.")],
    mode: Annotated[
        SearchMode, typer.Option("--mode", help="Search mode: fts, vec, fts-vec, vec-fts.")
    ] = "fts-vec",
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="Number of results.")] = 10,
    candidates: Annotated[
        int, typer.Option("--candidates", min=1, help="Candidate count before rerank.")
    ] = 50,
    model: Annotated[str, typer.Option("--model", help="sentence-transformers model name.")] = (
        DEFAULT_MODEL
    ),
    explain: Annotated[bool, typer.Option("--explain", help="Show component scores.")] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print JSON instead of a table.")
    ] = False,
) -> None:
    """Search indexed files."""
    embedder = None
    if mode in {"vec", "fts-vec", "vec-fts"}:
        embedder = SentenceTransformerEmbeddingProvider(model_name=model)
    with connect(db) as con:
        rows = search(
            con,
            query=query,
            mode=mode,
            limit=limit,
            candidates=max(candidates, limit),
            embedder=embedder,
        )
    if json_output:
        console.print(as_json([row.__dict__ for row in rows]))
        return
    print_results(rows, explain=explain)


app.command(name="search")(search_cmd)


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
    for key, value in data["metadata"].items():
        table.add_row(str(key), str(value))
    table.add_row("file_count", str(data["file_count"]))
    table.add_row("chunk_count", str(data["chunk_count"]))
    console.print(table)


def print_results(rows: list[object], *, explain: bool) -> None:
    table = Table("score", "path", "chunk", "snippet")
    if explain:
        table.add_column("fts_rank")
        table.add_column("vec_distance")
    for row in rows:
        snippet = " ".join(row.text.split())
        if len(snippet) > 160:
            snippet = f"{snippet[:157]}..."
        values = [f"{row.score:.4f}", row.path, str(row.chunk_index), snippet]
        if explain:
            values.extend(
                [
                    "" if row.fts_rank is None else f"{row.fts_rank:.4f}",
                    "" if row.vec_distance is None else f"{row.vec_distance:.4f}",
                ]
            )
        table.add_row(*values)
    console.print(table)
