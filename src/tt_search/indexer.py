from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .db import ensure_schema, serialize_vector, set_embedding_metadata
from .embeddings import EmbeddingProvider

DEFAULT_EXTENSIONS = [".txt", ".md", ".markdown", ".rst"]
MAX_CHARS = 1200
OVERLAP_CHARS = 120


@dataclass(frozen=True)
class CandidateFile:
    path: Path
    root_path: Path
    relative_path: Path


@dataclass(frozen=True)
class IndexedFile:
    path: Path
    root_path: Path
    relative_path: Path
    size: int
    mtime_ns: int
    content_hash: str
    text: str


@dataclass(frozen=True)
class Chunk:
    index: int
    start_offset: int
    end_offset: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class IndexStats:
    scanned_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    chunks: int = 0
    removed_files: int = 0


class IndexProgress(Protocol):
    def on_scan_complete(self, total_files: int) -> None:
        """Called after candidate files are discovered."""

    def on_file_done(self, *, path: Path, status: str, chunks: int = 0) -> None:
        """Called after a file is skipped, indexed, or rejected."""

    def on_embedding_start(self, *, path: Path, chunks: int) -> None:
        """Called before embedding chunks for a file."""


def normalize_extensions(exts: list[str] | None) -> set[str]:
    values = exts or DEFAULT_EXTENSIONS
    return {ext if ext.startswith(".") else f".{ext}" for ext in values}


def iter_candidate_files(roots: list[Path], extensions: set[str]) -> list[CandidateFile]:
    paths: dict[Path, CandidateFile] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file():
            if root.suffix in extensions:
                paths[root] = CandidateFile(root, root.parent, Path(root.name))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {".git", ".venv", "__pycache__"} and not name.startswith(".")
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix in extensions:
                    resolved = path.resolve()
                    paths.setdefault(
                        resolved, CandidateFile(resolved, root, resolved.relative_to(root))
                    )
    return [paths[path] for path in sorted(paths)]


def read_text_file(candidate: CandidateFile) -> IndexedFile | None:
    stat = candidate.path.stat()
    data = candidate.path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None
    return IndexedFile(
        path=candidate.path,
        root_path=candidate.root_path,
        relative_path=candidate.relative_path,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        content_hash=hashlib.sha256(data).hexdigest(),
        text=text,
    )


def chunk_text(text: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
    paragraphs = split_paragraphs(text)
    chunks: list[Chunk] = []
    for start, end, paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(
                Chunk(
                    len(chunks),
                    start,
                    end,
                    line_number_at_offset(text, start),
                    line_number_at_offset(text, max(start, end - 1)),
                    paragraph,
                )
            )
            continue
        cursor = 0
        while cursor < len(paragraph):
            part = paragraph[cursor : cursor + max_chars]
            part_start = start + cursor
            part_end = part_start + len(part)
            chunks.append(
                Chunk(
                    len(chunks),
                    part_start,
                    part_end,
                    line_number_at_offset(text, part_start),
                    line_number_at_offset(text, max(part_start, part_end - 1)),
                    part,
                )
            )
            if cursor + max_chars >= len(paragraph):
                break
            cursor += max_chars - OVERLAP_CHARS
    return [chunk for chunk in chunks if chunk.text.strip()]


def line_number_at_offset(text: str, offset: int) -> int:
    if offset <= 0:
        return 1
    return text.count("\n", 0, min(offset, len(text))) + 1


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    paragraphs: list[tuple[int, int, str]] = []
    start = 0
    current: list[str] = []
    current_start = 0
    offset = 0
    for block in text.splitlines(keepends=True):
        if block.strip():
            if not current:
                current_start = offset
            current.append(block)
        elif current:
            value = "".join(current).strip()
            end = offset
            paragraphs.append((current_start, end, value))
            current = []
        offset += len(block)
    if current:
        value = "".join(current).strip()
        paragraphs.append((current_start, len(text), value))
    if not paragraphs and text.strip():
        paragraphs.append((start, len(text), text.strip()))
    return paragraphs


def index_paths(
    con: sqlite3.Connection,
    *,
    roots: list[Path],
    extensions: list[str] | None,
    embedder: EmbeddingProvider,
    rebuild: bool = False,
    progress: IndexProgress | None = None,
) -> IndexStats:
    ensure_schema(con, embedding_dim=embedder.dim, embedding_model=embedder.model_name)
    set_embedding_metadata(
        con,
        backend=getattr(embedder, "backend", "unknown"),
        device=getattr(embedder, "device", "unknown"),
        batch_size=getattr(embedder, "batch_size", 0),
        prefix_policy=getattr(embedder, "prefix_policy", "unknown"),
    )
    if rebuild:
        clear_index(con)

    allowed = normalize_extensions(extensions)
    paths = iter_candidate_files(roots, allowed)
    if progress is not None:
        progress.on_scan_complete(len(paths))
    seen = {str(candidate.path) for candidate in paths}
    stats = IndexStats(scanned_files=len(paths))
    removed = remove_missing_files(con, seen)

    indexed_files = 0
    skipped_files = 0
    chunk_count = 0
    for candidate in paths:
        indexed = read_text_file(candidate)
        if indexed is None:
            skipped_files += 1
            if progress is not None:
                progress.on_file_done(path=candidate.path, status="skipped")
            continue
        if is_unchanged(con, indexed):
            if progress is not None:
                progress.on_file_done(path=indexed.path, status="unchanged")
            continue
        chunks = chunk_text(indexed.text)
        if progress is not None:
            progress.on_embedding_start(path=indexed.path, chunks=len(chunks))
        upsert_file(con, indexed, chunks, embedder)
        indexed_files += 1
        chunk_count += len(chunks)
        if progress is not None:
            progress.on_file_done(path=indexed.path, status="indexed", chunks=len(chunks))

    return IndexStats(
        scanned_files=stats.scanned_files,
        indexed_files=indexed_files,
        skipped_files=skipped_files,
        chunks=chunk_count,
        removed_files=removed,
    )


def clear_index(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM chunks_fts")
    con.execute("DELETE FROM chunk_vec")
    con.execute("DELETE FROM chunks")
    con.execute("DELETE FROM files")


def remove_missing_files(con: sqlite3.Connection, seen_paths: set[str]) -> int:
    rows = con.execute("SELECT id, path FROM files").fetchall()
    removed = 0
    for row in rows:
        if row["path"] not in seen_paths:
            delete_file(con, int(row["id"]))
            removed += 1
    return removed


def is_unchanged(con: sqlite3.Connection, indexed: IndexedFile) -> bool:
    row = con.execute(
        """
        SELECT root_path, relative_path, size, mtime_ns, content_hash
        FROM files
        WHERE path = ?
        """,
        (str(indexed.path),),
    ).fetchone()
    return (
        row is not None
        and row["root_path"] == str(indexed.root_path)
        and row["relative_path"] == str(indexed.relative_path)
        and row["size"] == indexed.size
        and row["mtime_ns"] == indexed.mtime_ns
        and row["content_hash"] == indexed.content_hash
    )


def delete_file(con: sqlite3.Connection, file_id: int) -> None:
    chunk_ids = [
        row["id"] for row in con.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
    ]
    for chunk_id in chunk_ids:
        con.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
        con.execute("DELETE FROM chunk_vec WHERE rowid = ?", (chunk_id,))
    con.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    con.execute("DELETE FROM files WHERE id = ?", (file_id,))


def upsert_file(
    con: sqlite3.Connection,
    indexed: IndexedFile,
    chunks: list[Chunk],
    embedder: EmbeddingProvider,
) -> None:
    old = con.execute("SELECT id FROM files WHERE path = ?", (str(indexed.path),)).fetchone()
    if old is not None:
        delete_file(con, int(old["id"]))

    cursor = con.execute(
        """
        INSERT INTO files(path, root_path, relative_path, size, mtime_ns, content_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(indexed.path),
            str(indexed.root_path),
            str(indexed.relative_path),
            indexed.size,
            indexed.mtime_ns,
            indexed.content_hash,
        ),
    )
    file_id = int(cursor.lastrowid)
    embeddings = embedder.embed_passages([chunk.text for chunk in chunks]) if chunks else []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        cursor = con.execute(
            """
            INSERT INTO chunks(
                file_id,
                chunk_index,
                start_offset,
                end_offset,
                start_line,
                end_line,
                text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                chunk.index,
                chunk.start_offset,
                chunk.end_offset,
                chunk.start_line,
                chunk.end_line,
                chunk.text,
            ),
        )
        chunk_id = int(cursor.lastrowid)
        con.execute(
            "INSERT INTO chunks_fts(rowid, path, text) VALUES (?, ?, ?)",
            (chunk_id, str(indexed.path), chunk.text),
        )
        con.execute(
            "INSERT INTO chunk_vec(rowid, embedding) VALUES (?, ?)",
            (chunk_id, serialize_vector(embedding)),
        )
