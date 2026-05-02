from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .db import ensure_schema, serialize_vector, set_embedding_metadata
from .embeddings import EmbeddingProvider

DEFAULT_EXTENSIONS = [".txt", ".md", ".markdown", ".rst"]
MAX_CHARS = 600
OVERLAP_CHARS = 120


@dataclass(frozen=True)
class CandidateFile:
    path: Path
    root_path: Path
    relative_path: Path


@dataclass(frozen=True)
class MarkdownSection:
    start: int
    end: int
    headings: tuple[str, ...]


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
    session_id: str | None = None
    cwd: str | None = None
    role: str | None = None
    turn_id: str | None = None
    timestamp: str | None = None
    session_path: str | None = None
    line_no: int | None = None


@dataclass(frozen=True)
class IndexStats:
    scanned_files: int = 0
    excluded_files: int = 0
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

    def on_embedding_batch_done(
        self, *, path: Path, embedded_chunks: int, total_chunks: int
    ) -> None:
        """Called after embedding a batch of chunks for a file."""


class ChunkingStrategy(Protocol):
    name: str

    def chunk(self, text: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
        """Split file text into searchable chunks."""


class ParagraphPackingStrategy:
    name = "paragraph-pack"

    def chunk(self, text: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
        return chunk_paragraphs(text, split_paragraphs(text), max_chars=max_chars)


class MarkdownSectionStrategy:
    name = "markdown-section"

    def chunk(self, text: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
        chunks: list[Chunk] = []
        fenced_ranges = split_fenced_code_ranges(text)
        for section in split_markdown_sections(text):
            paragraphs = split_paragraphs_in_range(
                text,
                section.start,
                section.end,
                skip_ranges=fenced_ranges,
            )
            chunks.extend(
                add_markdown_heading_context(
                    chunk_paragraphs(
                        text,
                        paragraphs,
                        max_chars=max_chars,
                        overlap_paragraph=False,
                    ),
                    section.headings,
                )
            )
        return reindex_chunks(chunks)


def chunk_paragraphs(
    text: str,
    paragraphs: list[tuple[int, int, str]],
    *,
    max_chars: int = MAX_CHARS,
    overlap_paragraph: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_parts: list[tuple[int, int, str]] = []

    def flush_current(*, keep_last_paragraph: bool = False) -> None:
        if not current_parts:
            return
        start = current_parts[0][0]
        end = current_parts[-1][1]
        overlap_part = current_parts[-1] if keep_last_paragraph else None
        chunk_body = "\n\n".join(part for _, _, part in current_parts)
        chunks.append(
            Chunk(
                len(chunks),
                start,
                end,
                line_number_at_offset(text, start),
                line_number_at_offset(text, max(start, end - 1)),
                chunk_body,
            )
        )
        current_parts.clear()
        if overlap_part is not None:
            current_parts.append(overlap_part)

    def trim_overlap_for(paragraph: str) -> None:
        while current_parts:
            candidate = "\n\n".join([*(part for _, _, part in current_parts), paragraph])
            if len(candidate) <= max_chars:
                return
            current_parts.pop(0)

    for start, end, paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush_current()
            chunks.extend(split_long_paragraph(text, paragraph, start, max_chars))
            continue

        candidate = "\n\n".join([*(part for _, _, part in current_parts), paragraph])
        if current_parts and len(candidate) > max_chars:
            flush_current(keep_last_paragraph=overlap_paragraph)
            trim_overlap_for(paragraph)
        current_parts.append((start, end, paragraph))

    flush_current()
    return reindex_chunks(chunks)


DEFAULT_CHUNKING_STRATEGY = ParagraphPackingStrategy()
MARKDOWN_CHUNKING_STRATEGY = MarkdownSectionStrategy()
CHUNKING_STRATEGIES_BY_EXTENSION: dict[str, ChunkingStrategy] = {
    ".txt": DEFAULT_CHUNKING_STRATEGY,
    ".md": MARKDOWN_CHUNKING_STRATEGY,
    ".markdown": MARKDOWN_CHUNKING_STRATEGY,
    ".rst": DEFAULT_CHUNKING_STRATEGY,
}


def normalize_extensions(exts: list[str] | None) -> set[str]:
    values = exts or DEFAULT_EXTENSIONS
    return {ext if ext.startswith(".") else f".{ext}" for ext in values}


def compile_exclude_patterns(patterns: list[str] | None) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns or []]


def is_excluded(relative_path: Path, patterns: list[re.Pattern[str]]) -> bool:
    value = relative_path.as_posix()
    return any(pattern.search(value) for pattern in patterns)


def iter_candidate_files(
    roots: list[Path],
    extensions: set[str],
    exclude_patterns: list[re.Pattern[str]] | None = None,
) -> tuple[list[CandidateFile], int]:
    paths: dict[Path, CandidateFile] = {}
    excluded = 0
    exclude_patterns = exclude_patterns or []
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file():
            relative_path = Path(root.name)
            if root.suffix in extensions and not is_excluded(relative_path, exclude_patterns):
                paths[root] = CandidateFile(root, root.parent, relative_path)
            elif root.suffix in extensions:
                excluded += 1
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
                    relative_path = resolved.relative_to(root)
                    if is_excluded(relative_path, exclude_patterns):
                        excluded += 1
                        continue
                    paths.setdefault(resolved, CandidateFile(resolved, root, relative_path))
    return [paths[path] for path in sorted(paths)], excluded


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
    return DEFAULT_CHUNKING_STRATEGY.chunk(text, max_chars=max_chars)


def chunk_file(path: Path, text: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
    return strategy_for_path(path).chunk(text, max_chars=max_chars)


def strategy_for_path(path: Path) -> ChunkingStrategy:
    return CHUNKING_STRATEGIES_BY_EXTENSION.get(path.suffix.lower(), DEFAULT_CHUNKING_STRATEGY)


def reindex_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [
        Chunk(
            index,
            chunk.start_offset,
            chunk.end_offset,
            chunk.start_line,
            chunk.end_line,
            chunk.text,
        )
        for index, chunk in enumerate(chunk for chunk in chunks if chunk.text.strip())
    ]


def split_long_paragraph(text: str, paragraph: str, start: int, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
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
        cursor += max(1, max_chars - OVERLAP_CHARS)
    return chunks


def line_number_at_offset(text: str, offset: int) -> int:
    if offset <= 0:
        return 1
    return text.count("\n", 0, min(offset, len(text))) + 1


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    return split_paragraphs_in_range(text, 0, len(text), skip_ranges=[])


def split_paragraphs_in_range(
    text: str,
    start: int,
    end: int,
    *,
    skip_ranges: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    paragraphs: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = start
    offset = start
    range_index = 0

    def flush(until: int) -> None:
        nonlocal current
        if not current:
            return
        value = "".join(current).strip()
        paragraphs.append((current_start, until, value))
        current = []

    for block in text[start:end].splitlines(keepends=True):
        while range_index < len(skip_ranges) and skip_ranges[range_index][1] <= offset:
            range_index += 1
        in_skip_range = (
            range_index < len(skip_ranges)
            and skip_ranges[range_index][0] <= offset < skip_ranges[range_index][1]
        )
        if in_skip_range:
            flush(offset)
            offset += len(block)
            continue
        if block.strip():
            if not current:
                current_start = offset
            current.append(block)
        elif current:
            flush(offset)
        offset += len(block)
    if current:
        flush(end)
    if not paragraphs and not skip_ranges and text[start:end].strip():
        paragraphs.append((start, end, text[start:end].strip()))
    return paragraphs


def split_fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    range_start: int | None = None
    fence_marker: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence_marker is None:
            marker = markdown_fence_marker(stripped)
            if marker is not None:
                range_start = offset
                fence_marker = marker
        elif stripped.startswith(fence_marker):
            assert range_start is not None
            ranges.append((range_start, offset + len(line)))
            range_start = None
            fence_marker = None
        offset += len(line)
    if range_start is not None:
        ranges.append((range_start, len(text)))
    return ranges


def split_markdown_sections(text: str) -> list[MarkdownSection]:
    if not text:
        return []

    sections: list[MarkdownSection] = []
    section_start = 0
    section_headings: tuple[str, ...] = ()
    heading_stack: list[str] = []
    offset = 0
    fence_marker: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence_marker is None:
            marker = markdown_fence_marker(stripped)
            if marker is not None:
                fence_marker = marker
            else:
                heading = markdown_heading(stripped)
                if heading is not None:
                    if offset != section_start:
                        sections.append(
                            MarkdownSection(section_start, offset, section_headings)
                        )
                    level, heading_text = heading
                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(heading_text)
                    section_start = offset
                    section_headings = tuple(heading_stack)
        elif stripped.startswith(fence_marker):
            fence_marker = None
        offset += len(line)
    sections.append(MarkdownSection(section_start, len(text), section_headings))
    return [section for section in sections if section.start < section.end]


def is_markdown_heading(stripped_line: str) -> bool:
    return markdown_heading(stripped_line) is not None


def markdown_heading(stripped_line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+\S.*$", stripped_line.rstrip())
    if match is None:
        return None
    return len(match.group(1)), match.group(0)


def add_markdown_heading_context(chunks: list[Chunk], headings: tuple[str, ...]) -> list[Chunk]:
    if not headings:
        return chunks
    prefix = "\n".join(headings)
    return [replace_chunk_text(chunk, prefix, headings) for chunk in chunks]


def replace_chunk_text(chunk: Chunk, prefix: str, headings: tuple[str, ...]) -> Chunk:
    body = strip_leading_context_headings(chunk.text, headings)
    text = prefix if not body else f"{prefix}\n{body}"
    return Chunk(
        index=chunk.index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        text=text,
    )


def strip_leading_context_headings(text: str, headings: tuple[str, ...]) -> str:
    lines = text.splitlines()
    index = 0
    for heading in headings:
        if index < len(lines) and lines[index].strip() == heading:
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
    return "\n".join(lines[index:]).strip()


def markdown_fence_marker(stripped_line: str) -> str | None:
    match = re.match(r"^(`{3,}|~{3,})", stripped_line)
    if match is None:
        return None
    marker = match.group(1)
    return marker[0] * len(marker)


def index_paths(
    con: sqlite3.Connection,
    *,
    roots: list[Path],
    extensions: list[str] | None,
    embedder: EmbeddingProvider,
    rebuild: bool = False,
    progress: IndexProgress | None = None,
    exclude_patterns: list[str] | None = None,
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
    compiled_exclude_patterns = compile_exclude_patterns(exclude_patterns)
    paths, excluded_files = iter_candidate_files(roots, allowed, compiled_exclude_patterns)
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
        chunks = chunk_file(indexed.path, indexed.text)
        if progress is not None:
            progress.on_embedding_start(path=indexed.path, chunks=len(chunks))
        upsert_file(con, indexed, chunks, embedder, progress=progress)
        indexed_files += 1
        chunk_count += len(chunks)
        if progress is not None:
            progress.on_file_done(path=indexed.path, status="indexed", chunks=len(chunks))

    return IndexStats(
        scanned_files=stats.scanned_files,
        excluded_files=excluded_files,
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
    progress: IndexProgress | None = None,
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
    embedded_chunks = 0
    for start in range(0, len(chunks), embedder.batch_size):
        chunk_batch = chunks[start : start + embedder.batch_size]
        embeddings = embedder.embed_passages([chunk.text for chunk in chunk_batch])
        for chunk, embedding in zip(chunk_batch, embeddings, strict=True):
            insert_chunk(con, indexed, file_id, chunk, embedding)
        embedded_chunks += len(chunk_batch)
        if progress is not None:
            progress.on_embedding_batch_done(
                path=indexed.path,
                embedded_chunks=embedded_chunks,
                total_chunks=len(chunks),
            )


def insert_chunk(
    con: sqlite3.Connection,
    indexed: IndexedFile,
    file_id: int,
    chunk: Chunk,
    embedding: list[float],
) -> None:
    cursor = con.execute(
        """
        INSERT INTO chunks(
            file_id,
            chunk_index,
            start_offset,
            end_offset,
            start_line,
            end_line,
            text,
            session_id,
            cwd,
            role,
            turn_id,
            timestamp,
            session_path,
            line_no
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_id,
            chunk.index,
            chunk.start_offset,
            chunk.end_offset,
            chunk.start_line,
            chunk.end_line,
            chunk.text,
            chunk.session_id,
            chunk.cwd,
            chunk.role,
            chunk.turn_id,
            chunk.timestamp,
            chunk.session_path,
            chunk.line_no,
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
