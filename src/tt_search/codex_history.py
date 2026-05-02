from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .db import ensure_schema, set_embedding_metadata, set_metadata
from .embeddings import EmbeddingProvider
from .indexer import (
    MAX_CHARS,
    Chunk,
    IndexedFile,
    IndexProgress,
    IndexStats,
    chunk_text,
    clear_index,
    is_unchanged,
    remove_missing_files,
    upsert_file,
)

CODEX_HISTORY_INDEX_KIND = "codex-history"
CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"
CODEX_HISTORY_DB = Path.home() / ".codex" / "tt-search" / "codex-history.sqlite"
CODEX_HISTORY_MODEL = "cl-nagoya/ruri-v3-310m"
CODEX_TURN_MAX_CHARS = MAX_CHARS


@dataclass(frozen=True)
class CodexTurn:
    session_id: str
    cwd: str
    role: str
    text: str
    timestamp: str
    turn_id: str | None
    session_path: Path
    line_no: int


def iter_codex_session_files(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file() and root.suffix == ".jsonl":
            paths.add(root)
            continue
        if root.is_dir():
            paths.update(path.resolve() for path in root.rglob("*.jsonl"))
    return sorted(paths)


def parse_codex_session_file(path: Path) -> list[CodexTurn]:
    session_id: str | None = None
    cwd = ""
    current_turn_id: str | None = None
    turns: list[CodexTurn] = []
    is_subagent_session = False

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_type = item.get("type")
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue

            if item_type == "session_meta":
                session_id = optional_payload_str(payload, "id")
                cwd = optional_payload_str(payload, "cwd") or ""
                is_subagent_session = is_subagent_source(payload.get("source"))
                continue
            if is_subagent_session:
                continue
            if item_type == "event_msg" and payload.get("type") == "task_started":
                current_turn_id = optional_payload_str(payload, "turn_id")
                continue
            if item_type != "response_item" or payload.get("type") != "message":
                continue
            if session_id is None:
                continue

            role = optional_payload_str(payload, "role")
            if role not in {"user", "assistant"}:
                continue
            if role == "assistant" and payload.get("phase") != "final_answer":
                continue
            text = message_text(payload.get("content"))
            if not text or should_skip_user_message(role, text):
                continue
            turns.append(
                CodexTurn(
                    session_id=session_id,
                    cwd=cwd,
                    role=role,
                    text=text,
                    timestamp=str(item.get("timestamp", "")),
                    turn_id=current_turn_id,
                    session_path=path,
                    line_no=line_no,
                )
            )
    return turns


def index_codex_sessions(
    con: sqlite3.Connection,
    *,
    roots: list[Path],
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
    set_metadata(con, "index_kind", CODEX_HISTORY_INDEX_KIND)
    if rebuild:
        clear_index(con)

    paths = iter_codex_session_files(roots)
    if progress is not None:
        progress.on_scan_complete(len(paths))
    removed = remove_missing_files(con, {str(path) for path in paths})
    indexed_files = 0
    skipped_files = 0
    chunk_count = 0

    for path in paths:
        indexed, chunks = codex_indexed_file(path, roots)
        if indexed is None:
            skipped_files += 1
            if progress is not None:
                progress.on_file_done(path=path, status="skipped")
            continue
        if is_unchanged(con, indexed):
            if progress is not None:
                progress.on_file_done(path=indexed.path, status="unchanged")
            continue
        if progress is not None:
            progress.on_embedding_start(path=indexed.path, chunks=len(chunks))
        upsert_file(con, indexed, chunks, embedder, progress=progress)
        indexed_files += 1
        chunk_count += len(chunks)
        if progress is not None:
            progress.on_file_done(path=indexed.path, status="indexed", chunks=len(chunks))

    return IndexStats(
        scanned_files=len(paths),
        excluded_files=0,
        indexed_files=indexed_files,
        skipped_files=skipped_files,
        chunks=chunk_count,
        removed_files=removed,
    )


def codex_indexed_file(path: Path, roots: list[Path]) -> tuple[IndexedFile | None, list[Chunk]]:
    data = path.read_bytes()
    turns = parse_codex_session_file(path)
    if not turns:
        return None, []
    root_path = matching_root(path, roots)
    relative_path = path.relative_to(root_path) if root_path is not None else Path(path.name)
    stat = path.stat()
    chunks: list[Chunk] = []
    for turn in turns:
        for chunk in chunk_codex_turn(turn):
            chunks.append(replace(chunk, index=len(chunks)))
    return (
        IndexedFile(
            path=path,
            root_path=root_path or path.parent,
            relative_path=relative_path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            content_hash=hashlib.sha256(data).hexdigest(),
            text="",
        ),
        chunks,
    )


def chunk_codex_turn(
    turn: CodexTurn, *, max_chars: int = CODEX_TURN_MAX_CHARS
) -> list[Chunk]:
    if len(turn.text) <= max_chars:
        source_chunks = [
            Chunk(
                index=0,
                start_offset=0,
                end_offset=len(turn.text),
                start_line=1,
                end_line=max(1, turn.text.count("\n") + 1),
                text=turn.text,
            )
        ]
    else:
        source_chunks = chunk_text(turn.text, max_chars=max_chars)
    return [
        Chunk(
            index=source.index,
            start_offset=source.start_offset,
            end_offset=source.end_offset,
            start_line=source.start_line,
            end_line=source.end_line,
            text=source.text,
            session_id=turn.session_id,
            cwd=turn.cwd,
            role=turn.role,
            turn_id=turn.turn_id,
            timestamp=turn.timestamp,
            session_path=str(turn.session_path),
            line_no=turn.line_no,
        )
        for source in source_chunks
    ]


def matching_root(path: Path, roots: list[Path]) -> Path | None:
    for root in sorted((root.expanduser().resolve() for root in roots), key=lambda p: len(str(p))):
        if root.is_file() and path == root:
            return root.parent
        try:
            path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(part for part in parts if part).strip()


def should_skip_user_message(role: str, text: str) -> bool:
    if role != "user":
        return False
    stripped = text.lstrip()
    return (
        stripped.startswith("# AGENTS.md instructions")
        or "<environment_context>" in stripped
        or stripped.startswith("The following is the Codex agent history")
    )


def is_subagent_source(source: object) -> bool:
    return isinstance(source, dict) and "subagent" in source


def optional_payload_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return None if value is None else str(value)
