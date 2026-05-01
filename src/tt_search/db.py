from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = "1"


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def ensure_schema(con: sqlite3.Connection, *, embedding_dim: int, embedding_model: str) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            root_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            content_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            text TEXT NOT NULL,
            UNIQUE(file_id, chunk_index)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            path UNINDEXED,
            text,
            tokenize='trigram'
        );
        """
    )
    ensure_files_columns(con)
    con.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(embedding float[{embedding_dim}])"
    )
    existing_dim = get_metadata(con, "embedding_dim")
    if existing_dim is not None and existing_dim != str(embedding_dim):
        raise ValueError(
            f"DB embedding dimension is {existing_dim}, but current provider is {embedding_dim}"
        )
    set_metadata(con, "schema_version", SCHEMA_VERSION)
    set_metadata(con, "embedding_model", embedding_model)
    set_metadata(con, "embedding_dim", str(embedding_dim))
    if get_metadata(con, "created_at") is None:
        set_metadata(con, "created_at", datetime.now(UTC).isoformat())


def ensure_files_columns(con: sqlite3.Connection) -> None:
    columns = {row["name"] for row in con.execute("PRAGMA table_info(files)")}
    if "root_path" not in columns:
        con.execute("ALTER TABLE files ADD COLUMN root_path TEXT")
        con.execute("UPDATE files SET root_path = '' WHERE root_path IS NULL")
    if "relative_path" not in columns:
        con.execute("ALTER TABLE files ADD COLUMN relative_path TEXT")
        con.execute("UPDATE files SET relative_path = path WHERE relative_path IS NULL")


def set_metadata(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def get_metadata(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def serialize_vector(vector: list[float]) -> bytes:
    return sqlite_vec.serialize_float32(vector)


def decode_metadata(con: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in con.execute("SELECT key, value FROM metadata")}


def format_info(con: sqlite3.Connection) -> dict[str, object]:
    metadata = decode_metadata(con)
    file_count = con.execute("SELECT count(*) FROM files").fetchone()[0]
    chunk_count = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {
        "metadata": metadata,
        "file_count": file_count,
        "chunk_count": chunk_count,
    }


def as_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
