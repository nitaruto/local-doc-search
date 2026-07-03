from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .db import connect, ensure_chunks_columns, ensure_files_columns, serialize_vector
from .embeddings import EmbeddingProvider

SearchMode = Literal["fts", "vec", "fts-vec", "vec-fts"]
VECTOR_MODES = {"vec", "fts-vec", "vec-fts"}
FTS_MODES = {"fts", "fts-vec", "vec-fts"}


@dataclass(frozen=True)
class SearchResult:
    db_path: str | None
    chunk_id: int
    path: str
    relative_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    start_line: int
    end_line: int
    text: str
    score: float
    fts_rank: float | None = None
    vec_distance: float | None = None
    source: str = ""
    session_id: str | None = None
    cwd: str | None = None
    role: str | None = None
    turn_id: str | None = None
    timestamp: str | None = None
    session_path: str | None = None
    line_no: int | None = None


@dataclass(frozen=True)
class ResolvedSearch:
    mode: SearchMode
    vector_query: str | None
    fts_query: str | None
    fts_is_pattern: bool


def resolve_search(
    *,
    query: str | None,
    pattern: str | None,
    mode: SearchMode | None,
) -> ResolvedSearch:
    query = normalize_search_text(query)
    pattern = normalize_search_text(pattern)
    if query is None and pattern is None:
        raise ValueError("Either query or pattern is required")
    if mode is None:
        if query is not None and pattern is not None:
            mode = "vec-fts"
        elif query is not None:
            mode = "vec"
        else:
            mode = "fts"
    if mode in VECTOR_MODES and query is None:
        raise ValueError("Vector search modes require --query")
    fts_query = pattern if pattern is not None else query
    if mode in FTS_MODES and fts_query is None:
        raise ValueError("FTS search modes require --query or --pattern")
    return ResolvedSearch(
        mode=mode,
        vector_query=query if mode in VECTOR_MODES else None,
        fts_query=fts_query if mode in FTS_MODES else None,
        fts_is_pattern=pattern is not None,
    )


def normalize_search_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def search(
    con: sqlite3.Connection,
    *,
    query: str | None = None,
    vector_query: str | None = None,
    fts_query: str | None = None,
    fts_is_pattern: bool = False,
    mode: SearchMode,
    limit: int,
    candidates: int,
    embedder: EmbeddingProvider | None,
    db_path: str | None = None,
    query_vector: bytes | None = None,
) -> list[SearchResult]:
    ensure_files_columns(con)
    ensure_chunks_columns(con)
    vector_query = vector_query or query
    fts_query = fts_query or query
    if mode in VECTOR_MODES and embedder is None:
        raise ValueError("Embedding provider is required for vector search")
    if mode in VECTOR_MODES and vector_query is None:
        raise ValueError("Vector search modes require a vector query")
    if mode in FTS_MODES and fts_query is None:
        raise ValueError("FTS search modes require an FTS query")
    if mode == "fts":
        assert fts_query is not None
        return with_db_path(
            fts_candidates(
                con,
                fts_query,
                candidates=limit,
                is_pattern=fts_is_pattern,
            ),
            db_path,
        )
    if mode == "vec":
        assert embedder is not None
        assert vector_query is not None
        return with_db_path(
            vec_candidates(
                con,
                vector_query,
                candidates=limit,
                embedder=embedder,
                query_vector=query_vector,
            ),
            db_path,
        )
    if mode == "fts-vec":
        assert embedder is not None
        assert vector_query is not None
        assert fts_query is not None
        rows = fts_candidates(
            con,
            fts_query,
            candidates=candidates,
            is_pattern=fts_is_pattern,
        )
        return with_db_path(
            rerank_by_vector(
                con,
                vector_query,
                rows,
                limit=limit,
                embedder=embedder,
                query_vector=query_vector,
            ),
            db_path,
        )
    if mode == "vec-fts":
        assert embedder is not None
        assert vector_query is not None
        assert fts_query is not None
        rows = vec_candidates(
            con,
            vector_query,
            candidates=candidates,
            embedder=embedder,
            query_vector=query_vector,
        )
        return with_db_path(
            rerank_by_fts(
                con,
                fts_query,
                rows,
                limit=limit,
                is_pattern=fts_is_pattern,
            ),
            db_path,
        )
    raise ValueError(f"Unknown mode: {mode}")


def search_many(
    db_paths: list[Path],
    *,
    query: str | None = None,
    vector_query: str | None = None,
    fts_query: str | None = None,
    fts_is_pattern: bool = False,
    mode: SearchMode,
    limit: int,
    candidates: int,
    embedder: EmbeddingProvider | None,
) -> list[SearchResult]:
    vector_query = vector_query or query
    fts_query = fts_query or query
    query_vector = None
    if mode in VECTOR_MODES:
        if embedder is None:
            raise ValueError("Embedding provider is required for vector search")
        if vector_query is None:
            raise ValueError("Vector search modes require a vector query")
        query_vector = serialize_vector(embedder.embed_query(vector_query))

    all_results: list[SearchResult] = []
    per_db_candidates = max(candidates, limit)
    for db_path in db_paths:
        with connect(db_path) as con:
            all_results.extend(
                search(
                    con,
                    query=query,
                    vector_query=vector_query,
                    fts_query=fts_query,
                    fts_is_pattern=fts_is_pattern,
                    mode=mode,
                    limit=per_db_candidates,
                    candidates=per_db_candidates,
                    embedder=embedder,
                    db_path=str(db_path),
                    query_vector=query_vector,
                )
            )
    return sorted(all_results, key=lambda result: result.score, reverse=True)[:limit]


def with_db_path(results: list[SearchResult], db_path: str | None) -> list[SearchResult]:
    if db_path is None:
        return results
    return [SearchResult(**{**result.__dict__, "db_path": db_path}) for result in results]


def fts_candidates(
    con: sqlite3.Connection,
    query: str,
    *,
    candidates: int,
    is_pattern: bool,
) -> list[SearchResult]:
    if not is_pattern and use_like_fallback(query):
        return like_candidates(con, query, candidates=candidates)
    rows = con.execute(
        """
        SELECT
            c.id AS chunk_id,
            f.path AS path,
            f.relative_path AS relative_path,
            c.chunk_index AS chunk_index,
            c.start_offset AS start_offset,
            c.end_offset AS end_offset,
            c.start_line AS start_line,
            c.end_line AS end_line,
            c.text AS text,
            c.session_id AS session_id,
            c.cwd AS cwd,
            c.role AS role,
            c.turn_id AS turn_id,
            c.timestamp AS timestamp,
            c.session_path AS session_path,
            c.line_no AS line_no,
            bm25(chunks_fts) AS fts_rank
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN files f ON f.id = c.file_id
        WHERE chunks_fts MATCH ?
        ORDER BY fts_rank
        LIMIT ?
        """,
        (fts_match_query(query, is_pattern=is_pattern), candidates),
    ).fetchall()
    return [
        SearchResult(
            db_path=None,
            chunk_id=int(row["chunk_id"]),
            path=str(row["path"]),
            relative_path=str(row["relative_path"]),
            chunk_index=int(row["chunk_index"]),
            start_offset=int(row["start_offset"]),
            end_offset=int(row["end_offset"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            text=str(row["text"]),
            score=-float(row["fts_rank"]),
            fts_rank=float(row["fts_rank"]),
            source="fts",
            session_id=optional_str(row["session_id"]),
            cwd=optional_str(row["cwd"]),
            role=optional_str(row["role"]),
            turn_id=optional_str(row["turn_id"]),
            timestamp=optional_str(row["timestamp"]),
            session_path=optional_str(row["session_path"]),
            line_no=optional_int(row["line_no"]),
        )
        for row in rows
    ]


def like_candidates(con: sqlite3.Connection, query: str, *, candidates: int) -> list[SearchResult]:
    pattern = f"%{escape_like(query)}%"
    rows = con.execute(
        """
        SELECT
            c.id AS chunk_id,
            f.path AS path,
            f.relative_path AS relative_path,
            c.chunk_index AS chunk_index,
            c.start_offset AS start_offset,
            c.end_offset AS end_offset,
            c.start_line AS start_line,
            c.end_line AS end_line,
            c.text AS text,
            c.session_id AS session_id,
            c.cwd AS cwd,
            c.role AS role,
            c.turn_id AS turn_id,
            c.timestamp AS timestamp,
            c.session_path AS session_path,
            c.line_no AS line_no
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        WHERE c.text LIKE ? ESCAPE '\\'
        ORDER BY length(c.text), c.id
        LIMIT ?
        """,
        (pattern, candidates),
    ).fetchall()
    return [
        SearchResult(
            db_path=None,
            chunk_id=int(row["chunk_id"]),
            path=str(row["path"]),
            relative_path=str(row["relative_path"]),
            chunk_index=int(row["chunk_index"]),
            start_offset=int(row["start_offset"]),
            end_offset=int(row["end_offset"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            text=str(row["text"]),
            score=1.0,
            fts_rank=None,
            source="like",
            session_id=optional_str(row["session_id"]),
            cwd=optional_str(row["cwd"]),
            role=optional_str(row["role"]),
            turn_id=optional_str(row["turn_id"]),
            timestamp=optional_str(row["timestamp"]),
            session_path=optional_str(row["session_path"]),
            line_no=optional_int(row["line_no"]),
        )
        for row in rows
    ]


def vec_candidates(
    con: sqlite3.Connection,
    query: str,
    *,
    candidates: int,
    embedder: EmbeddingProvider,
    query_vector: bytes | None = None,
) -> list[SearchResult]:
    query_vector = query_vector or serialize_vector(embedder.embed_query(query))
    rows = con.execute(
        """
        SELECT
            c.id AS chunk_id,
            f.path AS path,
            f.relative_path AS relative_path,
            c.chunk_index AS chunk_index,
            c.start_offset AS start_offset,
            c.end_offset AS end_offset,
            c.start_line AS start_line,
            c.end_line AS end_line,
            c.text AS text,
            c.session_id AS session_id,
            c.cwd AS cwd,
            c.role AS role,
            c.turn_id AS turn_id,
            c.timestamp AS timestamp,
            c.session_path AS session_path,
            c.line_no AS line_no,
            v.distance AS vec_distance
        FROM chunk_vec v
        JOIN chunks c ON c.id = v.rowid
        JOIN files f ON f.id = c.file_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (query_vector, candidates),
    ).fetchall()
    return [
        SearchResult(
            db_path=None,
            chunk_id=int(row["chunk_id"]),
            path=str(row["path"]),
            relative_path=str(row["relative_path"]),
            chunk_index=int(row["chunk_index"]),
            start_offset=int(row["start_offset"]),
            end_offset=int(row["end_offset"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            text=str(row["text"]),
            score=distance_to_score(vec_distance),
            vec_distance=vec_distance,
            source="vec",
            session_id=optional_str(row["session_id"]),
            cwd=optional_str(row["cwd"]),
            role=optional_str(row["role"]),
            turn_id=optional_str(row["turn_id"]),
            timestamp=optional_str(row["timestamp"]),
            session_path=optional_str(row["session_path"]),
            line_no=optional_int(row["line_no"]),
        )
        for row in rows
        for vec_distance in [require_vec_distance(row["vec_distance"])]
    ]


def optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def optional_int(value: object) -> int | None:
    if not isinstance(value, int | str | bytes | bytearray):
        return None
    return None if value is None else int(value)


def rerank_by_vector(
    con: sqlite3.Connection,
    query: str,
    rows: list[SearchResult],
    *,
    limit: int,
    embedder: EmbeddingProvider,
    query_vector: bytes | None = None,
) -> list[SearchResult]:
    if not rows:
        return []
    query_vector = query_vector or serialize_vector(embedder.embed_query(query))
    chunk_ids = [row.chunk_id for row in rows]
    placeholders = ",".join("?" for _ in chunk_ids)
    distances = {
        int(row["rowid"]): require_vec_distance(row["distance"])
        for row in con.execute(
            f"""
            SELECT rowid, vec_distance_l2(embedding, ?) AS distance
            FROM chunk_vec
            WHERE rowid IN ({placeholders})
            """,
            (query_vector, *chunk_ids),
        )
    }
    reranked = [
        SearchResult(
            **{
                **result.__dict__,
                "vec_distance": distances.get(result.chunk_id),
                "score": hybrid_score(result.fts_rank, distances.get(result.chunk_id)),
                "source": "fts-vec",
            }
        )
        for result in rows
    ]
    return sorted(reranked, key=lambda result: result.score, reverse=True)[:limit]


def rerank_by_fts(
    con: sqlite3.Connection,
    query: str,
    rows: list[SearchResult],
    *,
    limit: int,
    is_pattern: bool,
) -> list[SearchResult]:
    if not rows:
        return []
    ranks = fts_rank_for_ids(
        con,
        query,
        [row.chunk_id for row in rows],
        is_pattern=is_pattern,
    )
    reranked = [
        SearchResult(
            **{
                **result.__dict__,
                "fts_rank": ranks.get(result.chunk_id),
                "score": hybrid_score(ranks.get(result.chunk_id), result.vec_distance),
                "source": "vec-fts",
            }
        )
        for result in rows
    ]
    return sorted(reranked, key=lambda result: result.score, reverse=True)[:limit]


def fts_rank_for_ids(
    con: sqlite3.Connection,
    query: str,
    chunk_ids: list[int],
    *,
    is_pattern: bool,
) -> dict[int, float | None]:
    if not is_pattern and use_like_fallback(query):
        pattern = f"%{escape_like(query)}%"
        rows = con.execute(
            f"""
            SELECT id AS chunk_id, 1.0 AS rank
            FROM chunks
            WHERE id IN ({",".join("?" for _ in chunk_ids)}) AND text LIKE ? ESCAPE '\\'
            """,
            (*chunk_ids, pattern),
        ).fetchall()
        return {int(row["chunk_id"]): -float(row["rank"]) for row in rows}
    rows = con.execute(
        f"""
        SELECT rowid AS chunk_id, bm25(chunks_fts) AS rank
        FROM chunks_fts
        WHERE chunks_fts MATCH ? AND rowid IN ({",".join("?" for _ in chunk_ids)})
        """,
        (fts_match_query(query, is_pattern=is_pattern), *chunk_ids),
    ).fetchall()
    return {int(row["chunk_id"]): float(row["rank"]) for row in rows}


def hybrid_score(fts_rank: float | None, vec_distance: float | None) -> float:
    vec_score = 0.0 if vec_distance is None else distance_to_score(vec_distance)
    text_score = 0.0 if fts_rank is None else min(1.0, max(0.0, -fts_rank))
    return (0.7 * vec_score) + (0.3 * text_score)


def distance_to_score(distance: float) -> float:
    if math.isnan(distance):
        return 0.0
    return 1.0 / (1.0 + max(distance, 0.0))


def require_vec_distance(value: object) -> float:
    if value is None:
        raise ValueError(
            "sqlite-vec returned NULL distance. The DB may contain non-finite embeddings; "
            "rebuild it with `local-doc-search index --rebuild`."
        )
    if not isinstance(value, int | float | str | bytes | bytearray):
        raise ValueError(f"sqlite-vec returned unsupported distance value: {value!r}")
    return float(value)


def use_like_fallback(query: str) -> bool:
    return len(query.strip()) < 3


def escape_like(query: str) -> str:
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def fts_match_query(query: str, *, is_pattern: bool) -> str:
    if is_pattern:
        return query
    return " OR ".join(f'"{escape_fts_phrase(term)}"' for term in trigram_query_terms(query))


def escape_fts_phrase(query: str) -> str:
    return query.replace('"', '""')


def trigram_query_terms(query: str, *, max_terms: int = 128) -> list[str]:
    query = query.strip()
    terms: list[str] = []
    seen: set[str] = set()
    for index in range(max(0, len(query) - 2)):
        term = query[index : index + 3]
        if term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= max_terms:
            break
    return terms
