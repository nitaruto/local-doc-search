from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .db import connect, serialize_vector
from .embeddings import EmbeddingProvider

SearchMode = Literal["fts", "vec", "fts-vec", "vec-fts"]


@dataclass(frozen=True)
class SearchResult:
    db_path: str | None
    chunk_id: int
    path: str
    relative_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    text: str
    score: float
    fts_rank: float | None = None
    vec_distance: float | None = None
    source: str = ""


def search(
    con: sqlite3.Connection,
    *,
    query: str,
    mode: SearchMode,
    limit: int,
    candidates: int,
    embedder: EmbeddingProvider | None,
    db_path: str | None = None,
    query_vector: bytes | None = None,
) -> list[SearchResult]:
    if mode in {"vec", "fts-vec", "vec-fts"} and embedder is None:
        raise ValueError("Embedding provider is required for vector search")
    if mode == "fts":
        return with_db_path(fts_candidates(con, query, candidates=limit), db_path)
    if mode == "vec":
        assert embedder is not None
        return with_db_path(
            vec_candidates(
                con,
                query,
                candidates=limit,
                embedder=embedder,
                query_vector=query_vector,
            ),
            db_path,
        )
    if mode == "fts-vec":
        assert embedder is not None
        rows = fts_candidates(con, query, candidates=candidates)
        return with_db_path(
            rerank_by_vector(
                con,
                query,
                rows,
                limit=limit,
                embedder=embedder,
                query_vector=query_vector,
            ),
            db_path,
        )
    if mode == "vec-fts":
        assert embedder is not None
        rows = vec_candidates(
            con,
            query,
            candidates=candidates,
            embedder=embedder,
            query_vector=query_vector,
        )
        return with_db_path(rerank_by_fts(con, query, rows, limit=limit), db_path)
    raise ValueError(f"Unknown mode: {mode}")


def search_many(
    db_paths: list[Path],
    *,
    query: str,
    mode: SearchMode,
    limit: int,
    candidates: int,
    embedder: EmbeddingProvider | None,
) -> list[SearchResult]:
    query_vector = None
    if mode in {"vec", "fts-vec", "vec-fts"}:
        if embedder is None:
            raise ValueError("Embedding provider is required for vector search")
        query_vector = serialize_vector(embedder.embed_query(query))

    all_results: list[SearchResult] = []
    per_db_candidates = max(candidates, limit)
    for db_path in db_paths:
        with connect(db_path) as con:
            all_results.extend(
                search(
                    con,
                    query=query,
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


def fts_candidates(con: sqlite3.Connection, query: str, *, candidates: int) -> list[SearchResult]:
    if use_like_fallback(query):
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
            c.text AS text,
            bm25(chunks_fts) AS fts_rank
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN files f ON f.id = c.file_id
        WHERE chunks_fts MATCH ?
        ORDER BY fts_rank
        LIMIT ?
        """,
        (escape_fts_query(query), candidates),
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
            text=str(row["text"]),
            score=-float(row["fts_rank"]),
            fts_rank=float(row["fts_rank"]),
            source="fts",
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
            c.text AS text
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
            text=str(row["text"]),
            score=1.0,
            fts_rank=None,
            source="like",
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
            c.text AS text,
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
            text=str(row["text"]),
            score=distance_to_score(float(row["vec_distance"])),
            vec_distance=float(row["vec_distance"]),
            source="vec",
        )
        for row in rows
    ]


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
        int(row["rowid"]): float(row["distance"])
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
) -> list[SearchResult]:
    if not rows:
        return []
    ranks = fts_rank_for_ids(con, query, [row.chunk_id for row in rows])
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
    con: sqlite3.Connection, query: str, chunk_ids: list[int]
) -> dict[int, float | None]:
    if use_like_fallback(query):
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
        (escape_fts_query(query), *chunk_ids),
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


def use_like_fallback(query: str) -> bool:
    return len(query.strip()) < 3


def escape_like(query: str) -> str:
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def escape_fts_query(query: str) -> str:
    return query.replace('"', '""')
