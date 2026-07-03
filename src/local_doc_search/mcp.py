from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .codex_history import CODEX_HISTORY_DB, CODEX_HISTORY_INDEX_KIND
from .db import (
    as_json,
    connect,
    get_metadata,
    list_indexed_roots,
    normalize_db_paths,
)
from .embeddings import DeviceOption, EmbeddingProvider, create_embedding_provider
from .reload import ReloadableDbSet
from .search import SearchMode, SearchResult, resolve_search, search_many

MCP_PROTOCOL_VERSION = "2024-11-05"
SEARCH_MODES = {"fts", "vec", "fts-vec", "vec-fts"}


class McpSearchServer:
    def __init__(self, db_paths: list[Path], *, device: DeviceOption = "auto") -> None:
        self.db_paths = normalize_db_paths(db_paths)
        if not self.db_paths:
            raise ValueError("At least one --db is required")
        self.device: DeviceOption = device
        self._db_set = ReloadableDbSet(self.db_paths)
        self._codex_db_set: ReloadableDbSet | None = None
        self._embedder: EmbeddingProvider | None = None
        self._codex_embedder: EmbeddingProvider | None = None

    def serve(self) -> None:
        for message, framing in read_stdio_messages():
            response = self.handle_message(message)
            if response is not None:
                write_stdio_message(response, framing=framing)

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        request_id = message.get("id")
        method = message.get("method")
        if request_id is None:
            return None
        try:
            if method == "initialize":
                result = self.initialize_result()
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {
                    "tools": [
                        search_tool_definition(),
                        codex_session_search_tool_definition(),
                        roots_tool_definition(),
                    ]
                }
            elif method == "resources/list":
                result = {"resources": []}
            elif method == "prompts/list":
                result = {"prompts": []}
            elif method == "tools/call":
                result = self.handle_tool_call(message.get("params", {}))
            else:
                return json_rpc_error(request_id, -32601, f"Method not found: {method}")
        except Exception as exc:  # noqa: BLE001
            return json_rpc_error(request_id, -32000, str(exc))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def initialize_result(self) -> dict[str, Any]:
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "local-doc-search", "version": "0.1.0"},
        }

    def handle_tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if name == "roots":
            results = self.roots()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": as_json(results),
                    }
                ],
                "isError": False,
            }
        if name not in {"search", "codex_session_search"}:
            raise ValueError(f"Unknown tool: {name}")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        if name == "codex_session_search":
            results = self.codex_session_search(arguments)
        else:
            results = self.search(arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": as_json(results),
                }
            ],
            "isError": False,
        }

    def search(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        return self.search_db_paths(
            self.db_paths,
            arguments,
            embedder_factory=self.embedder_for_configured_dbs,
        )

    def codex_session_search(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        codex_db = CODEX_HISTORY_DB.expanduser()
        validate_codex_history_db(codex_db)
        return self.search_db_paths(
            [codex_db],
            arguments,
            embedder_factory=self.embedder_for_codex_db,
        )

    def search_db_paths(
        self,
        db_paths: list[Path],
        arguments: dict[str, Any],
        *,
        embedder_factory: Callable[[SearchMode], EmbeddingProvider | None],
    ) -> list[dict[str, Any]]:
        query = optional_argument_str(arguments, "query")
        pattern = optional_argument_str(arguments, "pattern")
        mode = parse_optional_mode(arguments.get("mode"))
        resolved = resolve_search(query=query, pattern=pattern, mode=mode)
        limit = int(arguments.get("limit", 10))
        candidates = max(int(arguments.get("candidates", 50)), limit)
        explain = bool(arguments.get("explain", False))
        self.reload_db_paths(db_paths)
        embedder = embedder_factory(resolved.mode)
        rows = search_many(
            db_paths,
            vector_query=resolved.vector_query,
            fts_query=resolved.fts_query,
            fts_is_pattern=resolved.fts_is_pattern,
            mode=resolved.mode,
            limit=limit,
            candidates=candidates,
            embedder=embedder,
        )
        return [result_to_mcp_dict(row, explain=explain) for row in rows]

    def roots(self) -> list[dict[str, Any]]:
        self._db_set.refresh_if_compatible()
        results: list[dict[str, Any]] = []
        for db_path in self.db_paths:
            with connect(db_path) as con:
                roots = list_indexed_roots(con)
            results.append(
                {
                    "db_path": str(db_path),
                    "roots": [root.__dict__ for root in roots],
                }
            )
        return results

    def embedder_for_configured_dbs(self, mode: SearchMode) -> EmbeddingProvider | None:
        if mode == "fts":
            return None
        if self._embedder is None:
            self._embedder = self.create_embedder_for_paths(self.db_paths)
        return self._embedder

    def embedder_for_codex_db(self, mode: SearchMode) -> EmbeddingProvider | None:
        if mode == "fts":
            return None
        if self._codex_embedder is None:
            self._codex_embedder = self.create_embedder_for_paths([CODEX_HISTORY_DB.expanduser()])
        return self._codex_embedder

    def create_embedder_for_paths(self, db_paths: list[Path]) -> EmbeddingProvider:
        db_set = self.db_set_for_paths(db_paths)
        model = db_set.embedding_metadata.get("embedding_model")
        if model is None:
            raise ValueError(
                "DB does not contain embedding metadata. "
                "Rebuild it with `local-doc-search index`."
            )
        return create_embedding_provider(model_name=model, device=self.device)

    def reload_db_paths(self, db_paths: list[Path]) -> None:
        self.db_set_for_paths(db_paths).refresh_if_compatible()

    def db_set_for_paths(self, db_paths: list[Path]) -> ReloadableDbSet:
        normalized = normalize_db_paths(db_paths)
        if normalized == self.db_paths:
            return self._db_set
        codex_db = CODEX_HISTORY_DB.expanduser().resolve()
        if normalized == [codex_db]:
            if self._codex_db_set is None:
                self._codex_db_set = ReloadableDbSet(normalized)
            return self._codex_db_set
        return ReloadableDbSet(normalized)


def run_mcp_server(db_paths: list[Path], *, device: DeviceOption = "auto") -> None:
    McpSearchServer(db_paths, device=device).serve()


def search_tool_definition() -> dict[str, Any]:
    return {
        "name": "search",
        "description": (
            "Search local-doc-search SQLite indexes. Provide query, pattern, or both."
        ),
        "inputSchema": search_input_schema(),
    }


def codex_session_search_tool_definition() -> dict[str, Any]:
    return {
        "name": "codex_session_search",
        "description": (
            "Search indexed Codex session history from the fixed local-doc-search "
            "Codex history DB. Provide query, pattern, or both."
        ),
        "inputSchema": search_input_schema(),
    }


def search_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Semantic/vector search query. In FTS modes without pattern, "
                    "it is converted to an OR query over up to 128 trigram tokens."
                ),
            },
            "pattern": {
                "type": "string",
                "description": (
                    "FTS5 MATCH pattern. Supports FTS5 operators such as "
                    "AND, OR, NOT, and NEAR."
                ),
            },
            "mode": {
                "type": "string",
                "enum": sorted(SEARCH_MODES),
                "description": (
                    "When both query and pattern are provided, vec-fts gets vector "
                    "candidates with query then reranks/filters with pattern; "
                    "fts-vec gets FTS candidates with pattern then reranks with query."
                ),
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "default": 10,
                "description": "Number of results.",
            },
            "candidates": {
                "type": "integer",
                "minimum": 1,
                "default": 50,
                "description": "Candidate count before rerank.",
            },
            "explain": {
                "type": "boolean",
                "default": False,
                "description": "Include component scores.",
            },
        },
        "additionalProperties": False,
    }


def roots_tool_definition() -> dict[str, Any]:
    return {
        "name": "roots",
        "description": (
            "List indexed root directories for the SQLite DBs configured for this MCP server."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }


def parse_mode(value: object) -> SearchMode:
    mode = str(value)
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return mode  # type: ignore[return-value]


def parse_optional_mode(value: object) -> SearchMode | None:
    if value is None:
        return None
    return parse_mode(value)


def optional_argument_str(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    return str(value)


def validate_codex_history_db(db_path: Path) -> None:
    if not db_path.exists():
        raise ValueError(
            f"Codex history DB does not exist: {db_path}. "
            "Run `local-doc-search codex-index` first."
        )
    with connect(db_path) as con:
        index_kind = get_metadata(con, "index_kind")
    if index_kind != CODEX_HISTORY_INDEX_KIND:
        raise ValueError(
            f"DB is not a Codex history index: {db_path}. "
            "Run `local-doc-search codex-index --rebuild`."
        )


def result_to_mcp_dict(result: SearchResult, *, explain: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": result.score,
        "db_path": result.db_path,
        "path": result.path,
        "relative_path": result.relative_path,
        "start_line": result.start_line,
        "end_line": result.end_line,
        "chunk_index": result.chunk_index,
        "start_offset": result.start_offset,
        "end_offset": result.end_offset,
        "source": result.source,
        "text": result.text,
    }
    if result.session_id is not None:
        payload.update(
            {
                "session_id": result.session_id,
                "cwd": result.cwd,
                "role": result.role,
                "turn_id": result.turn_id,
                "timestamp": result.timestamp,
                "session_path": result.session_path,
                "line_no": result.line_no,
            }
        )
    if explain:
        payload["fts_rank"] = result.fts_rank
        payload["vec_distance"] = result.vec_distance
    return payload


def json_rpc_error(request_id: object, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def read_stdio_messages() -> Iterator[tuple[dict[str, Any], str]]:
    stream = sys.stdin.buffer
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.strip():
            continue
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
            while True:
                header = stream.readline()
                if header in {b"\r\n", b"\n", b""}:
                    break
            body = stream.read(length)
            yield json.loads(body.decode("utf-8")), "content-length"
            continue
        yield json.loads(line.decode("utf-8")), "json-lines"


def write_stdio_message(message: dict[str, Any], *, framing: str) -> None:
    data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if framing == "content-length":
        sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    sys.stdout.write(data.decode("utf-8"))
    sys.stdout.write("\n")
    sys.stdout.flush()
