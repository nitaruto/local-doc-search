from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .db import as_json, fingerprint_many, normalize_db_paths, validate_embedding_compatible
from .embeddings import DeviceOption, EmbeddingProvider, create_embedding_provider
from .search import SearchMode, SearchResult, search_many

MCP_PROTOCOL_VERSION = "2024-11-05"
SEARCH_MODES = {"fts", "vec", "fts-vec", "vec-fts"}


class McpSearchServer:
    def __init__(self, db_paths: list[Path], *, device: DeviceOption = "auto") -> None:
        self.db_paths = normalize_db_paths(db_paths)
        if not self.db_paths:
            raise ValueError("At least one --db is required")
        self.device = device
        self._embedder: EmbeddingProvider | None = None

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
                result = {"tools": [search_tool_definition()]}
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
            "serverInfo": {"name": "tt-search", "version": "0.1.0"},
        }

    def handle_tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if name != "search":
            raise ValueError(f"Unknown tool: {name}")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
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
        query = str(arguments["query"])
        mode = parse_mode(arguments.get("mode", "fts-vec"))
        limit = int(arguments.get("limit", 10))
        candidates = max(int(arguments.get("candidates", 50)), limit)
        explain = bool(arguments.get("explain", False))
        embedder = self.embedder_for(mode)
        rows = search_many(
            self.db_paths,
            query=query,
            mode=mode,
            limit=limit,
            candidates=candidates,
            embedder=embedder,
        )
        return [result_to_mcp_dict(row, explain=explain) for row in rows]

    def embedder_for(self, mode: SearchMode) -> EmbeddingProvider | None:
        if mode == "fts":
            return None
        if self._embedder is None:
            fingerprints = fingerprint_many(self.db_paths)
            metadata = validate_embedding_compatible(fingerprints)
            model = metadata.get("embedding_model")
            if model is None:
                raise ValueError(
                    "DB does not contain embedding metadata. Rebuild it with `tt-search index`."
                )
            self._embedder = create_embedding_provider(model_name=model, device=self.device)
        return self._embedder


def run_mcp_server(db_paths: list[Path], *, device: DeviceOption = "auto") -> None:
    McpSearchServer(db_paths, device=device).serve()


def search_tool_definition() -> dict[str, Any]:
    return {
        "name": "search",
        "description": "Search local tt-search SQLite indexes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "mode": {
                    "type": "string",
                    "enum": sorted(SEARCH_MODES),
                    "default": "fts-vec",
                    "description": "Search mode.",
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
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def parse_mode(value: object) -> SearchMode:
    mode = str(value)
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return mode  # type: ignore[return-value]


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
