from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .client import write_registry
from .db import fingerprint_many, fingerprints_match, validate_embedding_compatible
from .embeddings import DeviceOption, create_embedding_provider
from .search import resolve_search, search_many


class SearchServerState:
    def __init__(self, db_paths: list[Path], *, device: DeviceOption) -> None:
        self.db_paths = db_paths
        self.device = device
        self.fingerprints = fingerprint_many(db_paths)
        metadata = validate_embedding_compatible(self.fingerprints)
        self.embedder = create_embedding_provider(
            model_name=metadata["embedding_model"],
            device=device,
        )

    def assert_fresh(self) -> None:
        current = fingerprint_many(self.db_paths)
        expected = [fingerprint.__dict__ for fingerprint in self.fingerprints]
        if not fingerprints_match(current, expected):
            raise ValueError("DB files changed after server startup. Restart tt-search server.")


class SearchRequestHandler(BaseHTTPRequestHandler):
    server: SearchHTTPServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self.write_json({"ok": True})

    def do_POST(self) -> None:
        if self.path != "/search":
            self.send_error(404)
            return
        try:
            payload = self.read_json()
            results = self.handle_search(payload)
        except Exception as exc:  # noqa: BLE001
            self.write_json({"error": str(exc)}, status=400)
            return
        self.write_json({"results": [result.__dict__ for result in results]})

    def handle_search(self, payload: dict[str, Any]):
        self.server.state.assert_fresh()
        mode = payload.get("mode", "fts-vec")
        if mode not in {"fts", "vec", "fts-vec", "vec-fts"}:
            raise ValueError(f"Unknown mode: {mode}")
        resolved = resolve_search(
            query=optional_payload_str(payload, "query"),
            pattern=None,
            mode=mode,
        )
        if payload.get("vector_query") is not None or payload.get("fts_query") is not None:
            resolved = resolve_search(
                query=optional_payload_str(payload, "vector_query"),
                pattern=(
                    optional_payload_str(payload, "fts_query")
                    if bool(payload.get("fts_is_pattern", False))
                    else None
                ),
                mode=mode,
            )
            if not bool(payload.get("fts_is_pattern", False)):
                resolved = resolve_search(
                    query=optional_payload_str(payload, "vector_query")
                    or optional_payload_str(payload, "fts_query"),
                    pattern=None,
                    mode=mode,
                )
        embedder = None if resolved.mode == "fts" else self.server.state.embedder
        return search_many(
            self.server.state.db_paths,
            vector_query=resolved.vector_query,
            fts_query=resolved.fts_query,
            fts_is_pattern=resolved.fts_is_pattern,
            mode=resolved.mode,
            limit=int(payload.get("limit", 10)),
            candidates=int(payload.get("candidates", 50)),
            embedder=embedder,
        )

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def optional_payload_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


class SearchHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: SearchServerState) -> None:
        super().__init__(server_address, SearchRequestHandler)
        self.state = state


def run_server(db_paths: list[Path], *, host: str, port: int, device: DeviceOption) -> None:
    state = SearchServerState(db_paths, device=device)
    httpd = SearchHTTPServer((host, port), state)
    actual_host, actual_port = httpd.server_address
    write_registry(
        db_paths,
        host=str(actual_host),
        port=int(actual_port),
        device=state.embedder.device,
        fingerprints=state.fingerprints,
    )
    print(f"tt-search server listening on http://{actual_host}:{actual_port}")
    print("Press Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
