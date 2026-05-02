from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from .client import find_live_server, write_registry
from .db import DbFingerprint
from .embeddings import DeviceOption, create_embedding_provider
from .reload import ReloadableDbSet
from .search import resolve_search, search_many


class SearchServerState:
    def __init__(self, db_paths: list[Path], *, device: DeviceOption) -> None:
        self.db_paths = db_paths
        self.device = device
        self.db_set = ReloadableDbSet(db_paths)
        self.embedder = create_embedding_provider(
            model_name=self.db_set.embedding_metadata["embedding_model"],
            device=device,
        )

    @property
    def fingerprints(self) -> list[DbFingerprint]:
        return self.db_set.fingerprints

    def refresh_if_compatible(self) -> None:
        self.db_set.refresh_if_compatible()

    def resolve_requested_db_paths(self, payload: dict[str, Any]) -> list[Path]:
        requested = payload.get("db_paths")
        if requested is None:
            return self.db_paths
        if not isinstance(requested, list):
            raise ValueError("db_paths must be a list")
        configured = {str(path.expanduser().resolve()): path for path in self.db_paths}
        db_paths: list[Path] = []
        for item in requested:
            resolved = Path(str(item)).expanduser().resolve()
            db_path = configured.get(str(resolved))
            if db_path is None:
                raise ValueError(f"Requested DB is not served by this server: {resolved}")
            db_paths.append(db_path)
        if not db_paths:
            raise ValueError("At least one requested DB is required")
        return db_paths


class SearchRequestHandler(BaseHTTPRequestHandler):
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
        server = cast(SearchHTTPServer, self.server)
        server.state.refresh_if_compatible()
        db_paths = server.state.resolve_requested_db_paths(payload)
        mode = payload.get("mode", "fts-vec")
        if mode not in {"fts", "vec", "fts-vec", "vec-fts"}:
            raise ValueError(f"Unknown mode: {mode}")
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
        else:
            resolved = resolve_search(
                query=optional_payload_str(payload, "query"),
                pattern=None,
                mode=mode,
            )
        embedder = None if resolved.mode == "fts" else server.state.embedder
        return search_many(
            db_paths,
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
    existing = find_live_server(db_paths)
    if existing is not None:
        raise ValueError(
            "A local-doc-search server is already running for this DB set at "
            f"http://{existing['host']}:{existing['port']}"
        )
    state = SearchServerState(db_paths, device=device)
    httpd = SearchHTTPServer((host, port), state)
    actual_host, actual_port = cast(tuple[str, int], httpd.server_address)
    write_registry(
        db_paths,
        host=str(actual_host),
        port=int(actual_port),
        device=state.embedder.device,
        fingerprints=state.fingerprints,
    )
    print(f"local-doc-search server listening on http://{actual_host}:{actual_port}")
    print("Press Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
