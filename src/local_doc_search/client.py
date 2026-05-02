from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .db import DbFingerprint, normalize_db_paths
from .search import SearchMode, SearchResult

REGISTRY_DIR = Path.home() / ".cache" / "local-doc-search" / "servers"
HEALTH_RETRY_SECONDS = 3.0
HEALTH_RETRY_INTERVAL_SECONDS = 0.1


class ServerSearchError(RuntimeError):
    pass


def db_set_hash(db_paths: list[Path]) -> str:
    normalized = [str(path) for path in normalize_db_paths(db_paths)]
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:24]


def registry_path(db_paths: list[Path]) -> Path:
    return REGISTRY_DIR / f"{db_set_hash(db_paths)}.json"


def write_registry(
    db_paths: list[Path],
    *,
    host: str,
    port: int,
    device: str,
    fingerprints: list[DbFingerprint],
) -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = registry_path(db_paths)
    payload = {
        "host": host,
        "port": port,
        "device": device,
        "db_paths": [fingerprint.path for fingerprint in fingerprints],
        "fingerprints": [fingerprint.__dict__ for fingerprint in fingerprints],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_registry(db_paths: list[Path]) -> dict[str, Any] | None:
    path = registry_path(db_paths)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_live_server(db_paths: list[Path]) -> dict[str, Any] | None:
    registry = read_registry(db_paths)
    if registry is None:
        return None
    if not wait_for_server_health(registry):
        return None
    return registry


def find_subset_live_servers(db_paths: list[Path]) -> list[dict[str, Any]]:
    requested = normalize_db_paths(db_paths)
    requested_set = {str(path) for path in requested}
    registries: list[dict[str, Any]] = []
    for registry in find_live_servers():
        server_paths = registry_db_paths(registry)
        server_set = {str(path) for path in server_paths}
        if not requested_set.issubset(server_set):
            continue
        registry = dict(registry)
        registry["requested_db_paths"] = [str(path) for path in requested]
        registries.append(registry)
    return registries


def find_live_servers() -> list[dict[str, Any]]:
    registries: list[dict[str, Any]] = []
    for registry in read_all_registries():
        db_paths = registry_db_paths(registry)
        if not db_paths:
            continue
        if not wait_for_server_health(registry):
            continue
        registries.append(registry)
    return registries


def read_all_registries() -> list[dict[str, Any]]:
    if not REGISTRY_DIR.exists():
        return []
    registries: list[dict[str, Any]] = []
    for path in sorted(REGISTRY_DIR.glob("*.json")):
        try:
            registries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return registries


def registry_db_paths(registry: dict[str, Any]) -> list[Path]:
    db_paths = registry.get("db_paths")
    if not isinstance(db_paths, list):
        return []
    return [Path(str(db_path)) for db_path in db_paths]


def wait_for_server_health(registry: dict[str, Any]) -> bool:
    deadline = time.monotonic() + HEALTH_RETRY_SECONDS
    while True:
        if server_health(registry):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(HEALTH_RETRY_INTERVAL_SECONDS)


def server_health(registry: dict[str, Any]) -> bool:
    url = f"http://{registry['host']}:{registry['port']}/health"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def search_via_server(
    registry: dict[str, Any],
    *,
    db_paths: list[Path] | None = None,
    query: str | None = None,
    vector_query: str | None = None,
    fts_query: str | None = None,
    fts_is_pattern: bool = False,
    mode: SearchMode,
    limit: int,
    candidates: int,
) -> list[SearchResult]:
    url = f"http://{registry['host']}:{registry['port']}/search"
    body = json.dumps(
        {
            "query": query,
            "db_paths": (
                [str(path) for path in normalize_db_paths(db_paths)]
                if db_paths is not None
                else registry.get("requested_db_paths")
            ),
            "vector_query": vector_query,
            "fts_query": fts_query,
            "fts_is_pattern": fts_is_pattern,
            "mode": mode,
            "limit": limit,
            "candidates": candidates,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ServerSearchError(
            f"local-doc-search server returned HTTP {exc.code}: {body}"
        ) from exc
    return [SearchResult(**item) for item in payload["results"]]
