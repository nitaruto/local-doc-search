from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .db import DbFingerprint, fingerprint_many, fingerprints_match, normalize_db_paths
from .search import SearchMode, SearchResult

REGISTRY_DIR = Path.home() / ".cache" / "tt-search" / "servers"


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
    try:
        current = fingerprint_many(db_paths)
    except OSError:
        return None
    if not fingerprints_match(current, registry.get("fingerprints", [])):
        return None
    if not server_health(registry):
        return None
    return registry


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
    query: str,
    mode: SearchMode,
    limit: int,
    candidates: int,
) -> list[SearchResult]:
    url = f"http://{registry['host']}:{registry['port']}/search"
    body = json.dumps(
        {
            "query": query,
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
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [SearchResult(**item) for item in payload["results"]]
