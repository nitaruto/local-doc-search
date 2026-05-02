from __future__ import annotations

from pathlib import Path

from .db import DbFingerprint, fingerprint_many, validate_embedding_compatible


class ReloadableDbSet:
    def __init__(self, db_paths: list[Path]) -> None:
        self.db_paths = db_paths
        self.fingerprints = fingerprint_many(db_paths)
        self.embedding_metadata = validate_embedding_compatible(self.fingerprints)

    def refresh_if_compatible(self) -> bool:
        current = fingerprint_many(self.db_paths)
        if fingerprints_unchanged(current, self.fingerprints):
            return False
        current_metadata = validate_embedding_compatible(current)
        if current_metadata != self.embedding_metadata:
            raise ValueError(
                "DB embedding metadata changed after server startup. "
                "Restart local-doc-search server."
            )
        self.fingerprints = current
        return True


def fingerprints_unchanged(
    current: list[DbFingerprint],
    expected: list[DbFingerprint],
) -> bool:
    if len(current) != len(expected):
        return False
    for current_item, expected_item in zip(current, expected, strict=True):
        if current_item.path != expected_item.path:
            return False
        if current_item.size != expected_item.size:
            return False
        if current_item.mtime_ns != expected_item.mtime_ns:
            return False
    return True
