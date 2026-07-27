"""Small shared index for validated Task, Tool, and VQA artifacts.

Artifact-specific validation stays with TaskGen, ToolGen, or Execution VQA.
This module only records the common lookup fields after that validation has
completed.  It is intentionally an exact semantic-key index, not another
admission or provenance layer.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_KINDS = frozenset({"task", "tool", "vqa"})


class ArtifactRegistryError(ValueError):
    """Raised when the lightweight artifact index is malformed or ambiguous."""


def _canonical_value(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactRegistryError(f"{field} must be an object")
    try:
        canonical = json.loads(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactRegistryError(f"{field} must be JSON serializable") from exc
    if not isinstance(canonical, dict):
        raise ArtifactRegistryError(f"{field} must be an object")
    return canonical


def _canonical_key(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _canonical_value(value, field="semantic_key"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_entry(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactRegistryError("artifact entry must be an object")
    expected = {
        "kind",
        "semantic_key",
        "artifact_path",
        "validation",
        "source_query",
        "reuse_count",
    }
    if set(value) != expected:
        raise ArtifactRegistryError("artifact entry fields are invalid")
    kind = value["kind"]
    if kind not in ARTIFACT_KINDS:
        raise ArtifactRegistryError(f"unsupported artifact kind: {kind!r}")
    artifact_path = value["artifact_path"]
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ArtifactRegistryError("artifact_path must be a non-empty string")
    source_query = value["source_query"]
    if not isinstance(source_query, str) or not source_query.strip():
        raise ArtifactRegistryError("source_query must be a non-empty string")
    reuse_count = value["reuse_count"]
    if isinstance(reuse_count, bool) or not isinstance(reuse_count, int):
        raise ArtifactRegistryError("reuse_count must be an integer")
    if reuse_count < 0:
        raise ArtifactRegistryError("reuse_count must be non-negative")
    return {
        "kind": kind,
        "semantic_key": _canonical_value(
            value["semantic_key"], field="semantic_key"
        ),
        "artifact_path": artifact_path,
        "validation": _canonical_value(value["validation"], field="validation"),
        "source_query": source_query,
        "reuse_count": reuse_count,
    }


class ArtifactRegistry:
    """Persist and retrieve the six fields shared by generated artifacts."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)

    def _load(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactRegistryError(
                f"invalid artifact registry: {self.index_path}"
            ) from exc
        if not isinstance(payload, list):
            raise ArtifactRegistryError("artifact registry must be a JSON list")
        entries = [_validate_entry(item) for item in payload]
        identities = [
            (item["kind"], _canonical_key(item["semantic_key"]))
            for item in entries
        ]
        if len(identities) != len(set(identities)):
            raise ArtifactRegistryError(
                "artifact registry contains duplicate semantic keys"
            )
        return entries

    def _write(self, entries: list[dict[str, Any]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def register(
        self,
        *,
        kind: str,
        semantic_key: Mapping[str, Any],
        artifact_path: str | Path,
        validation: Mapping[str, Any],
        source_query: str,
    ) -> dict[str, Any]:
        """Add a validated artifact, or return the existing exact match."""

        entry = _validate_entry(
            {
                "kind": kind,
                "semantic_key": semantic_key,
                "artifact_path": str(artifact_path),
                "validation": validation,
                "source_query": source_query,
                "reuse_count": 0,
            }
        )
        entries = self._load()
        identity = (kind, _canonical_key(entry["semantic_key"]))
        for current in entries:
            current_identity = (
                current["kind"],
                _canonical_key(current["semantic_key"]),
            )
            if current_identity != identity:
                continue
            if (
                current["artifact_path"] != entry["artifact_path"]
                or current["validation"] != entry["validation"]
            ):
                raise ArtifactRegistryError(
                    "semantic key is already bound to a different artifact"
                )
            return deepcopy(current)
        entries.append(entry)
        self._write(entries)
        return deepcopy(entry)

    def retrieve(
        self,
        *,
        kind: str,
        semantic_key: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return an exact match and record one successful reuse."""

        return self.mark_reuse(kind=kind, semantic_key=semantic_key)

    def find(
        self,
        *,
        kind: str,
        semantic_key: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return an exact match without recording a completed reuse."""

        if kind not in ARTIFACT_KINDS:
            raise ArtifactRegistryError(f"unsupported artifact kind: {kind!r}")
        key = _canonical_key(semantic_key)
        entries = self._load()
        for entry in entries:
            if (
                entry["kind"] == kind
                and _canonical_key(entry["semantic_key"]) == key
            ):
                return deepcopy(entry)
        return None

    def mark_reuse(
        self,
        *,
        kind: str,
        semantic_key: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Record one successfully materialized exact reuse."""

        if kind not in ARTIFACT_KINDS:
            raise ArtifactRegistryError(f"unsupported artifact kind: {kind!r}")
        key = _canonical_key(semantic_key)
        entries = self._load()
        for entry in entries:
            if (
                entry["kind"] == kind
                and _canonical_key(entry["semantic_key"]) == key
            ):
                entry["reuse_count"] += 1
                self._write(entries)
                return deepcopy(entry)
        return None

    def entries(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        """Return a copy of all entries, optionally limited to one kind."""

        if kind is not None and kind not in ARTIFACT_KINDS:
            raise ArtifactRegistryError(f"unsupported artifact kind: {kind!r}")
        return [
            deepcopy(entry)
            for entry in self._load()
            if kind is None or entry["kind"] == kind
        ]


__all__ = [
    "ARTIFACT_KINDS",
    "ArtifactRegistry",
    "ArtifactRegistryError",
]
