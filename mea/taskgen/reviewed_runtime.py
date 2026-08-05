"""Runtime dependency revalidation and materialization for reviewed Tasks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .reviewed_schema import (
    RUNTIME_DEPENDENCY_PATHS,
    ReviewedTaskRegistryError,
    _runtime_dependency_hashes,
    _safe_artifact,
    _unresolved_root,
    _validate_runtime_dependency_hashes,
    _write_bytes_atomic,
)
from .reviewed_storage import _load_entry, _match_from_loaded, load_reviewed_task_registry


def validate_reviewed_task_runtime(
    match: Mapping[str, Any], repo_root: str | Path
) -> dict[str, str]:
    """Verify imported official task/utils bytes before materialization."""

    expected = _validate_runtime_dependency_hashes(
        match.get("runtime_dependency_hashes")
        if isinstance(match, Mapping)
        else None
    )
    root = Path(repo_root).expanduser().resolve()
    actual = _runtime_dependency_hashes(root)
    if actual != expected:
        changed = sorted(
            relative
            for relative in RUNTIME_DEPENDENCY_PATHS
            if actual.get(relative) != expected.get(relative)
        )
        raise ReviewedTaskRegistryError(
            f"reviewed Task runtime dependencies changed: {changed}"
        )
    return actual


def copy_reviewed_task_artifacts(
    match: Mapping[str, Any],
    destination: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy only verified task artifacts into an absent or empty destination."""

    if not isinstance(match, Mapping):
        raise ReviewedTaskRegistryError("reviewed task match must be an object")
    for field in ("registry_dir", "registration_id", "artifact_id"):
        if not isinstance(match.get(field), str) or not match[field]:
            raise ReviewedTaskRegistryError(f"reviewed task match lacks {field}")
    root = _unresolved_root(match["registry_dir"], label="reviewed task registry")
    index = load_reviewed_task_registry(root)
    entry = next(
        (
            item
            for item in index["entries"]
            if item["registration_id"] == match["registration_id"]
        ),
        None,
    )
    if entry is None or entry["artifact_id"] != match["artifact_id"]:
        raise ReviewedTaskRegistryError("reviewed task match is not registered")
    loaded = _load_entry(root, entry)
    expected_match = _match_from_loaded(loaded)
    for field in (
        "semantic_key",
        "semantic_key_sha256",
        "runtime_dependency_hashes",
        "review_authority",
        "reviewed_at",
        "review_attestation_paper_eligible",
    ):
        if match.get(field) != expected_match[field]:
            raise ReviewedTaskRegistryError(f"reviewed task match {field} differs")
    if repo_root is not None:
        validate_reviewed_task_runtime(expected_match, repo_root)

    raw_destination = Path(destination).expanduser()
    if raw_destination.is_symlink():
        raise ReviewedTaskRegistryError("artifact destination must not be a symlink")
    target = raw_destination.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise ReviewedTaskRegistryError("artifact destination must be outside registry")
    if target.exists():
        if not target.is_dir():
            raise ReviewedTaskRegistryError("artifact destination must be a directory")
        if any(target.iterdir()):
            raise ReviewedTaskRegistryError("artifact destination must be empty")

    payloads: dict[str, bytes] = {}
    for relative, descriptor in loaded["verified_artifacts"].items():
        source_path = Path(descriptor["path"])
        payload = source_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ReviewedTaskRegistryError(
                f"reviewed task artifact changed before copy: {relative}"
            )
        payloads[relative] = payload
    target.mkdir(parents=True, exist_ok=True)
    for relative, payload in payloads.items():
        output = _safe_artifact(target, relative, label="artifact destination")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(output, payload)
        if hashlib.sha256(output.read_bytes()).hexdigest() != (
            loaded["verified_artifacts"][relative]["sha256"]
        ):
            raise ReviewedTaskRegistryError(
                f"copied task artifact integrity failed: {relative}"
            )
    return {
        "registration_id": match["registration_id"],
        "artifact_id": match["artifact_id"],
        "runtime_dependency_hashes": expected_match[
            "runtime_dependency_hashes"
        ],
        "review_authority": expected_match["review_authority"],
        "reviewed_at": expected_match["reviewed_at"],
        "review_attestation_paper_eligible": expected_match[
            "review_attestation_paper_eligible"
        ],
        "destination": str(target),
        "files": {
            relative: loaded["verified_artifacts"][relative]["sha256"]
            for relative in sorted(payloads)
        },
    }
