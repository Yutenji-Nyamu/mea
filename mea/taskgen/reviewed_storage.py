"""Persistent storage and exact lookup for reviewed generated Tasks."""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .reviewed_schema import (
    ADMISSION_POLICY,
    BASE_ARTIFACTS,
    HASH_FIELDS,
    INDEX_FIELDS,
    REGISTRATION_FIELDS,
    REGISTRATION_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    REGISTRY_SCOPE,
    SUCCESS_SPEC_ARTIFACT,
    ReviewedTaskRegistryError,
    _canonical_sha256,
    _file_sha256,
    _pretty_json_bytes,
    _read_json,
    _require_hash,
    _safe_artifact,
    _unresolved_root,
    _validate_runtime_dependency_hashes,
    _validate_semantic_key,
    _write_bytes_atomic,
)
from .reviewed_source import _source_artifacts, validate_task_review_manifest


def _empty_index() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "scope": REGISTRY_SCOPE,
        "admission_policy": ADMISSION_POLICY,
        "entries": [],
    }


def _read_index(root: Path) -> dict[str, Any]:
    index_path = _safe_artifact(root, "index.json", label="registry index")
    if not index_path.exists():
        if root.exists() and any(root.iterdir()):
            raise ReviewedTaskRegistryError(
                "non-empty reviewed registry has no index.json"
            )
        return _empty_index()
    if not index_path.is_file():
        raise ReviewedTaskRegistryError("reviewed registry index must be a file")
    index = _read_json(index_path, label="reviewed task registry index")
    if set(index) != {"schema_version", "scope", "admission_policy", "entries"}:
        raise ReviewedTaskRegistryError("reviewed task registry index fields are invalid")
    if (
        index.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or index.get("scope") != REGISTRY_SCOPE
        or index.get("admission_policy") != ADMISSION_POLICY
        or not isinstance(index.get("entries"), list)
    ):
        raise ReviewedTaskRegistryError("unsupported reviewed task registry index")
    registration_ids: set[str] = set()
    semantic_hashes: set[str] = set()
    for entry in index["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != INDEX_FIELDS:
            raise ReviewedTaskRegistryError("reviewed task index entry fields are invalid")
        registration_id = entry.get("registration_id")
        artifact_id = entry.get("artifact_id")
        if not isinstance(registration_id, str) or re.fullmatch(
            r"reviewed_task_[0-9a-f]{20}", registration_id
        ) is None:
            raise ReviewedTaskRegistryError("invalid reviewed task registration_id")
        if not isinstance(artifact_id, str) or re.fullmatch(
            r"task_artifact_[0-9a-f]{20}", artifact_id
        ) is None:
            raise ReviewedTaskRegistryError("invalid reviewed task artifact_id")
        semantic_hash = _require_hash(
            entry.get("semantic_key_sha256"), field="entry semantic_key_sha256"
        )
        _validate_runtime_dependency_hashes(entry.get("runtime_dependency_hashes"))
        if registration_id in registration_ids or semantic_hash in semantic_hashes:
            raise ReviewedTaskRegistryError(
                "reviewed task registry contains ambiguous duplicate entries"
            )
        registration_ids.add(registration_id)
        semantic_hashes.add(semantic_hash)
    return index


def _expected_entry_paths(registration_id: str, artifact_names: set[str]) -> set[str]:
    files = {"registration.json", "review_manifest.json"}
    files.update(f"artifacts/{name}" for name in artifact_names)
    nodes = set(files)
    for file_name in files:
        parent = Path(file_name).parent
        while str(parent) not in {"", "."}:
            nodes.add(parent.as_posix())
            parent = parent.parent
    return nodes


def _validate_entry_layout(entry_dir: Path, artifact_names: set[str]) -> None:
    if entry_dir.is_symlink() or not entry_dir.is_dir():
        raise ReviewedTaskRegistryError("reviewed task entry must be a real directory")
    actual: set[str] = set()
    for path in entry_dir.rglob("*"):
        relative = path.relative_to(entry_dir).as_posix()
        if path.is_symlink():
            raise ReviewedTaskRegistryError(
                f"reviewed task entry must not contain symlinks: {relative}"
            )
        actual.add(relative)
    expected = _expected_entry_paths(entry_dir.name, artifact_names)
    if actual != expected:
        raise ReviewedTaskRegistryError(
            "reviewed task entry contains missing or unapproved files"
        )


def _artifact_names_from_hashes(hashes: Any) -> set[str]:
    if not isinstance(hashes, Mapping):
        raise ReviewedTaskRegistryError("artifact_hashes must be an object")
    expected = set(BASE_ARTIFACTS)
    if hashes.get(SUCCESS_SPEC_ARTIFACT) is not None:
        expected.add(SUCCESS_SPEC_ARTIFACT)
    if set(hashes) != expected:
        raise ReviewedTaskRegistryError("artifact_hashes contains an invalid artifact set")
    for relative, value in hashes.items():
        _require_hash(value, field=f"artifact hash {relative}")
    return expected


def _load_entry(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    if entry.get("scope") != REGISTRY_SCOPE or entry.get("status") != "approved":
        raise ReviewedTaskRegistryError("reviewed task entry is not approved")
    registration_id = entry["registration_id"]
    artifact_names = _artifact_names_from_hashes(entry.get("artifact_hashes"))
    entry_dir = _safe_artifact(
        root, f"entries/{registration_id}", label="reviewed task entry"
    )
    _validate_entry_layout(entry_dir, artifact_names)
    expected_registration = f"entries/{registration_id}/registration.json"
    expected_review = f"entries/{registration_id}/review_manifest.json"
    if (
        entry.get("registration_artifact") != expected_registration
        or entry.get("review_manifest_artifact") != expected_review
    ):
        raise ReviewedTaskRegistryError("reviewed task metadata path is not fixed")
    registration_path = _safe_artifact(
        root, expected_registration, label="reviewed task registration"
    )
    review_path = _safe_artifact(root, expected_review, label="task review manifest")
    if _file_sha256(registration_path) != entry.get("registration_artifact_sha256"):
        raise ReviewedTaskRegistryError("reviewed task registration was tampered")
    if _file_sha256(review_path) != entry.get("review_manifest_artifact_sha256"):
        raise ReviewedTaskRegistryError("reviewed task review manifest was tampered")
    registration = _read_json(registration_path, label="reviewed task registration")
    if set(registration) != REGISTRATION_FIELDS:
        raise ReviewedTaskRegistryError("reviewed task registration fields are invalid")
    review = validate_task_review_manifest(
        _read_json(review_path, label="task review manifest")
    )
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_names:
        raise ReviewedTaskRegistryError("reviewed task artifact map is invalid")
    verified: dict[str, dict[str, str]] = {}
    for relative in sorted(artifact_names):
        descriptor = artifacts[relative]
        expected_path = f"entries/{registration_id}/artifacts/{relative}"
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256"}
            or descriptor.get("path") != expected_path
            or descriptor.get("sha256") != entry["artifact_hashes"].get(relative)
        ):
            raise ReviewedTaskRegistryError(
                f"reviewed task artifact descriptor is invalid: {relative}"
            )
        path = _safe_artifact(root, expected_path, label=f"reviewed {relative}")
        if not path.is_file() or _file_sha256(path) != descriptor["sha256"]:
            raise ReviewedTaskRegistryError(
                f"reviewed task artifact was tampered: {relative}"
            )
        verified[relative] = {"path": str(path), "sha256": descriptor["sha256"]}

    review_hash = _canonical_sha256(review)
    expected_artifact_id = "task_artifact_" + _canonical_sha256(
        {
            "semantic_key_sha256": entry.get("semantic_key_sha256"),
            "artifact_hashes": entry.get("artifact_hashes"),
            "runtime_dependency_hashes": entry.get("runtime_dependency_hashes"),
        }
    )[:20]
    expected_registration_id = "reviewed_task_" + _canonical_sha256(
        {
            "artifact_id": expected_artifact_id,
            "review_manifest_sha256": review_hash,
        }
    )[:20]
    checks = {
        "registration_schema": registration.get("schema_version")
        == REGISTRATION_SCHEMA_VERSION,
        "registration_id": registration.get("registration_id") == registration_id,
        "artifact_id": registration.get("artifact_id") == entry.get("artifact_id"),
        "derived_artifact_id": entry.get("artifact_id") == expected_artifact_id,
        "derived_registration_id": registration_id == expected_registration_id,
        "scope": registration.get("scope") == REGISTRY_SCOPE,
        "status": registration.get("status") == "approved",
        "task_name": registration.get("task_name") == entry.get("task_name"),
        "semantic_hash": registration.get("semantic_key_sha256")
        == entry.get("semantic_key_sha256"),
        "semantic_key": _canonical_sha256(registration.get("semantic_key"))
        == entry.get("semantic_key_sha256"),
        "artifact_hashes": registration.get("artifact_hashes")
        == entry.get("artifact_hashes"),
        "runtime_dependency_hashes": registration.get(
            "runtime_dependency_hashes"
        )
        == entry.get("runtime_dependency_hashes"),
        "review_runtime_dependencies": review.get("runtime_dependency_hashes")
        == entry.get("runtime_dependency_hashes"),
        "review_hash": registration.get("review_manifest_sha256") == review_hash,
        "review_semantic": review.get("semantic_key_sha256")
        == entry.get("semantic_key_sha256"),
        "review_source_run": review.get("source_run_id")
        == registration.get("source_run_id"),
        "reviewer": review.get("reviewer") == registration.get("reviewer"),
        "reviewed_at": review.get("reviewed_at") == registration.get("reviewed_at"),
    }
    for artifact, field in HASH_FIELDS.items():
        checks[f"review_{field}"] = review.get(field) == entry["artifact_hashes"].get(
            artifact
        )
    failed = sorted(key for key, passed in checks.items() if passed is not True)
    if failed:
        raise ReviewedTaskRegistryError(
            f"reviewed task registration hashes are inconsistent: {failed}"
        )
    semantic_key = _validate_semantic_key(registration["semantic_key"])
    if registration["task_name"] != semantic_key["task_name"]:
        raise ReviewedTaskRegistryError(
            "reviewed task registration task differs from semantic key"
        )
    revalidated = _source_artifacts(
        entry_dir / "artifacts",
        semantic_key,
        expected_runtime_dependencies=entry["runtime_dependency_hashes"],
    )
    if revalidated["artifact_hashes"] != entry["artifact_hashes"]:
        raise ReviewedTaskRegistryError(
            "reviewed task artifact set differs after semantic revalidation"
        )
    return {
        "registration": registration,
        "review_manifest": review,
        "registry_dir": root,
        "entry_dir": entry_dir,
        "verified_artifacts": verified,
    }


def _audit_registry_layout(root: Path, index: Mapping[str, Any]) -> None:
    if not root.exists():
        return
    allowed_root = {"index.json"}
    entries_path = root / "entries"
    if entries_path.exists() or entries_path.is_symlink():
        allowed_root.add("entries")
    if {path.name for path in root.iterdir()} != allowed_root:
        raise ReviewedTaskRegistryError("reviewed task registry has unindexed files")
    expected = {entry["registration_id"] for entry in index["entries"]}
    if entries_path.is_symlink():
        raise ReviewedTaskRegistryError("reviewed task entries directory must not be a symlink")
    if expected and not entries_path.is_dir():
        raise ReviewedTaskRegistryError("reviewed task entries directory is missing")
    if entries_path.exists():
        actual = {path.name for path in entries_path.iterdir()}
        if actual != expected:
            raise ReviewedTaskRegistryError(
                "reviewed task entries directory has unindexed content"
            )


def load_reviewed_task_registry(registry_dir: str | Path) -> dict[str, Any]:
    """Load and fully verify every persistent generated-task entry."""

    root = _unresolved_root(registry_dir, label="reviewed task registry")
    if root.exists() and not root.is_dir():
        raise ReviewedTaskRegistryError(
            "reviewed task registry must be a directory"
        )
    index = _read_index(root)
    _audit_registry_layout(root, index)
    for entry in index["entries"]:
        _load_entry(root, entry)
    return deepcopy(index)


def _read_manifest_input(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ReviewedTaskRegistryError("review manifest path must not be a symlink")
    return _read_json(path.resolve(), label="task review manifest")


def install_reviewed_task(
    source_run_dir: str | Path,
    semantic_key: Mapping[str, Any],
    review_manifest: Mapping[str, Any] | str | Path,
    registry_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Install one separately approved immutable generated-task artifact set."""

    source = _source_artifacts(
        source_run_dir, semantic_key, repo_root=repo_root
    )
    review = validate_task_review_manifest(_read_manifest_input(review_manifest))
    expected = {
        "source_run_id": source["source_run_id"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
    }
    for artifact, field in HASH_FIELDS.items():
        expected[field] = source["artifact_hashes"].get(artifact)
    mismatched = sorted(key for key, value in expected.items() if review.get(key) != value)
    if mismatched:
        raise ReviewedTaskRegistryError(
            f"review manifest does not match source artifacts: {mismatched}"
        )

    artifact_id = "task_artifact_" + _canonical_sha256(
        {
            "semantic_key_sha256": source["semantic_key_sha256"],
            "artifact_hashes": source["artifact_hashes"],
            "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        }
    )[:20]
    review_hash = _canonical_sha256(review)
    registration_id = "reviewed_task_" + _canonical_sha256(
        {"artifact_id": artifact_id, "review_manifest_sha256": review_hash}
    )[:20]
    root = _unresolved_root(registry_dir, label="reviewed task registry")
    index = load_reviewed_task_registry(root)
    for entry in index["entries"]:
        if entry["semantic_key_sha256"] == source["semantic_key_sha256"]:
            loaded = _load_entry(root, entry)
            if entry["artifact_id"] == artifact_id:
                if (
                    loaded["registration"].get("review_manifest_sha256")
                    == review_hash
                ):
                    return _match_from_loaded(loaded)
                raise ReviewedTaskRegistryError(
                    "task artifact already has a different review attestation; "
                    "review upgrades require an explicit multi-attestation workflow"
                )
            raise ReviewedTaskRegistryError(
                "semantic key is already approved for a different task artifact"
            )

    registration = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "registration_id": registration_id,
        "artifact_id": artifact_id,
        "scope": REGISTRY_SCOPE,
        "status": "approved",
        "source_run_id": source["source_run_id"],
        "task_name": source["semantic_key"]["task_name"],
        "semantic_key": source["semantic_key"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "artifact_hashes": source["artifact_hashes"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        "review_manifest_sha256": review_hash,
        "reviewer": review["reviewer"],
        "reviewed_at": review["reviewed_at"],
        "installed_at": datetime.now().astimezone().isoformat(),
    }
    entries_root = root / "entries"
    entry_dir = entries_root / registration_id
    temporary_dir = entries_root / (registration_id + ".tmp")
    if entries_root.is_symlink():
        raise ReviewedTaskRegistryError(
            "reviewed task entries directory must not be a symlink"
        )
    if entries_root.exists() and not entries_root.is_dir():
        raise ReviewedTaskRegistryError("reviewed task entries path must be a directory")
    if any(path.exists() or path.is_symlink() for path in (entry_dir, temporary_dir)):
        raise ReviewedTaskRegistryError(
            f"unindexed reviewed task entry already exists: {entry_dir}"
        )
    temporary_dir.mkdir(parents=True)
    (temporary_dir / "registration.json").write_bytes(
        _pretty_json_bytes(registration)
    )
    (temporary_dir / "review_manifest.json").write_bytes(_pretty_json_bytes(review))
    for relative, payload in source["artifacts"].items():
        path = temporary_dir / "artifacts" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    temporary_dir.replace(entry_dir)

    registration_path = entry_dir / "registration.json"
    review_path = entry_dir / "review_manifest.json"
    descriptors = {
        relative: {
            "path": f"entries/{registration_id}/artifacts/{relative}",
            "sha256": digest,
        }
        for relative, digest in source["artifact_hashes"].items()
    }
    entry = {
        "registration_id": registration_id,
        "artifact_id": artifact_id,
        "scope": REGISTRY_SCOPE,
        "status": "approved",
        "task_name": source["semantic_key"]["task_name"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "artifact_hashes": source["artifact_hashes"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        "registration_artifact": f"entries/{registration_id}/registration.json",
        "registration_artifact_sha256": _file_sha256(registration_path),
        "review_manifest_artifact": f"entries/{registration_id}/review_manifest.json",
        "review_manifest_artifact_sha256": _file_sha256(review_path),
        "artifacts": descriptors,
    }
    index["entries"].append(entry)
    index["entries"].sort(key=lambda item: item["registration_id"])
    _write_bytes_atomic(root / "index.json", _pretty_json_bytes(index))
    return _match_from_loaded(_load_entry(root, entry))


def _match_from_loaded(loaded: Mapping[str, Any]) -> dict[str, Any]:
    registration = loaded["registration"]
    return {
        "schema_version": 1,
        "registration_id": registration["registration_id"],
        "artifact_id": registration["artifact_id"],
        "status": "approved",
        "semantic_key": deepcopy(registration["semantic_key"]),
        "semantic_key_sha256": registration["semantic_key_sha256"],
        "runtime_dependency_hashes": deepcopy(
            registration["runtime_dependency_hashes"]
        ),
        "review_authority": deepcopy(registration["reviewer"]),
        "reviewed_at": registration["reviewed_at"],
        "review_attestation_paper_eligible": (
            registration["reviewer"].get("kind") == "human"
        ),
        "registry_dir": str(loaded["registry_dir"]),
        "verified_artifacts": deepcopy(loaded["verified_artifacts"]),
    }


def find_reviewed_task(
    registry_dir: str | Path,
    query: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Find one approved entry for the resolver's exact semantic-key query."""

    if not isinstance(query, Mapping) or set(query) != {
        "schema_version",
        "semantic_key",
        "semantic_key_sha256",
    }:
        raise ReviewedTaskRegistryError("reviewed task query fields are invalid")
    if query.get("schema_version") != 1:
        raise ReviewedTaskRegistryError("reviewed task query schema_version must be 1")
    semantic_key = _validate_semantic_key(query.get("semantic_key"))
    semantic_hash = _canonical_sha256(semantic_key)
    if query.get("semantic_key_sha256") != semantic_hash:
        raise ReviewedTaskRegistryError("reviewed task query semantic hash differs")
    root = _unresolved_root(registry_dir, label="reviewed task registry")
    index = load_reviewed_task_registry(root)
    matches = [
        entry
        for entry in index["entries"]
        if entry["semantic_key_sha256"] == semantic_hash
    ]
    if not matches:
        return None
    if len(matches) != 1:  # Defensive; load already rejects ambiguity.
        raise ReviewedTaskRegistryError("reviewed task query is ambiguous")
    loaded = _load_entry(root, matches[0])
    if loaded["registration"]["semantic_key"] != semantic_key:
        raise ReviewedTaskRegistryError("reviewed task semantic contract differs")
    match = _match_from_loaded(loaded)
    if repo_root is not None:
        # Local import keeps persistent storage independent from runtime
        # materialization while preserving the public convenience check.
        from .reviewed_runtime import validate_reviewed_task_runtime

        validate_reviewed_task_runtime(match, repo_root)
    return match
