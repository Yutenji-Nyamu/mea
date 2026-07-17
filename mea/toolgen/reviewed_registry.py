"""Explicitly reviewed, cross-evaluation registry for generated Tools.

Unlike the evaluation-local registry, this registry is never populated by the
generation path.  Admission requires a separate, explicit ``approved`` review
manifest pinned to the exact source, ToolSpec, telemetry schema, and source
registration hashes.  Entries remain generated Tools: they are not added to
the Trusted Tool catalog and are revalidated against current trajectories on
every reuse by the orchestration layer.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .prototype import validate_generated_tool
from .registry import (
    RunLocalRegistryError,
    canonical_sha256,
    file_sha256,
    load_registry,
    telemetry_schema_compatibility,
    tool_contract_sha256,
)


REVIEWED_REGISTRY_SCHEMA_VERSION = 1
REVIEWED_REGISTRATION_SCHEMA_VERSION = 1
REVIEW_MANIFEST_SCHEMA_VERSION = 1
REVIEW_SCOPE = "persistent_generated_tool_reuse"
REVIEWED_SCOPE = "reviewed_persistent"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

REVIEW_MANIFEST_KEYS = {
    "schema_version",
    "decision",
    "review_scope",
    "reviewer",
    "reviewed_at",
    "source_registration_id",
    "source_registration_sha256",
    "code_sha256",
    "tool_spec_sha256",
    "tool_contract_sha256",
    "telemetry_schema_sha256",
    "checks",
    "notes",
}
REVIEW_CHECK_KEYS = {
    "source_read",
    "tool_spec_matches_intent",
    "validation_evidence_reviewed",
    "tests_reviewed",
}
REVIEWER_KINDS = {"human", "development_agent"}


class ReviewedRegistryError(RuntimeError):
    """Raised when reviewed admission or exact persistent lookup is invalid."""


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ReviewedRegistryError(
            f"unfinished registry write already exists: {temporary}"
        )
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, _pretty_json_bytes(value))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewedRegistryError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewedRegistryError(f"{label} must be a JSON object: {path}")
    return value


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ReviewedRegistryError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_artifact(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReviewedRegistryError(f"{label} path must be a non-empty string")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ReviewedRegistryError(f"{label} path escapes registry root")
    current = root
    for part in candidate_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReviewedRegistryError(f"{label} path must not contain symlinks")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReviewedRegistryError(f"{label} path escapes registry root") from exc
    return candidate


def _empty_reviewed_index() -> dict[str, Any]:
    return {
        "schema_version": REVIEWED_REGISTRY_SCHEMA_VERSION,
        "scope": REVIEWED_SCOPE,
        "admission_policy": "explicit_approved_manifest_exact_hashes",
        "entries": [],
    }


def load_reviewed_registry(registry_dir: str | Path) -> dict[str, Any]:
    """Load the persistent index without treating missing storage as approval."""

    root = Path(registry_dir).expanduser().resolve()
    index_path = root / "index.json"
    if not index_path.is_file():
        return _empty_reviewed_index()
    index = _read_json(index_path, label="reviewed registry index")
    if (
        index.get("schema_version") != REVIEWED_REGISTRY_SCHEMA_VERSION
        or index.get("scope") != REVIEWED_SCOPE
        or index.get("admission_policy")
        != "explicit_approved_manifest_exact_hashes"
        or not isinstance(index.get("entries"), list)
    ):
        raise ReviewedRegistryError("unsupported reviewed registry index")
    return index


def validate_review_manifest(value: Any) -> dict[str, Any]:
    """Require an explicit, honest approval pinned to all executable hashes."""

    if not isinstance(value, dict):
        raise ReviewedRegistryError("review manifest must be a JSON object")
    if set(value) != REVIEW_MANIFEST_KEYS:
        missing = sorted(REVIEW_MANIFEST_KEYS - set(value))
        extra = sorted(set(value) - REVIEW_MANIFEST_KEYS)
        raise ReviewedRegistryError(
            f"review manifest fields do not match: missing={missing}, extra={extra}"
        )
    if value.get("schema_version") != REVIEW_MANIFEST_SCHEMA_VERSION:
        raise ReviewedRegistryError("review manifest schema_version must be 1")
    if value.get("decision") != "approved":
        raise ReviewedRegistryError("review manifest decision must be approved")
    if value.get("review_scope") != REVIEW_SCOPE:
        raise ReviewedRegistryError(
            f"review manifest review_scope must be {REVIEW_SCOPE}"
        )

    reviewer = value.get("reviewer")
    if not isinstance(reviewer, dict) or set(reviewer) != {"id", "kind"}:
        raise ReviewedRegistryError("reviewer must contain exactly id and kind")
    if not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip():
        raise ReviewedRegistryError("reviewer.id must be a non-empty identifier")
    if len(reviewer["id"]) > 120:
        raise ReviewedRegistryError("reviewer.id is too long")
    if reviewer.get("kind") not in REVIEWER_KINDS:
        raise ReviewedRegistryError(
            "reviewer.kind must be human or development_agent"
        )

    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ReviewedRegistryError("reviewed_at must be an ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewedRegistryError("reviewed_at is not valid ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ReviewedRegistryError("reviewed_at must include a timezone")

    registration_id = value.get("source_registration_id")
    if not isinstance(registration_id, str) or not registration_id.startswith(
        "runlocal_"
    ):
        raise ReviewedRegistryError(
            "source_registration_id must identify a run-local registration"
        )
    for field in (
        "source_registration_sha256",
        "code_sha256",
        "tool_spec_sha256",
        "tool_contract_sha256",
        "telemetry_schema_sha256",
    ):
        _require_hash(value.get(field), field=field)

    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != REVIEW_CHECK_KEYS:
        raise ReviewedRegistryError(
            "review checks must contain exactly the required manual checks"
        )
    failed = sorted(key for key in REVIEW_CHECK_KEYS if checks.get(key) is not True)
    if failed:
        raise ReviewedRegistryError(f"review checks were not approved: {failed}")
    if not isinstance(value.get("notes"), str):
        raise ReviewedRegistryError("review notes must be a string")
    return json.loads(json.dumps(value, ensure_ascii=False))


def _source_registration(
    source_registry_dir: str | Path,
    registration_id: str,
) -> dict[str, Any]:
    root = Path(source_registry_dir).expanduser().resolve()
    try:
        index = load_registry(root)
    except RunLocalRegistryError as exc:
        raise ReviewedRegistryError(f"invalid source run-local registry: {exc}") from exc
    entry = next(
        (
            item
            for item in index["entries"]
            if item.get("registration_id") == registration_id
        ),
        None,
    )
    if entry is None:
        raise ReviewedRegistryError(
            f"source registration does not exist: {registration_id}"
        )
    if entry.get("scope") != "run_local" or entry.get("status") != "validated":
        raise ReviewedRegistryError("source registration is not validated run-local code")

    registration_path = _safe_artifact(
        root, entry.get("registration_artifact"), label="source registration"
    )
    source_path = _safe_artifact(
        root, entry.get("source_artifact"), label="source generated Tool"
    )
    if not registration_path.is_file() or not source_path.is_file():
        raise ReviewedRegistryError("source registration artifacts are missing")
    registration = _read_json(
        registration_path, label="source registration artifact"
    )
    if (
        registration.get("registration_id") != registration_id
        or registration.get("scope") != "run_local"
        or registration.get("status") != "validated"
    ):
        raise ReviewedRegistryError("source registration identity is inconsistent")

    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    code_hash = file_sha256(source_path)
    if (
        code_hash != entry.get("code_sha256")
        or code_hash != registration.get("code_sha256")
    ):
        raise ReviewedRegistryError("source generated Tool integrity failed")
    static = validate_generated_tool(source)
    if static.get("source_sha256") != code_hash:
        raise ReviewedRegistryError("source static-validation hash mismatch")

    tool_contract = registration.get("tool_contract")
    tool_spec = tool_contract.get("tool_spec") if isinstance(tool_contract, dict) else None
    if not isinstance(tool_spec, dict):
        raise ReviewedRegistryError("source registration has no exact ToolSpec")
    tool_spec_hash = canonical_sha256(tool_spec)
    contract_hash = tool_contract_sha256(tool_spec)
    if (
        contract_hash != entry.get("contract_sha256")
        or contract_hash != registration.get("contract_sha256")
    ):
        raise ReviewedRegistryError("source ToolSpec contract integrity failed")
    compatibility = registration.get("telemetry_schema_compatibility")
    schema_hash = (
        compatibility.get("compatibility_sha256")
        if isinstance(compatibility, dict)
        else None
    )
    _require_hash(schema_hash, field="source telemetry schema hash")
    if schema_hash != entry.get("telemetry_schema_sha256"):
        raise ReviewedRegistryError("source telemetry schema integrity failed")

    validation = registration.get("validation")
    minimum_episodes = tool_spec.get("validation_requirements", {}).get(
        "min_episodes", 1
    )
    if (
        not isinstance(validation, dict)
        or validation.get("determinism_required") is not True
        or validation.get("oracle_agreement_required") is not True
        or int(validation.get("validated_episode_count") or 0)
        < int(minimum_episodes)
    ):
        raise ReviewedRegistryError(
            "source registration does not attest the existing validation gates"
        )
    return {
        "root": root,
        "entry": entry,
        "registration": registration,
        "registration_path": registration_path,
        "source_path": source_path,
        "source_bytes": source_bytes,
        "source": source,
        "code_sha256": code_hash,
        "tool_spec": tool_spec,
        "tool_spec_sha256": tool_spec_hash,
        "tool_contract_sha256": contract_hash,
        "telemetry_schema_sha256": schema_hash,
        "source_registration_sha256": file_sha256(registration_path),
    }


def build_review_manifest_template(
    source_registry_dir: str | Path,
    registration_id: str,
) -> dict[str, Any]:
    """Return a non-executable pending template; approval is never synthesized."""

    source = _source_registration(source_registry_dir, registration_id)
    return {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "decision": "pending",
        "review_scope": REVIEW_SCOPE,
        "reviewer": {"id": "", "kind": "development_agent"},
        "reviewed_at": None,
        "source_registration_id": registration_id,
        "source_registration_sha256": source[
            "source_registration_sha256"
        ],
        "code_sha256": source["code_sha256"],
        "tool_spec_sha256": source["tool_spec_sha256"],
        "tool_contract_sha256": source["tool_contract_sha256"],
        "telemetry_schema_sha256": source["telemetry_schema_sha256"],
        "checks": {key: False for key in sorted(REVIEW_CHECK_KEYS)},
        "notes": "",
    }


def _load_reviewed_entry(
    root: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    if entry.get("scope") != REVIEWED_SCOPE or entry.get("status") != "approved":
        raise ReviewedRegistryError("reviewed registry entry is not approved")
    paths = {
        "registration_path": _safe_artifact(
            root, entry.get("registration_artifact"), label="reviewed registration"
        ),
        "source_path": _safe_artifact(
            root, entry.get("source_artifact"), label="reviewed source"
        ),
        "tool_spec_path": _safe_artifact(
            root, entry.get("tool_spec_artifact"), label="reviewed ToolSpec"
        ),
        "review_manifest_path": _safe_artifact(
            root, entry.get("review_manifest_artifact"), label="review manifest"
        ),
    }
    if any(not path.is_file() for path in paths.values()):
        raise ReviewedRegistryError("reviewed registry entry artifacts are missing")
    artifact_hash_fields = {
        "registration_path": "registration_artifact_sha256",
        "source_path": "code_sha256",
        "tool_spec_path": "tool_spec_artifact_sha256",
        "review_manifest_path": "review_manifest_artifact_sha256",
    }
    for path_name, hash_field in artifact_hash_fields.items():
        actual = file_sha256(paths[path_name])
        if actual != entry.get(hash_field):
            raise ReviewedRegistryError(
                f"reviewed registry artifact integrity failed: {path_name}"
            )

    registration = _read_json(
        paths["registration_path"], label="reviewed registration"
    )
    tool_spec = _read_json(paths["tool_spec_path"], label="reviewed ToolSpec")
    review_manifest = validate_review_manifest(
        _read_json(paths["review_manifest_path"], label="review manifest")
    )
    source = paths["source_path"].read_text(encoding="utf-8")
    static = validate_generated_tool(source)

    checks = {
        "registration_id": registration.get("registration_id")
        == entry.get("registration_id"),
        "scope": registration.get("scope") == REVIEWED_SCOPE,
        "status": registration.get("status") == "approved",
        "code": static.get("source_sha256") == entry.get("code_sha256"),
        "tool_spec": canonical_sha256(tool_spec)
        == entry.get("tool_spec_sha256"),
        "tool_contract": tool_contract_sha256(tool_spec)
        == entry.get("tool_contract_sha256"),
        "review_manifest": canonical_sha256(review_manifest)
        == entry.get("review_manifest_sha256"),
    }
    for field in (
        "registration_id",
        "scope",
        "status",
        "code_sha256",
        "tool_spec_sha256",
        "tool_contract_sha256",
        "telemetry_schema_sha256",
        "review_manifest_sha256",
    ):
        if registration.get(field) != entry.get(field):
            checks[f"registration_{field}"] = False
    if review_manifest.get("code_sha256") != entry.get("code_sha256"):
        checks["review_code"] = False
    if review_manifest.get("tool_spec_sha256") != entry.get(
        "tool_spec_sha256"
    ):
        checks["review_tool_spec"] = False
    if review_manifest.get("tool_contract_sha256") != entry.get(
        "tool_contract_sha256"
    ):
        checks["review_contract"] = False
    if review_manifest.get("telemetry_schema_sha256") != entry.get(
        "telemetry_schema_sha256"
    ):
        checks["review_schema"] = False
    if any(value is not True for value in checks.values()):
        failed = sorted(key for key, value in checks.items() if value is not True)
        raise ReviewedRegistryError(
            f"reviewed registry entry hashes are inconsistent: {failed}"
        )
    return {
        "registration": registration,
        "review_manifest": review_manifest,
        "source": source,
        "tool_spec": tool_spec,
        "registry_dir": root,
        **paths,
    }


def install_reviewed_registration(
    source_registry_dir: str | Path,
    registration_id: str,
    review_manifest: dict[str, Any] | str | Path,
    reviewed_registry_dir: str | Path,
) -> dict[str, Any]:
    """Copy one explicitly approved run-local Tool into persistent storage."""

    source = _source_registration(source_registry_dir, registration_id)
    if isinstance(review_manifest, (str, Path)):
        manifest_path = Path(review_manifest).expanduser().resolve()
        manifest = _read_json(manifest_path, label="review manifest")
    else:
        manifest = review_manifest
    manifest = validate_review_manifest(manifest)
    expected = {
        "source_registration_id": registration_id,
        "source_registration_sha256": source["source_registration_sha256"],
        "code_sha256": source["code_sha256"],
        "tool_spec_sha256": source["tool_spec_sha256"],
        "tool_contract_sha256": source["tool_contract_sha256"],
        "telemetry_schema_sha256": source["telemetry_schema_sha256"],
    }
    mismatched = sorted(
        key for key, value in expected.items() if manifest.get(key) != value
    )
    if mismatched:
        raise ReviewedRegistryError(
            f"review manifest does not match source registration: {mismatched}"
        )

    root = Path(reviewed_registry_dir).expanduser().resolve()
    index = load_reviewed_registry(root)
    review_hash = canonical_sha256(manifest)
    reviewed_id = "reviewed_" + canonical_sha256(
        {
            "code_sha256": source["code_sha256"],
            "tool_contract_sha256": source["tool_contract_sha256"],
            "telemetry_schema_sha256": source["telemetry_schema_sha256"],
            "review_manifest_sha256": review_hash,
        }
    )[:20]
    existing = next(
        (
            item
            for item in index["entries"]
            if item.get("registration_id") == reviewed_id
        ),
        None,
    )
    if existing is not None:
        return _load_reviewed_entry(root, existing)

    registration = {
        "schema_version": REVIEWED_REGISTRATION_SCHEMA_VERSION,
        "registration_id": reviewed_id,
        "scope": REVIEWED_SCOPE,
        "status": "approved",
        "tool_id": source["registration"].get("tool_id"),
        "target_metric": source["registration"].get("target_metric"),
        "task_name": source["tool_spec"].get("task_name"),
        "code_sha256": source["code_sha256"],
        "tool_spec_sha256": source["tool_spec_sha256"],
        "tool_contract_sha256": source["tool_contract_sha256"],
        "telemetry_schema_sha256": source["telemetry_schema_sha256"],
        "review_manifest_sha256": review_hash,
        "source_registration_id": registration_id,
        "source_registration_sha256": source[
            "source_registration_sha256"
        ],
        "reviewer": manifest["reviewer"],
        "reviewed_at": manifest["reviewed_at"],
        "installed_at": datetime.now().astimezone().isoformat(),
    }

    entries_root = root / "entries"
    entry_dir = entries_root / reviewed_id
    temporary_dir = entries_root / (reviewed_id + ".tmp")
    if entries_root.is_symlink():
        raise ReviewedRegistryError(
            "reviewed registry entries directory must not be a symlink"
        )
    if entries_root.exists() and not entries_root.is_dir():
        raise ReviewedRegistryError(
            "reviewed registry entries path must be a directory"
        )
    if (
        entry_dir.is_symlink()
        or temporary_dir.is_symlink()
        or entry_dir.exists()
        or temporary_dir.exists()
    ):
        raise ReviewedRegistryError(
            f"unindexed reviewed entry directory already exists: {entry_dir}"
        )
    temporary_dir.mkdir(parents=True)
    source_path = temporary_dir / "generated_tool.py"
    tool_spec_path = temporary_dir / "tool_spec.json"
    review_path = temporary_dir / "reviewed_manifest.json"
    registration_path = temporary_dir / "registration.json"
    source_path.write_bytes(source["source_bytes"])
    tool_spec_path.write_bytes(_pretty_json_bytes(source["tool_spec"]))
    review_path.write_bytes(_pretty_json_bytes(manifest))
    registration_path.write_bytes(_pretty_json_bytes(registration))
    temporary_dir.replace(entry_dir)

    stored_source = entry_dir / source_path.name
    stored_tool_spec = entry_dir / tool_spec_path.name
    stored_review = entry_dir / review_path.name
    stored_registration = entry_dir / registration_path.name
    entry = {
        "registration_id": reviewed_id,
        "scope": REVIEWED_SCOPE,
        "status": "approved",
        "tool_id": registration["tool_id"],
        "target_metric": registration["target_metric"],
        "task_name": registration["task_name"],
        "code_sha256": registration["code_sha256"],
        "tool_spec_sha256": registration["tool_spec_sha256"],
        "tool_contract_sha256": registration["tool_contract_sha256"],
        "telemetry_schema_sha256": registration["telemetry_schema_sha256"],
        "review_manifest_sha256": review_hash,
        "registration_artifact": str(stored_registration.relative_to(root)),
        "registration_artifact_sha256": file_sha256(stored_registration),
        "source_artifact": str(stored_source.relative_to(root)),
        "tool_spec_artifact": str(stored_tool_spec.relative_to(root)),
        "tool_spec_artifact_sha256": file_sha256(stored_tool_spec),
        "review_manifest_artifact": str(stored_review.relative_to(root)),
        "review_manifest_artifact_sha256": file_sha256(stored_review),
    }
    index["entries"].append(entry)
    index["entries"].sort(key=lambda item: item["registration_id"])
    _write_json_atomic(root / "index.json", index)
    return _load_reviewed_entry(root, entry)


def find_reviewed_registration(
    registry_dir: str | Path,
    *,
    tool_spec: dict[str, Any],
    episode_dirs: Iterable[str | Path],
) -> dict[str, Any] | None:
    """Find one exact approved Tool for the current contract and schemas."""

    root = Path(registry_dir).expanduser().resolve()
    index = load_reviewed_registry(root)
    contract_hash = tool_contract_sha256(tool_spec)
    tool_spec_hash = canonical_sha256(tool_spec)
    schema = telemetry_schema_compatibility(
        episode_dirs, required_signals=tool_spec.get("required_signals", [])
    )
    schema_hash = schema["compatibility_sha256"]
    candidates = sorted(
        (
            item
            for item in index["entries"]
            if item.get("scope") == REVIEWED_SCOPE
            and item.get("status") == "approved"
            and item.get("task_name") == tool_spec.get("task_name")
            and item.get("target_metric") == tool_spec.get("metric")
            and item.get("tool_spec_sha256") == tool_spec_hash
            and item.get("tool_contract_sha256") == contract_hash
            and item.get("telemetry_schema_sha256") == schema_hash
        ),
        key=lambda item: item.get("registration_id", ""),
        reverse=True,
    )
    for entry in candidates:
        match = _load_reviewed_entry(root, entry)
        if match["tool_spec"] == tool_spec:
            return match
    return None


def public_reviewed_registration_summary(
    match: dict[str, Any],
) -> dict[str, Any]:
    registration = match["registration"]
    return {
        "registration_id": registration["registration_id"],
        "scope": registration["scope"],
        "status": registration["status"],
        "tool_id": registration["tool_id"],
        "target_metric": registration["target_metric"],
        "task_name": registration["task_name"],
        "code_sha256": registration["code_sha256"],
        "tool_spec_sha256": registration["tool_spec_sha256"],
        "tool_contract_sha256": registration["tool_contract_sha256"],
        "telemetry_schema_sha256": registration[
            "telemetry_schema_sha256"
        ],
        "review_manifest_sha256": registration["review_manifest_sha256"],
        "reviewer": registration["reviewer"],
        "reviewed_at": registration["reviewed_at"],
    }
