"""Evaluation-local registry for validated generated trajectory Tools.

This module deliberately stops at ``run_local`` scope.  It exposes promotion
metadata for later candidate/trusted workflows, but it never promotes code.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REGISTRY_SCHEMA_VERSION = 1
REGISTRATION_SCHEMA_VERSION = 3


class RunLocalRegistryError(RuntimeError):
    """Raised when an evaluation-local registration is malformed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 for one JSON-compatible value."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_registry_dir(output_dir: str | Path) -> Path | None:
    """Infer ``<evaluation>/tool_registry`` from a standard round output."""

    output = Path(output_dir).expanduser().resolve()
    for parent in output.parents:
        if parent.name == "execution":
            return parent.parent / "tool_registry"
    return None


def tool_contract(tool_spec: dict[str, Any]) -> dict[str, Any]:
    """Build the exact routeful ToolSpec contract used for local matching."""

    return {
        "tool_spec": tool_spec,
        "required_signals": list(tool_spec.get("required_signals", [])),
    }


def tool_contract_sha256(tool_spec: dict[str, Any]) -> str:
    return canonical_sha256(tool_contract(tool_spec))


def telemetry_schema_compatibility(
    episode_dirs: Iterable[str | Path],
    *,
    required_signals: Iterable[str],
) -> dict[str, Any]:
    """Fingerprint exact schema snapshots for the current trajectory set."""

    schemas: list[dict[str, Any]] = []
    for episode_dir in sorted(
        (Path(item).expanduser().resolve() for item in episode_dirs),
        key=str,
    ):
        schema_path = episode_dir / "schema.json"
        if not schema_path.is_file():
            raise RunLocalRegistryError(
                f"telemetry schema is missing: {schema_path}"
            )
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunLocalRegistryError(
                f"invalid telemetry schema: {schema_path}: {exc}"
            ) from exc
        schemas.append(
            {
                "schema_sha256": canonical_sha256(schema),
                "schema_version": schema.get("schema_version"),
                "task_name": schema.get("task_name"),
            }
        )
    if not schemas:
        raise RunLocalRegistryError("telemetry schema set must not be empty")
    unique_schemas = sorted(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in schemas
        }
    )
    compatibility = {
        "matching_policy": "exact_schema_snapshot_set",
        "required_signals": list(required_signals),
        "schemas": [json.loads(item) for item in unique_schemas],
    }
    compatibility["compatibility_sha256"] = canonical_sha256(compatibility)
    return compatibility


def _empty_index() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "scope": "evaluation_local",
        "promotion_policy": "explicit_only_not_implemented",
        "entries": [],
    }


def load_registry(registry_dir: str | Path) -> dict[str, Any]:
    root = Path(registry_dir).expanduser().resolve()
    index_path = root / "index.json"
    if not index_path.is_file():
        return _empty_index()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunLocalRegistryError(f"invalid registry index: {exc}") from exc
    if (
        index.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or index.get("scope") != "evaluation_local"
        or not isinstance(index.get("entries"), list)
    ):
        raise RunLocalRegistryError("unsupported evaluation-local registry index")
    return index


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _entry_paths(
    registry_root: Path,
    entry: dict[str, Any],
) -> tuple[Path, Path]:
    registration_path = registry_root / entry["registration_artifact"]
    source_path = registry_root / entry["source_artifact"]
    return registration_path, source_path


def find_run_local_registration(
    registry_dir: str | Path,
    *,
    tool_spec: dict[str, Any],
    episode_dirs: Iterable[str | Path],
) -> dict[str, Any] | None:
    """Return a validated exact match, rejecting stale or modified code."""

    root = Path(registry_dir).expanduser().resolve()
    index = load_registry(root)
    contract_hash = tool_contract_sha256(tool_spec)
    schema_compatibility = telemetry_schema_compatibility(
        episode_dirs,
        required_signals=tool_spec.get("required_signals", []),
    )
    schema_hash = schema_compatibility["compatibility_sha256"]
    candidates = sorted(
        (
            item
            for item in index["entries"]
            if item.get("scope") == "run_local"
            and item.get("status") == "validated"
            and item.get("target_metric") == tool_spec.get("metric")
            and item.get("contract_sha256") == contract_hash
            and item.get("telemetry_schema_sha256") == schema_hash
        ),
        key=lambda item: (int(item.get("version", 0)), item.get("registration_id", "")),
        reverse=True,
    )
    for entry in candidates:
        try:
            registration_path, source_path = _entry_paths(root, entry)
        except (KeyError, TypeError):
            continue
        if not registration_path.is_file() or not source_path.is_file():
            continue
        try:
            registration = json.loads(
                registration_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        code_hash = file_sha256(source_path)
        if (
            registration.get("registration_id") != entry.get("registration_id")
            or registration.get("contract_sha256") != contract_hash
            or registration.get("telemetry_schema_compatibility", {}).get(
                "compatibility_sha256"
            )
            != schema_hash
            or registration.get("code_sha256") != code_hash
            or entry.get("code_sha256") != code_hash
        ):
            continue
        return {
            "registration": registration,
            "registration_path": registration_path,
            "source_path": source_path,
            "registry_dir": root,
        }
    return None


def register_run_local_tool(
    registry_dir: str | Path,
    *,
    tool_spec: dict[str, Any],
    episode_dirs: Iterable[str | Path],
    source_path: str | Path,
    generation_registration: dict[str, Any],
    generation_manifest: dict[str, Any],
    validation_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy a validated generated Tool into one evaluation-local registry."""

    root = Path(registry_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    episode_paths = [Path(item).expanduser().resolve() for item in episode_dirs]
    if not source.is_file():
        raise RunLocalRegistryError(f"generated Tool source is missing: {source}")
    contract = tool_contract(tool_spec)
    contract_hash = canonical_sha256(contract)
    schema_compatibility = telemetry_schema_compatibility(
        episode_paths,
        required_signals=tool_spec.get("required_signals", []),
    )
    code_hash = file_sha256(source)
    registration_id = "runlocal_" + canonical_sha256(
        {
            "contract_sha256": contract_hash,
            "telemetry_schema_sha256": schema_compatibility[
                "compatibility_sha256"
            ],
            "code_sha256": code_hash,
        }
    )[:20]
    successful_attempt = generation_manifest.get("successful_attempt")
    prompt_path = None
    prompt_hash = None
    if successful_attempt is not None:
        candidate = source.parent / "attempts" / f"attempt_{successful_attempt}" / "prompt.md"
        if candidate.is_file():
            prompt_path = candidate
            prompt_hash = file_sha256(candidate)
    registration = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "registration_id": registration_id,
        "scope": "run_local",
        "status": "validated",
        "tool_id": generation_registration.get("tool"),
        "version": 1,
        "target_metric": tool_spec.get("metric"),
        "code_sha256": code_hash,
        "source_sha256": code_hash,
        "contract_sha256": contract_hash,
        "tool_contract": contract,
        "required_signals": list(tool_spec.get("required_signals", [])),
        "generation": {
            "model": generation_manifest.get("model_requested"),
            "prompt_sha256": prompt_hash,
            "prompt_artifact": str(prompt_path) if prompt_path else None,
            "generator_source_sha256": generation_manifest.get(
                "generator_source_sha256"
            ),
            "generator_contract_sha256": generation_manifest.get(
                "contract_sha256"
            ),
            "source_examples": list(
                generation_manifest.get("example_validation", [])
            ),
        },
        "telemetry_schema_compatibility": schema_compatibility,
        "validation": {
            "episodes": validation_episodes,
            "validated_episode_count": generation_registration.get(
                "validated_episode_count"
            ),
            "validated_property_scenario_count": generation_registration.get(
                "validated_property_scenario_count"
            ),
            "oracle_kind": generation_registration.get("oracle_kind"),
            "determinism_required": True,
            "oracle_agreement_required": True,
        },
        "promotion": {
            "current_scope": "run_local",
            "candidate": {"status": "not_requested"},
            "trusted": {"status": "not_requested"},
        },
        "created_at": datetime.now().astimezone().isoformat(),
    }
    index = load_registry(root)
    existing = next(
        (
            item
            for item in index["entries"]
            if item.get("registration_id") == registration_id
        ),
        None,
    )
    if existing is not None:
        match = find_run_local_registration(
            root,
            tool_spec=tool_spec,
            episode_dirs=episode_paths,
        )
        if match is not None:
            return match
        # Keep the stale entry for audit and install a new validated version.
        # Matching always re-checks code integrity and prefers the latest valid
        # version, so modified code can never be executed implicitly.
        version = max(
            (
                int(item.get("version", 0))
                for item in index["entries"]
                if item.get("target_metric") == tool_spec.get("metric")
            ),
            default=1,
        ) + 1
        registration_id = f"{registration_id}_v{version}"
        registration["registration_id"] = registration_id
        registration["version"] = version

    entry_dir = root / "entries" / registration_id
    stored_source = entry_dir / "generated_tool.py"
    stored_registration = entry_dir / "registration.json"

    entry_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, stored_source)
    _write_json(stored_registration, registration)
    entry = {
        "registration_id": registration_id,
        "scope": "run_local",
        "status": "validated",
        "tool_id": registration["tool_id"],
        "version": registration["version"],
        "target_metric": registration["target_metric"],
        "contract_sha256": contract_hash,
        "telemetry_schema_sha256": schema_compatibility[
            "compatibility_sha256"
        ],
        "code_sha256": code_hash,
        "registration_artifact": str(stored_registration.relative_to(root)),
        "source_artifact": str(stored_source.relative_to(root)),
        "promotion": registration["promotion"],
    }
    index["entries"].append(entry)
    index["entries"].sort(key=lambda item: item["registration_id"])
    _write_json(root / "index.json", index)
    return {
        "registration": registration,
        "registration_path": stored_registration,
        "source_path": stored_source,
        "registry_dir": root,
    }


def public_registration_summary(match: dict[str, Any]) -> dict[str, Any]:
    registration = match["registration"]
    return {
        "registration_id": registration["registration_id"],
        "scope": registration["scope"],
        "status": registration["status"],
        "tool_id": registration["tool_id"],
        "version": registration["version"],
        "target_metric": registration["target_metric"],
        "contract_sha256": registration["contract_sha256"],
        "telemetry_schema_sha256": registration[
            "telemetry_schema_compatibility"
        ]["compatibility_sha256"],
        "code_sha256": registration["code_sha256"],
        "promotion": registration["promotion"],
    }


def request_candidate_promotion(
    registry_dir: str | Path,
    registration_id: str,
    validation_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate an explicit run-local -> candidate promotion request.

    Candidate eligibility is deliberately conservative and deterministic.  It
    never changes the executable scope and never promotes directly to Trusted;
    Trusted still requires separate code review and repository tests.
    """

    root = Path(registry_dir).expanduser().resolve()
    index = load_registry(root)
    entry = next(
        (
            item
            for item in index["entries"]
            if item.get("registration_id") == registration_id
        ),
        None,
    )
    if entry is None:
        raise RunLocalRegistryError(
            f"registration does not exist: {registration_id}"
        )
    if not isinstance(validation_evidence, dict):
        raise RunLocalRegistryError("validation_evidence must be an object")

    registration_path, source_path = _entry_paths(root, entry)
    try:
        registration = json.loads(
            registration_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RunLocalRegistryError(
            f"invalid registration artifact: {registration_path}: {exc}"
        ) from exc

    positive_examples = validation_evidence.get("positive_examples")
    negative_examples = validation_evidence.get("negative_examples")
    real_rollouts = validation_evidence.get("real_rollouts")
    reasons: list[str] = []
    if not isinstance(positive_examples, list) or len(positive_examples) < 2:
        reasons.append("at_least_two_positive_examples_required")
    if not isinstance(negative_examples, list) or len(negative_examples) < 2:
        reasons.append("at_least_two_negative_examples_required")
    if validation_evidence.get("determinism_passed") is not True:
        reasons.append("determinism_validation_required")
    if validation_evidence.get("oracle_agreement_passed") is not True:
        reasons.append("oracle_agreement_required")
    if not isinstance(real_rollouts, list) or not real_rollouts:
        reasons.append("at_least_one_real_rollout_required")

    actual_code_hash = file_sha256(source_path) if source_path.is_file() else None
    if (
        actual_code_hash is None
        or actual_code_hash != entry.get("code_sha256")
        or actual_code_hash != registration.get("code_sha256")
    ):
        reasons.append("registered_code_integrity_failed")

    now = datetime.now().astimezone()
    evidence_summary = {
        "positive_example_count": (
            len(positive_examples) if isinstance(positive_examples, list) else 0
        ),
        "negative_example_count": (
            len(negative_examples) if isinstance(negative_examples, list) else 0
        ),
        "real_rollout_count": (
            len(real_rollouts) if isinstance(real_rollouts, list) else 0
        ),
        "determinism_passed": validation_evidence.get(
            "determinism_passed"
        )
        is True,
        "oracle_agreement_passed": validation_evidence.get(
            "oracle_agreement_passed"
        )
        is True,
    }
    decision = {
        "schema_version": 1,
        "registration_id": registration_id,
        "requested_scope": "candidate",
        "status": "rejected" if reasons else "eligible",
        "reasons": reasons,
        "evidence_summary": evidence_summary,
        "evidence_sha256": canonical_sha256(validation_evidence),
        "code_sha256": actual_code_hash,
        "evaluated_at": now.isoformat(),
        "scope_changed": False,
        "trusted_promotion": "requires_code_review_and_tests",
    }
    decision_path = (
        root
        / "promotion_decisions"
        / (
            registration_id
            + "_"
            + now.strftime("%Y%m%dT%H%M%S%f%z")
            + ".json"
        )
    )
    decision["artifact"] = str(decision_path.relative_to(root))
    _write_json(decision_path, decision)

    if reasons:
        return decision

    candidate = {
        "status": "eligible",
        "decision_artifact": decision["artifact"],
        "evidence_sha256": decision["evidence_sha256"],
        "evaluated_at": decision["evaluated_at"],
    }
    registration["promotion"]["candidate"] = candidate
    registration["promotion"]["trusted"] = {
        "status": "requires_code_review_and_tests"
    }
    entry["promotion"] = registration["promotion"]
    _write_json(registration_path, registration)
    _write_json(root / "index.json", index)
    return decision
