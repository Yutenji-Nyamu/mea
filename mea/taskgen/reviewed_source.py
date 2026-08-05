"""Semantic validation and review manifests for persistent generated Tasks."""

from __future__ import annotations

import hashlib
import textwrap
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from .artifacts import TaskArtifactBundleError, validate_task_artifact_bundle
from .capabilities import CapabilityError, validate_variant_spec_envelope
from .prototype import TaskGenError, compile_overlay, validate_load_actors
from .scene_checks import SceneCheckSpecError, validate_scene_check_spec
from .success_spec import (
    SuccessSpecError,
    success_spec_validation_report,
    validate_compiled_success_method,
    validate_success_spec,
)
from .reviewed_schema import (
    BASE_ARTIFACTS,
    HASH_FIELDS,
    IDENTIFIER_PATTERN,
    REVIEW_CHECKS,
    REVIEW_MANIFEST_FIELDS,
    REVIEW_MANIFEST_SCHEMA_VERSION,
    REVIEW_SCOPE,
    SUCCESS_SPEC_ARTIFACT,
    ReviewedTaskRegistryError,
    _canonical_bytes,
    _canonical_sha256,
    _method_source,
    _read_json,
    _require_hash,
    _runtime_dependency_hashes,
    _safe_artifact,
    _unresolved_root,
    _validate_runtime_dependency_hashes,
    _validate_semantic_key,
    _validate_static_validation,
)


def _source_artifacts(
    source_run_dir: str | Path,
    semantic_key: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    expected_runtime_dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _unresolved_root(source_run_dir, label="source run directory")
    if not root.is_dir():
        raise ReviewedTaskRegistryError(f"source run directory does not exist: {root}")
    key = _validate_semantic_key(semantic_key)
    paths = {
        relative: _safe_artifact(root, relative, label=f"source {relative}")
        for relative in BASE_ARTIFACTS
    }
    missing = sorted(relative for relative, path in paths.items() if not path.is_file())
    if missing:
        raise ReviewedTaskRegistryError(f"source task artifacts are missing: {missing}")

    variant = _read_json(paths["variant_spec.json"], label="VariantSpec")
    bundle = _read_json(
        paths["generation/task_artifact_bundle.json"],
        label="TaskArtifactBundle",
    )
    scene_check = _read_json(
        paths["generation/scene_check_spec.json"], label="SceneCheckSpec"
    )
    static = _read_json(paths["validation/static.json"], label="static validation")
    try:
        normalized_variant = validate_variant_spec_envelope(variant)
        normalized_bundle = validate_task_artifact_bundle(bundle)
        normalized_scene_check = validate_scene_check_spec(scene_check)
    except (CapabilityError, TaskArtifactBundleError, SceneCheckSpecError) as exc:
        raise ReviewedTaskRegistryError(f"invalid generated task evidence: {exc}") from exc
    if normalized_variant != variant or normalized_bundle != bundle or normalized_scene_check != scene_check:
        raise ReviewedTaskRegistryError("generated task evidence is not canonical")
    try:
        overlay = yaml.safe_load(paths["overlay.yml"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ReviewedTaskRegistryError(f"invalid generated overlay.yml: {exc}") from exc
    if overlay != compile_overlay(variant):
        raise ReviewedTaskRegistryError("overlay.yml does not match VariantSpec")
    controlled_axis = variant["controlled_axis"]
    aspect_id = key["aspect_id"]
    if (
        variant["task_name"] != key["task_name"]
        or variant["capability_id"] != key["capability_id"]
        or not (
            aspect_id == controlled_axis
            or aspect_id.startswith(controlled_axis + ".")
            or key["capability_id"].startswith(controlled_axis + ".")
        )
        or variant["changes"] != key["changes"]
        or variant["generation_mode"] != "force_codegen"
    ):
        raise ReviewedTaskRegistryError(
            "VariantSpec does not match the exact semantic key"
        )
    if bundle.get("task_name") != key["task_name"]:
        raise ReviewedTaskRegistryError("TaskArtifactBundle task identity differs")
    if bundle.get("scene_method", {}).get("origin") != "generated_code":
        raise ReviewedTaskRegistryError("reviewed generated task must bind generated scene code")
    if bundle.get("scene_check_spec", {}).get("artifact") != (
        "generation/scene_check_spec.json"
    ):
        raise ReviewedTaskRegistryError("TaskArtifactBundle scene check path is not fixed")
    if bundle.get("variant_spec_sha256") != _canonical_sha256(variant):
        raise ReviewedTaskRegistryError("TaskArtifactBundle VariantSpec hash differs")
    if bundle.get("scene_check_spec", {}).get("sha256") != _canonical_sha256(scene_check):
        raise ReviewedTaskRegistryError("TaskArtifactBundle SceneCheckSpec hash differs")
    if scene_check.get("task_name") != key["task_name"]:
        raise ReviewedTaskRegistryError("SceneCheckSpec task identity differs")

    task_bytes = paths["task.py"].read_bytes()
    try:
        task_source = task_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ReviewedTaskRegistryError("task.py must be UTF-8") from exc
    task_hash = hashlib.sha256(task_bytes).hexdigest()
    compiled_success = bundle.get("success_method", {}).get("origin") == (
        "compiled_success_spec"
    )
    allowed_methods = (
        {"load_actors", "check_success"}
        if compiled_success
        else {"load_actors"}
    )
    load_source, load_symbol_hash = _method_source(
        task_source,
        key["task_name"],
        "load_actors",
        allowed_methods=allowed_methods,
    )
    try:
        repair_source = paths["generation/load_actors.py.txt"].read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError) as exc:
        raise ReviewedTaskRegistryError(
            f"invalid generated load_actors repair source: {exc}"
        ) from exc
    normalized_load = textwrap.dedent(load_source).strip() + "\n"
    normalized_repair = textwrap.dedent(repair_source).strip() + "\n"
    if normalized_repair != normalized_load:
        raise ReviewedTaskRegistryError(
            "generation/load_actors.py.txt differs from task.py load_actors"
        )
    try:
        recomputed_load = validate_load_actors(load_source, variant)
    except TaskGenError as exc:
        raise ReviewedTaskRegistryError(f"generated load_actors is invalid: {exc}") from exc
    for field in ("valid", "complete_method_generated", "calls_super"):
        if static.get("load_actors_ast", {}).get(field) != recomputed_load.get(field):
            raise ReviewedTaskRegistryError(
                f"recorded load_actors validation differs for {field}"
            )
    scene_binding = bundle["scene_method"]
    if (
        scene_binding.get("source_sha256") != task_hash
        or scene_binding.get("symbol_sha256") != load_symbol_hash
        or scene_binding.get("symbol_declared") is not True
    ):
        raise ReviewedTaskRegistryError("task.py scene binding hash differs")

    success_spec = None
    success_report: dict[str, Any] | None = None
    if compiled_success:
        success_path = _safe_artifact(
            root, SUCCESS_SPEC_ARTIFACT, label="source SuccessSpec"
        )
        if not success_path.is_file():
            raise ReviewedTaskRegistryError("compiled task SuccessSpec is missing")
        paths[SUCCESS_SPEC_ARTIFACT] = success_path
        success_spec = _read_json(success_path, label="SuccessSpec")
        try:
            normalized_success = validate_success_spec(success_spec)
            success_report = success_spec_validation_report(success_spec)
            success_source, success_symbol_hash = _method_source(
                task_source,
                key["task_name"],
                "check_success",
                allowed_methods=allowed_methods,
            )
            recomputed_success = validate_compiled_success_method(
                success_source, success_spec
            )
        except SuccessSpecError as exc:
            raise ReviewedTaskRegistryError(
                f"compiled SuccessSpec binding is invalid: {exc}"
            ) from exc
        if normalized_success != success_spec:
            raise ReviewedTaskRegistryError("SuccessSpec is not canonical")
        success_hash = _canonical_sha256(success_spec)
        semantics = bundle.get("success_semantics", {})
        success_binding = bundle["success_method"]
        if (
            semantics.get("success_spec") != SUCCESS_SPEC_ARTIFACT
            or semantics.get("success_spec_sha256") != success_hash
            or success_binding.get("source_sha256") != task_hash
            or success_binding.get("symbol_sha256") != success_symbol_hash
            or success_binding.get("symbol_declared") is not True
        ):
            raise ReviewedTaskRegistryError("task.py SuccessSpec binding hash differs")
        expected_authority = (
            "compiled_success_spec_experimental_bounded"
            if success_report["experimental_bounded"]
            else "compiled_success_spec_official_equivalent"
        )
        if (
            not success_report["act_eligible"]
            or semantics.get("authority") != expected_authority
            or semantics.get("preserved")
            is not bool(success_report["official_equivalent"])
        ):
            raise ReviewedTaskRegistryError(
                "TaskArtifactBundle mislabels SuccessSpec execution authority"
            )
        if key["preserve_success_semantics"] is not bool(
            success_report["official_equivalent"]
        ):
            raise ReviewedTaskRegistryError(
                "semantic key preservation claim differs from compiled SuccessSpec"
            )
        for field in (
            "valid",
            "act_eligible",
            "complete_method_generated",
            "arbitrary_code_accepted",
        ):
            recorded = static.get("success_spec", {}).get(field)
            legacy_official_act = (
                field == "act_eligible"
                and recorded is None
                and success_report["official_equivalent"]
            )
            if not legacy_official_act and recorded != recomputed_success.get(field):
                raise ReviewedTaskRegistryError(
                    f"recorded SuccessSpec validation differs for {field}"
                )
        if key["success_spec"] is not None and key["success_spec"] != success_spec:
            raise ReviewedTaskRegistryError(
                "generated SuccessSpec differs from semantic key"
            )
    elif key["success_spec"] is not None:
        raise ReviewedTaskRegistryError(
            "semantic key requests SuccessSpec but bundle reuses official success"
        )
    _validate_static_validation(
        static,
        compiled_success=compiled_success,
        official_equivalent_success=bool(
            success_report and success_report["official_equivalent"]
        ),
    )

    artifacts = {relative: path.read_bytes() for relative, path in paths.items()}
    artifact_hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in artifacts.items()
    }
    runtime_dependency_hashes = (
        _validate_runtime_dependency_hashes(expected_runtime_dependencies)
        if expected_runtime_dependencies is not None
        else _runtime_dependency_hashes(
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else root.parents[2]
        )
    )
    return {
        "root": root,
        "source_run_id": root.name,
        "semantic_key": key,
        "semantic_key_sha256": _canonical_sha256(key),
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "runtime_dependency_hashes": runtime_dependency_hashes,
        "variant_spec": variant,
        "success_spec": success_spec,
        "bundle": bundle,
    }


def build_task_review_manifest_template(
    source_run_dir: str | Path,
    semantic_key: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a pending template; this function never synthesizes approval."""

    source = _source_artifacts(
        source_run_dir, semantic_key, repo_root=repo_root
    )
    result = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "decision": "pending",
        "review_scope": REVIEW_SCOPE,
        "reviewer": {"id": "", "kind": "development_agent"},
        "reviewed_at": None,
        "source_run_id": source["source_run_id"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        "checks": {key: False for key in sorted(REVIEW_CHECKS)},
        "notes": "",
    }
    for artifact, field in HASH_FIELDS.items():
        result[field] = source["artifact_hashes"].get(artifact)
    return result


def validate_task_review_manifest(value: Any) -> dict[str, Any]:
    """Require explicit approval pinned to every reusable artifact byte."""

    if not isinstance(value, Mapping) or set(value) != REVIEW_MANIFEST_FIELDS:
        raise ReviewedTaskRegistryError(
            "review manifest fields do not match the strict schema"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != REVIEW_MANIFEST_SCHEMA_VERSION:
        raise ReviewedTaskRegistryError("review manifest schema_version must be 1")
    if result.get("decision") != "approved":
        raise ReviewedTaskRegistryError("review manifest decision must be approved")
    if result.get("review_scope") != REVIEW_SCOPE:
        raise ReviewedTaskRegistryError(
            f"review manifest review_scope must be {REVIEW_SCOPE}"
        )
    reviewer = result.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {"id", "kind"}:
        raise ReviewedTaskRegistryError("reviewer must contain exactly id and kind")
    if (
        not isinstance(reviewer.get("id"), str)
        or not reviewer["id"].strip()
        or len(reviewer["id"]) > 120
    ):
        raise ReviewedTaskRegistryError("reviewer.id must be a non-empty identifier")
    if reviewer.get("kind") not in {"human", "development_agent"}:
        raise ReviewedTaskRegistryError(
            "reviewer.kind must be human or development_agent"
        )
    reviewed_at = result.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ReviewedTaskRegistryError("reviewed_at must be an ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewedTaskRegistryError("reviewed_at is not valid ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ReviewedTaskRegistryError("reviewed_at must include a timezone")
    if not isinstance(result.get("source_run_id"), str) or IDENTIFIER_PATTERN.fullmatch(
        result["source_run_id"]
    ) is None:
        raise ReviewedTaskRegistryError("source_run_id must be a safe identifier")
    _require_hash(result.get("semantic_key_sha256"), field="semantic_key_sha256")
    result["runtime_dependency_hashes"] = _validate_runtime_dependency_hashes(
        result.get("runtime_dependency_hashes")
    )
    for artifact, field in HASH_FIELDS.items():
        _require_hash(
            result.get(field),
            field=field,
            allow_none=artifact == SUCCESS_SPEC_ARTIFACT,
        )
    checks = result.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != REVIEW_CHECKS:
        raise ReviewedTaskRegistryError(
            "review checks must contain exactly the required checks"
        )
    failed = sorted(key for key in REVIEW_CHECKS if checks.get(key) is not True)
    if failed:
        raise ReviewedTaskRegistryError(f"review checks were not approved: {failed}")
    if not isinstance(result.get("notes"), str):
        raise ReviewedTaskRegistryError("review notes must be a string")
    _canonical_bytes(result)
    return result
