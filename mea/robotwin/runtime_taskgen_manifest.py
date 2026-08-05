"""Validated TaskGen manifest binding for RoboTwin policy execution."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import (
    BackendTaskBinding,
    CandidateRequest,
    MaterializedCandidate,
)
from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    resolve_robotwin_task_context,
)
from mea.taskgen.generic_backend import GenericRoboTwinTaskAdapter
from mea.taskgen.semantic_review import (
    CheckerSemanticReviewError,
    validate_checker_semantic_review_binding,
)

from .runtime_contracts import _RoboTwinNativeCandidate, _json_object, _required_text
from .task_identity import RoboTwinTaskIdentity


def _validate_taskgen_checker_artifacts(
    *,
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    task_source_path: Path,
) -> dict[str, Any] | None:
    """Bind an approved checker review to current Proposal and source bytes."""

    try:
        candidate_manifest = json.loads(
            candidate_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "TaskGen candidate manifest is invalid"
        ) from exc
    if not isinstance(candidate_manifest, Mapping):
        raise ValueError("TaskGen candidate manifest must be an object")
    if candidate.get("checker_need") is None:
        if candidate_manifest.get("checker_semantic_review") is not None:
            raise ValueError(
                "official checker reuse carries an unexpected semantic review"
            )
        return None
    try:
        source_sha256 = hashlib.sha256(
            task_source_path.read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise ValueError("TaskGen task source is unavailable") from exc
    if source_sha256 != candidate_manifest.get("module_sha256"):
        raise ValueError(
            "TaskGen task source differs from its candidate manifest"
        )
    try:
        review = validate_checker_semantic_review_binding(
            candidate_manifest.get("checker_semantic_review"),
            candidate=candidate,
            checker_sha256=str(
                candidate_manifest.get("success_method_sha256") or ""
            ),
        )
    except CheckerSemanticReviewError as exc:
        raise ValueError(
            f"TaskGen checker semantic review is invalid: {exc}"
        ) from exc
    if manifest.get("checker_semantic_review") not in (None, review):
        raise ValueError(
            "TaskGen manifest checker review differs from candidate manifest"
        )
    acceptance = manifest.get("task_generation_acceptance")
    if isinstance(acceptance, Mapping) and (
        acceptance.get("checker_semantic_review") != review
    ):
        raise ValueError(
            "TaskGen acceptance checker review differs from source binding"
        )
    return review


def bind_validated_taskgen_candidate(
    backend: Any,
    binding: BackendTaskBinding,
    request: CandidateRequest,
    taskgen_manifest: Mapping[str, Any],
) -> MaterializedCandidate:
    """Verify all TaskGen gates before exposing a rollout candidate."""

    adapter = binding.native_task
    if not isinstance(
        adapter,
        (GenericRoboTwinTaskAdapter, RoboTwinTaskIdentity),
    ):
        raise TypeError(
            "RoboTwin binding native_task has the wrong runtime type"
        )
    try:
        candidate = validate_experiment_candidate(
            request.proposal_bundle
        )
    except ExperimentCandidateError as exc:
        raise ValueError(f"invalid Proposal: {exc}") from exc
    if (
        candidate["candidate_id"] != request.candidate_id
        or candidate["source_query"] != request.source_query
        or candidate["base_task"] != adapter.task_name
    ):
        raise ValueError(
            "validated TaskGen artifact differs from CandidateRequest"
        )
    manifest = _json_object(
        taskgen_manifest,
        "validated TaskGen manifest",
    )
    manifest_candidate = validate_experiment_candidate(
        manifest.get("proposal")
    )
    if manifest_candidate != candidate:
        raise ValueError(
            "validated TaskGen manifest Proposal differs from request"
        )
    if (
        manifest.get("status") != "generated"
        or manifest.get("task_name") != adapter.task_name
    ):
        raise ValueError(
            "validated TaskGen manifest has the wrong task or status"
        )

    acceptance = _json_object(
        manifest.get("task_generation_acceptance"),
        "TaskGen task_generation_acceptance",
    )
    if (
        acceptance.get("status") != "accepted"
        or acceptance.get("act_rollouts_started_before_acceptance") != 0
    ):
        raise ValueError(
            "TaskGen artifact was not accepted before policy rollout"
        )
    scene_validation = _json_object(
        manifest.get("scene_validation"),
        "TaskGen scene_validation",
    )
    preflight = _json_object(
        scene_validation.get("generic_preflight"),
        "TaskGen generic_preflight",
    )
    fixtures = preflight.get("checker_fixtures")
    if (
        preflight.get("render_passed") is not True
        or preflight.get("expert_passed") is not True
        or not isinstance(fixtures, list)
        or len(fixtures) < 2
        or any(
            not isinstance(item, Mapping)
            or item.get("passed") is not True
            for item in fixtures
        )
    ):
        raise ValueError(
            "TaskGen artifact lacks passed render/expert/checker gates"
        )
    if (
        candidate["scene_need"] is not None
        and preflight.get("scene_change_passed") is not True
    ):
        raise ValueError(
            "scene-generating TaskGen artifact lacks scene-change evidence"
        )
    vision = _json_object(
        manifest.get("vision_validation"),
        "TaskGen vision_validation",
    )
    if (
        acceptance.get("visual_self_check_required") is not True
        or vision.get("status") != "passed"
        or vision.get("passed") is not True
    ):
        raise ValueError(
            "production TaskGen artifact lacks passed VLM diagnosis"
        )

    generated_checker = candidate["checker_need"] is not None
    artifact_summary = _json_object(
        manifest.get("task_artifact_summary"),
        "TaskGen task_artifact_summary",
    )
    if generated_checker:
        valid_success_semantics = (
            artifact_summary.get("success_origin")
            == "provider_generated_python"
            and artifact_summary.get("success_official_equivalent")
            is False
        )
    else:
        valid_success_semantics = (
            artifact_summary.get("success_origin")
            == "official_method_reuse"
            and artifact_summary.get("success_official_equivalent")
            is True
        )
    if not valid_success_semantics:
        raise ValueError(
            "TaskGen checker semantics differ from the Proposal"
        )

    run_dir = request.output_dir.expanduser().resolve()
    if manifest.get("run_id") != run_dir.name:
        raise ValueError(
            "TaskGen manifest run_id differs from its artifact directory"
        )
    task_module = _required_text(
        manifest.get("task_module"),
        "TaskGen manifest.task_module",
    )
    task_source = run_dir / "task.py"
    candidate_manifest = run_dir / "candidate_manifest.json"
    manifest_path = run_dir / "manifest.json"
    overlay = run_dir / "overlay.yml"
    for artifact in (
        task_source,
        candidate_manifest,
        manifest_path,
        overlay,
    ):
        if not artifact.is_file():
            raise ValueError(
                f"accepted TaskGen artifact is missing: {artifact}"
            )
    checker_semantic_review = _validate_taskgen_checker_artifacts(
        candidate=candidate,
        manifest=manifest,
        candidate_manifest_path=candidate_manifest,
        task_source_path=task_source,
    )
    if generated_checker and (
        acceptance.get("checker_semantic_review")
        != checker_semantic_review
    ):
        raise ValueError(
            "TaskGen checker semantic review was not accepted before "
            "policy rollout"
        )
    execution_schema = binding.task_contract.get("task_schema")
    task_context_artifact: Path | None = None
    task_context_value: dict[str, Any] | None = None
    context_summary = manifest.get("task_context")
    if isinstance(context_summary, Mapping):
        context_relative = _required_text(
            context_summary.get("path"),
            "TaskGen manifest.task_context.path",
        )
        task_context_artifact = (run_dir / context_relative).resolve()
        try:
            task_context_artifact.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(
                "TaskGen TaskContext artifact escapes its run directory"
            ) from exc
        try:
            task_context_value = json.loads(
                task_context_artifact.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "TaskGen TaskContext artifact is invalid"
            ) from exc
        if not isinstance(task_context_value, Mapping):
            raise ValueError(
                "TaskGen TaskContext artifact must be an object"
            )
        runtime_probe = task_context_value.get("runtime_probe")
        try:
            verified_context = resolve_robotwin_task_context(
                backend.repo_root,
                adapter.task_name,
                runtime_probe=(
                    runtime_probe
                    if isinstance(runtime_probe, Mapping)
                    else None
                ),
            )
        except RoboTwinTaskContextError as exc:
            raise ValueError(
                f"TaskGen TaskContext authority is invalid: {exc}"
            ) from exc
        if (
            task_context_value.get("official_source_sha256")
            != verified_context.official_source_sha256
            or task_context_value.get("task_schema")
            != verified_context.task_schema
            or verified_context.task_schema is None
        ):
            raise ValueError(
                "TaskGen TaskContext differs from current authority"
            )
        execution_schema = deepcopy(
            dict(verified_context.task_schema)
        )
        task_context_value = verified_context.to_dict()
    if not isinstance(execution_schema, Mapping):
        raise ValueError(
            "accepted TaskGen artifact lacks a validated TaskContext"
        )

    resolution = {
        "schema_version": 1,
        "status": "generated",
        "route": "validated_taskgen_artifact",
        "run_dir": str(run_dir),
        "candidate_manifest": manifest,
        "provider_call_count": int(
            (manifest.get("provider") or {}).get(
                "provider_call_count"
            )
            or 0
        ),
    }
    native = _RoboTwinNativeCandidate(
        adapter=adapter,
        experiment_candidate=candidate,
        taskgen_resolution=resolution,
        rollout_manifest=manifest,
    )
    return MaterializedCandidate(
        benchmark=backend.benchmark,
        candidate_id=request.candidate_id,
        binding_id=binding.binding_id,
        source_query=request.source_query,
        task_contract={
            **dict(binding.task_contract),
            "candidate_id": request.candidate_id,
            "semantic_concern": candidate["semantic_concern"],
            "task_module": task_module,
            "task_schema": deepcopy(dict(execution_schema)),
            "task_schema_available": True,
            "task_context": task_context_value,
        },
        native_task=native,
        artifacts={
            **binding.artifacts,
            "run_dir": str(run_dir),
            "overlay": str(overlay),
            "task_module": task_module,
            "task_source": str(task_source),
            "candidate_manifest": str(candidate_manifest),
            "manifest": str(manifest_path),
            **(
                {"task_context": str(task_context_artifact)}
                if task_context_artifact is not None
                else {}
            ),
        },
        validation={
            "route": resolution["route"],
            "status": resolution["status"],
            "provider_call_count": resolution[
                "provider_call_count"
            ],
            "taskgen": {
                "acceptance": acceptance,
                "scene_validation": scene_validation,
                "vision_validation": vision,
                "task_context": task_context_value,
            },
        },
        metadata={
            "official_control": False,
            "official_task_reused": False,
            "generated_checker": generated_checker,
            "taskgen_route": resolution["route"],
        },
    )

def taskgen_rollout_manifest(
    backend: Any,
    resolution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    status = resolution.get("status")
    if status == "generated":
        manifest = _json_object(
            resolution.get("candidate_manifest"),
            "TaskGen candidate_manifest",
        )
        run_dir = Path(
            _required_text(
                resolution.get("run_dir"),
                "TaskGen run_dir",
            )
        ).resolve()
        return manifest, {
            "task_module": _required_text(
                manifest.get("task_module"),
                "candidate_manifest.task_module",
            ),
            "task_source": str(run_dir / "task.py"),
            "candidate_manifest": str(
                run_dir / "candidate_manifest.json"
            ),
            "taskgen_resolution": str(
                run_dir / "generic_taskgen_resolution.json"
            ),
        }
    if status == "reused":
        exact = _json_object(
            resolution.get("exact_match"),
            "TaskGen exact_match",
        )
        relative = Path(
            _required_text(
                exact.get("artifact_manifest"),
                "exact_match.artifact_manifest",
            )
        )
        manifest_path = (
            relative
            if relative.is_absolute()
            else backend.repo_root / relative
        ).resolve()
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "reused TaskGen artifact manifest is invalid"
            ) from exc
        return _json_object(
            manifest, "reused TaskGen artifact manifest"
        ), {
            "task_module": _required_text(
                manifest.get("task_module"),
                "reused manifest.task_module",
            ),
            "task_source": str(manifest_path.parent / "task.py"),
            "candidate_manifest": str(
                manifest_path.parent / "candidate_manifest.json"
            ),
            "taskgen_resolution": str(manifest_path),
        }
    raise ValueError(
        f"unsupported generic TaskGen resolution status: {status!r}"
    )
