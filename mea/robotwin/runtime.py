"""RoboTwin implementation of the simulator-neutral MEA method runtime.

This module is deliberately an adapter, not a second orchestration stack.
Semantic planning remains above :class:`mea.method_runtime.MethodRuntime`;
scene/checker generation remains in
``GenericRoboTwinTaskGenBackend``; ACT mechanics remain in an injected rollout
runner.  The backend only preserves the typed hand-off:

``bind -> materialize -> rollout -> evidence``.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from mea.method_runtime import (
    BackendBindingRequest,
    BackendTaskBinding,
    CandidateRequest,
    EvidenceRequest,
    MaterializedCandidate,
    RolloutObservation,
    RolloutRequest,
    RoundEvidence,
    build_round_evidence,
)
from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.taskgen.generic_backend import (
    GenericRoboTwinTaskAdapter,
    GenericRoboTwinTaskGenBackend,
)
from mea.taskgen.semantic_review import (
    CheckerSemanticReviewError,
    validate_checker_semantic_review_binding,
)
from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    probe_official_robotwin_task_context,
    resolve_robotwin_task_context,
)

from .task_identity import (
    RoboTwinTaskIdentity,
    discover_robotwin_task_identity,
)

RuntimeTaskIdentity = GenericRoboTwinTaskAdapter | RoboTwinTaskIdentity
TaskAdapterFactory = Callable[[str], RuntimeTaskIdentity]
TaskContextProbeRunner = Callable[..., Mapping[str, Any]]


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


class RoboTwinRolloutRunner(Protocol):
    """Translate one materialized candidate into a native policy episode.

    A production ACT adapter should call ``mea.taskgen.act_runtime.run_act``
    and then project the resulting telemetry into this compact return shape.
    ``passed`` from the ACT launcher is transport validity, not policy success;
    therefore the runner must supply an explicit boolean ``success``.
    """

    def __call__(
        self,
        *,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class _RoboTwinNativeCandidate:
    adapter: RuntimeTaskIdentity
    experiment_candidate: Mapping[str, Any]
    taskgen_resolution: Mapping[str, Any]
    rollout_manifest: Mapping[str, Any]


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _json_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return deepcopy(dict(value))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class RoboTwinMethodBackend:
    """Bind generic RoboTwin TaskGen and a policy runner to MethodRuntime."""

    benchmark = "robotwin"

    def __init__(
        self,
        *,
        repo_root: str | Path,
        task_adapter_factory: TaskAdapterFactory | None = None,
        taskgen_backend: GenericRoboTwinTaskGenBackend | None = None,
        rollout_runner: RoboTwinRolloutRunner,
        task_context_probe_runner: TaskContextProbeRunner | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.task_adapter_factory = task_adapter_factory or (
            lambda task_name: discover_robotwin_task_identity(
                self.repo_root,
                task_name,
            )
        )
        self.taskgen_backend = taskgen_backend
        self.rollout_runner = rollout_runner
        self.task_context_probe_runner = (
            task_context_probe_runner
            or probe_official_robotwin_task_context
        )

    def bind_task(
        self,
        request: BackendBindingRequest,
    ) -> BackendTaskBinding:
        task_name = _required_text(
            request.task_reference.get("task_name"),
            "task_reference.task_name",
        )
        adapter = self.task_adapter_factory(task_name)
        if (
            not isinstance(
                adapter,
                (GenericRoboTwinTaskAdapter, RoboTwinTaskIdentity),
            )
            or adapter.task_name != task_name
        ):
            raise TypeError(
                "task_adapter_factory must return a matching "
                "RoboTwin task identity"
            )
        policy = request.task_reference.get("policy", {})
        if not isinstance(policy, Mapping):
            raise TypeError("task_reference.policy must be an object")
        policy_contract = deepcopy(dict(policy))
        policy_name = str(policy_contract.get("name") or "bound_policy").strip()
        binding_id = str(
            request.task_reference.get("binding_id")
            or f"{task_name}/{policy_name}"
        ).strip()
        try:
            task_context = resolve_robotwin_task_context(
                self.repo_root,
                task_name,
            )
        except RoboTwinTaskContextError as exc:
            raise ValueError(
                f"cannot bind RoboTwin TaskContext: {exc}"
            ) from exc
        # A Generic adapter loaded by the production discovery path already
        # carries this exact context.  Keep hand-constructed test/compat
        # adapters usable without treating their injected schema as source
        # authority.
        adapter_context = (
            deepcopy(dict(adapter.task_context))
            if isinstance(adapter, GenericRoboTwinTaskAdapter)
            and isinstance(adapter.task_context, Mapping)
            else task_context.to_dict()
        )
        return BackendTaskBinding(
            benchmark=self.benchmark,
            binding_id=binding_id,
            task_contract={
                "schema_version": 1,
                "task_name": task_name,
                "official_source": adapter.official_source,
                "official_class": adapter.official_class,
                "task_schema": (
                    deepcopy(dict(adapter.task_schema))
                    if adapter.task_schema is not None
                    else None
                ),
                "task_schema_available": adapter.task_schema is not None,
                "task_context": adapter_context,
                "policy": policy_contract,
            },
            native_task=adapter,
            artifacts={
                "official_source": adapter.official_source,
                **request.artifacts,
            },
            metadata={
                "task_name": task_name,
                "policy": policy_contract,
                **request.metadata,
            },
        )

    @staticmethod
    def official_candidate(
        binding: BackendTaskBinding,
        *,
        source_query: str,
        candidate_id: str = "official_control",
    ) -> MaterializedCandidate:
        adapter = binding.native_task
        if not isinstance(
            adapter,
            (GenericRoboTwinTaskAdapter, RoboTwinTaskIdentity),
        ):
            raise TypeError(
                "RoboTwin binding native_task has the wrong runtime type"
            )
        manifest = {
            "schema_version": 1,
            "status": "official",
            "task_name": adapter.task_name,
            "task_module": f"envs.{adapter.task_name}",
            "generation_kind": "official_passthrough",
        }
        native = _RoboTwinNativeCandidate(
            adapter=adapter,
            experiment_candidate={},
            taskgen_resolution={
                "schema_version": 1,
                "status": "bypassed",
                "route": "official_control",
            },
            rollout_manifest=manifest,
        )
        return MaterializedCandidate(
            benchmark=binding.benchmark,
            candidate_id=candidate_id,
            binding_id=binding.binding_id,
            source_query=source_query,
            task_contract={
                **dict(binding.task_contract),
                "task_module": manifest["task_module"],
            },
            native_task=native,
            artifacts={
                **binding.artifacts,
                "task_module": manifest["task_module"],
            },
            validation={"route": "official_control"},
            metadata={"official_control": True},
        )

    def materialize_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
    ) -> MaterializedCandidate:
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
        if candidate["candidate_id"] != request.candidate_id:
            raise ValueError(
                "CandidateRequest.candidate_id differs from "
                "Proposal.candidate_id"
            )
        if candidate["source_query"] != request.source_query:
            raise ValueError(
                "CandidateRequest.source_query differs from "
                "Proposal.source_query"
            )
        if candidate["base_task"] != adapter.task_name:
            raise ValueError(
                "Proposal.base_task differs from the bound task"
            )

        taskgen_required = bool(
            candidate["scene_need"] is not None
            or candidate["checker_need"] is not None
        )
        task_context_value = binding.task_contract.get("task_context")
        execution_schema = binding.task_contract.get("task_schema")
        task_context_artifact: Path | None = None
        tool_observation_required = bool(
            candidate["rule_tool_need"] is not None
            or candidate["vqa_tool_need"] is not None
        )
        if (
            not taskgen_required
            and tool_observation_required
            and not isinstance(execution_schema, Mapping)
        ):
            supplied_probe = request.context.get("runtime_task_context_probe")
            if supplied_probe is not None and not isinstance(
                supplied_probe, Mapping
            ):
                raise ValueError(
                    "runtime_task_context_probe must be an object"
                )
            policy = binding.task_contract.get("policy")
            action_dimension = (
                policy.get("action_dimension", 0)
                if isinstance(policy, Mapping)
                else 0
            )
            if (
                isinstance(action_dimension, bool)
                or not isinstance(action_dimension, int)
                or action_dimension < 1
            ):
                raise ValueError(
                    "schema-less Tool Proposal requires the bound policy "
                    "action_dimension"
                )
            try:
                runtime_probe = (
                    deepcopy(dict(supplied_probe))
                    if isinstance(supplied_probe, Mapping)
                    else deepcopy(
                        dict(
                            self.task_context_probe_runner(
                                repo_root=self.repo_root,
                                task_name=adapter.task_name,
                                seed=request.seed,
                                action_dimension=action_dimension,
                            )
                        )
                    )
                )
                task_context = resolve_robotwin_task_context(
                    self.repo_root,
                    adapter.task_name,
                    runtime_probe=runtime_probe,
                )
            except (RoboTwinTaskContextError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Tool Proposal could not establish pre-rollout "
                    f"TaskContext authority: {exc}"
                ) from exc
            if task_context.task_schema is None:
                raise ValueError(
                    "Tool Proposal pre-rollout TaskContext has no telemetry "
                    "schema"
                )
            task_context_value = task_context.to_dict()
            execution_schema = deepcopy(dict(task_context.task_schema))
            task_context_artifact = (
                request.output_dir.resolve()
                / "validation"
                / "task_context.json"
            )
            _write_json(task_context_artifact, task_context_value)
        if taskgen_required:
            if not isinstance(adapter, GenericRoboTwinTaskAdapter):
                raise ValueError(
                    "generated scene/checker requires a validated TaskSchema"
                )
            if self.taskgen_backend is None:
                raise ValueError(
                    "generated scene/checker requires a TaskGen backend"
                )
            run_id = str(
                request.context.get("taskgen_run_id")
                or request.output_dir.name
            )
            resolution = self.taskgen_backend.materialize(
                candidate,
                adapter,
                run_id=run_id,
                max_regenerations=int(
                    request.context.get("max_regenerations", 1)
                ),
            )
            manifest, artifacts = self._taskgen_rollout_manifest(resolution)
            _validate_taskgen_checker_artifacts(
                candidate=candidate,
                manifest=manifest,
                candidate_manifest_path=Path(
                    artifacts["candidate_manifest"]
                ),
                task_source_path=Path(artifacts["task_source"]),
            )
            validation = {
                "route": resolution["route"],
                "status": resolution["status"],
                "provider_call_count": int(
                    resolution.get("provider_call_count") or 0
                ),
                "taskgen": deepcopy(
                    dict(resolution.get("validation") or {})
                ),
            }
            run_dir = Path(artifacts["task_source"]).parent
        else:
            resolution = {
                "schema_version": 1,
                "status": "bypassed",
                "route": "official_task_tool_only",
                "candidate": candidate,
                "provider_required": False,
                "provider_call_count": 0,
            }
            manifest = {
                "schema_version": 1,
                "status": "official",
                "task_name": adapter.task_name,
                "task_module": f"envs.{adapter.task_name}",
                "generation_kind": "official_passthrough",
                "proposal": candidate,
                **(
                    {
                        "task_context": {
                            "path": str(task_context_artifact),
                            "schema_origin": "runtime_probe",
                        }
                    }
                    if task_context_artifact is not None
                    else {}
                ),
            }
            artifacts = {
                "official_source": adapter.official_source,
                "task_module": manifest["task_module"],
                **(
                    {"task_context": str(task_context_artifact)}
                    if task_context_artifact is not None
                    else {}
                ),
            }
            validation = {
                "route": resolution["route"],
                "status": resolution["status"],
                "provider_call_count": 0,
            }
            run_dir = request.output_dir.resolve()

        run_dir.mkdir(parents=True, exist_ok=True)
        overlay = run_dir / "overlay.yml"
        if not overlay.is_file():
            overlay.write_text("{}\n", encoding="utf-8")
        manifest = {
            **manifest,
            "overlay": str(overlay),
        }
        artifacts.update(
            {
                "run_dir": str(run_dir),
                "overlay": str(overlay),
            }
        )

        native = _RoboTwinNativeCandidate(
            adapter=adapter,
            experiment_candidate=candidate,
            taskgen_resolution=resolution,
            rollout_manifest=manifest,
        )
        return MaterializedCandidate(
            benchmark=self.benchmark,
            candidate_id=request.candidate_id,
            binding_id=binding.binding_id,
            source_query=request.source_query,
            task_contract={
                **dict(binding.task_contract),
                "candidate_id": request.candidate_id,
                "semantic_concern": candidate["semantic_concern"],
                "task_module": manifest["task_module"],
                "task_schema": (
                    deepcopy(dict(execution_schema))
                    if isinstance(execution_schema, Mapping)
                    else None
                ),
                "task_schema_available": isinstance(
                    execution_schema,
                    Mapping,
                ),
                "task_context": (
                    deepcopy(dict(task_context_value))
                    if isinstance(task_context_value, Mapping)
                    else None
                ),
            },
            native_task=native,
            artifacts={**binding.artifacts, **artifacts},
            validation=validation,
            metadata={
                "official_control": False,
                "official_task_reused": not taskgen_required,
                "taskgen_route": resolution["route"],
                "task_context_bound_before_rollout": isinstance(
                    execution_schema,
                    Mapping,
                ),
            },
        )

    def bind_validated_taskgen_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
        taskgen_manifest: Mapping[str, Any],
    ) -> MaterializedCandidate:
        """Bind an accepted TaskGen artifact without invoking generation.

        The production TaskGen runtime has already performed code fixtures,
        render/VLM diagnosis, simulator-state preservation, and an expert
        terminal probe.  This method verifies that acceptance boundary and
        projects the existing artifact into MethodRuntime for policy rollout.
        """

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
                    self.repo_root,
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
            benchmark=self.benchmark,
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

    def _taskgen_rollout_manifest(
        self,
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
                else self.repo_root / relative
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

    def rollout(
        self,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
    ) -> RolloutObservation:
        native = candidate.native_task
        if not isinstance(native, _RoboTwinNativeCandidate):
            raise TypeError(
                "RoboTwin candidate native_task has the wrong runtime type"
            )
        manifest = deepcopy(dict(native.rollout_manifest))
        if not manifest.get("overlay"):
            request.output_dir.mkdir(parents=True, exist_ok=True)
            overlay = request.output_dir / "overlay.yml"
            if not overlay.is_file():
                overlay.write_text("{}\n", encoding="utf-8")
            manifest["overlay"] = str(overlay)
        result = self.rollout_runner(
            candidate=candidate,
            request=request,
            manifest=manifest,
        )
        value = _json_object(result, "RoboTwin rollout result")
        if not isinstance(value.get("success"), bool):
            raise TypeError(
                "RoboTwin rollout result requires explicit boolean success"
            )
        episode = _json_object(
            value.get("episode"),
            "RoboTwin rollout result.episode",
        )
        raw_artifacts = value.get("artifacts") or {}
        if not isinstance(raw_artifacts, Mapping):
            raise TypeError(
                "RoboTwin rollout result.artifacts must be an object"
            )
        artifacts = {
            str(key): str(item)
            for key, item in raw_artifacts.items()
        }
        metadata = _json_object(
            value.get("metadata") or {},
            "RoboTwin rollout result.metadata",
        )
        return RolloutObservation(
            benchmark=self.benchmark,
            round_id=request.round_id,
            candidate_id=candidate.candidate_id,
            seed=request.seed,
            success=value["success"],
            episode=episode,
            native_episode=result,
            artifacts=artifacts,
            metadata={
                "taskgen_route": native.taskgen_resolution["route"],
                **metadata,
            },
        )

    def evidence(
        self,
        rollout: RolloutObservation,
        request: EvidenceRequest,
    ) -> RoundEvidence:
        return build_round_evidence(rollout, request)


__all__ = [
    "RoboTwinMethodBackend",
    "RoboTwinRolloutRunner",
    "TaskAdapterFactory",
    "TaskContextProbeRunner",
]
