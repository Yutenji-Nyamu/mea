"""RoboTwin implementation of the simulator-neutral MEA method runtime.

This module is deliberately an adapter, not a second orchestration stack.
Semantic planning remains above :class:`mea.method_runtime.MethodRuntime`;
scene/checker generation remains in
``GenericRoboTwinTaskGenBackend``; ACT mechanics remain in an injected rollout
runner.  The backend only preserves the typed hand-off:

``bind -> materialize -> rollout -> evidence``.
"""

from __future__ import annotations

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


TaskAdapterFactory = Callable[[str], GenericRoboTwinTaskAdapter]


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
    adapter: GenericRoboTwinTaskAdapter
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


class RoboTwinMethodBackend:
    """Bind generic RoboTwin TaskGen and a policy runner to MethodRuntime."""

    benchmark = "robotwin"

    def __init__(
        self,
        *,
        repo_root: str | Path,
        task_adapter_factory: TaskAdapterFactory,
        taskgen_backend: GenericRoboTwinTaskGenBackend,
        rollout_runner: RoboTwinRolloutRunner,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.task_adapter_factory = task_adapter_factory
        self.taskgen_backend = taskgen_backend
        self.rollout_runner = rollout_runner

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
            not isinstance(adapter, GenericRoboTwinTaskAdapter)
            or adapter.task_name != task_name
        ):
            raise TypeError(
                "task_adapter_factory must return a matching "
                "GenericRoboTwinTaskAdapter"
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
        return BackendTaskBinding(
            benchmark=self.benchmark,
            binding_id=binding_id,
            task_contract={
                "schema_version": 1,
                "task_name": task_name,
                "official_source": adapter.official_source,
                "official_class": adapter.official_class,
                "task_schema": deepcopy(dict(adapter.task_schema)),
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
        if not isinstance(adapter, GenericRoboTwinTaskAdapter):
            raise TypeError(
                "RoboTwin binding native_task must be a "
                "GenericRoboTwinTaskAdapter"
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
        if not isinstance(adapter, GenericRoboTwinTaskAdapter):
            raise TypeError(
                "RoboTwin binding native_task must be a "
                "GenericRoboTwinTaskAdapter"
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
        if taskgen_required:
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
            }
            artifacts = {
                "official_source": adapter.official_source,
                "task_module": manifest["task_module"],
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
            },
            native_task=native,
            artifacts={**binding.artifacts, **artifacts},
            validation=validation,
            metadata={
                "official_control": False,
                "official_task_reused": not taskgen_required,
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
]
