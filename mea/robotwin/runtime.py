"""RoboTwin backend for the simulator-neutral MEA MethodRuntime.

The public backend owns configuration while focused modules implement task
binding, TaskGen materialization, manifest validation, and policy invocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
from mea.robotwin_task_context import probe_official_robotwin_task_context
from mea.taskgen.generic_backend import GenericRoboTwinTaskGenBackend

from .runtime_binding import bind_task, official_candidate
from .runtime_contracts import (
    AcceptedTaskGenMaterializer,
    RoboTwinRolloutRunner,
    TaskAdapterFactory,
    TaskContextProbeRunner,
    _required_text,
)
from .runtime_materialization import materialize_candidate
from .runtime_rollout import rollout_candidate
from .runtime_taskgen_manifest import (
    bind_validated_taskgen_candidate as bind_validated_taskgen_candidate_impl,
    taskgen_rollout_manifest,
)
from .task_identity import discover_robotwin_task_identity


class RoboTwinMethodBackend:
    """Bind generic RoboTwin TaskGen and a policy runner to MethodRuntime."""

    benchmark = "robotwin"

    def __init__(
        self,
        *,
        repo_root: str | Path,
        task_adapter_factory: TaskAdapterFactory | None = None,
        taskgen_backend: GenericRoboTwinTaskGenBackend | None = None,
        accepted_taskgen_materializer: AcceptedTaskGenMaterializer | None = None,
        taskgen_provider: Any = None,
        taskgen_text_model: str = "",
        taskgen_vision_model: str = "",
        taskgen_telemetry_profile: str = "balanced_v1",
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
        self.accepted_taskgen_materializer = accepted_taskgen_materializer
        self.taskgen_provider = taskgen_provider
        self.taskgen_text_model = str(taskgen_text_model or "").strip()
        self.taskgen_vision_model = str(taskgen_vision_model or "").strip()
        self.taskgen_telemetry_profile = _required_text(
            taskgen_telemetry_profile,
            "taskgen_telemetry_profile",
        )
        self.rollout_runner = rollout_runner
        self.task_context_probe_runner = (
            task_context_probe_runner or probe_official_robotwin_task_context
        )

    def bind_task(self, request: BackendBindingRequest) -> BackendTaskBinding:
        return bind_task(self, request)

    def official_candidate(
        self,
        binding: BackendTaskBinding,
        *,
        source_query: str,
        seed: int,
        candidate_id: str = "official_control",
    ) -> MaterializedCandidate:
        return official_candidate(
            self,
            binding,
            source_query=source_query,
            seed=seed,
            candidate_id=candidate_id,
        )

    def materialize_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
    ) -> MaterializedCandidate:
        return materialize_candidate(self, binding, request)

    def bind_validated_taskgen_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
        taskgen_manifest: Mapping[str, Any],
    ) -> MaterializedCandidate:
        """Compatibility entry for an already accepted TaskGen artifact."""

        return bind_validated_taskgen_candidate_impl(
            self,
            binding,
            request,
            taskgen_manifest,
        )

    def _bind_validated_taskgen_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
        taskgen_manifest: Mapping[str, Any],
    ) -> MaterializedCandidate:
        return bind_validated_taskgen_candidate_impl(
            self,
            binding,
            request,
            taskgen_manifest,
        )

    def _taskgen_rollout_manifest(
        self,
        resolution: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        return taskgen_rollout_manifest(self, resolution)

    def rollout(
        self,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
    ) -> RolloutObservation:
        return rollout_candidate(self, candidate, request)

    def evidence(
        self,
        rollout: RolloutObservation,
        request: EvidenceRequest,
    ) -> RoundEvidence:
        return build_round_evidence(rollout, request)


__all__ = [
    "AcceptedTaskGenMaterializer",
    "RoboTwinMethodBackend",
    "RoboTwinRolloutRunner",
    "TaskAdapterFactory",
    "TaskContextProbeRunner",
]
