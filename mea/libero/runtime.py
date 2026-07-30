"""LIBERO implementation of the simulator-neutral MEA method runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from mea.method_runtime import (
    CandidateRequest,
    EvidenceRequest,
    MaterializedCandidate,
    RolloutObservation,
    RolloutRequest,
    RoundEvidence,
    BackendBindingRequest,
    BackendTaskBinding,
    build_round_evidence,
)

from .benchmark import (
    LiberoBenchmarkAdapter,
    TaskContract,
    build_official_task_contract,
)
from .policy import LeRobotPolicyAdapter
from .retrieval import BDDLRetrieval, ControlledChangeContract
from .taskgen import LiberoTaskGenBackend


class LiberoMethodBackend:
    """Keep BDDL/env/policy details behind the shared method contract."""

    benchmark = "libero"

    def __init__(
        self,
        *,
        benchmark_adapter: LiberoBenchmarkAdapter | None = None,
        policy_adapter: LeRobotPolicyAdapter | None = None,
        taskgen_backend: LiberoTaskGenBackend | None = None,
        task_contract_factory: Callable[..., TaskContract] = build_official_task_contract,
    ) -> None:
        self.benchmark_adapter = benchmark_adapter
        self.policy_adapter = policy_adapter
        self.taskgen_backend = taskgen_backend
        self.task_contract_factory = task_contract_factory

    def bind_task(
        self,
        request: BackendBindingRequest,
    ) -> BackendTaskBinding:
        suite = str(request.task_reference.get("suite", "")).strip()
        task_id = int(request.task_reference.get("task_id"))
        contract = self.task_contract_factory(suite=suite, task_id=task_id)
        binding_id = f"{contract.suite}/task{contract.official_task_id}"
        return BackendTaskBinding(
            benchmark=self.benchmark,
            binding_id=binding_id,
            task_contract=contract.to_dict(),
            native_task=contract,
            artifacts=request.artifacts,
            metadata={
                "suite": contract.suite,
                "official_task_id": contract.official_task_id,
                **request.metadata,
            },
        )

    @staticmethod
    def official_candidate(
        binding: BackendTaskBinding,
        *,
        source_query: str,
        task_contract_path: str | Path,
    ) -> MaterializedCandidate:
        return MaterializedCandidate(
            benchmark=binding.benchmark,
            candidate_id="official_control",
            binding_id=binding.binding_id,
            source_query=source_query,
            task_contract=binding.task_contract,
            native_task=binding.native_task,
            artifacts={"task_contract": str(Path(task_contract_path))},
            validation={"route": "official_task_contract"},
            metadata={"official_control": True},
        )

    def materialize_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
    ) -> MaterializedCandidate:
        if self.taskgen_backend is None:
            raise RuntimeError("LIBERO TaskGen backend is not configured")
        retrieval = request.context.get("retrieval")
        change_contract = request.context.get("change_contract")
        if not isinstance(retrieval, BDDLRetrieval):
            raise TypeError("candidate context requires a BDDLRetrieval")
        if not isinstance(change_contract, ControlledChangeContract):
            raise TypeError(
                "candidate context requires a ControlledChangeContract"
            )
        contract, result = self.taskgen_backend.generate(
            user_query=request.source_query,
            proposal_bundle=request.proposal_bundle,
            output_dir=request.output_dir,
            seed=request.seed,
            retrieval=retrieval,
            change_contract=change_contract,
        )
        artifacts = {
            str(key): str(value)
            for key, value in dict(result.get("artifacts", {})).items()
        }
        return MaterializedCandidate(
            benchmark=self.benchmark,
            candidate_id=request.candidate_id,
            binding_id=binding.binding_id,
            source_query=request.source_query,
            task_contract=contract.to_dict(),
            native_task=contract,
            artifacts=artifacts,
            validation={
                "planner_taskgen_alignment": bool(
                    result.get("planner_taskgen_alignment")
                ),
                "checks": dict(result.get("checks", {})),
            },
            metadata={
                "official_control": False,
                "taskgen_result": result,
            },
        )

    def rollout(
        self,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
    ) -> RolloutObservation:
        if self.benchmark_adapter is None or self.policy_adapter is None:
            raise RuntimeError("LIBERO benchmark and policy adapters are not configured")
        contract = candidate.native_task
        if not isinstance(contract, TaskContract):
            raise TypeError("LIBERO candidate native_task must be a TaskContract")
        task_contract_path = candidate.artifacts.get("task_contract")
        if not task_contract_path:
            raise ValueError("LIBERO candidate has no task_contract artifact")
        official_control = bool(candidate.metadata.get("official_control"))
        if official_control:
            env_factory = self.benchmark_adapter.make_official_env
            task_suffix = "official"
        else:
            env_factory = lambda: self.benchmark_adapter.make_custom_env(contract)
            task_suffix = "mea_custom"
        record = self.policy_adapter.run(
            env_factory=env_factory,
            seed=request.seed,
            output_dir=request.output_dir,
            task_id=(
                f"{contract.suite}/task{contract.official_task_id}/{task_suffix}"
            ),
            task_contract_path=task_contract_path,
            bddl_path=contract.bddl_path,
            provenance=dict(request.provenance),
            use_stock_official_env=official_control,
        )
        artifacts = {
            "task_contract": str(task_contract_path),
            "actions": record.actions_path,
        }
        if record.video_path:
            artifacts["video"] = record.video_path
        return RolloutObservation(
            benchmark=self.benchmark,
            round_id=request.round_id,
            candidate_id=candidate.candidate_id,
            seed=request.seed,
            success=record.success,
            episode=record.to_dict(),
            native_episode=record,
            artifacts=artifacts,
            metadata={
                "goal_predicate_satisfied": record.goal_predicate_satisfied,
                "official_control": official_control,
            },
        )

    def evidence(
        self,
        rollout: RolloutObservation,
        request: EvidenceRequest,
    ) -> RoundEvidence:
        return build_round_evidence(rollout, request)


__all__ = ["LiberoMethodBackend"]
