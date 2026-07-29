"""Compatibility projection from an executed RoboTwin child into MethodRuntime.

The production Agent still owns the subprocess that launches TaskGen and ACT.
This module validates that already-completed child bundle through the same
typed method boundary used by native RoboTwin and LIBERO backends, without
invoking TaskGen, a provider, a simulator, or a policy for a second time.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import (
    BackendBindingRequest,
    BackendTaskBinding,
    CandidateRequest,
    EvidenceRequest,
    MaterializedCandidate,
    MethodRuntime,
    RolloutObservation,
    RolloutRequest,
    RoundEvidence,
    build_round_evidence,
)
from mea.planner.experiment_candidate import validate_experiment_candidate


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


class _ExecutedRoundProjectionBackend:
    """Project one already executed legacy round through MethodRuntime."""

    benchmark = "robotwin"

    def __init__(
        self,
        *,
        task_name: str,
        experiment_candidate: Mapping[str, Any],
        task_module: str,
        route: str,
        success: bool,
        episode: Mapping[str, Any],
        artifacts: Mapping[str, str],
        validation: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        self.task_name = _required_text(task_name, "task_name")
        self.experiment_candidate = validate_experiment_candidate(
            experiment_candidate
        )
        self.task_module = _required_text(task_module, "task_module")
        self.route = _required_text(route, "route")
        if not isinstance(success, bool):
            raise TypeError("executed RoboTwin round success must be boolean")
        self.success = success
        self.episode = deepcopy(dict(episode))
        self.artifacts = {
            str(key): str(value) for key, value in artifacts.items()
        }
        self.validation = deepcopy(dict(validation))
        self.metadata = deepcopy(dict(metadata))

    def bind_task(
        self,
        request: BackendBindingRequest,
    ) -> BackendTaskBinding:
        requested_task = _required_text(
            request.task_reference.get("task_name"),
            "task_reference.task_name",
        )
        if requested_task != self.task_name:
            raise ValueError(
                "executed round task differs from the requested binding"
            )
        policy = request.task_reference.get("policy") or {}
        if not isinstance(policy, Mapping):
            raise TypeError("task_reference.policy must be an object")
        policy_contract = deepcopy(dict(policy))
        policy_name = str(
            policy_contract.get("name") or "bound_policy"
        ).strip()
        return BackendTaskBinding(
            benchmark=self.benchmark,
            binding_id=f"{self.task_name}/{policy_name}",
            task_contract={
                "schema_version": 1,
                "task_name": self.task_name,
                "task_module": self.task_module,
                "policy": policy_contract,
            },
            native_task={
                "task_name": self.task_name,
                "task_module": self.task_module,
            },
            artifacts=request.artifacts,
            metadata={
                "task_name": self.task_name,
                "policy": policy_contract,
                "projection": "executed_child_bundle",
                **request.metadata,
            },
        )

    def materialize_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
    ) -> MaterializedCandidate:
        candidate = validate_experiment_candidate(
            request.proposal_bundle
        )
        if candidate != self.experiment_candidate:
            raise ValueError(
                "executed child candidate differs from CandidateRequest"
            )
        if candidate["candidate_id"] != request.candidate_id:
            raise ValueError(
                "executed child candidate_id differs from CandidateRequest"
            )
        if candidate["source_query"] != request.source_query:
            raise ValueError(
                "executed child source_query differs from CandidateRequest"
            )
        if candidate["base_task"] != self.task_name:
            raise ValueError(
                "executed child base_task differs from the bound task"
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
                "task_module": self.task_module,
            },
            native_task={
                "experiment_candidate": candidate,
                "task_module": self.task_module,
            },
            artifacts={**binding.artifacts, **self.artifacts},
            validation={
                "route": self.route,
                "projection": "executed_child_bundle",
                **self.validation,
            },
            metadata={
                "official_control": False,
                "projection": "executed_child_bundle",
            },
        )

    def rollout(
        self,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
    ) -> RolloutObservation:
        return RolloutObservation(
            benchmark=self.benchmark,
            round_id=request.round_id,
            candidate_id=candidate.candidate_id,
            seed=request.seed,
            success=self.success,
            episode=self.episode,
            native_episode=self.episode,
            artifacts=self.artifacts,
            metadata={
                "route": self.route,
                "projection": "executed_child_bundle",
                **self.metadata,
            },
        )

    def evidence(
        self,
        rollout: RolloutObservation,
        request: EvidenceRequest,
    ) -> RoundEvidence:
        return build_round_evidence(rollout, request)


def project_executed_round_through_method_runtime(
    *,
    task_name: str,
    round_plan: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    round_summary: Mapping[str, Any],
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate one completed open-world round through the shared runtime."""

    raw_candidate = round_plan.get("experiment_candidate")
    if not isinstance(raw_candidate, Mapping):
        raise ValueError(
            "RoboTwin MethodRuntime projection requires ExperimentCandidate"
        )
    candidate = validate_experiment_candidate(raw_candidate)
    plan_candidate_id = str(
        round_plan.get("candidate_id") or ""
    ).strip()
    if plan_candidate_id != candidate["candidate_id"]:
        raise ValueError(
            "round plan candidate_id differs from ExperimentCandidate"
        )
    plan_task = _required_text(round_plan.get("task_name"), "task_name")
    requested_task = _required_text(task_name, "task_name")
    if plan_task != requested_task or candidate["base_task"] != requested_task:
        raise ValueError(
            "round plan, ExperimentCandidate, and bound task disagree"
        )
    observations = round_summary.get("observations")
    if not isinstance(observations, Mapping):
        raise TypeError("round_summary.observations must be an object")
    raw_success = observations.get("policy_success")
    if isinstance(raw_success, bool):
        success = raw_success
        success_rate = 1.0 if raw_success else 0.0
    elif (
        isinstance(raw_success, (int, float))
        and not isinstance(raw_success, bool)
        and math.isfinite(float(raw_success))
        and 0.0 <= float(raw_success) <= 1.0
    ):
        # The ACT result file stores the round success rate.  The typed
        # projection's boolean means every requested episode passed.
        success_rate = float(raw_success)
        success = success_rate >= 1.0
    else:
        raise TypeError(
            "open-world MethodRuntime projection requires policy_success "
            "in [0, 1]"
        )
    actual_seeds = observations.get("actual_seeds")
    if (
        not isinstance(actual_seeds, list)
        or not actual_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in actual_seeds
        )
    ):
        raise ValueError(
            "open-world ACT round requires at least one integer actual seed"
        )
    round_id = _required_text(round_plan.get("round_id"), "round_id")
    if round_summary.get("round_id") != round_id:
        raise ValueError(
            "round summary identity differs from the executed round plan"
        )
    route = _required_text(round_plan.get("route"), "route")
    task_module = _required_text(
        child_manifest.get("task_module")
        or round_plan.get("task_module")
        or f"envs.{requested_task}",
        "task_module",
    )
    policy_outcome = observations.get("policy_outcome")
    policy_outcome = (
        deepcopy(dict(policy_outcome))
        if isinstance(policy_outcome, Mapping)
        else {}
    )
    compact_episode = {
        "child_run_id": child_manifest.get("run_id"),
        "actual_seeds": list(actual_seeds),
        "policy_success": success,
        "policy_success_rate": success_rate,
        "policy_outcome": policy_outcome,
        "planned_tool": deepcopy(observations.get("planned_tool")),
        "aggregate": deepcopy(observations.get("aggregate")),
        "execution_vqa": deepcopy(observations.get("execution_vqa")),
    }
    artifact_map = {
        str(key): str(value)
        for key, value in dict(artifacts or {}).items()
    }
    backend = _ExecutedRoundProjectionBackend(
        task_name=requested_task,
        experiment_candidate=candidate,
        task_module=task_module,
        route=route,
        success=success,
        episode=compact_episode,
        artifacts=artifact_map,
        validation={
            "child_status": child_manifest.get("status"),
            "pipeline_passed": round_summary.get("pipeline_passed"),
        },
        metadata={
            "child_run_id": child_manifest.get("run_id"),
            "actual_seeds": list(actual_seeds),
        },
    )
    runtime = MethodRuntime(backend)
    binding = runtime.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": requested_task,
                "policy": {"name": "ACT"},
            },
            artifacts=artifact_map,
            metadata={"source": "production_open_world_round"},
        )
    )
    materialized = runtime.materialize_candidate(
        binding,
        CandidateRequest(
            candidate_id=candidate["candidate_id"],
            source_query=candidate["source_query"],
            proposal_bundle=candidate,
            output_dir=Path("."),
            seed=actual_seeds[0],
            context={"projection": "executed_child_bundle"},
        ),
    )
    rollout = runtime.rollout(
        materialized,
        RolloutRequest(
            round_id=round_id,
            seed=actual_seeds[0],
            output_dir=Path("."),
            provenance={
                "child_run_id": child_manifest.get("run_id"),
                "actual_seeds": list(actual_seeds),
            },
        ),
    )
    perturbations = [
        str(need["description"])
        for need_name in (
            "scene_need",
            "checker_need",
            "rule_tool_need",
            "vqa_tool_need",
        )
        for need in [candidate.get(need_name)]
        if isinstance(need, Mapping)
    ]
    limitations = [f"N={len(actual_seeds)} executed seed(s)."]
    if policy_outcome.get("official_equivalent") is False:
        limitations.append(
            "The generated checker is not certified as official-equivalent."
        )
    aggregate = observations.get("aggregate")
    aggregate_status = (
        aggregate.get("status")
        if isinstance(aggregate, Mapping)
        else None
    )
    evidence = runtime.evidence(
        rollout,
        EvidenceRequest(
            sub_aspect=_required_text(
                round_plan.get("sub_aspect"),
                "sub_aspect",
            ),
            hypothesis=_required_text(
                round_plan.get("task_instruction"),
                "task_instruction",
            ),
            perturbation=(
                "; ".join(perturbations)
                if perturbations
                else "reuse official task; evaluate requested Tool/VQA only"
            ),
            summary=(
                f"Executed child {child_manifest.get('run_id')}; "
                f"policy_success_rate={success_rate}; "
                f"aggregate_status={aggregate_status}."
            ),
            limitations=tuple(limitations),
            metadata={
                "pipeline_passed": round_summary.get("pipeline_passed"),
                "actual_seeds": list(actual_seeds),
            },
        ),
    )
    return {
        "schema_version": 1,
        "runtime": "MethodRuntime",
        "backend": "robotwin_executed_child_projection",
        "execution_reused": True,
        "taskgen_reinvoked": False,
        "policy_rollout_reinvoked": False,
        "binding": binding.to_dict(),
        "candidate": materialized.to_dict(),
        "rollout": rollout.to_dict(),
        "evidence": evidence.to_dict(),
    }


__all__ = ["project_executed_round_through_method_runtime"]
