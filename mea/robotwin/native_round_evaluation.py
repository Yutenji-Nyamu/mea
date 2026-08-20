"""Trusted Tool/checker evaluation and evidence projection for a round."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import EvidenceRequest
from mea.taskgen.rollout_evidence import evaluate_generic_task_rollout_telemetry

from .native_round_contracts import (
    NativeAgentRoundError,
    NativeRoundEvaluation,
    NativeRoundPreparation,
)

def _artifact_exists(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).expanduser().is_file()
    except OSError:
        return False


def _trusted_checker_result(
    evaluation: Mapping[str, Any],
    *,
    expected_metric: str,
    expected_seed: int,
) -> dict[str, Any]:
    """Return the ToolResult bound to one requested policy seed."""

    if evaluation.get("status") != "passed":
        raise NativeAgentRoundError(
            "trusted checker evaluation did not pass"
        )
    if evaluation.get("outcome_metric") != expected_metric:
        raise NativeAgentRoundError(
            "trusted checker metric differs from the executed checker"
        )
    policy_episodes = [
        episode
        for episode in evaluation.get("episodes", [])
        if isinstance(episode, Mapping)
        and episode.get("role") == "policy_under_evaluation"
    ]
    matching_episodes = [
        episode
        for episode in policy_episodes
        if episode.get("seed") == expected_seed
    ]
    if len(matching_episodes) != 1:
        raise NativeAgentRoundError(
            "trusted checker evaluation requires exactly one policy episode "
            f"for seed {expected_seed}"
        )
    results = [
        result
        for result in matching_episodes[0].get("tool_results", [])
        if isinstance(result, Mapping)
        and result.get("tool") == expected_metric
    ]
    if len(results) != 1:
        raise NativeAgentRoundError(
            "trusted checker evaluation requires exactly one bound ToolResult"
        )
    result = deepcopy(dict(results[0]))
    if not isinstance(result.get("value"), bool):
        raise NativeAgentRoundError(
            "trusted checker ToolResult requires a boolean value"
        )
    if result.get("passed") is not result["value"]:
        raise NativeAgentRoundError(
            "trusted checker ToolResult passed/value semantics disagree"
        )
    return result


def _project_trusted_checker_outcome(
    rollout: Any,
    evaluation: Mapping[str, Any],
    *,
    expected_metric: str,
    policy_backend: str,
) -> tuple[Any, dict[str, Any]]:
    """Project MethodRuntime evidence onto the same result Aggregate consumes."""

    result = _trusted_checker_result(
        evaluation,
        expected_metric=expected_metric,
        expected_seed=rollout.seed,
    )
    episode = rollout.episode
    if policy_backend in {"smolvla", "hyvla"}:
        if episode.get("active_checker_metric") != expected_metric:
            raise NativeAgentRoundError(
                f"{policy_backend} active checker differs from trusted ToolResult"
            )
        if not isinstance(
            episode.get("episode_latched_success"),
            bool,
        ):
            raise NativeAgentRoundError(
                f"{policy_backend} result lacks an explicit episode latch"
            )
        if expected_metric == "generated_check_success":
            generated = episode.get("generated_checker_success")
            official_core = episode.get(
                "official_core_predicate_satisfied"
            )
            details = result.get("details")
            if (
                not isinstance(generated, bool)
                or not isinstance(official_core, bool)
                or not isinstance(details, Mapping)
                or details.get("generated_checker_success") is not generated
                or details.get("official_core_predicate_satisfied")
                is not official_core
            ):
                raise NativeAgentRoundError(
                    f"{policy_backend} generated/official checker channels disagree "
                    "with the trusted ToolResult"
                )
        elif episode.get("official_check_success") is not result["value"]:
            raise NativeAgentRoundError(
                f"{policy_backend} official checker differs from trusted ToolResult"
            )
    projected = replace(
        rollout,
        success=result["value"],
        metadata={
            **dict(rollout.metadata),
            "trusted_checker": {
                "metric": expected_metric,
                "authority": evaluation.get("outcome_authority"),
                "value": result["value"],
            },
        },
    )
    return projected, result



def evaluate_robotwin_method_round(
    prepared: NativeRoundPreparation,
    rollouts: tuple[Any, ...],
    *,
    round_plan: Mapping[str, Any],
    policy_backend: str,
    policy_name: str,
) -> NativeRoundEvaluation:
    if not rollouts:
        raise NativeAgentRoundError(
            "native method round requires at least one policy rollout"
        )
    actual_seeds = tuple(rollout.seed for rollout in rollouts)
    if actual_seeds != prepared.seeds:
        raise NativeAgentRoundError(
            "native policy rollouts differ from the requested seed sequence"
        )
    root = prepared.root
    child_dir = prepared.child_dir
    proposal = prepared.proposal
    candidate = prepared.candidate
    taskgen_manifest = prepared.taskgen_manifest
    runtime = prepared.runtime
    query = prepared.query
    generated_checker = bool(
        proposal is not None and proposal["checker_need"] is not None
    )
    executed_schema_available = bool(
        candidate.task_contract.get("task_schema_available")
    )
    executed_task_context = candidate.task_contract.get("task_context")
    executed_schema_origin = (
        executed_task_context.get("schema_origin")
        if isinstance(executed_task_context, Mapping)
        else None
    )
    execution_scope = (
        "generated_check_success"
        if generated_checker
        else "official_check_success"
    )
    limitations: tuple[str, ...] = ()
    if len(rollouts) == 1:
        limitations += (
            "M=1 is a mechanism/debug observation, not a stable policy "
            "judgment.",
        )
    if generated_checker:
        limitations += (
            "The generated checker is experimental, not certified "
            "as official-equivalent.",
        )
    elif not executed_schema_available:
        limitations += (
            "No reviewed TaskSchema; the Task context is limited to official "
            "source identity and executed telemetry.",
        )
    elif executed_schema_origin == "runtime_probe":
        limitations += (
            "The TaskContext was derived from a fresh official reset rather "
            "than a reviewed task-specific schema; semantic roles and "
            "thresholds remain unavailable unless directly observed.",
        )
    semantic_ready = all(
        bool(rollout.metadata.get("semantic_telemetry_ready"))
        for rollout in rollouts
    )
    trusted_tool_evaluation = (
        {
            "schema_version": 1,
            "status": "passed",
            **evaluate_generic_task_rollout_telemetry(
                root,
                child_dir,
                taskgen_manifest,
            ),
        }
        if taskgen_manifest is not None and semantic_ready
        else {
            "schema_version": 1,
            "status": "pending" if semantic_ready else "skipped",
            "outcome_metric": execution_scope,
            "outcome_authority": (
                "llm_generated_python_ast_validated"
                if generated_checker
                else "official_check_success"
            ),
            "episode_count": 0,
            "episodes": [],
        }
    )
    authoritative_rollouts = rollouts
    checker_results: tuple[Mapping[str, Any], ...] = ()
    if taskgen_manifest is not None and semantic_ready:
        policy_episode_seeds = [
            episode.get("seed")
            for episode in trusted_tool_evaluation.get("episodes", [])
            if isinstance(episode, Mapping)
            and episode.get("role") == "policy_under_evaluation"
        ]
        if (
            len(policy_episode_seeds) != len(prepared.seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in policy_episode_seeds
            )
            or len(set(policy_episode_seeds)) != len(policy_episode_seeds)
            or set(policy_episode_seeds) != set(prepared.seeds)
        ):
            raise NativeAgentRoundError(
                "trusted checker policy episodes do not match the requested "
                "seed set"
            )
        projected = [
            _project_trusted_checker_outcome(
                rollout,
                trusted_tool_evaluation,
                expected_metric=execution_scope,
                policy_backend=policy_backend,
            )
            for rollout in rollouts
        ]
        authoritative_rollouts = tuple(item[0] for item in projected)
        checker_results = tuple(item[1] for item in projected)
    success_count = sum(
        1 for rollout in authoritative_rollouts if rollout.success
    )
    trial_count = len(authoritative_rollouts)
    success_rate = success_count / trial_count
    aggregate_outcome = (
        "success"
        if success_count == trial_count
        else "failure"
        if success_count == 0
        else "ambiguous"
    )
    aggregate_rollout = replace(
        authoritative_rollouts[0],
        success=success_count == trial_count,
        metadata={
            **dict(authoritative_rollouts[0].metadata),
            "trial_aggregate": {
                "seeds": list(prepared.seeds),
                "trial_count": trial_count,
                "success_count": success_count,
                "success_rate": success_rate,
            },
        },
    )
    evidence = runtime.evidence(
        aggregate_rollout,
        EvidenceRequest(
            sub_aspect=str(round_plan["sub_aspect"]),
            hypothesis=query,
            perturbation=(
                str(proposal["semantic_concern"])
                if proposal is not None
                else "unchanged official-scene control"
            ),
            summary=(
                f"{policy_name} completed {trial_count} RoboTwin policy "
                f"trials on seeds {list(prepared.seeds)}; "
                f"{execution_scope} success_rate={success_rate:.3f} "
                f"({success_count}/{trial_count})."
            ),
            limitations=limitations,
            metadata={
                "policy_backend": policy_backend,
                "semantic_telemetry_ready": semantic_ready,
                "seeds": list(prepared.seeds),
                "trial_count": trial_count,
                "success_count": success_count,
                "success_rate": success_rate,
                "trusted_checker_results": list(checker_results),
            },
        ),
    )
    evidence = replace(evidence, outcome=aggregate_outcome)
    return NativeRoundEvaluation(
        authoritative_rollouts=authoritative_rollouts,
        trusted_tool_evaluation=trusted_tool_evaluation,
        checker_results=checker_results,
        evidence=evidence,
        execution_scope=execution_scope,
        semantic_ready=semantic_ready,
        success_rate=success_rate,
    )
