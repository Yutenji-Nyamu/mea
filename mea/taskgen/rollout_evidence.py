"""Authoritative TaskGen checker evidence from an executed rollout.

The generic TaskGen CLI and native MethodRuntime must derive checker evidence
from the same telemetry/toolkit path.  This module owns that small bridge so a
generated ``check_success()`` is never approximated by a newly generated metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mea.toolkit import evaluate_telemetry_root


class GenericRolloutEvidenceError(RuntimeError):
    """Raised when generic TaskGen rollout authority is incomplete."""


def _policy_role(policy_name: Any) -> str:
    normalized = str(policy_name or "").casefold()
    if normalized in {"act", "smolvla"}:
        return "policy_under_evaluation"
    if normalized == "expert":
        return "expert_validation"
    return "validation_control"


def evaluate_generic_task_rollout_telemetry(
    repo_root: str | Path,
    run_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the bound official/generated checker over one generic Task rollout."""

    root = Path(repo_root).expanduser().resolve()
    candidate_dir = Path(run_dir).expanduser().resolve()
    if manifest.get("generation_kind") != (
        "generic_provider_scene_checker_codegen"
    ):
        raise GenericRolloutEvidenceError(
            "generic rollout evidence requires a generic TaskGen manifest"
        )
    artifact_summary = manifest.get("task_artifact_summary")
    artifact_summary = (
        artifact_summary if isinstance(artifact_summary, Mapping) else {}
    )
    outcome_metric = str(
        artifact_summary.get("success_outcome_label")
        or "generated_check_success"
    )
    if outcome_metric not in {
        "generated_check_success",
        "official_check_success",
    }:
        raise GenericRolloutEvidenceError(
            f"unsupported generic TaskGen outcome metric: {outcome_metric!r}"
        )
    generated_checker = outcome_metric == "generated_check_success"
    outcome_authority = (
        "llm_generated_python_ast_validated"
        if generated_checker
        else "official_check_success_reused"
    )
    outcome_binding = (
        {
            "metric": outcome_metric,
            "authority": outcome_authority,
            "module_sha256": manifest.get("candidate_module_sha256"),
            "task_module": manifest.get("task_module"),
        }
        if generated_checker
        else None
    )
    summary = evaluate_telemetry_root(
        candidate_dir / "evaluation/telemetry",
        user_request=str(manifest.get("user_request") or ""),
        task_name=str(manifest.get("task_name") or ""),
        outcome_metric=outcome_metric,
        outcome_binding=outcome_binding,
    )
    episodes = []
    for episode in summary["episodes"]:
        metadata = episode["metadata"]
        policy_name = metadata.get("policy_name")
        outcome_results = [
            result
            for result in episode["tool_results"]
            if result.get("tool") == outcome_metric
        ]
        if len(outcome_results) != 1:
            raise GenericRolloutEvidenceError(
                "generic TaskGen rollout must yield exactly one bound "
                f"{outcome_metric} result per episode"
            )
        episodes.append(
            {
                "episode_dir": episode["episode_dir"],
                "policy_name": policy_name,
                "role": _policy_role(policy_name),
                "seed": metadata.get("seed"),
                "success": metadata.get("success"),
                "tool_results": outcome_results,
            }
        )
    artifact = candidate_dir / "evaluation/telemetry/tool_results.json"
    try:
        artifact_path = artifact.relative_to(root).as_posix()
    except ValueError:
        artifact_path = str(artifact)
    return {
        "artifact": artifact_path,
        "episode_count": summary["episode_count"],
        "outcome_metric": outcome_metric,
        "outcome_authority": outcome_authority,
        "outcome_binding": outcome_binding,
        "tool_retrieval": {
            "route": (
                "bound_llm_generated_checker"
                if generated_checker
                else "official_checker_reuse"
            ),
            "generated_new_tool": False,
        },
        "episodes": episodes,
    }


__all__ = [
    "GenericRolloutEvidenceError",
    "evaluate_generic_task_rollout_telemetry",
]
