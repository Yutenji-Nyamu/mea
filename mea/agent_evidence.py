"""Compact evidence projections for the production Agent runtime.

This module owns artifact-to-evidence shaping only.  It does not execute a
provider, simulator, policy, Tool, VQA model, or planner, and it deliberately
does not import the Agent CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mea.planner.evidence_policy import validate_round_evidence
from mea.tool_results import episode_tool_results


def round_execution_backend(round_plan: dict[str, Any]) -> str:
    """Resolve policy execution independently from the TaskGen route."""

    raw = (round_plan.get("execution") or {}).get("backend")
    if raw is None:
        raw = "expert" if round_plan.get("route") == "official" else "act"
    backend = str(raw).casefold()
    if backend not in {"expert", "act", "both"}:
        raise ValueError(f"unsupported execution backend: {raw!r}")
    return backend


def compact_trusted_tools(
    child_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep numerical Toolkit evidence small enough for planner/feedback use."""

    evaluation = child_manifest.get("trusted_tool_evaluation") or {}
    episodes = []
    for episode in evaluation.get("episodes", []):
        raw_results = episode_tool_results(episode)
        episodes.append(
            {
                "episode_dir": episode.get("episode_dir"),
                "policy_name": episode.get("policy_name"),
                "seed": episode.get("seed"),
                "success": episode.get("success"),
                "results": [
                    {
                        "tool": result.get("tool"),
                        "value": result.get("value"),
                        "unit": result.get("unit"),
                        "passed": result.get("passed"),
                        "evidence_steps": result.get("evidence_steps", []),
                        "details": result.get("details", {}),
                    }
                    for result in raw_results
                ],
            }
        )
    return episodes


def compact_aggregate_result(
    aggregate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip repeated provenance before sending aggregate evidence to an LLM."""

    if not aggregate:
        return None

    def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "episode_result_count": summary.get("episode_result_count"),
            "quality": {
                key: value.get("value")
                for key, value in summary.get("quality", {}).items()
            },
            "statistics": {
                key: {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key != "provenance"
                }
                for key, value in summary.get("statistics", {}).items()
            },
        }

    return {
        "schema_version": aggregate.get("schema_version"),
        "status": aggregate.get("status"),
        "source_count": aggregate.get("source_count"),
        "unique_episode_count": aggregate.get("unique_episode_count"),
        "input_issues": aggregate.get("input_issues", []),
        "metrics": [
            {
                "metric": metric.get("metric"),
                "value_kind": metric.get("value_kind"),
                "unit": metric.get("unit"),
                "cohorts": [
                    {
                        "role": cohort.get("role"),
                        "policy_names": cohort.get("policy_names", []),
                        "summary": compact_summary(cohort.get("summary", {})),
                        "passed_summary": (
                            compact_summary(cohort["passed_summary"])
                            if cohort.get("passed_summary")
                            else None
                        ),
                        "groups": {
                            dimension: [
                                {
                                    "value": group.get("value"),
                                    "summary": compact_summary(
                                        group.get("summary", {})
                                    ),
                                    "passed_summary": (
                                        compact_summary(group["passed_summary"])
                                        if group.get("passed_summary")
                                        else None
                                    ),
                                }
                                for group in groups
                            ]
                            for dimension, groups in cohort.get(
                                "groups", {}
                            ).items()
                        },
                    }
                    for cohort in metric.get("cohorts", [])
                ],
            }
            for metric in aggregate.get("metrics", [])
        ],
    }


def build_evidence_bundle(
    repo_root: Path,
    evaluation_id: str,
    user_request: str,
    plan: dict[str, Any],
    round_runs: list[dict[str, Any]],
    evaluation_aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del repo_root, evaluation_aggregate
    rounds: list[dict[str, Any]] = []
    for index, item in enumerate(round_runs):
        round_summary = item.get("round_summary")
        observations = (
            round_summary.get("observations")
            if isinstance(round_summary, dict)
            else None
        )
        raw = (
            observations.get("round_evidence")
            if isinstance(observations, dict)
            else None
        )
        if not isinstance(raw, dict):
            raise ValueError(
                f"round_runs[{index}] has no observations.round_evidence"
            )
        evidence = validate_round_evidence(raw)
        round_plan = item.get("round_plan")
        if (
            isinstance(round_plan, dict)
            and str(round_plan.get("round_id")) != evidence["round_id"]
        ):
            raise ValueError(
                f"round_runs[{index}] plan and RoundEvidence ids disagree"
            )
        rounds.append(evidence)

    evaluation_relative = Path("mea/evaluation_runs") / evaluation_id
    max_rounds = int(plan["max_rounds"])
    return {
        "schema_version": 3,
        "evaluation_id": evaluation_id,
        "query": user_request,
        "plan": {
            "max_rounds": max_rounds,
            "executed_rounds": len(rounds),
            "planning_state": plan.get("planning_state"),
            "round_budget_remaining": max(
                max_rounds - len(rounds), 0
            ),
        },
        "rounds": rounds,
        "total_policy_episodes": sum(
            len(item["policy"]["seeds"]) for item in rounds
        ),
        "artifacts": {
            "evaluation_plan": str(
                evaluation_relative / "plan/evaluation_plan.json"
            ),
            "summary": str(
                evaluation_relative / "summary/summary.json"
            ),
            "aggregate": str(
                evaluation_relative / "summary/aggregate_result.json"
            ),
            "round_evidence": [
                str(
                    evaluation_relative
                    / "execution"
                    / item["round_id"]
                    / "round_evidence.json"
                )
                for item in rounds
            ],
        },
    }


__all__ = [
    "build_evidence_bundle",
    "compact_aggregate_result",
    "compact_trusted_tools",
    "round_execution_backend",
]
