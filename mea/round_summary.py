"""Normalize one completed round into evidence for the Plan Agent."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.tool_results import episode_tool_results
from mea.agent_evidence import (
    compact_aggregate_result,
    compact_trusted_tools,
    round_execution_backend,
)
from mea.execution_vqa.runtime import compact_execution_vqa
from mea.round_evidence import compact_tool_evaluation


def read_policy_success(result_path: Path) -> float | None:
    if not result_path.is_file():
        return None
    for line in reversed(result_path.read_text(encoding="utf-8").splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


def normalize_outcome_semantics(
    trusted_tool_evaluation: Mapping[str, Any],
    task_artifact_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare generated and official predicates without conflating semantics."""

    raw_official_equivalent = task_artifact_summary.get(
        "success_official_equivalent"
    )
    outcome_authority = trusted_tool_evaluation.get("outcome_authority")
    official_authority = outcome_authority in {
        "official_check_success",
        "official_check_success_reused",
    }
    official_equivalent = (
        raw_official_equivalent
        if isinstance(raw_official_equivalent, bool)
        else (
            True
            if official_authority
            else None
        )
    )
    episodes: list[dict[str, Any]] = []
    for episode in trusted_tool_evaluation.get("episodes", []):
        if not isinstance(episode, Mapping):
            continue
        for result in episode_tool_results(episode):
            details = result.get("details")
            if not isinstance(details, Mapping):
                continue
            raw_generated = details.get("generated_checker_success")
            raw_official = details.get("official_success")
            raw_official_core = details.get("official_core_predicate_satisfied")
            generated = raw_generated if isinstance(raw_generated, bool) else None
            official_core = (
                raw_official_core if isinstance(raw_official_core, bool) else None
            )
            official = (
                raw_official
                if isinstance(raw_official, bool)
                else official_core
            )
            if generated is None and official is None and official_core is None:
                continue

            if (
                generated is not None
                and outcome_authority
                != "llm_generated_python_ast_validated"
            ):
                status = "non_comparable"
                reason = "generated_result_has_untrusted_outcome_authority"
            elif generated is not None and official_equivalent is None:
                status = "non_comparable"
                reason = "generated_checker_equivalence_not_declared"
            elif (
                generated is not None
                and official is not None
                and official_equivalent is True
            ):
                status = (
                    "equivalent_agreement"
                    if generated == official
                    else "conflict"
                )
                reason = (
                    "generated_and_official_equivalent_predicates_agree"
                    if generated == official
                    else "generated_and_official_equivalent_predicates_disagree"
                )
            elif generated is not None and official_equivalent is False:
                if generated is True and official_core is False:
                    status = "conflict"
                    reason = "generated_success_without_official_core_predicate"
                elif official_core is not None:
                    status = "expected_semantic_extension"
                    reason = (
                        "generated_checker_adds_constraints_beyond_official_core"
                    )
                else:
                    status = "non_comparable"
                    reason = "non_equivalent_checker_has_no_official_core_projection"
            elif generated is None and official is not None:
                status = "official_only"
                reason = "no_generated_checker_result"
            else:
                status = "non_comparable"
                reason = "insufficient_dual_predicate_results"

            episodes.append(
                {
                    "seed": episode.get("seed"),
                    "generated_checker_success": generated,
                    "official_success": official,
                    "official_core_predicate_satisfied": official_core,
                    "official_equivalent": official_equivalent,
                    "status": status,
                    "reason": reason,
                }
            )

    statuses = {item["status"] for item in episodes}
    if (
        not episodes
        and official_authority
        and official_equivalent is True
    ):
        status = "official_only"
        reason_codes = ["official_outcome_has_no_generated_checker"]
    elif "conflict" in statuses:
        status = "conflict"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    elif "expected_semantic_extension" in statuses:
        status = "expected_semantic_extension"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    elif statuses == {"equivalent_agreement"}:
        status = "equivalent_agreement"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    elif statuses == {"official_only"}:
        status = "official_only"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    else:
        status = "non_comparable"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    return {
        "schema_version": 1,
        "status": status,
        "evidence_conflict": status == "conflict",
        "official_equivalent": official_equivalent,
        "outcome_authority": outcome_authority,
        "episodes": episodes,
        "reason_codes": reason_codes,
    }


def _compact_scene_change(
    scene_validation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Expose only simulator-authoritative executable scene-change facts."""

    preflight = scene_validation.get("generic_preflight")
    preflight = preflight if isinstance(preflight, Mapping) else {}
    scene_change = preflight.get("scene_change")
    if not isinstance(scene_change, Mapping):
        return None
    changes = scene_change.get("tracked_actor_changes")
    if not isinstance(changes, list) or not changes:
        return None
    return {
        "tracked_actor_changes": deepcopy(changes),
    }


def summarize_round(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    tool_evaluation: dict[str, Any] | None = None,
    aggregate_result: dict[str, Any] | None = None,
    execution_vqa: dict[str, Any] | None = None,
    taskgen_returncode: int = 0,
) -> dict[str, Any]:
    scene = child_manifest.get("scene_validation", {})
    scene_change = _compact_scene_change(scene)
    vision = child_manifest.get("vision_validation", {})
    act = child_manifest.get("act_evaluation", {})
    expert = scene.get("expert", {})
    positions = child_manifest.get("position_samples", {})
    position_metrics = positions.get("metrics", {})
    variant_samples = positions.get("samples", [])
    observed_bell_ids = sorted(
        {
            int(item["bell_id"])
            for item in variant_samples
            if isinstance(item, dict)
            and not isinstance(item.get("bell_id"), bool)
            and isinstance(item.get("bell_id"), int)
        }
    )
    clutter_counts = [
        int(item["clutter_count"])
        for item in variant_samples
        if isinstance(item, dict)
        and not isinstance(item.get("clutter_count"), bool)
        and isinstance(item.get("clutter_count"), int)
    ]
    policy_success = read_policy_success(child_dir / "evaluation/_result.txt")
    trusted_tools = compact_trusted_tools(child_manifest)
    trusted_tool_evaluation = child_manifest.get("trusted_tool_evaluation") or {}
    task_artifact_summary = child_manifest.get("task_artifact_summary") or {}
    is_official = round_plan.get("route") == "official"
    execution_backend = round_execution_backend(round_plan)
    uses_act = execution_backend in {"act", "both"}
    uses_expert = execution_backend in {"expert", "both"}
    outcome_semantics = normalize_outcome_semantics(
        trusted_tool_evaluation,
        task_artifact_summary,
    )
    raw_official_equivalent = task_artifact_summary.get(
        "success_official_equivalent"
    )
    official_equivalent = (
        raw_official_equivalent
        if isinstance(raw_official_equivalent, bool)
        else None
    )
    policy_outcome = {
        "metric": trusted_tool_evaluation.get("outcome_metric"),
        "authority": trusted_tool_evaluation.get("outcome_authority"),
        "binding": deepcopy(trusted_tool_evaluation.get("outcome_binding")),
        "value": policy_success if uses_act else None,
        "official_equivalent": official_equivalent,
        "execution_scope": task_artifact_summary.get(
            "success_execution_scope"
        ),
        "outcome_semantics": deepcopy(outcome_semantics),
    }
    if uses_act:
        actual_seeds = [int(value) for value in act.get("actual_seeds", [])]
    else:
        actual_seeds = [
            int(item["seed"])
            for item in scene.get("expert_batch", {}).get("episodes", [])
            if item.get("seed") is not None
        ]
    summary = {
        "round_id": round_plan["round_id"],
        "variant_id": (
            round_plan.get("task_variant_id") or round_plan.get("template_id")
        ),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "semantic_need_execution": deepcopy(
            round_plan.get("semantic_need_execution")
        ),
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "taskgen_run_id": child_manifest.get("run_id"),
        "taskgen_returncode": taskgen_returncode,
        "execution": round_plan["execution"],
        "observations": {
            "execution_backend": {
                "expert": "expert",
                "act": "ACT",
                "both": "ACT+expert",
            }[execution_backend],
            "requested_seeds": [
                int(value) for value in round_plan["execution"].get("seeds", [])
            ],
            "actual_seeds": actual_seeds,
            "scene_alignment": bool(scene.get("rule_check", {}).get("passed")),
            "scene_change": scene_change,
            "observed_color": vision.get("observed_color"),
            "bell_visible": vision.get("bell_visible"),
            "position_authority": vision.get("position_authority"),
            "expert_solvable": (
                bool((scene.get("expert_batch") or expert).get("passed"))
                if uses_expert or not is_official
                else None
            ),
            "policy_success": policy_success if uses_act else None,
            "policy_outcome": policy_outcome,
            "outcome_semantics": outcome_semantics,
            "semantic_need_execution": deepcopy(
                round_plan.get("semantic_need_execution")
            ),
            "position_samples": positions.get("samples", []),
            "position_metrics": position_metrics,
            "controlled_axis": positions.get("controlled_axis"),
            "variant_samples": variant_samples,
            "variant_metrics": position_metrics,
            "observed_bell_ids": observed_bell_ids,
            "bell_instance_id": (
                observed_bell_ids[0] if len(observed_bell_ids) == 1 else None
            ),
            "scene_clutter": {
                "expected": bool(position_metrics.get("expected_clutter")),
                "counts": clutter_counts,
                "all_matched": position_metrics.get("all_clutter_matched"),
                "authority": (
                    "simulator_task_info:cluttered_table_info"
                    if clutter_counts
                    else None
                ),
            },
            "trusted_tools": trusted_tools,
            "planned_tool": compact_tool_evaluation(tool_evaluation),
            "aggregate": compact_aggregate_result(aggregate_result),
            "execution_vqa": compact_execution_vqa(execution_vqa),
        },
        "interpretation": (
            "任务、策略、Rule 与 VQA 各自记录事实；不再压成整轮 AND。"
        ),
    }
    return summary


__all__ = [
    "normalize_outcome_semantics",
    "summarize_round",
]
