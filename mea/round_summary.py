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
from mea.execution_vqa.runtime import (
    _legacy_round_requests_execution_vqa,
    compact_execution_vqa,
)
from mea.planner import (
    advance_implementation_trace_with_tool,
    build_evidence_aggregate,
    build_implementation_trace,
)
from mea.round_contract import validate_round_capability_contract
from mea.round_evidence import compact_tool_evaluation


_STRUCTURALLY_COMPLETED_VQA_STATUSES = {"passed", "skipped", "abstained"}


def read_policy_success(result_path: Path) -> float | None:
    if not result_path.is_file():
        return None
    for line in reversed(result_path.read_text(encoding="utf-8").splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None



def taskgen_ast_gate_passed(static_validation: Mapping[str, Any]) -> bool:
    """Accept either legacy TaskGen AST output or provider scene+checker AST."""

    legacy = static_validation.get("load_actors_ast") or {}
    if isinstance(legacy, Mapping) and legacy.get("valid") is True:
        return True
    provider = static_validation.get("provider_scene_checker") or {}
    return bool(
        isinstance(provider, Mapping)
        and provider.get("valid") is True
        and provider.get("model_written_python") is True
        and provider.get("restricted_success_spec_compiler_used") is False
        and isinstance(provider.get("ast_policy"), str)
        and provider["ast_policy"].strip()
    )


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


def summarize_round(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    tool_evaluation: dict[str, Any] | None = None,
    aggregate_result: dict[str, Any] | None = None,
    execution_vqa: dict[str, Any] | None = None,
    taskgen_returncode: int = 0,
) -> dict[str, Any]:
    capability_contract = validate_round_capability_contract(round_plan)
    scene = child_manifest.get("scene_validation", {})
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
    is_generic_provider = (
        round_plan.get("route")
        == "generic_provider_scene_checker_codegen"
    )
    execution_backend = round_execution_backend(round_plan)
    uses_act = execution_backend in {"act", "both"}
    uses_expert = execution_backend in {"expert", "both"}
    outcome_semantics = normalize_outcome_semantics(
        trusted_tool_evaluation,
        task_artifact_summary,
    )
    static = child_manifest.get("static_validation") or {}
    policy_outcome = {
        "metric": trusted_tool_evaluation.get("outcome_metric"),
        "authority": trusted_tool_evaluation.get("outcome_authority"),
        "binding": deepcopy(trusted_tool_evaluation.get("outcome_binding")),
        "value": policy_success if uses_act else None,
        "official_equivalent": bool(
            task_artifact_summary.get("success_official_equivalent", True)
        ),
        "execution_scope": task_artifact_summary.get(
            "success_execution_scope", "official_equivalent"
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
    vqa_explicitly_omitted = not _legacy_round_requests_execution_vqa(round_plan)
    gate_status = {
        "variant_spec": (
            (child_manifest.get("capability_contract_validation") or {}).get(
                "status"
            )
            == "passed"
        ),
        "ast": taskgen_ast_gate_passed(static),
        "render": bool(scene.get("render_success")),
        "rule": bool((scene.get("rule_check") or {}).get("passed")),
        "scene_variant": bool(positions.get("passed")),
        "vision": bool(vision.get("passed")),
        "expert": bool((scene.get("expert_batch") or expert).get("passed")),
        "act": bool((not uses_act and is_official) or act.get("passed")),
        "toolkit": bool(
            (child_manifest.get("trusted_tool_evaluation") or {}).get(
                "episode_count"
            )
        ),
        "planned_tool": bool(
            tool_evaluation and tool_evaluation.get("status") == "passed"
        ),
        "aggregate": bool(
            aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
        ),
        "execution_vqa": bool(
            execution_vqa
            and (
                execution_vqa.get("status") in {"passed", "abstained"}
                or (
                    (not uses_act or vqa_explicitly_omitted)
                    and execution_vqa.get("status") == "skipped"
                )
            )
        ),
    }
    required_gates = (
        list(capability_contract["required_gates"])
        if capability_contract is not None
        else []
    )
    required_gate_status = {
        "required": required_gates,
        "by_gate": {gate: bool(gate_status.get(gate, False)) for gate in required_gates},
    }
    required_gate_status["passed"] = all(
        required_gate_status["by_gate"].values()
    )
    if is_official:
        expert_batch = scene.get("expert_batch") or expert
        pipeline_passed = bool(
            child_manifest.get("status")
            == ("completed" if uses_act else "completed_without_act")
            and taskgen_returncode == 0
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and (not uses_expert or expert_batch.get("passed"))
            and (not uses_act or act.get("passed"))
            and child_manifest.get("trusted_tool_evaluation", {}).get("episode_count")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in _STRUCTURALLY_COMPLETED_VQA_STATUSES
        )
    elif is_generic_provider:
        preflight = scene.get("generic_preflight") or {}
        fixtures = preflight.get("checker_fixtures") or []
        acceptance = child_manifest.get("task_generation_acceptance") or {}
        visual_required = acceptance.get(
            "visual_self_check_required", True
        )
        pipeline_passed = bool(
            child_manifest.get("status") == "completed"
            and taskgen_returncode == 0
            and taskgen_ast_gate_passed(static)
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and expert.get("passed")
            and preflight.get("render_passed") is True
            and preflight.get("expert_passed") is True
            and preflight.get("scene_change_passed") is True
            and (
                not visual_required
                or (
                    vision.get("status") == "passed"
                    and vision.get("passed") is True
                )
            )
            and fixtures
            and all(item.get("passed") is True for item in fixtures)
            and act.get("passed")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in _STRUCTURALLY_COMPLETED_VQA_STATUSES
        )
    else:
        # Generated rounds keep their expert, visual, and task-specific
        # position gates while ACT remains the policy under evaluation.
        pipeline_passed = bool(
            child_manifest.get("status") == "completed"
            and taskgen_returncode == 0
            and scene.get("rule_check", {}).get("passed")
            and vision.get("passed")
            and expert.get("passed")
            and positions.get("passed")
            and act.get("passed")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in _STRUCTURALLY_COMPLETED_VQA_STATUSES
        )
    if capability_contract is not None:
        pipeline_passed = bool(pipeline_passed and required_gate_status["passed"])
    implementation_trace = child_manifest.get("implementation_trace")
    if (
        not isinstance(implementation_trace, Mapping)
        and isinstance(
            round_plan.get("proposal")
            or round_plan.get("experiment_candidate"),
            Mapping,
        )
    ):
        implementation_trace = build_implementation_trace(
            round_plan.get("proposal")
            or round_plan["experiment_candidate"]
        )
    if isinstance(implementation_trace, Mapping):
        semantic_needs = round_plan.get("semantic_need_execution")
        rule_need = (
            semantic_needs.get("rule_tool")
            if isinstance(semantic_needs, Mapping)
            else None
        )
        checker_need = (
            semantic_needs.get("checker")
            if isinstance(semantic_needs, Mapping)
            else None
        )
        vqa_need = (
            semantic_needs.get("vqa_tool")
            if isinstance(semantic_needs, Mapping)
            else None
        )
        rule_requested = bool(
            isinstance(rule_need, Mapping)
            and rule_need.get("requested") is True
        )
        checker_requested = bool(
            isinstance(checker_need, Mapping)
            and checker_need.get("requested") is True
        )
        vqa_requested = bool(
            isinstance(vqa_need, Mapping)
            and vqa_need.get("requested") is True
        )
        implementation_trace = advance_implementation_trace_with_tool(
            implementation_trace,
            tool_evaluation,
            # The default task checker remains the Rule observation for
            # scene-only rounds.  VQA-only rounds must not depend on it.
            rule_required=bool(
                not isinstance(semantic_needs, Mapping)
                or rule_requested
                or checker_requested
                or not vqa_requested
            ),
            vqa_evaluation=execution_vqa,
            vqa_required=vqa_requested,
        )
    summary = {
        "round_id": round_plan["round_id"],
        "variant_id": (
            round_plan.get("task_variant_id") or round_plan.get("template_id")
        ),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "capability_contract": round_plan.get("capability_contract"),
        "semantic_need_execution": deepcopy(
            round_plan.get("semantic_need_execution")
        ),
        "required_gate_status": required_gate_status,
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
            "observed_color": vision.get("observed_color"),
            "bell_visible": vision.get("bell_visible"),
            "position_authority": vision.get("position_authority"),
            "expert_solvable": (
                bool((scene.get("expert_batch") or expert).get("passed"))
                if uses_expert or not is_official
                else None
            ),
            "act_pipeline_status": bool(act.get("passed")) if uses_act else None,
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
            "implementation_trace": implementation_trace,
            "required_gate_status": required_gate_status,
        },
        "pipeline_passed": pipeline_passed,
        "interpretation": (
            "任务路由与执行后端分别记录；ACT 策略结果和流水线状态分开报告，"
            "策略失败不会被误记为 pipeline failure。"
        ),
    }
    summary["observations"]["evidence_aggregate"] = (
        build_evidence_aggregate(round_plan, summary)
    )
    return summary


__all__ = [
    "normalize_outcome_semantics",
    "summarize_round",
    "taskgen_ast_gate_passed",
]
