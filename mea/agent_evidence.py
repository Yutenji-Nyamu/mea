"""Compact evidence projections for the production Agent runtime.

This module owns artifact-to-evidence shaping only.  It does not execute a
provider, simulator, policy, Tool, VQA model, or planner, and it deliberately
does not import the Agent CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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


def _round_evidence(
    repo_root: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    round_summary: dict[str, Any],
    tool_evaluation: dict[str, Any],
) -> dict[str, Any]:
    static = child_manifest.get("static_validation", {})
    scene = child_manifest.get("scene_validation", {})
    vision = child_manifest.get("vision_validation", {})
    reflection = child_manifest.get("visual_self_reflection", {})
    retrieval = child_manifest.get("task_retrieval") or {}
    knowledge = child_manifest.get("knowledge_retrieval") or {}
    trusted_tool_evaluation = child_manifest.get("trusted_tool_evaluation") or {}
    child_relative = child_dir.relative_to(repo_root)
    task_module = str(child_manifest.get("task_module") or "")
    module_source = repo_root / (task_module.replace(".", "/") + ".py")
    generated_task_artifact = (
        str(module_source.relative_to(repo_root))
        if task_module and module_source.is_file()
        else str(child_relative / "task.py")
    )
    act_videos = sorted(
        str(path.relative_to(repo_root))
        for path in (child_dir / "evaluation").glob("episode*.mp4")
    )
    rollout_video_paths = {
        child_dir
        / "evaluation/telemetry"
        / str(episode.get("episode_dir") or "")
        / "video.mp4"
        for episode in trusted_tool_evaluation.get("episodes", [])
    }
    rollout_videos = sorted(
        str(path.relative_to(repo_root))
        for path in rollout_video_paths
        if path.is_file()
    )
    variant_spec_path = child_dir / "variant_spec.json"
    variant_spec = (
        json.loads(variant_spec_path.read_text(encoding="utf-8"))
        if variant_spec_path.is_file()
        else None
    )
    feedback_observations = {
        key: value
        for key, value in round_summary["observations"].items()
        if key
        not in {
            "trusted_tools",
            "planned_tool",
            "aggregate",
            "execution_vqa",
        }
    }

    round_execution_value = round_summary.get("execution_artifact_dir")
    round_execution = (
        Path(str(round_execution_value))
        if round_execution_value
        else Path("mea/evaluation_runs")
        / evaluation_id
        / "execution"
        / round_plan["round_id"]
    )
    execution_vqa_observation = (
        round_summary["observations"].get("execution_vqa") or {}
    )
    execution_vqa_artifacts = execution_vqa_observation.get("artifacts") or {}
    execution_vqa_artifact = execution_vqa_artifacts.get(
        "result"
    ) or execution_vqa_artifacts.get("execution_vqa")
    if not execution_vqa_artifact:
        if execution_vqa_observation.get("status") == "skipped":
            execution_vqa_artifact = str(
                round_execution / "execution_vqa_skipped.json"
            )
        elif execution_vqa_observation.get("status") == "failed":
            execution_vqa_artifact = str(
                round_execution / "execution_vqa_error.json"
            )
    requested_seeds = [
        int(value) for value in round_plan["execution"].get("seeds", [])
    ]
    requested_num_episodes = int(
        round_plan["execution"].get("num_episodes", len(requested_seeds))
    )
    actual_policy_seeds = (
        [
            int(value)
            for value in (
                round_summary["observations"].get("actual_seeds") or []
            )
        ]
        if round_execution_backend(round_plan) in {"act", "both"}
        else []
    )
    implementation_trace = round_summary["observations"].get(
        "implementation_trace"
    )
    round_proposal = (
        round_plan.get("proposal")
        or round_plan.get("experiment_candidate")
    )
    round_candidate_id = (
        round_plan.get("candidate_id")
        or (
            round_proposal.get("candidate_id")
            if isinstance(round_proposal, Mapping)
            else None
        )
        or round_plan.get("template_id")
    )
    if isinstance(implementation_trace, Mapping):
        trace_candidate_id = implementation_trace.get("candidate_id")
        if (
            round_candidate_id
            and trace_candidate_id
            and str(trace_candidate_id) != str(round_candidate_id)
        ):
            raise RuntimeError(
                "implementation trace candidate_id conflicts with the "
                f"executed round: {trace_candidate_id!r} != "
                f"{round_candidate_id!r}"
            )
    return {
        "round_id": round_plan["round_id"],
        "candidate_id": (
            str(round_candidate_id) if round_candidate_id else None
        ),
        "child_run_id": child_manifest.get("run_id"),
        "variant_id": (
            round_plan.get("task_variant_id") or round_plan.get("template_id")
        ),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "capability_contract": round_plan.get("capability_contract"),
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "seeds": actual_policy_seeds,
        "actual_seeds": actual_policy_seeds,
        "requested_seeds": requested_seeds,
        "num_episodes": len(actual_policy_seeds),
        "requested_num_episodes": requested_num_episodes,
        "task_retrieval": {
            "catalog_size": retrieval.get("catalog_size"),
            "selected_tasks": retrieval.get("selected_tasks", []),
            "reasoning": retrieval.get("reasoning"),
        },
        "knowledge_retrieval": {
            "selected_ids": knowledge.get("selected_ids", []),
            "context_character_count": knowledge.get(
                "context_character_count"
            ),
            "committed_index_current": knowledge.get(
                "committed_index_current"
            ),
        },
        "generation": {
            "variant_spec": variant_spec,
            "complete_method_generated": static.get(
                "load_actors_ast", {}
            ).get("complete_method_generated"),
            "generated_color": static.get("load_actors_ast", {}).get(
                "generated_color"
            ),
        },
        "visual_observation": {
            "render_success": scene.get("render_success"),
            "aligned": vision.get("aligned"),
            "target_actor": vision.get("target_actor"),
            "bell_visible": vision.get("bell_visible"),
            "observed_color": vision.get("observed_color"),
            "unexpected_changes": vision.get("unexpected_changes"),
            "confidence": vision.get("confidence"),
            "position_authority": vision.get("position_authority"),
        },
        "visual_self_reflection": {
            "passed": reflection.get("passed"),
            "max_repairs": reflection.get("max_repairs"),
            "repairs_used": reflection.get("repairs_used"),
            "final_attempt": reflection.get("final_attempt"),
            "attempt_count": len(reflection.get("attempts", [])),
        },
        "observations": {
            **feedback_observations,
            "pipeline_passed": round_summary["pipeline_passed"],
        },
        "tool_evaluation": tool_evaluation,
        "aggregate": round_summary["observations"].get("aggregate"),
        "execution_vqa": round_summary["observations"].get("execution_vqa"),
        "implementation_trace": implementation_trace,
        "trusted_tool_evaluation": {
            "artifact": trusted_tool_evaluation.get("artifact"),
            "episode_count": trusted_tool_evaluation.get("episode_count"),
            "episodes": compact_trusted_tools(child_manifest),
        },
        "artifacts": {
            "generated_task": generated_task_artifact,
            "scene_image": str(child_relative / "evidence/initial_head.png"),
            "vision_result": str(child_relative / "validation/vision.json"),
            "position_samples": str(
                child_relative / "validation/position_samples.json"
            ),
            "reflection_summary": str(
                child_relative / "reflection/summary.json"
            ),
            "act_videos": act_videos,
            "rollout_videos": rollout_videos,
            "act_result": str(child_relative / "evaluation/_result.txt"),
            "trusted_tools": trusted_tool_evaluation.get("artifact"),
            "planned_tool": tool_evaluation.get("artifacts", {}).get(
                "tool_execution"
            ),
            "aggregate": str(round_execution / "aggregate_result.json"),
            "evidence_aggregate": str(
                round_execution / "evidence_aggregate.json"
            ),
            "method_runtime": (
                (
                    round_summary["observations"].get("method_runtime")
                    or {}
                ).get("artifact")
            ),
            "execution_vqa": execution_vqa_artifact,
            "execution_vqa_query": str(
                round_execution / "execution_vqa_query.json"
            ),
            "execution_vqa_montage": execution_vqa_artifacts.get("montage"),
            "execution_vqa_selection": execution_vqa_artifacts.get(
                "selection"
            ),
            "child_manifest": str(child_relative / "manifest.json"),
        },
    }


def build_evidence_bundle(
    repo_root: Path,
    evaluation_id: str,
    user_request: str,
    plan: dict[str, Any],
    round_runs: list[dict[str, Any]],
    evaluation_aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rounds = [
        _round_evidence(
            repo_root,
            evaluation_id,
            item["round_plan"],
            item["child_manifest"],
            item["child_dir"],
            item["round_summary"],
            item["tool_evaluation"],
        )
        for item in round_runs
    ]
    planning_rounds = [
        item
        for item in rounds
        if isinstance(
            item.get("observations", {}).get("planning_observation"),
            Mapping,
        )
    ]
    policy_rounds = [item for item in rounds if item not in planning_rounds]
    total_episodes = sum(item["num_episodes"] for item in rounds)
    requested_total_episodes = sum(
        item["requested_num_episodes"] for item in rounds
    )
    weighted_success = 0.0
    measured_episodes = 0
    for item in rounds:
        rate = item["observations"].get("policy_success")
        if rate is not None:
            weighted_success += float(rate) * item["num_episodes"]
            measured_episodes += item["num_episodes"]
    policy_success = (
        weighted_success / measured_episodes if measured_episodes else None
    )
    position_rounds = [
        item
        for item in rounds
        if str(item["sub_aspect"]).startswith("object_position")
    ]
    position_metrics_by_round = {
        item["round_id"]: item["observations"].get("position_metrics", {})
        for item in position_rounds
    }
    sampled_xy: list[list[float]] = []
    for item in position_rounds:
        for sample in item["observations"].get("position_samples", []):
            # Existing TaskGen samples do not yet carry one common controlled
            # actor/semantic-key field.  Preserve the established projection
            # until that producer contract is explicit.
            position = sample.get("bell_position") or sample.get(
                "block_position"
            )
            if isinstance(position, list) and len(position) >= 2:
                sampled_xy.append(
                    [float(position[0]), float(position[1])]
                )
    unique_xy = {
        (round(item[0], 8), round(item[1], 8)) for item in sampled_xy
    }
    position_metrics = (
        {
            "sample_count": len(sampled_xy),
            "unique_xy_count": len(unique_xy),
            "x_span": (
                max(item[0] for item in sampled_xy)
                - min(item[0] for item in sampled_xy)
            ),
            "y_span": (
                max(item[1] for item in sampled_xy)
                - min(item[1] for item in sampled_xy)
            ),
            "position_varied": len(unique_xy) > 1,
            "by_round": position_metrics_by_round,
        }
        if sampled_xy
        else {}
    )
    evaluation_relative = Path("mea/evaluation_runs") / evaluation_id
    completed_template_ids = [
        item["round_plan"]["template_id"] for item in round_runs
    ]
    completed_aspect_ids: list[str] = []
    for item in round_runs:
        round_plan = item["round_plan"]
        proposal = round_plan.get("task_proposal") or {}
        aspect_id = str(
            proposal.get("aspect_id")
            or round_plan.get("aspect_id")
            or round_plan.get("sub_aspect")
            or ""
        )
        if aspect_id and aspect_id not in completed_aspect_ids:
            completed_aspect_ids.append(aspect_id)
    remaining_template_ids = [
        item
        for item in plan.get("requested_template_ids", [])
        if item not in completed_template_ids
    ]
    required_aspect_ids = list(
        plan.get("requested_aspect_ids") or completed_aspect_ids
    )
    initial_requested_aspect_ids = list(
        plan.get("initial_requested_aspect_ids") or required_aspect_ids
    )
    discovered_aspect_ids = [
        item
        for item in completed_aspect_ids
        if item not in initial_requested_aspect_ids
    ]
    uncovered_required_aspect_ids = [
        item
        for item in required_aspect_ids
        if item not in completed_aspect_ids
    ]
    decision_artifacts = [
        str(
            evaluation_relative
            / f"plan/decision_after_round_{round_number}.json"
        )
        for round_number in range(
            1, len(plan.get("round_decisions", [])) + 1
        )
    ]
    evidence_assessment_artifacts = [
        str(
            evaluation_relative
            / f"plan/evidence_after_round_{round_number}.json"
        )
        for round_number in range(
            1, len(plan.get("round_decisions", [])) + 1
        )
    ]
    history_path = (
        repo_root / evaluation_relative / "plan/history_retrieval.json"
    )
    history_retrieval = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else {"status": "missing", "matches": []}
    )
    global_route_path = (
        repo_root / evaluation_relative / "plan/global_query_route.json"
    )
    global_route = (
        json.loads(global_route_path.read_text(encoding="utf-8"))
        if global_route_path.is_file()
        else None
    )
    execution_backends = sorted(
        {
            str(item["observations"].get("execution_backend") or "ACT")
            for item in rounds
        }
    )
    act_statuses = [
        item["observations"].get("act_pipeline_status") for item in rounds
    ]
    measured_act_statuses = [
        bool(value) for value in act_statuses if value is not None
    ]
    expert_statuses = [
        item["observations"].get("expert_solvable") for item in rounds
    ]
    measured_expert_statuses = [
        bool(value) for value in expert_statuses if value is not None
    ]
    return {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "user_request": user_request,
        "plan": {
            "max_rounds": plan["max_rounds"],
            "executed_rounds": len(rounds),
            "planning_state": plan.get("planning_state"),
            "round_decisions": plan.get("round_decisions", []),
            "requested_template_ids": plan.get(
                "requested_template_ids", []
            ),
            "completed_template_ids": completed_template_ids,
            "remaining_template_ids": remaining_template_ids,
            "round_budget_remaining": max(
                int(plan["max_rounds"]) - len(rounds), 0
            ),
            "aspect_coverage": {
                "schema_version": 1,
                "initial_requested_aspect_ids": (
                    initial_requested_aspect_ids
                ),
                "required_aspect_ids": required_aspect_ids,
                "covered_aspect_ids": completed_aspect_ids,
                "discovered_aspect_ids": discovered_aspect_ids,
                "uncovered_required_aspect_ids": (
                    uncovered_required_aspect_ids
                ),
                "coverage_status": (
                    "complete"
                    if not uncovered_required_aspect_ids
                    else "partial"
                    if completed_aspect_ids
                    else "not_started"
                ),
            },
        },
        "rounds": rounds,
        "observations": {
            "execution_backends": execution_backends,
            "scene_alignment": bool(policy_rounds)
            and all(
                item["observations"]["scene_alignment"]
                for item in policy_rounds
            ),
            "observed_color_by_round": [
                item["observations"]["observed_color"] for item in rounds
            ],
            "expert_solvable": (
                all(measured_expert_statuses)
                if measured_expert_statuses
                else None
            ),
            "act_pipeline_status": (
                all(measured_act_statuses)
                if measured_act_statuses
                else None
            ),
            "policy_success": policy_success,
            "policy_success_by_round": [
                item["observations"]["policy_success"] for item in rounds
            ],
            "position_varied": position_metrics.get("position_varied"),
            "position_metrics": position_metrics,
            "position_metrics_by_round": position_metrics_by_round,
            # A TaskGen candidate rejected before policy execution is planning
            # evidence, not a failed policy/evaluation pipeline.  Preserve the
            # rejected round below while computing policy-pipeline health only
            # from rounds that actually started policy execution.
            "pipeline_passed": bool(policy_rounds)
            and all(
                item["observations"]["pipeline_passed"]
                for item in policy_rounds
            ),
            "policy_round_count": len(policy_rounds),
            "planning_round_count": len(planning_rounds),
            "planning_observations": [
                item["observations"]["planning_observation"]
                for item in planning_rounds
            ],
            "aggregate": compact_aggregate_result(evaluation_aggregate),
            "execution_vqa_conflict": any(
                bool(
                    item.get("execution_vqa", {}).get(
                        "evidence_conflict"
                    )
                )
                for item in rounds
            ),
        },
        "total_episodes": total_episodes,
        "requested_total_episodes": requested_total_episodes,
        "history_retrieval": history_retrieval,
        "global_query_route": (
            {
                "selection": global_route.get("selection"),
                "resolved": global_route.get("resolved"),
                "provider_called": global_route.get("provider_called"),
                "attempt_count": global_route.get("attempt_count"),
            }
            if global_route is not None
            else None
        ),
        "limitations": {
            "bounded_three_round_prototype": True,
            "few_episodes_are_not_a_generalization_benchmark": True,
            "policy_result_is_not_pipeline_status": True,
            "global_route_unsupported_capabilities": (
                (global_route.get("selection") or {}).get(
                    "unsupported_capabilities", []
                )
                if global_route is not None
                else []
            ),
        },
        "artifacts": {
            "evaluation_plan": str(
                evaluation_relative / "plan/evaluation_plan.json"
            ),
            "plan_decisions": decision_artifacts,
            "evidence_assessments": evidence_assessment_artifacts,
            "history_retrieval": str(
                evaluation_relative / "plan/history_retrieval.json"
            ),
            "global_query_route": (
                str(evaluation_relative / "plan/global_query_route.json")
                if global_route is not None
                else None
            ),
            "summary": str(
                evaluation_relative / "summary/summary.json"
            ),
            "aggregate": str(
                evaluation_relative / "summary/aggregate_result.json"
            ),
            "round_artifacts": [item["artifacts"] for item in rounds],
        },
    }


__all__ = [
    "_round_evidence",
    "build_evidence_bundle",
    "compact_aggregate_result",
    "compact_trusted_tools",
    "round_execution_backend",
]
