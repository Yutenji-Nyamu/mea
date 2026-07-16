"""Plan and execute a bounded, evidence-driven multi-round MEA evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.execution_vqa import build_execution_vqa_query, run_execution_vqa
from mea.feedback import FeedbackAgent, render_evaluation_report
from mea.history import EvaluationHistoryDB
from mea.planner import OfficialTaskPlanAgent, PlanAgentPrototype
from mea.providers import (
    OpenAICompatibleProvider,
    available_model_profiles,
    resolve_model_profile,
)
from mea.toolgen import execute_tool_request
from mea.toolkit import aggregate_tool_executions


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_manifest(evaluation_dir: Path, **updates: Any) -> dict[str, Any]:
    path = evaluation_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def child_run_id(evaluation_id: str, round_id: str) -> str:
    return f"run_{evaluation_id.removeprefix('eval_')}_{round_id}"


def build_taskgen_command(
    repo_root: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    telemetry_profile: str = "balanced_v1",
) -> tuple[list[str], str]:
    run_id = child_run_id(evaluation_id, round_plan["round_id"])
    execution = round_plan["execution"]
    seed = execution["seeds"][0]
    task_name = str(round_plan.get("task_name") or "beat_block_hammer")
    task_module = round_plan.get("task_module")
    route = str(round_plan["route"])
    command = [
        sys.executable,
        str(repo_root / "scripts/manipeval_taskgen.py"),
        "--repo-root",
        str(repo_root),
        "--request",
        round_plan["task_instruction"],
        "--run-id",
        run_id,
        "--task-name",
        task_name,
        "--mode",
        route,
        "--text-model",
        text_model,
        "--vision-model",
        vision_model,
        "--seed",
        str(seed),
        "--num-episodes",
        str(execution["num_episodes"]),
        "--gpu",
        str(gpu),
        "--telemetry-profile",
        telemetry_profile,
        "--probe",
        "--expert",
        "--max-reflections",
        str(max_reflections),
    ]
    if task_module:
        command.extend(["--task-module", str(task_module)])
    if route != "official":
        command.extend(["--vision-check", "--run-act"])
    if base_url:
        command.extend(["--base-url", base_url])
    return command, run_id


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def read_policy_success(result_path: Path) -> float | None:
    if not result_path.is_file():
        return None
    for line in reversed(result_path.read_text(encoding="utf-8").splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


def compact_trusted_tools(child_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the numerical Toolkit evidence small enough for planner/feedback use."""

    evaluation = child_manifest.get("trusted_tool_evaluation") or {}
    episodes = []
    for episode in evaluation.get("episodes", []):
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
                    for result in episode.get("tool_results", [])
                ],
            }
        )
    return episodes


def compact_tool_evaluation(
    tool_evaluation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep planned Tool evidence compact while preserving ACT/expert roles."""

    if not tool_evaluation:
        return None
    return {
        "status": tool_evaluation.get("status"),
        "requested_route": tool_evaluation.get("requested_route"),
        "route": tool_evaluation.get("route"),
        "reference_tool": tool_evaluation.get("reference_tool"),
        "route_decision": tool_evaluation.get("route_decision", {}),
        "source": tool_evaluation.get("source", {}),
        "episodes": [
            {
                "policy_name": item.get("policy_name"),
                "seed": item.get("seed"),
                "role": item.get("role"),
                "value": item.get("result", {}).get("value"),
                "passed": item.get("result", {}).get("passed"),
                "evidence_steps": item.get("result", {}).get(
                    "evidence_steps", []
                ),
                "details": item.get("result", {}).get("details", {}),
            }
            for item in tool_evaluation.get("episodes", [])
        ],
        "validation": tool_evaluation.get("validation", {}),
    }


def _aggregate_sources(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build one de-duplicated set of episode ToolResult sources."""

    context = {
        "round_id": round_plan["round_id"],
        "variant": round_plan.get("template_id")
        or round_plan.get("sub_aspect")
        or round_plan.get("route"),
    }
    sources: list[dict[str, Any]] = []
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    trusted_tools = {
        result.get("tool")
        for episode in trusted.get("episodes", [])
        for result in episode.get("tool_results", [])
        if result.get("tool")
    }
    if trusted.get("episodes"):
        sources.append(
            {
                **trusted,
                "context": {
                    **context,
                    "source_artifact": trusted.get("artifact"),
                },
            }
        )
    if tool_evaluation and tool_evaluation.get("episodes"):
        request = tool_evaluation.get("tool_request") or tool_evaluation.get(
            "tool_spec", {}
        )
        metric = request.get("metric") if isinstance(request, dict) else None
        if metric not in trusted_tools:
            sources.append(
                {
                    "tool_execution": tool_evaluation,
                    "context": {
                        **context,
                        "source_artifact": tool_evaluation.get(
                            "artifacts", {}
                        ).get("tool_execution"),
                    },
                }
            )
    return sources


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
                                        compact_summary(
                                            group["passed_summary"]
                                        )
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


def aggregate_round_results(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    sources = _aggregate_sources(round_plan, child_manifest, tool_evaluation)
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no episode ToolResult rows were available",
            "metrics": [],
        }
        write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


def aggregate_evaluation_results(
    round_runs: list[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    sources = [
        source
        for item in round_runs
        for source in _aggregate_sources(
            item["round_plan"], item["child_manifest"], item["tool_evaluation"]
        )
    ]
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no completed round ToolResult rows were available",
            "metrics": [],
        }
        write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


def _policy_episode_for_execution_vqa(
    child_manifest: dict[str, Any], child_dir: Path
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]] | None:
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    candidates = sorted(
        (
            episode
            for episode in trusted.get("episodes", [])
            if str(episode.get("policy_name", "")).casefold() == "act"
        ),
        key=lambda episode: (
            int(episode.get("seed") or 0),
            str(episode.get("episode_dir") or ""),
        ),
    )
    if not candidates:
        return None
    episode = candidates[0]
    episode_dir = child_dir / "evaluation/telemetry" / episode["episode_dir"]
    return episode_dir, episode, list(episode.get("tool_results", []))


def _same_telemetry_episode(
    candidate: dict[str, Any], representative: dict[str, Any]
) -> bool:
    """Match generated and Trusted Tool rows to one physical rollout."""

    candidate_dir = candidate.get("episode_dir")
    representative_dir = representative.get("episode_dir")
    if candidate_dir and representative_dir:
        return str(candidate_dir) == str(representative_dir)
    return (
        candidate.get("seed") == representative.get("seed")
        and str(candidate.get("policy_name", "")).casefold()
        == str(representative.get("policy_name", "")).casefold()
    )


def run_round_execution_vqa(
    *,
    repo_root: Path,
    child_manifest: dict[str, Any],
    child_dir: Path,
    tool_evaluation: dict[str, Any] | None,
    execution_dir: Path,
    provider: Any,
    model: str,
    round_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query = build_execution_vqa_query(
        task_name=(
            str((round_plan or {}).get("task_name") or child_manifest.get("task_name"))
            if (round_plan or {}).get("task_name") or child_manifest.get("task_name")
            else None
        ),
        template_id=(round_plan or {}).get("template_id"),
        sub_aspect=(round_plan or {}).get("sub_aspect"),
        tool_contract=(round_plan or {}).get("tool_request"),
    )
    write_json(execution_dir / "execution_vqa_query.json", query)
    selected = _policy_episode_for_execution_vqa(child_manifest, child_dir)
    if selected is None:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no completed ACT telemetry episode was available",
            "evidence_conflict": False,
            "query": query,
        }
        write_json(execution_dir / "execution_vqa_skipped.json", result)
        return result
    episode_dir, representative, numeric_results = selected
    representative_path = str(episode_dir.relative_to(repo_root))
    if not (episode_dir / "video.mp4").is_file():
        result = {
            "schema_version": 1,
            "status": "failed",
            "reason": "completed ACT telemetry episode is missing video.mp4",
            "representative_episode": representative_path,
            "evidence_conflict": False,
            "query": query,
        }
        write_json(execution_dir / "execution_vqa_error.json", result)
        return result
    known_tools = {item.get("tool") for item in numeric_results}
    for episode in (tool_evaluation or {}).get("episodes", []):
        if episode.get("role") != "policy_under_evaluation":
            continue
        if not _same_telemetry_episode(episode, representative):
            continue
        result = episode.get("result", {})
        if result.get("tool") not in known_tools:
            numeric_results.append(result)
            known_tools.add(result.get("tool"))
    try:
        result = run_execution_vqa(
            provider=provider,
            model=model,
            video_path=episode_dir / "video.mp4",
            output_dir=execution_dir / "execution_vqa",
            numeric_tool_results=numeric_results,
            events_path=episode_dir / "events.jsonl",
            semantic_trace_path=episode_dir / "semantic_trace.npz",
            reference_scene=child_dir / "evidence/initial_head.png",
            query=query,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "representative_episode": representative_path,
            "evidence_conflict": False,
            "query": query,
        }
        write_json(execution_dir / "execution_vqa_error.json", result)
        return result
    result["status"] = "passed"
    result["representative_episode"] = representative_path
    result["artifacts"] = {
        key: (
            str(Path(value).resolve().relative_to(repo_root))
            if isinstance(value, str)
            and Path(value).is_absolute()
            and Path(value).resolve().is_relative_to(repo_root)
            else value
        )
        for key, value in result.get("artifacts", {}).items()
    }
    write_json(execution_dir / "execution_vqa/execution_vqa.json", result)
    return result


def compact_execution_vqa(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not result:
        return None
    return {
        "status": result.get("status"),
        "model_requested": result.get("model_requested"),
        "representative_episode": result.get("representative_episode"),
        "evidence_conflict": bool(result.get("evidence_conflict")),
        "observation": result.get("observation"),
        "selected_frames": result.get("selection", {}).get(
            "selected_frames", []
        ),
        "artifacts": result.get("artifacts", {}),
        "reason": result.get("reason"),
        "query": result.get("query"),
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
    vision = child_manifest.get("vision_validation", {})
    act = child_manifest.get("act_evaluation", {})
    expert = scene.get("expert", {})
    positions = child_manifest.get("position_samples", {})
    policy_success = read_policy_success(child_dir / "evaluation/_result.txt")
    trusted_tools = compact_trusted_tools(child_manifest)
    is_official = round_plan.get("route") == "official"
    if is_official:
        expert_batch = scene.get("expert_batch") or expert
        pipeline_passed = bool(
            child_manifest.get("status") == "completed_without_act"
            and taskgen_returncode == 0
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and expert_batch.get("passed")
            and child_manifest.get("trusted_tool_evaluation", {}).get(
                "episode_count"
            )
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in {"passed", "skipped"}
        )
    else:
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
            and execution_vqa.get("status") in {"passed", "skipped"}
        )
    return {
        "round_id": round_plan["round_id"],
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "taskgen_run_id": child_manifest.get("run_id"),
        "taskgen_returncode": taskgen_returncode,
        "execution": round_plan["execution"],
        "observations": {
            "execution_backend": "expert" if is_official else "ACT",
            "scene_alignment": bool(scene.get("rule_check", {}).get("passed")),
            "observed_color": vision.get("observed_color"),
            "expert_solvable": bool(
                (scene.get("expert_batch") or expert).get("passed")
            ),
            "act_pipeline_status": (
                None if is_official else bool(act.get("passed"))
            ),
            "policy_success": None if is_official else policy_success,
            "position_samples": positions.get("samples", []),
            "position_metrics": positions.get("metrics", {}),
            "trusted_tools": trusted_tools,
            "planned_tool": compact_tool_evaluation(tool_evaluation),
            "aggregate": compact_aggregate_result(aggregate_result),
            "execution_vqa": compact_execution_vqa(execution_vqa),
        },
        "pipeline_passed": pipeline_passed,
        "interpretation": (
            "official route 使用 expert backend，ACT/policy 字段为 N/A；"
            if is_official
            else "场景生成和评估流水线状态与 policy_success 分开报告；"
            "策略失败不会被误记为 pipeline failure。"
        ),
    }


def execute_round(
    repo_root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    provider: Any,
    toolgen_model: str,
    telemetry_profile: str = "balanced_v1",
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
    int,
]:
    round_id = round_plan["round_id"]
    command, run_id = build_taskgen_command(
        repo_root,
        evaluation_id,
        round_plan,
        text_model=text_model,
        vision_model=vision_model,
        base_url=base_url,
        gpu=gpu,
        max_reflections=max_reflections,
        telemetry_profile=telemetry_profile,
    )
    execution_dir = evaluation_dir / "execution" / round_id
    write_json(
        execution_dir / "taskgen_command.json",
        {"command": command, "child_run_id": run_id},
    )
    update_manifest(
        evaluation_dir,
        status=f"executing_{round_id}",
        active_child_run_id=run_id,
    )
    returncode = run_logged(
        command,
        cwd=repo_root,
        log_path=execution_dir / "taskgen.log",
    )
    child_dir = repo_root / "mea/generated_tasks" / run_id
    child_manifest_path = child_dir / "manifest.json"
    if not child_manifest_path.is_file():
        raise RuntimeError(f"child TaskGen manifest 不存在: {child_manifest_path}")
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    write_json(
        execution_dir / "child_run.json",
        {
            "run_id": run_id,
            "returncode": returncode,
            "manifest_path": str(child_manifest_path.relative_to(repo_root)),
            "status": child_manifest.get("status"),
        },
    )
    if child_manifest.get("status") in {
        "completed",
        "completed_without_act",
    } and returncode == 0:
        tool_evaluation = execute_tool_request(
            repo_root,
            child_dir,
            execution_dir / "planned_tool",
            round_plan["tool_request"],
            provider=provider,
            model=toolgen_model,
        )
    else:
        skip_reason = (
            f"child TaskGen exited with code {returncode}"
            if returncode != 0
            else "child TaskGen pipeline did not complete"
        )
        tool_evaluation = {
            "schema_version": 1,
            "status": "skipped",
            "requested_route": "auto",
            "route": None,
            "reference_tool": None,
            "tool_request": round_plan["tool_request"],
            "route_decision": {
                "status": "skipped",
                "requested_route": "auto",
                "resolved_route": None,
                "reason": skip_reason,
                "provider_required": None,
                "provider_called": False,
            },
            "source": {},
            "episodes": [],
            "validation": {"reason": skip_reason},
            "artifacts": {},
        }
        write_json(
            execution_dir / "planned_tool_skipped.json", tool_evaluation
        )
    aggregate_result = aggregate_round_results(
        round_plan,
        child_manifest,
        tool_evaluation,
        execution_dir / "aggregate_result.json",
    )
    execution_vqa = run_round_execution_vqa(
        repo_root=repo_root,
        child_manifest=child_manifest,
        child_dir=child_dir,
        tool_evaluation=tool_evaluation,
        execution_dir=execution_dir,
        provider=provider,
        model=vision_model,
        round_plan=round_plan,
    )
    round_summary = summarize_round(
        round_plan,
        child_manifest,
        child_dir,
        tool_evaluation,
        aggregate_result,
        execution_vqa,
        returncode,
    )
    write_json(evaluation_dir / "summary" / f"{round_id}.json", round_summary)
    return child_manifest, child_dir, round_summary, tool_evaluation, returncode


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
    episode_videos = sorted(
        str(path.relative_to(repo_root))
        for path in (child_dir / "evaluation").glob("episode*.mp4")
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

    round_execution = (
        Path("mea/evaluation_runs")
        / evaluation_id
        / "execution"
        / round_plan["round_id"]
    )
    execution_vqa_observation = round_summary["observations"].get(
        "execution_vqa"
    ) or {}
    execution_vqa_artifacts = execution_vqa_observation.get("artifacts") or {}
    execution_vqa_artifact = (
        execution_vqa_artifacts.get("result")
        or execution_vqa_artifacts.get("execution_vqa")
    )
    if not execution_vqa_artifact:
        if execution_vqa_observation.get("status") == "skipped":
            execution_vqa_artifact = str(
                round_execution / "execution_vqa_skipped.json"
            )
        elif execution_vqa_observation.get("status") == "failed":
            execution_vqa_artifact = str(
                round_execution / "execution_vqa_error.json"
            )
    return {
        "round_id": round_plan["round_id"],
        "child_run_id": child_manifest.get("run_id"),
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "seeds": round_plan["execution"]["seeds"],
        "num_episodes": round_plan["execution"]["num_episodes"],
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
            "complete_method_generated": static.get("load_actors_ast", {}).get(
                "complete_method_generated"
            ),
            "generated_color": static.get("load_actors_ast", {}).get(
                "generated_color"
            ),
        },
        "visual_observation": {
            "render_success": scene.get("render_success"),
            "aligned": vision.get("aligned"),
            "observed_color": vision.get("observed_color"),
            "unexpected_changes": vision.get("unexpected_changes"),
            "confidence": vision.get("confidence"),
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
        "trusted_tool_evaluation": {
            "artifact": trusted_tool_evaluation.get("artifact"),
            "episode_count": trusted_tool_evaluation.get("episode_count"),
            "episodes": compact_trusted_tools(child_manifest),
        },
        "artifacts": {
            "generated_task": str(child_relative / "task.py"),
            "scene_image": str(child_relative / "evidence/initial_head.png"),
            "vision_result": str(child_relative / "validation/vision.json"),
            "position_samples": str(
                child_relative / "validation/position_samples.json"
            ),
            "reflection_summary": str(child_relative / "reflection/summary.json"),
            "act_videos": episode_videos,
            "act_result": str(child_relative / "evaluation/_result.txt"),
            "trusted_tools": trusted_tool_evaluation.get("artifact"),
            "planned_tool": tool_evaluation.get("artifacts", {}).get(
                "tool_execution"
            ),
            "aggregate": str(
                round_execution / "aggregate_result.json"
            ),
            "execution_vqa": execution_vqa_artifact,
            "execution_vqa_query": str(
                round_execution / "execution_vqa_query.json"
            ),
            "execution_vqa_montage": execution_vqa_artifacts.get(
                "montage"
            ),
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
    total_episodes = sum(item["num_episodes"] for item in rounds)
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
    position_metrics = next(
        (
            item["observations"].get("position_metrics", {})
            for item in rounds
            if item["sub_aspect"] == "object_position"
        ),
        {},
    )
    evaluation_relative = Path("mea/evaluation_runs") / evaluation_id
    completed_template_ids = [item["round_plan"]["template_id"] for item in round_runs]
    remaining_template_ids = [
        item
        for item in plan.get("requested_template_ids", [])
        if item not in completed_template_ids
    ]
    decision_artifacts = [
        str(
            evaluation_relative
            / f"plan/decision_after_round_{round_number}.json"
        )
        for round_number in range(1, len(plan.get("round_decisions", [])) + 1)
    ]
    evidence_assessment_artifacts = [
        str(
            evaluation_relative
            / f"plan/evidence_after_round_{round_number}.json"
        )
        for round_number in range(1, len(plan.get("round_decisions", [])) + 1)
    ]
    history_path = repo_root / evaluation_relative / "plan/history_retrieval.json"
    history_retrieval = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else {"status": "missing", "matches": []}
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
    return {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "user_request": user_request,
        "plan": {
            "max_rounds": plan["max_rounds"],
            "executed_rounds": len(rounds),
            "planning_state": plan.get("planning_state"),
            "round_decisions": plan.get("round_decisions", []),
            "requested_template_ids": plan.get("requested_template_ids", []),
            "completed_template_ids": completed_template_ids,
            "remaining_template_ids": remaining_template_ids,
            "round_budget_remaining": max(
                int(plan["max_rounds"]) - len(rounds), 0
            ),
        },
        "rounds": rounds,
        "observations": {
            "execution_backends": execution_backends,
            "scene_alignment": all(
                item["observations"]["scene_alignment"] for item in rounds
            ),
            "observed_color_by_round": [
                item["observations"]["observed_color"] for item in rounds
            ],
            "expert_solvable": all(
                item["observations"]["expert_solvable"] for item in rounds
            ),
            "act_pipeline_status": (
                all(measured_act_statuses) if measured_act_statuses else None
            ),
            "policy_success": policy_success,
            "policy_success_by_round": [
                item["observations"]["policy_success"] for item in rounds
            ],
            "position_varied": position_metrics.get("position_varied"),
            "position_metrics": position_metrics,
            "pipeline_passed": all(
                item["observations"]["pipeline_passed"] for item in rounds
            ),
            "aggregate": compact_aggregate_result(evaluation_aggregate),
            "execution_vqa_conflict": any(
                bool(item.get("execution_vqa", {}).get("evidence_conflict"))
                for item in rounds
            ),
        },
        "total_episodes": total_episodes,
        "history_retrieval": history_retrieval,
        "limitations": {
            "bounded_three_round_prototype": True,
            "few_episodes_are_not_a_generalization_benchmark": True,
            "policy_result_is_not_pipeline_status": True,
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
            "summary": str(evaluation_relative / "summary/summary.json"),
            "aggregate": str(
                evaluation_relative / "summary/aggregate_result.json"
            ),
            "round_artifacts": [item["artifacts"] for item in rounds],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id")
    parser.add_argument(
        "--task-name",
        default="beat_block_hammer",
        help=(
            "Canonical RoboTwin task identity. beat_block_hammer uses the "
            "bounded Plan/TaskGen flow; other schema-backed tasks use the "
            "unchanged official expert-probe route."
        ),
    )
    parser.add_argument(
        "--task-module",
        help="Optional Python module for an official schema-backed task.",
    )
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument(
        "--model-profile",
        choices=available_model_profiles(),
        default="legacy",
        help=(
            "Named per-stage model defaults. Individual --*-model arguments "
            "override the selected profile."
        ),
    )
    parser.add_argument("--planner-model")
    parser.add_argument("--taskgen-model")
    parser.add_argument("--toolgen-model")
    parser.add_argument("--vision-model")
    parser.add_argument("--feedback-model")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum visual diagnosis-driven CodeGen repairs per TaskGen run.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--history-database",
        type=Path,
        help=(
            "SQLite planning-history cache. Defaults to "
            "mea/evaluation_runs/history.sqlite3 under --repo-root."
        ),
    )
    parser.add_argument("--history-limit", type=int, default=3)
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable cross-evaluation planning retrieval and indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    models = resolve_model_profile(
        args.model_profile,
        {
            "planner": args.planner_model,
            "taskgen": args.taskgen_model,
            "toolgen": args.toolgen_model,
            "vision": args.vision_model,
            "feedback": args.feedback_model,
        },
    )
    # The deterministic official planner can materialize --plan-only without
    # any provider credential. Full execution still creates the provider for
    # final Feedback (and for VQA when an ACT video exists).
    provider = None
    if args.task_name == "beat_block_hammer" or not args.plan_only:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=models["planner"],
            vision_model=models["vision"],
            timeout=180.0,
        )
    if args.task_name == "beat_block_hammer":
        assert provider is not None
        planner = PlanAgentPrototype(
            repo_root,
            provider,
            model=models["planner"],
        )
    else:
        planner = OfficialTaskPlanAgent(
            repo_root,
            task_name=args.task_name,
            task_module=args.task_module,
            start_seed=args.start_seed,
            num_episodes=args.num_episodes,
            telemetry_profile=args.telemetry_profile,
        )
    history_database = None
    history_context: list[dict[str, Any]] = []
    history_retrieval: dict[str, Any] = {
        "schema_version": 1,
        "status": "disabled" if args.no_history else "empty",
        "candidates": [],
    }
    history_path = (
        args.history_database.expanduser().resolve()
        if args.history_database
        else repo_root / "mea/evaluation_runs/history.sqlite3"
    )
    if not args.no_history:
        try:
            history_database = EvaluationHistoryDB(
                history_path,
                repo_root=repo_root,
            )
            history_retrieval = history_database.retrieve_similar(
                args.request,
                task_name=args.task_name,
                policy_name=(
                    "ACT" if args.task_name == "beat_block_hammer" else "expert"
                ),
                checkpoint_setting="demo_clean",
                limit=args.history_limit,
                exclude_evaluation_id=args.evaluation_id,
            )
            history_retrieval["status"] = "passed"
            history_context = list(history_retrieval.get("candidates", []))
        except Exception as exc:
            history_retrieval = {
                "schema_version": 1,
                "status": "failed",
                "candidates": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
    manifest = planner.plan(
        args.request,
        evaluation_id=args.evaluation_id,
        history_context=history_context,
        history_metadata={
            key: value
            for key, value in history_retrieval.items()
            if key != "candidates"
        },
    )
    evaluation_id = manifest["evaluation_id"]
    evaluation_dir = repo_root / "mea/evaluation_runs" / evaluation_id
    plan = manifest["plan"]
    update_manifest(
        evaluation_dir,
        model_profile=args.model_profile,
        resolved_models=models,
        history_database=(
            str(history_path.relative_to(repo_root))
            if history_path.is_relative_to(repo_root)
            else str(history_path)
        ),
        history_retrieval_status=history_retrieval.get("status"),
        task_name=args.task_name,
        task_module=args.task_module,
        telemetry_profile=args.telemetry_profile,
    )

    if args.plan_only:
        update_manifest(evaluation_dir, status="planned_only")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    assert provider is not None

    round_runs: list[dict[str, Any]] = []
    try:
        executed_rounds = 0
        while executed_rounds < len(plan["rounds"]):
            round_plan = plan["rounds"][executed_rounds]
            (
                child_manifest,
                child_dir,
                round_summary,
                tool_evaluation,
                returncode,
            ) = execute_round(
                repo_root,
                evaluation_dir,
                evaluation_id,
                round_plan,
                text_model=models["taskgen"],
                vision_model=models["vision"],
                base_url=args.base_url,
                gpu=args.gpu,
                max_reflections=args.max_reflections,
                provider=provider,
                toolgen_model=models["toolgen"],
                telemetry_profile=args.telemetry_profile,
            )
            round_runs.append(
                {
                    "round_plan": round_plan,
                    "child_manifest": child_manifest,
                    "child_dir": child_dir,
                    "round_summary": round_summary,
                    "tool_evaluation": tool_evaluation,
                    "returncode": returncode,
                }
            )
            executed_rounds += 1

            plan, decision = planner.decide_next_round(
                evaluation_id=evaluation_id,
                user_request=args.request,
                current_plan=plan,
                observation_history=[
                    item["round_summary"] for item in round_runs
                ],
            )
            if decision["action"] == "stop":
                break

        evaluation_aggregate = aggregate_evaluation_results(
            round_runs,
            evaluation_dir / "summary/aggregate_result.json",
        )
        summary = {
            "schema_version": 2,
            "evaluation_id": evaluation_id,
            "status": (
                "completed"
                if round_runs
                and all(item["round_summary"]["pipeline_passed"] for item in round_runs)
                else "completed_with_pipeline_failure"
            ),
            "rounds": [item["round_summary"] for item in round_runs],
            "aggregate": compact_aggregate_result(evaluation_aggregate),
        }
        write_json(evaluation_dir / "summary/summary.json", summary)
        evidence = build_evidence_bundle(
            repo_root,
            evaluation_id,
            args.request,
            plan,
            round_runs,
            evaluation_aggregate,
        )
        write_json(evaluation_dir / "summary/evidence_bundle.json", evidence)
        update_manifest(
            evaluation_dir,
            status="generating_feedback",
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            summary=summary,
        )
        feedback = FeedbackAgent(
            repo_root,
            provider,
            model=models["feedback"],
        ).generate(
            evidence,
            output_dir=evaluation_dir / "feedback",
        )
        report_path = evaluation_dir / "evaluation_report.md"
        report_path.write_text(
            render_evaluation_report(evidence, feedback),
            encoding="utf-8",
        )
        update_manifest(
            evaluation_dir,
            status=summary["status"],
            lifecycle_status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            feedback_path="feedback/feedback.json",
            report_path="evaluation_report.md",
            child_run_ids=[
                item["child_manifest"].get("run_id") for item in round_runs
            ],
            summary=summary,
            feedback=feedback,
        )
        history_index = {
            "status": "disabled" if args.no_history else "not_available"
        }
        if history_database is not None:
            try:
                history_index = {
                    "status": "passed",
                    **history_database.index_evaluation_dir(evaluation_dir),
                }
            except Exception as exc:
                history_index = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        update_manifest(evaluation_dir, history_index=history_index)
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "child_run_ids": [
                        item["child_manifest"].get("run_id") for item in round_runs
                    ],
                    "summary": summary,
                    "feedback": feedback,
                    "history_retrieval": {
                        "status": history_retrieval.get("status"),
                        "selected_count": len(history_context),
                    },
                    "history_index": history_index,
                    "report_path": str(report_path.relative_to(repo_root)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        update_manifest(
            evaluation_dir,
            status="failed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
