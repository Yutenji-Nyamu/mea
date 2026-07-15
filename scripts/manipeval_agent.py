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

from mea.feedback import FeedbackAgent, render_evaluation_report
from mea.planner import PlanAgentPrototype
from mea.providers import OpenAICompatibleProvider


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
) -> tuple[list[str], str]:
    run_id = child_run_id(evaluation_id, round_plan["round_id"])
    execution = round_plan["execution"]
    seed = execution["seeds"][0]
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
        "beat_block_hammer",
        "--mode",
        round_plan["route"],
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
        "--probe",
        "--vision-check",
        "--expert",
        "--run-act",
        "--max-reflections",
        str(max_reflections),
    ]
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


def summarize_round(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
) -> dict[str, Any]:
    scene = child_manifest.get("scene_validation", {})
    vision = child_manifest.get("vision_validation", {})
    act = child_manifest.get("act_evaluation", {})
    expert = scene.get("expert", {})
    positions = child_manifest.get("position_samples", {})
    policy_success = read_policy_success(child_dir / "evaluation/_result.txt")
    trusted_tools = compact_trusted_tools(child_manifest)
    pipeline_passed = bool(
        child_manifest.get("status") == "completed"
        and scene.get("rule_check", {}).get("passed")
        and vision.get("passed")
        and expert.get("passed")
        and positions.get("passed")
        and act.get("passed")
    )
    return {
        "round_id": round_plan["round_id"],
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "taskgen_run_id": child_manifest.get("run_id"),
        "execution": round_plan["execution"],
        "observations": {
            "scene_alignment": bool(scene.get("rule_check", {}).get("passed")),
            "observed_color": vision.get("observed_color"),
            "expert_solvable": bool(expert.get("passed")),
            "act_pipeline_status": bool(act.get("passed")),
            "policy_success": policy_success,
            "position_samples": positions.get("samples", []),
            "position_metrics": positions.get("metrics", {}),
            "trusted_tools": trusted_tools,
        },
        "pipeline_passed": pipeline_passed,
        "interpretation": (
            "场景生成和评估流水线状态与 policy_success 分开报告；"
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
) -> tuple[dict[str, Any], Path, dict[str, Any], int]:
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
    round_summary = summarize_round(round_plan, child_manifest, child_dir)
    write_json(evaluation_dir / "summary" / f"{round_id}.json", round_summary)
    return child_manifest, child_dir, round_summary, returncode


def _round_evidence(
    repo_root: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    round_summary: dict[str, Any],
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
            **round_summary["observations"],
            "pipeline_passed": round_summary["pipeline_passed"],
        },
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
            "child_manifest": str(child_relative / "manifest.json"),
        },
    }


def build_evidence_bundle(
    repo_root: Path,
    evaluation_id: str,
    user_request: str,
    plan: dict[str, Any],
    round_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    rounds = [
        _round_evidence(
            repo_root,
            evaluation_id,
            item["round_plan"],
            item["child_manifest"],
            item["child_dir"],
            item["round_summary"],
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
    return {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "user_request": user_request,
        "plan": {
            "max_rounds": plan["max_rounds"],
            "executed_rounds": len(rounds),
            "planning_state": plan.get("planning_state"),
            "round_decisions": plan.get("round_decisions", []),
        },
        "rounds": rounds,
        "observations": {
            "scene_alignment": all(
                item["observations"]["scene_alignment"] for item in rounds
            ),
            "observed_color_by_round": [
                item["observations"]["observed_color"] for item in rounds
            ],
            "expert_solvable": all(
                item["observations"]["expert_solvable"] for item in rounds
            ),
            "act_pipeline_status": all(
                item["observations"]["act_pipeline_status"] for item in rounds
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
        },
        "total_episodes": total_episodes,
        "limitations": {
            "bounded_two_round_prototype": True,
            "three_episodes_are_not_a_generalization_benchmark": True,
            "policy_result_is_not_pipeline_status": True,
        },
        "artifacts": {
            "evaluation_plan": str(
                evaluation_relative / "plan/evaluation_plan.json"
            ),
            "round_2_decision": str(
                evaluation_relative / "plan/round_2_decision.json"
            ),
            "summary": str(evaluation_relative / "summary/summary.json"),
            "round_artifacts": [item["artifacts"] for item in rounds],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id")
    parser.add_argument("--planner-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--taskgen-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--vision-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--feedback-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum visual diagnosis-driven CodeGen repairs per TaskGen run.",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=args.planner_model,
        vision_model=args.vision_model,
        timeout=180.0,
    )
    planner = PlanAgentPrototype(
        repo_root,
        provider,
        model=args.planner_model,
    )
    manifest = planner.plan(args.request, evaluation_id=args.evaluation_id)
    evaluation_id = manifest["evaluation_id"]
    evaluation_dir = repo_root / "mea/evaluation_runs" / evaluation_id
    plan = manifest["plan"]

    if args.plan_only:
        update_manifest(evaluation_dir, status="planned_only")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    round_runs: list[dict[str, Any]] = []
    try:
        round_1_plan = plan["rounds"][0]
        child_manifest, child_dir, round_1_summary, returncode = execute_round(
            repo_root,
            evaluation_dir,
            evaluation_id,
            round_1_plan,
            text_model=args.taskgen_model,
            vision_model=args.vision_model,
            base_url=args.base_url,
            gpu=args.gpu,
            max_reflections=args.max_reflections,
        )
        round_runs.append(
            {
                "round_plan": round_1_plan,
                "child_manifest": child_manifest,
                "child_dir": child_dir,
                "round_summary": round_1_summary,
                "returncode": returncode,
            }
        )

        plan, decision = planner.decide_next_round(
            evaluation_id=evaluation_id,
            user_request=args.request,
            current_plan=plan,
            round_1_observation=round_1_summary,
        )
        if decision["action"] == "continue":
            round_2_plan = decision["next_round"]
            child_manifest, child_dir, round_2_summary, returncode = execute_round(
                repo_root,
                evaluation_dir,
                evaluation_id,
                round_2_plan,
                text_model=args.taskgen_model,
                vision_model=args.vision_model,
                base_url=args.base_url,
                gpu=args.gpu,
                max_reflections=args.max_reflections,
            )
            round_runs.append(
                {
                    "round_plan": round_2_plan,
                    "child_manifest": child_manifest,
                    "child_dir": child_dir,
                    "round_summary": round_2_summary,
                    "returncode": returncode,
                }
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
        }
        write_json(evaluation_dir / "summary/summary.json", summary)
        evidence = build_evidence_bundle(
            repo_root,
            evaluation_id,
            args.request,
            plan,
            round_runs,
        )
        write_json(evaluation_dir / "summary/evidence_bundle.json", evidence)
        update_manifest(
            evaluation_dir,
            status="generating_feedback",
            summary_path="summary/summary.json",
            evidence_path="summary/evidence_bundle.json",
            summary=summary,
        )
        feedback = FeedbackAgent(
            repo_root,
            provider,
            model=args.feedback_model,
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
            status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            evidence_path="summary/evidence_bundle.json",
            feedback_path="feedback/feedback.json",
            report_path="evaluation_report.md",
            child_run_ids=[
                item["child_manifest"].get("run_id") for item in round_runs
            ],
            summary=summary,
            feedback=feedback,
        )
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "child_run_ids": [
                        item["child_manifest"].get("run_id") for item in round_runs
                    ],
                    "summary": summary,
                    "feedback": feedback,
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
