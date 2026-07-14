"""Plan and execute one single-round, blue-block MEA evaluation."""

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


def child_run_id(evaluation_id: str) -> str:
    return f"run_{evaluation_id.removeprefix('eval_')}_round_1"


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
    run_id = child_run_id(evaluation_id)
    seed = round_plan["execution"]["seeds"][0]
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


def summarize_child(
    evaluation_id: str,
    plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
) -> dict[str, Any]:
    scene = child_manifest.get("scene_validation", {})
    vision = child_manifest.get("vision_validation", {})
    act = child_manifest.get("act_evaluation", {})
    expert = scene.get("expert", {})
    policy_success = read_policy_success(child_dir / "evaluation/_result.txt")
    pipeline_passed = bool(
        child_manifest.get("status") == "completed"
        and scene.get("rule_check", {}).get("passed")
        and vision.get("passed")
        and expert.get("passed")
        and act.get("passed")
    )
    round_plan = plan["rounds"][0]
    return {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "status": "completed" if pipeline_passed else "failed",
        "rounds": [
            {
                "round_id": round_plan["round_id"],
                "sub_aspect": round_plan["sub_aspect"],
                "task_instruction": round_plan["task_instruction"],
                "taskgen_run_id": child_manifest.get("run_id"),
                "observations": {
                    "scene_alignment": bool(
                        scene.get("rule_check", {}).get("passed")
                    ),
                    "observed_color": vision.get("observed_color"),
                    "expert_solvable": bool(expert.get("passed")),
                    "act_pipeline_status": bool(act.get("passed")),
                    "policy_success": policy_success,
                },
                "pipeline_passed": pipeline_passed,
                "interpretation": (
                    "场景生成与评估流水线通过；policy_success 单独报告，"
                    "不把策略失败误判为 pipeline failure。"
                ),
            }
        ],
    }


def build_evidence_bundle(
    repo_root: Path,
    evaluation_id: str,
    user_request: str,
    plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    round_plan = plan["rounds"][0]
    round_summary = summary["rounds"][0]
    static = child_manifest.get("static_validation", {})
    scene = child_manifest.get("scene_validation", {})
    vision = child_manifest.get("vision_validation", {})
    reflection = child_manifest.get("visual_self_reflection", {})
    act = child_manifest.get("act_evaluation", {})
    retrieval = child_manifest.get("task_retrieval") or {}
    variant_spec_path = child_dir / "variant_spec.json"
    variant_spec = (
        json.loads(variant_spec_path.read_text(encoding="utf-8"))
        if variant_spec_path.is_file()
        else None
    )

    child_relative = child_dir.relative_to(repo_root)
    evaluation_relative = Path("mea/evaluation_runs") / evaluation_id
    observations = dict(round_summary["observations"])
    observations["pipeline_passed"] = round_summary["pipeline_passed"]
    return {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "child_run_id": child_manifest.get("run_id"),
        "user_request": user_request,
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "seed": round_plan["execution"]["seeds"][0],
        "num_episodes": round_plan["execution"]["num_episodes"],
        "task_retrieval": {
            "catalog_size": retrieval.get("catalog_size"),
            "selected_tasks": retrieval.get("selected_tasks", []),
            "reasoning": retrieval.get("reasoning"),
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
            "attempts": [
                {
                    "attempt_index": item.get("attempt_index"),
                    "passed": item.get("observation", {}).get("passed"),
                    "probe_passed": item.get("observation", {}).get(
                        "probe_passed"
                    ),
                    "observed_color": item.get("observation", {})
                    .get("vision", {})
                    .get("observed_color"),
                    "diagnosis": item.get("observation", {})
                    .get("vision", {})
                    .get("diagnosis"),
                    "suggestions": item.get("observation", {})
                    .get("vision", {})
                    .get("suggestions", []),
                    "repair_installed": bool(item.get("repair", {}).get("installed")),
                }
                for item in reflection.get("attempts", [])
            ],
        },
        "observations": observations,
        "limitations": {
            "single_round": True,
            "single_episode": round_plan["execution"]["num_episodes"] == 1,
            "policy_result_is_not_generalization_conclusion": True,
        },
        "artifacts": {
            "evaluation_plan": str(
                evaluation_relative / "plan/evaluation_plan.json"
            ),
            "task_catalog": str(child_relative / "generation/task_catalog.json"),
            "retrieval": str(child_relative / "generation/retrieval.json"),
            "generated_task": str(child_relative / "task.py"),
            "scene_image": str(child_relative / "evidence/initial_head.png"),
            "vision_result": str(child_relative / "validation/vision.json"),
            "reflection_summary": str(child_relative / "reflection/summary.json"),
            "act_video": str(child_relative / "evaluation/episode0.mp4"),
            "act_result": str(child_relative / "evaluation/_result.txt"),
            "child_manifest": str(child_relative / "manifest.json"),
        },
        "act_process_returncode": act.get("returncode"),
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
    manifest = PlanAgentPrototype(
        repo_root,
        provider,
        model=args.planner_model,
    ).plan(args.request, evaluation_id=args.evaluation_id)
    evaluation_id = manifest["evaluation_id"]
    evaluation_dir = repo_root / "mea/evaluation_runs" / evaluation_id
    plan = manifest["plan"]

    if args.plan_only:
        update_manifest(evaluation_dir, status="planned_only")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    command, run_id = build_taskgen_command(
        repo_root,
        evaluation_id,
        plan["rounds"][0],
        text_model=args.taskgen_model,
        vision_model=args.vision_model,
        base_url=args.base_url,
        gpu=args.gpu,
        max_reflections=args.max_reflections,
    )
    write_json(
        evaluation_dir / "execution/taskgen_command.json",
        {"command": command, "child_run_id": run_id},
    )
    update_manifest(
        evaluation_dir,
        status="executing_round_1",
        child_run_id=run_id,
        execution_started_at=datetime.now().astimezone().isoformat(),
    )

    try:
        returncode = run_logged(
            command,
            cwd=repo_root,
            log_path=evaluation_dir / "execution/taskgen.log",
        )
        child_dir = repo_root / "mea/generated_tasks" / run_id
        child_manifest_path = child_dir / "manifest.json"
        if not child_manifest_path.is_file():
            raise RuntimeError(f"child TaskGen manifest 不存在: {child_manifest_path}")
        child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
        write_json(
            evaluation_dir / "execution/child_run.json",
            {
                "run_id": run_id,
                "returncode": returncode,
                "manifest_path": str(child_manifest_path.relative_to(repo_root)),
                "status": child_manifest.get("status"),
            },
        )
        if returncode != 0:
            raise RuntimeError(f"child TaskGen 失败，returncode={returncode}")

        summary = summarize_child(
            evaluation_id,
            plan,
            child_manifest,
            child_dir,
        )
        write_json(evaluation_dir / "summary/summary.json", summary)
        if summary["status"] != "completed":
            raise RuntimeError(f"single-round evaluation 未通过: {summary}")

        evidence = build_evidence_bundle(
            repo_root,
            evaluation_id,
            args.request,
            plan,
            child_manifest,
            child_dir,
            summary,
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
            summary=summary,
            feedback=feedback,
        )
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "child_run_id": run_id,
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
