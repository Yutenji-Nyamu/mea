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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id")
    parser.add_argument("--planner-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--taskgen-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--vision-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--gpu", type=int, default=0)
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
        update_manifest(
            evaluation_dir,
            status=summary["status"],
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            summary=summary,
        )
        if summary["status"] != "completed":
            raise RuntimeError(f"single-round evaluation 未通过: {summary}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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
