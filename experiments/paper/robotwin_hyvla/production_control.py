"""Run one Hy-VLA official control through the production MEA round lifecycle.

The Hy-VLA policy server must already be running in its isolated environment.
This script intentionally bypasses open-Query admission and planning, but it
does not bypass RuntimeTaskBinding, MethodRuntime, RoundExecutor, Rule Tool,
Aggregate, or round-summary generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mea.planner.claim_first_initial import (
    build_plan_agent_control_round,
    build_plan_agent_execution_binding,
)
from mea.planner.runtime_task_binding import (
    build_hyvla_policy_spec,
    build_runtime_open_world_evaluation_target,
)
from mea.round_executor import RoundExecutionRequest
from mea.robotwin.production_round_executor import (
    build_production_round_executor,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _evaluation_id() -> str:
    return "eval_hyvla_control_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one official RoboTwin control through the production "
            "MEA RoundExecutor using an already-running Hy-VLA server."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--task", default="press_stapler")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--evaluation-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python-env", type=Path, required=True)
    parser.add_argument("--server-ready-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18781)
    parser.add_argument("--telemetry-profile", default="balanced_v1")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in [1, 65535]")

    repo_root = args.repo_root.expanduser().resolve()
    ready_path = args.server_ready_file.expanduser().resolve()
    try:
        server_ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read --server-ready-file: {exc}")
    expected_server = {
        "host": "127.0.0.1",
        "port": args.port,
        "source": str(args.source.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
    }
    for field, expected in expected_server.items():
        if server_ready.get(field) != expected:
            parser.error(
                f"server ready-file {field} does not match the requested binding"
            )
    evaluation_id = args.evaluation_id or _evaluation_id()
    evaluation_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else repo_root / "mea" / "evaluation_runs" / evaluation_id
    )
    if evaluation_dir.exists():
        parser.error(f"output directory already exists: {evaluation_dir}")
    evaluation_dir.mkdir(parents=True)

    policy_spec = build_hyvla_policy_spec(
        args.checkpoint,
        source_dir=args.source,
        python_env=args.python_env,
    )
    runtime_target = build_runtime_open_world_evaluation_target(
        repo_root,
        args.task,
        max_rounds=1,
        policy_spec=policy_spec,
    )
    execution_binding = build_plan_agent_execution_binding(
        start_seed=args.seed,
        num_episodes=1,
        execution_backend="act",
    )
    query = (
        f"Can Hy-VLA solve the unchanged official RoboTwin {args.task} task?"
    )
    round_plan = build_plan_agent_control_round(
        runtime_target,
        query,
        execution_binding=execution_binding,
        telemetry_profile=args.telemetry_profile,
    )
    _write_json(
        evaluation_dir / "manifest.json",
        {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "status": "bound_official_control",
            "entrypoint": "experiments/paper/robotwin_hyvla/production_control.py",
            "admission_bypassed": True,
            "round_lifecycle": "production_round_executor",
            "policy_backend": "hyvla",
            "task_name": args.task,
            "seed": args.seed,
            "policy_server_ready": server_ready,
            "runtime_target": runtime_target,
            "round_plan": round_plan,
        },
    )
    _write_json(evaluation_dir / "plan" / "round_1.json", round_plan)

    result = build_production_round_executor().execute(
        RoundExecutionRequest(
            repo_root=repo_root,
            evaluation_dir=evaluation_dir,
            evaluation_id=evaluation_id,
            round_plan=round_plan,
            text_model="not_used_for_official_control",
            vision_model="not_used_for_official_control",
            base_url=None,
            gpu=0,
            max_reflections=1,
            provider=None,
            toolgen_model="not_used_for_official_success_reuse",
            telemetry_profile=args.telemetry_profile,
            policy_backend="hyvla",
            runtime_target=runtime_target,
            policy_server_port=args.port,
        )
    )
    observations = result.round_summary.get("observations") or {}
    output = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "task": args.task,
        "seed": args.seed,
        "policy_backend": "hyvla",
        "admission_bypassed": True,
        "round_lifecycle": "production_round_executor",
        "returncode": result.returncode,
        "child_status": result.child_manifest.get("status"),
        "pipeline_passed": result.round_summary.get("pipeline_passed"),
        "policy_success": observations.get("policy_success"),
        "policy_outcome": observations.get("policy_outcome"),
        "semantic_telemetry_ready": observations.get(
            "semantic_telemetry_ready"
        ),
        "tool_status": result.tool_evaluation.get("status"),
        "method_runtime": observations.get("method_runtime"),
        "artifacts": {
            "evaluation_dir": str(evaluation_dir),
            "child_dir": str(result.child_dir),
            "round_summary": str(
                evaluation_dir / "summary" / "round_1.json"
            ),
        },
    }
    output_path = evaluation_dir / "production_control_result.json"
    _write_json(output_path, output)
    print("PRODUCTION_CONTROL_RESULT_JSON=" + json.dumps(output, sort_keys=True))
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
