#!/usr/bin/env python3
"""Replay Dynamic Execution VQA on an existing completed rollout.

This entrypoint never starts TaskGen, the simulator, or ACT.  It is intended
for cheaply validating a new bounded run-local question against already
recorded video, events, telemetry, Rule Tool results, and the original round
plan.  The output is always marked as replay/development evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.execution_vqa import validate_run_local_question_spec
from mea.providers import OpenAICompatibleProvider
from scripts.manipeval_agent import run_round_execution_vqa, write_json


def _read_object(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required replay artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"replay artifact must be an object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--round-id", default="round_1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--vision-model", default="gpt-5.6-luna")
    parser.add_argument(
        "--question-id",
        default="run_local.replay.click_bell.object_position.progress",
    )
    parser.add_argument(
        "--question",
        default=(
            "Does the cached rollout visibly show the robot making "
            "task-relevant progress toward the bell at this tested position?"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    evaluation_dir = repo_root / "mea/evaluation_runs" / args.evaluation_id
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    plan = _read_object(evaluation_dir / "plan/evaluation_plan.json")
    round_plan = next(
        (
            dict(item)
            for item in plan.get("rounds", [])
            if item.get("round_id") == args.round_id
        ),
        None,
    )
    if round_plan is None:
        raise SystemExit(f"round not found in evaluation plan: {args.round_id}")
    execution_dir = evaluation_dir / "execution" / args.round_id
    child_record = _read_object(execution_dir / "child_run.json")
    child_dir = repo_root / "mea/generated_tasks" / child_record["run_id"]
    child_manifest = _read_object(child_dir / "manifest.json")
    tool_evaluation = _read_object(
        execution_dir / "planned_tool/tool_execution.json"
    )
    question = validate_run_local_question_spec(
        {
            "id": args.question_id,
            "question_type": "visible_state_change",
            "target_role": "task_target",
            "question": args.question,
            "visual_scope": "rollout_change",
            "numeric_authority": "official_check_success_is_authoritative",
        }
    )
    task_name = str(round_plan.get("task_name") or child_manifest.get("task_name"))
    if task_name != "click_bell":
        raise SystemExit("the first replay profile currently supports click_bell")
    round_plan["tool_proposal"] = {
        "schema_version": 2,
        "proposal_id": f"{args.round_id}.execution_vqa_replay",
        "task_name": task_name,
        "aspect_id": str(
            round_plan.get("aspect_id") or round_plan.get("sub_aspect")
        ),
        "evaluation_goal": "replay a bounded run-local visual observation",
        "metric": round_plan["tool_request"]["metric"],
        "question": round_plan["tool_request"]["question"],
        "vqa_phenomenon_ids": ["bell_visibly_pressed", question["id"]],
        "vqa_question_specs": [question],
        "reuse_first": True,
    }
    output_dir.mkdir(parents=True)
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        vision_model=args.vision_model,
    )
    result = run_round_execution_vqa(
        repo_root=repo_root,
        child_manifest=child_manifest,
        child_dir=child_dir,
        tool_evaluation=tool_evaluation,
        execution_dir=output_dir,
        provider=provider,
        model=args.vision_model,
        round_plan=round_plan,
    )
    manifest = {
        "schema_version": 1,
        "status": result.get("status"),
        "evidence_kind": "cached_rollout_dynamic_vqa_replay",
        "development_evidence_only": True,
        "source_evaluation_id": args.evaluation_id,
        "source_round_id": args.round_id,
        "source_child_run_id": child_record["run_id"],
        "act_rollouts_started": 0,
        "question_spec": question,
        "model_requested": args.vision_model,
        "result": result,
    }
    write_json(output_dir / "replay_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if result.get("status") != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
