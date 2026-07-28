#!/usr/bin/env python3
"""Replay bounded Execution VQA over a completed rollout.

This paper-evidence utility never starts TaskGen, a simulator, or a policy
rollout.  Its default query is derived from the cached round's task, template,
sub-aspect, and Tool request through the production VQA query builder.  An
optional run-local question may extend that query after the normal validator
accepts it.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.execution_vqa import (
    build_execution_vqa_query,
    is_run_local_phenomenon_id,
    validate_run_local_question_spec,
)
from mea.providers import OpenAICompatibleProvider
from scripts.manipeval_agent import run_round_execution_vqa, write_json


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required replay artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"replay artifact must be an object: {path}")
    return value


def build_replay_query(
    round_plan: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the exact bounded query that the replay will execute."""

    task_name = round_plan.get("task_name") or child_manifest.get("task_name")
    tool_proposal = round_plan.get("tool_proposal") or {}
    return build_execution_vqa_query(
        task_name=str(task_name) if task_name else None,
        template_id=round_plan.get("template_id"),
        sub_aspect=round_plan.get("sub_aspect"),
        tool_contract=round_plan.get("tool_request"),
        proposed_phenomenon_ids=tool_proposal.get("vqa_phenomenon_ids"),
        proposed_question_specs=tool_proposal.get("vqa_question_specs"),
    )


def extend_round_with_run_local_question(
    round_plan: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    question_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a replay plan whose default query is extended by one local item."""

    normalized_question = validate_run_local_question_spec(question_spec)
    base_query = build_replay_query(round_plan, child_manifest)
    phenomenon_ids = list(base_query["phenomenon_ids"])
    if normalized_question["id"] not in phenomenon_ids:
        phenomenon_ids.append(normalized_question["id"])
    local_specs = [
        deepcopy(question)
        for question in base_query["questions"]
        if is_run_local_phenomenon_id(question["id"])
    ]
    local_specs.append(normalized_question)

    result = deepcopy(dict(round_plan))
    task_name = result.get("task_name") or child_manifest.get("task_name")
    tool_request = result.get("tool_request") or {}
    result["tool_proposal"] = {
        "schema_version": 2,
        "proposal_id": (
            f"{result.get('round_id', 'cached_round')}.execution_vqa_replay"
        ),
        "task_name": str(task_name),
        "aspect_id": str(
            result.get("aspect_id")
            or result.get("sub_aspect")
            or "open_world.visual_observation"
        ),
        "evaluation_goal": "replay a bounded run-local visual observation",
        "metric": str(tool_request.get("metric") or "official_check_success"),
        "question": str(
            tool_request.get("question")
            or "Did the cached rollout satisfy the selected task evidence?"
        ),
        "vqa_phenomenon_ids": phenomenon_ids,
        "vqa_question_specs": local_specs,
        "reuse_first": True,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--round-id", default="round_1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--vision-model", default="gpt-5.6-luna")
    parser.add_argument("--question-id")
    parser.add_argument("--question")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.question_id is None) != (args.question is None):
        raise SystemExit("--question-id and --question must be supplied together")

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

    question_spec = None
    if args.question_id is not None:
        question_spec = validate_run_local_question_spec(
            {
                "id": args.question_id,
                "question_type": "visible_state_change",
                "target_role": "task_target",
                "question": args.question,
                "visual_scope": "rollout_change",
                "numeric_authority": "official_check_success_is_authoritative",
            }
        )
        round_plan = extend_round_with_run_local_question(
            round_plan,
            child_manifest,
            question_spec,
        )

    replay_query = build_replay_query(round_plan, child_manifest)
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
    if result.get("query") != replay_query:
        raise RuntimeError("replay query drifted between preparation and execution")
    manifest = {
        "schema_version": 1,
        "status": result.get("status"),
        "evidence_kind": "cached_rollout_dynamic_vqa_replay",
        "development_evidence_only": True,
        "source_evaluation_id": args.evaluation_id,
        "source_round_id": args.round_id,
        "source_child_run_id": child_record["run_id"],
        "act_rollouts_started": 0,
        "query": replay_query,
        "question_spec": question_spec,
        "model_requested": args.vision_model,
        "result": result,
    }
    write_json(output_dir / "replay_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if result.get("status") != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
