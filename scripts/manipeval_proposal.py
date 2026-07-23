#!/usr/bin/env python3
"""Generate one Task/Tool proposal and optionally materialize its Task side."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.planner import BoundTaskPlanSession, build_act_catalog
from mea.proposal_agent import BoundedProposalAgent
from mea.proposals import tool_request_from_proposal
from mea.providers import OpenAICompatibleProvider, resolve_model_profile
from mea.taskgen.capabilities import get_capability


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--request", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--aspect-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=100405)
    parser.add_argument("--model-profile", default="economy")
    parser.add_argument("--planner-model")
    parser.add_argument("--taskgen-model")
    parser.add_argument("--vision-model")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--experimental-bbh-success-spec",
        action="store_true",
        help=(
            "Explicitly enable the bounded BBH appearance + experimental "
            "SuccessSpec v2 capability. Other task/aspect pairs fail closed."
        ),
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help=(
            "Run TaskGen materialization, expert probe, and production Task "
            "acceptance; never starts ACT."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir.is_absolute()
        else (root / args.output_dir).resolve()
    )
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise SystemExit("--output-dir must remain inside --repo-root") from exc
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)

    models = resolve_model_profile(
        args.model_profile,
        {
            "planner": args.planner_model,
            "taskgen": args.taskgen_model,
            "toolgen": None,
            "vision": args.vision_model,
            "feedback": None,
        },
    )
    catalog = build_act_catalog(root)
    session = BoundTaskPlanSession.from_catalog(
        catalog, args.task_name, max_rounds=1
    )
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=models["planner"],
        vision_model=models["vision"],
        timeout=180.0,
    )
    agent = BoundedProposalAgent(provider, model=models["taskgen"])
    bundle = agent.propose(
        args.request,
        target=session.target,
        aspect_id=args.aspect_id,
        require_novel_changes=True,
        capability_mode=(
            "experimental_success_bounded"
            if args.experimental_bbh_success_spec
            else None
        ),
    )
    (output / "proposal_prompt.md").write_text(
        agent.last_prompt or "", encoding="utf-8"
    )
    for index, response in enumerate(agent.last_responses, start=1):
        (output / f"proposal_response_{index}.txt").write_text(
            response + "\n", encoding="utf-8"
        )
    _write_json(output / "proposal_bundle.json", bundle)
    tool_resolution = {
        "schema_version": 1,
        "status": "validated_and_routed_not_executed",
        "tool_request": tool_request_from_proposal(bundle["tool_proposal"]),
        "route_preview": bundle["tool_route_preview"],
        "materialized": False,
        "act_rollouts_started": 0,
    }
    _write_json(output / "tool_resolution.json", tool_resolution)

    taskgen_result = None
    if args.materialize:
        task = bundle["task_proposal"]
        capability = get_capability(task["task_name"], task["capability_id"])
        generation_mode = capability["generation_mode"]
        route = {
            "bounded_variant_overlay": "reuse",
            "reuse": "reuse",
            "force_codegen": "force_codegen",
        }[generation_mode]
        run_id = args.run_id or (
            "run_proposal_"
            + task["proposal_id"].replace(".", "_").replace("-", "_")
        )
        command = [
            sys.executable,
            str(root / "scripts/manipeval_taskgen.py"),
            "--repo-root",
            str(root),
            "--request",
            args.request,
            "--run-id",
            run_id,
            "--task-name",
            args.task_name,
            "--mode",
            route,
            "--task-proposal-json",
            json.dumps(task, ensure_ascii=False, separators=(",", ":")),
            "--text-model",
            models["taskgen"],
            "--vision-model",
            models["vision"],
            "--seed",
            str(args.seed),
            "--num-episodes",
            "1",
            "--expert",
            "--accept-task-only",
        ]
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        process = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (output / "taskgen.log").write_text(process.stdout, encoding="utf-8")
        taskgen_result = {
            "returncode": process.returncode,
            "run_id": run_id,
            "route": route,
            "act_rollouts_started": 0,
            "production_acceptance": None,
            "act_runtime_eligible": (
                False
                if task.get("schema_version") == 2
                else None
            ),
            "log": str((output / "taskgen.log").relative_to(root)).replace("\\", "/"),
        }
        if process.returncode != 0:
            _write_json(output / "taskgen_result.json", taskgen_result)
            raise SystemExit(process.returncode)
        child_manifest_path = (
            root / "mea/generated_tasks" / run_id / "manifest.json"
        )
        child_manifest = json.loads(
            child_manifest_path.read_text(encoding="utf-8")
        )
        acceptance = child_manifest.get("task_generation_acceptance")
        if (
            not isinstance(acceptance, dict)
            or acceptance.get("status") != "accepted"
            or acceptance.get("scope") != "task_generation_only_no_act"
            or acceptance.get("act_rollouts_started_before_acceptance") != 0
        ):
            raise SystemExit(
                "TaskGen subprocess returned success without a valid 0-ACT "
                "production acceptance record"
            )
        taskgen_result["production_acceptance"] = acceptance
        taskgen_result["act_runtime_eligible"] = acceptance.get(
            "act_runtime_eligible"
        )
        _write_json(output / "taskgen_result.json", taskgen_result)

    result = {
        "schema_version": 1,
        "status": (
            "task_materialized_and_accepted_no_act_tool_routed"
            if taskgen_result
            else "task_and_tool_proposed"
        ),
        "evaluation_target": session.target,
        "proposal_bundle": bundle,
        "taskgen": taskgen_result,
        "tool": tool_resolution,
        "limitations": {
            "act_rollouts_started": 0,
            "experimental_v2_act_runtime_eligible": (
                False
                if bundle["task_proposal"]["schema_version"] == 2
                else None
            ),
            "taskgen_acceptance_only": bool(taskgen_result),
            "policy_performance_evidence": False,
            "development_validation_only": not bool(taskgen_result),
            "tool_execution_status": "not_executed",
            "tool_execution_blocker": (
                "dual-label runtime is required before experimental "
                "SuccessSpec evaluation"
                if bundle["task_proposal"]["schema_version"] == 2
                else "the proposal command routes but does not execute tools"
            ),
        },
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
