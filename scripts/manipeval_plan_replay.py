#!/usr/bin/env python3
"""Replay a bound PlanSession transition without starting simulator or ACT.

This development utility keeps the recorded non-policy evidence fixed and
changes only the explicit policy_success value.  Its output is counterfactual
control-flow evidence, not a new policy evaluation result.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.planner import BoundTaskPlanSession, build_act_catalog


def _inside(root: Path, value: Path, name: str) -> Path:
    path = value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{name} must remain inside --repo-root") from exc
    return path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON object required: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--observation-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--policy-success",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="Counterfactual success values in [0, 1]; default compares 0 and 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    plan_path = _inside(root, args.plan_json, "--plan-json")
    observation_path = _inside(root, args.observation_json, "--observation-json")
    output_path = _inside(root, args.output_json, "--output-json")
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    values = [float(value) for value in args.policy_success]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise SystemExit("--policy-success values must be in [0, 1]")

    plan = _read_json(plan_path)
    observation = _read_json(observation_path)
    session = BoundTaskPlanSession.from_catalog(
        build_act_catalog(root),
        args.task_name,
        max_rounds=args.max_rounds,
    )
    plan = session.normalize_plan(plan)
    results = []
    for success in values:
        counterfactual = copy.deepcopy(observation)
        observations = counterfactual.get("observations")
        if not isinstance(observations, dict):
            raise SystemExit("observation JSON has no observations object")
        observations["policy_success"] = success
        assessment = session.assess(plan, [counterfactual])
        results.append(
            {
                "counterfactual_policy_success": success,
                "required_action": assessment["required_action"],
                "required_transition": assessment["required_transition"],
                "required_next_aspect_id": assessment["required_next_aspect_id"],
                "reasons": assessment["reasons"],
                "assessment": assessment,
            }
        )

    output = {
        "schema_version": 1,
        "status": "counterfactual_plan_replay",
        "task_name": session.target["task_name"],
        "checkpoint": session.target["checkpoint"],
        "plan_source": str(plan_path.relative_to(root)).replace("\\", "/"),
        "observation_source": str(observation_path.relative_to(root)).replace("\\", "/"),
        "frozen_non_policy_evidence": True,
        "development_evidence_only": True,
        "act_rollouts_started": 0,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
