#!/usr/bin/env python3
"""Plan or replay a query-driven parent graph over fixed ACT children."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.evaluation_graph import (
    EvaluationGraphError,
    EvaluationGraphPlanner,
    EvaluationGraphSession,
    build_child_command_plan,
    child_outcome_from_evaluation,
    validate_evaluation_graph,
)
from mea.planner import build_act_catalog
from mea.providers import OpenAICompatibleProvider, resolve_model_profile


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationGraphError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _require_plan_identity(plan: dict, *, graph_id: str, query: str) -> None:
    if (
        plan.get("graph_id") != graph_id.strip()
        or plan.get("user_query") != query.strip()
    ):
        raise EvaluationGraphError(
            "plan graph_id/user_query must match the command-line identity"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--graph-id", required=True)
    plan.add_argument("--query", required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--proposal-json", type=Path)
    plan.add_argument("--model-profile", default="economy")
    plan.add_argument("--base-url")

    replay = sub.add_parser("replay")
    replay.add_argument("--plan", type=Path, required=True)
    replay.add_argument("--outcomes", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)

    outcome = sub.add_parser("outcome")
    outcome.add_argument("--plan", type=Path, required=True)
    outcome.add_argument("--node-id", required=True)
    outcome.add_argument("--evaluation-id", required=True)
    outcome.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        catalog = build_act_catalog(root)
        if args.command == "plan":
            if args.proposal_json:
                result = validate_evaluation_graph(_read(args.proposal_json), catalog)
                _require_plan_identity(
                    result,
                    graph_id=args.graph_id,
                    query=args.query,
                )
                provider_metadata = None
            else:
                profile = resolve_model_profile(args.model_profile)
                provider = OpenAICompatibleProvider(
                    base_url=args.base_url,
                    text_model=profile["planner"],
                )
                planner = EvaluationGraphPlanner(provider, model=profile["planner"])
                result = planner.plan(
                    args.query,
                    catalog,
                    graph_id=args.graph_id,
                )
                provider_metadata = dict(provider.last_metadata)
            payload = {
                "schema_version": 1,
                "plan": result,
                "provider_metadata": provider_metadata,
                "child_commands": build_child_command_plan(
                    result, catalog, repo_root=str(root)
                ),
            }
        elif args.command == "replay":
            plan_value = _read(args.plan)
            graph = plan_value.get("plan", plan_value)
            outcomes_value = json.loads(args.outcomes.read_text(encoding="utf-8"))
            if not isinstance(outcomes_value, list):
                raise EvaluationGraphError("--outcomes must contain a JSON list")
            session = EvaluationGraphSession(graph, catalog)
            for outcome in outcomes_value:
                session.record(outcome)
            payload = session.snapshot()
        else:
            plan_value = _read(args.plan)
            graph = validate_evaluation_graph(plan_value.get("plan", plan_value), catalog)
            node = next(
                (item for item in graph["nodes"] if item["node_id"] == args.node_id),
                None,
            )
            if node is None:
                raise EvaluationGraphError(f"unknown graph node: {args.node_id}")
            payload = child_outcome_from_evaluation(
                root,
                graph,
                catalog,
                node_id=node["node_id"],
                evaluation_id=args.evaluation_id,
            )
        _write(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError, EvaluationGraphError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
