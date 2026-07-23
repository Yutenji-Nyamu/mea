#!/usr/bin/env python3
"""Produce one claim-first open-Query semantic Plan proposal.

This command starts no simulator, expert, probe, or ACT rollout.  Without
``--proposal-json`` it makes one logical provider call (with one bounded retry
inside the Plan Agent) using only the Query, projected runtime capabilities,
and the explicitly supplied completed-round evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.planner import (
    BoundTaskPlanSession,
    ClaimFirstOpenQueryAgent,
    build_act_catalog,
    build_planning_context,
    open_query_input_digest,
    project_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
)
from mea.providers import OpenAICompatibleProvider, resolve_model_profile


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover one semantic sub-aspect from an open Query and completed "
            "evidence, without a predeclared aspect itinerary."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--evidence-json",
        type=Path,
        required=True,
        help="JSON list using the compact OpenQueryEvidence schema; use [] initially.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--proposal-json",
        type=Path,
        help="Validate a cached proposal without calling a provider.",
    )
    parser.add_argument("--model-profile", default="economy")
    parser.add_argument("--base-url")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    evidence_path = args.evidence_json.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    try:
        raw_evidence = _read_json(evidence_path)
        evidence = validate_open_query_evidence(raw_evidence)
        session = BoundTaskPlanSession.from_catalog(
            build_act_catalog(root), args.task_name
        )
        planning_context = build_planning_context(root, session.target)
        capabilities = project_open_query_capabilities(planning_context)
        digest = open_query_input_digest(args.query, capabilities, evidence)

        if args.proposal_json:
            proposal_path = args.proposal_json.expanduser().resolve()
            raw_proposal = _read_json(proposal_path)
            proposal = validate_open_query_plan_proposal(
                raw_proposal, has_evidence=bool(evidence)
            )
            result = {
                "schema_version": 1,
                "source": "validated_cached_claim_first_proposal",
                "input_digest": digest,
                "proposal": proposal,
                "provider": {
                    "called": False,
                    "attempt_count": 0,
                    "errors": [],
                },
            }
            execution_mode = "cached_validation_0_provider_0_act"
        else:
            profile = resolve_model_profile(args.model_profile)
            provider = OpenAICompatibleProvider(
                base_url=args.base_url,
                text_model=profile["planner"],
            )
            result = ClaimFirstOpenQueryAgent(
                provider, model=profile["planner"]
            ).propose(
                args.query,
                capabilities=capabilities,
                evidence_history=evidence,
            )
            execution_mode = "live_provider_semantic_plan_0_act"

        payload = {
            "schema_version": 1,
            "status": "claim_first_open_query_plan_completed",
            "execution_mode": execution_mode,
            "task_name": session.target["task_name"],
            "checkpoint": session.target["checkpoint"],
            "query": args.query.strip(),
            "evidence_rounds": len(evidence),
            "capabilities_exclude_navigation_catalog": True,
            "simulator_calls_started": 0,
            "expert_calls_started": 0,
            "act_rollouts_started": 0,
            "result": result,
        }
        _write_json(output_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except ValueError as exc:
        raise SystemExit(f"claim-first planning failed: {exc}") from exc


if __name__ == "__main__":
    main()
