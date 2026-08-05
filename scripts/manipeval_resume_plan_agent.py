#!/usr/bin/env python3
"""Retry one decision, or explicitly execute one persisted pending Proposal."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.plan_agent_decision_resume import (
    continue_pending_plan_agent_round,
    resume_plan_agent_decision,
)
from mea.providers import OpenAICompatibleProvider, resolve_model_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry only plan_agent_decision_after_round_N from immutable "
            "round artifacts. A continue decision persists the next Proposal "
            "without execution; --execute-pending-round explicitly executes "
            "exactly one persisted Proposal without replaying prior rounds."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--planner-model")
    parser.add_argument("--feedback-model")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--execute-pending-round",
        action="store_true",
        help=(
            "Execute exactly one Proposal previously persisted by a continue "
            "decision. Completed rounds are reconstructed, never replayed."
        ),
    )
    parser.add_argument("--policy-server-port", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-reflections", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"eval_[A-Za-z0-9_]+", args.evaluation_id):
        raise SystemExit("--evaluation-id must match eval_[A-Za-z0-9_]+")
    root = args.repo_root.expanduser().resolve()
    manifest = json.loads(
        (
            root
            / "mea/evaluation_runs"
            / args.evaluation_id
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    models = resolve_model_profile(manifest.get("model_profile", "balanced"))
    resolved = manifest.get("resolved_models")
    if isinstance(resolved, dict):
        for key in ("planner", "taskgen", "toolgen", "vision"):
            if isinstance(resolved.get(key), str):
                models[key] = resolved[key]
        if isinstance(resolved.get("answer"), str):
            models["feedback"] = resolved["answer"]
    if args.planner_model:
        models["planner"] = args.planner_model
    if args.feedback_model:
        models["feedback"] = args.feedback_model
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=models["planner"],
        timeout=args.timeout,
    )
    if args.execute_pending_round:
        if args.policy_server_port is None:
            raise SystemExit(
                "--execute-pending-round requires --policy-server-port"
            )
        result = continue_pending_plan_agent_round(
            root,
            args.evaluation_id,
            provider=provider,
            models=models,
            policy_server_port=args.policy_server_port,
            gpu=args.gpu,
            max_reflections=args.max_reflections,
        )
    else:
        result = resume_plan_agent_decision(
            root,
            args.evaluation_id,
            provider=provider,
            models=models,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
