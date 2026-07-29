"""Production Agent CLI arguments and side-effect-free option resolution.

This module deliberately contains no planner construction, provider calls, or
rollout execution.  Keeping the command-line surface here makes the executable
entry point a thin orchestrator while preserving the established
``scripts.manipeval_agent`` imports for compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from mea.planner.query_contract import infer_control_requirement
from mea.providers import available_model_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id")
    parser.add_argument(
        "--benchmark",
        choices=["robotwin", "libero"],
        default="robotwin",
        help="Select the existing RoboTwin chain or the bounded LIBERO backend.",
    )
    parser.add_argument(
        "--libero-checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/checkpoints/libero/smolvla_libero"),
    )
    parser.add_argument("--libero-seed", type=int, default=100800)
    parser.add_argument(
        "--auto-route",
        action="store_true",
        help=(
            "Explicitly request the production Query-first route (already the "
            "default unless a hidden paper-compatibility protocol is selected). "
            "With "
            "--bound-task-name, claim_first_v1 creates a catalog-free concern "
            "then checks it against the bound policy and the official RoboTwin "
            "task library. Without a bound task, the same catalog-free concern "
            "is created first and only then retrieves a checkpoint-ready base "
            "task; this portfolio convenience is not one policy executing "
            "arbitrary tasks."
        ),
    )
    parser.add_argument(
        "--bound-task-name",
        help=(
            "Bind one auto-routed evaluation to an already selected RoboTwin "
            "task/checkpoint. The Plan Agent may change sub-aspects but cannot "
            "route to another task."
        ),
    )
    parser.add_argument(
        "--bound-requested-aspect-id",
        dest="bound_requested_aspect_ids",
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--proposal-mode",
        choices=["catalog", "novel_first_round", "bounded_each_round"],
        default="catalog",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--task-name",
        default="beat_block_hammer",
        help=(
            "Canonical RoboTwin base-task identity used for checkpoint and "
            "simulator binding. Schema-backed tasks can use the generic "
            "retrieve/generate TaskGen path when the Query requires it."
        ),
    )
    parser.add_argument(
        "--task-module",
        help="Optional Python module for an official schema-backed task.",
    )
    parser.add_argument(
        "--task-profile",
        choices=["official", "position_lr", "adaptive_properties", "fixed_suite"],
        default="official",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--planning-policy",
        choices=["dynamic_evidence_v1", "fixed_predeclared_v1"],
        default="dynamic_evidence_v1",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--open-query-planner",
        choices=["catalog_step_v1", "claim_first_v1"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--query-sufficiency-contract",
        type=Path,
        help=(
            "Optional preregistered QuerySufficiencyContract JSON for "
            "claim_first_v1. Comparative Queries require this explicit "
            "two-group contract."
        ),
    )
    parser.add_argument(
        "--generated-rounds",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=2,
        help="Round budget for a bounded click_bell generated profile.",
    )
    parser.add_argument(
        "--max-agent-rounds",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help=(
            "Optional task-agnostic hard execution cap. After this many completed "
            "rounds the Agent writes an auditable budget stop without asking the "
            "planner to add another round."
        ),
    )
    parser.add_argument(
        "--execution-backend",
        choices=["expert", "act", "both"],
        help=(
            "Policy backend for schema-backed official tasks. Defaults to "
            "expert; both evaluates ACT and keeps expert as validation."
        ),
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=None,
        help=(
            "override trusted task seeds; omitted keeps the BBH catalog "
            "defaults and each other planner's existing default"
        ),
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument(
        "--model-profile",
        choices=available_model_profiles(),
        default="legacy",
        help=(
            "Named per-stage model defaults. Individual --*-model arguments "
            "override the selected profile."
        ),
    )
    parser.add_argument("--planner-model")
    parser.add_argument("--taskgen-model")
    parser.add_argument("--toolgen-model")
    parser.add_argument("--vision-model")
    parser.add_argument("--feedback-model")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum visual diagnosis-driven CodeGen repairs per TaskGen run.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--history-database",
        type=Path,
        help=(
            "SQLite planning-history cache. Defaults to "
            "mea/evaluation_runs/history.sqlite3 under --repo-root."
        ),
    )
    parser.add_argument("--history-limit", type=int, default=3)
    parser.add_argument(
        "--reviewed-task-registry",
        type=Path,
        help=(
            "Optional explicit reviewed generated-Task registry. Exact semantic "
            "and artifact-hash matches may be materialized without TaskGen text "
            "generation."
        ),
    )
    parser.add_argument(
        "--reviewed-tool-registry",
        type=Path,
        help=(
            "Optional explicit reviewed generated-Tool registry. Exact "
            "contract/schema/hash matches may be reused across evaluations "
            "without a ToolGen provider call."
        ),
    )
    parser.add_argument(
        "--reviewed-vqa-registry",
        type=Path,
        help=(
            "Optional hash-pinned reviewed VQAQuerySpec registry. Matching "
            "entries may only select existing trusted visual phenomena."
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable cross-evaluation planning retrieval and indexing.",
    )
    parser.add_argument(
        "--evidence-manifest",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--command-plan",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--registered-route",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--registered-strategy",
        choices=["fixed_predeclared_v1", "dynamic_evidence_v1"],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_default_open_query_planner(args: argparse.Namespace) -> str:
    """Choose production ClaimFirst unless a paper protocol is explicit."""

    selected = getattr(args, "open_query_planner", None)
    if selected is not None:
        return str(selected)
    paper_compatibility = bool(
        getattr(args, "registered_strategy", None) is not None
        or getattr(args, "task_profile", "official") != "official"
        or getattr(args, "planning_policy", "dynamic_evidence_v1")
        != "dynamic_evidence_v1"
        or getattr(args, "proposal_mode", "catalog") != "catalog"
    )
    return "catalog_step_v1" if paper_compatibility else "claim_first_v1"


def resolve_claim_first_control_required(
    user_request: str,
    *,
    query_contract: Mapping[str, Any] | None,
    semantic_context: Mapping[str, Any] | None,
) -> bool:
    """Resolve the control cost before screening the candidate budget."""

    if query_contract is not None:
        return query_contract["control_requirement"] == "required"
    return (
        infer_control_requirement(
            user_request,
            semantic_context=semantic_context,
        )
        == "required"
    )


def resolve_claim_first_allowed_aspects(
    explicit_aspect_ids: list[str] | None,
) -> list[str] | None:
    """Restrict planning only when the caller explicitly binds an aspect.

    A pre-control FreeConcern or catalog retrieval score is routing evidence,
    not permission to freeze the Plan Agent's later semantic search space.
    ``None`` keeps the complete retrieval inventory available while an
    unmatched concern may still enter open-world Task/Tool generation.
    """

    if explicit_aspect_ids is None:
        return None
    normalized = [
        str(item).strip()
        for item in explicit_aspect_ids
        if isinstance(item, str) and item.strip()
    ]
    if not normalized:
        raise ValueError("explicit ClaimFirst aspect binding cannot be empty")
    return list(dict.fromkeys(normalized))


def resolve_claim_first_candidate_budget(
    max_agent_rounds: int | None,
    *,
    user_request: str,
    query_contract: Mapping[str, Any] | None,
    semantic_context: Mapping[str, Any] | None,
) -> int | None:
    """Return candidate rounds after charging only a required control."""

    if max_agent_rounds is None:
        return None
    return int(max_agent_rounds) - int(
        resolve_claim_first_control_required(
            user_request,
            query_contract=query_contract,
            semantic_context=semantic_context,
        )
    )
