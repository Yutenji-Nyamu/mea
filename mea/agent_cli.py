"""Production Agent CLI arguments and side-effect-free option resolution.

This module deliberately contains no planner construction, provider calls, or
rollout execution.  Keeping the command-line surface here makes the executable
entry point a thin orchestrator while preserving the established
``scripts.manipeval_agent`` imports for compatibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from mea.planner.query_contract import (
    infer_claim_type,
    infer_control_requirement,
    validate_query_sufficiency_contract,
)
from mea.providers import available_model_profiles


def _positive_planning_allowance(raw: str) -> int:
    """Parse an execution allowance without turning it into method semantics."""

    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("planning allowance must be positive")
    return value


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
            "--bound-task-name, the Plan Agent interprets the Query without "
            "seeing the task inventory, then checks the proposed sub-aspect "
            "against the bound policy and the official RoboTwin "
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
        choices=["catalog_step_v1", "claim_first_v1", "plan_agent_v1"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--query-sufficiency-contract",
        type=Path,
        help=(
            "Optional preregistered QuerySufficiencyContract JSON for "
            "the Plan Agent. Comparative Queries require this explicit "
            "two-group contract."
        ),
    )
    parser.add_argument(
        "--generated-rounds",
        type=_positive_planning_allowance,
        default=5,
        help=(
            "Planning allowance for evidence-conditioned generated rounds. "
            "The Plan Agent should stop earlier when evidence is sufficient; "
            "this is not the preferred semantic stopping rule."
        ),
    )
    parser.add_argument(
        "--max-agent-rounds",
        type=_positive_planning_allowance,
        help=(
            "Optional emergency execution ceiling. Normal method runs should "
            "omit it and stop through the evidence-sufficiency contract."
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
        "--policy-backend",
        choices=["act", "smolvla", "hyvla"],
        default="act",
        help=(
            "RoboTwin policy implementation. ACT keeps the existing "
            "task-specific checkpoint path; SmolVLA uses one shared "
            "language-conditioned checkpoint through MethodRuntime."
        ),
    )
    parser.add_argument(
        "--smolvla-checkpoint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin"
        ),
        help=(
            "Shared RoboTwin SmolVLA checkpoint. Used only with "
            "--policy-backend smolvla."
        ),
    )
    parser.add_argument(
        "--smolvla-port",
        type=int,
        default=18771,
        help=(
            "Loopback SmolVLA policy-server port. Used only with "
            "--policy-backend smolvla."
        ),
    )
    parser.add_argument(
        "--hyvla-checkpoint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin"
        ),
        help="Official Hy-VLA RoboTwin checkpoint.",
    )
    parser.add_argument(
        "--hyvla-source",
        type=Path,
        default=Path(
            "/root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA"
        ),
        help="Pinned official Hy-VLA source used by the external server.",
    )
    parser.add_argument(
        "--hyvla-python-env",
        type=Path,
        default=Path("/root/autodl-tmp/envs/mea-hyvla"),
        help="Hy-VLA environment recorded in the binding; MEA does not launch it.",
    )
    parser.add_argument(
        "--hyvla-port",
        type=int,
        default=18781,
        help="Loopback port of an explicitly started Hy-VLA policy server.",
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
        default="balanced",
        help=(
            "Named per-stage model defaults. Individual --*-model arguments "
            "override the selected profile."
        ),
    )
    parser.add_argument("--planner-model")
    parser.add_argument("--taskgen-model")
    parser.add_argument("--toolgen-model")
    parser.add_argument("--vision-model")
    parser.add_argument("--answer-model", dest="feedback_model")
    parser.add_argument(
        "--feedback-model",
        dest="feedback_model",
        help=argparse.SUPPRESS,
    )
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
    """Choose the explicit planner or the production Plan Agent default.

    Hidden paper-profile overrides live in
    ``experiments.paper.compat_agent_profile`` and are applied lazily by the
    executable entry point.
    """

    selected = getattr(args, "open_query_planner", None)
    if selected is not None:
        return (
            "plan_agent_v1"
            if str(selected) == "claim_first_v1"
            else str(selected)
        )
    return "plan_agent_v1"


def paper_compat_profile_requested(
    args: argparse.Namespace,
    *,
    requested_open_query_planner: str | None,
) -> bool:
    """Return whether hidden paper-only options selected a compat profile."""

    return bool(
        requested_open_query_planner == "catalog_step_v1"
        or args.task_profile != "official"
        or args.planning_policy != "dynamic_evidence_v1"
        or args.proposal_mode != "catalog"
        or any(
            value is not None
            for value in (
                args.evidence_manifest,
                args.command_plan,
                args.registered_route,
                args.registered_strategy,
            )
        )
    )


def validate_and_normalize_agent_args(
    args: argparse.Namespace,
    *,
    plan_agent_mode: bool,
) -> bool:
    """Validate cross-option contracts and normalize the production route.

    The returned flag identifies the providerless, bound-task ``--plan-only``
    form.  This function performs no provider, simulator, or filesystem work;
    it only preserves the CLI's established argument semantics.
    """

    if args.policy_backend in {"smolvla", "hyvla"} and not plan_agent_mode:
        raise SystemExit(
            "external policy backends are available only on the production "
            "Plan Agent path"
        )
    if (
        args.policy_backend in {"smolvla", "hyvla"}
        and args.execution_backend not in {None, "act"}
    ):
        raise SystemExit(
            "external policy backends evaluate the bound policy only; "
            "--execution-backend expert/both remains an ACT compatibility path"
        )
    if args.smolvla_port < 1 or args.smolvla_port > 65535:
        raise SystemExit("--smolvla-port must be in [1, 65535]")
    if args.hyvla_port < 1 or args.hyvla_port > 65535:
        raise SystemExit("--hyvla-port must be in [1, 65535]")

    bound_plan_only = bool(
        plan_agent_mode
        and args.plan_only
        and args.bound_task_name is not None
        and not args.auto_route
    )
    if plan_agent_mode and not bound_plan_only:
        # Query-first routing is the production default. ``--auto-route`` is
        # retained as an explicit spelling for existing commands.
        args.auto_route = True
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes must be positive")
    if args.auto_route and args.task_module is not None:
        raise SystemExit(
            "--auto-route resolves a trusted task module; do not pass --task-module"
        )
    if (
        args.bound_task_name is not None
        and not args.auto_route
        and not bound_plan_only
    ):
        raise SystemExit("--bound-task-name requires --auto-route")
    if args.bound_requested_aspect_ids is not None and args.bound_task_name is None:
        raise SystemExit("--bound-requested-aspect-id requires --bound-task-name")
    if plan_agent_mode and not (args.auto_route or bound_plan_only):
        raise SystemExit(
            "plan_agent_v1 requires --auto-route, or --plan-only with "
            "--bound-task-name"
        )
    if bound_plan_only and args.bound_requested_aspect_ids is not None:
        raise SystemExit(
            "providerless Plan Agent plan-only owns the control anchor; "
            "do not predeclare aspect ids"
        )
    if args.query_sufficiency_contract is not None and not plan_agent_mode:
        raise SystemExit(
            "--query-sufficiency-contract requires the production Plan Agent"
        )
    return bound_plan_only


def load_query_sufficiency_contract(path: Path) -> dict[str, Any]:
    """Read and validate an explicitly supplied Query contract JSON file."""

    contract_path = path.expanduser().resolve()
    try:
        loaded_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"cannot read --query-sufficiency-contract: {exc}"
        ) from exc
    if not isinstance(loaded_contract, dict):
        raise SystemExit("--query-sufficiency-contract must contain a JSON object")
    try:
        return validate_query_sufficiency_contract(loaded_contract)
    except ValueError as exc:
        raise SystemExit(f"invalid --query-sufficiency-contract: {exc}") from exc


def resolve_plan_agent_control_required(
    user_request: str,
    *,
    query_contract: Mapping[str, Any] | None,
    semantic_context: Mapping[str, Any] | None,
    candidate_resolution: Mapping[str, Any] | None = None,
) -> bool:
    """Resolve whether a separate control anchor precedes the experiment.

    A trusted QueryContract remains authoritative.  Otherwise a typed
    unchanged-official candidate is already the requested experiment, so it
    must not be charged a second control round merely because the Query calls
    that episode a "control".
    """

    if query_contract is not None:
        return query_contract["control_requirement"] == "required"
    if (
        isinstance(candidate_resolution, Mapping)
        and candidate_resolution.get("resolution")
        == "official_execution_from_typed_needs"
        and candidate_resolution.get("execution_authorized") is True
        and infer_claim_type(user_request) == "diagnostic"
    ):
        return False
    return (
        infer_control_requirement(
            user_request,
            semantic_context=semantic_context,
        )
        == "required"
    )


def resolve_plan_agent_allowed_aspects(
    explicit_aspect_ids: list[str] | None,
) -> list[str] | None:
    """Restrict planning only when the caller explicitly binds an aspect.

    A pre-control Query interpretation or catalog retrieval score is routing
    evidence, not permission to freeze the Plan Agent's later semantic search
    space.
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
        raise ValueError("explicit Plan Agent aspect binding cannot be empty")
    return list(dict.fromkeys(normalized))


def resolve_plan_agent_candidate_budget(
    max_agent_rounds: int | None,
    *,
    user_request: str,
    query_contract: Mapping[str, Any] | None,
    semantic_context: Mapping[str, Any] | None,
    candidate_resolution: Mapping[str, Any] | None = None,
) -> int | None:
    """Return candidate rounds after charging only a required control."""

    if max_agent_rounds is None:
        return None
    return int(max_agent_rounds) - int(
        resolve_plan_agent_control_required(
            user_request,
            query_contract=query_contract,
            semantic_context=semantic_context,
            candidate_resolution=candidate_resolution,
        )
    )


# Read-only compatibility aliases for paper protocols and historical tests.
# Production imports use the paper-aligned Plan Agent names above.
resolve_claim_first_control_required = resolve_plan_agent_control_required
resolve_claim_first_allowed_aspects = resolve_plan_agent_allowed_aspects
resolve_claim_first_candidate_budget = resolve_plan_agent_candidate_budget
