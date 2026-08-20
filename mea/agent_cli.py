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

from mea.providers import available_model_profiles


def _positive_planning_allowance(raw: str) -> int:
    """Parse an execution allowance without turning it into method semantics."""

    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("planning allowance must be positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--libero-backbone-metadata",
        type=Path,
        help=(
            "Optional local SmolVLM metadata/tokenizer directory. This keeps "
            "LIBERO policy loading offline while the finetuned VLA weights "
            "continue to come from --libero-checkpoint."
        ),
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
            "first policy-trial seed; each executed Plan round uses the "
            "contiguous group starting here"
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help=(
            "policy trials aggregated within each executed task (paper "
            "default: 5; use 1 only for transport/mechanism debugging)"
        ),
    )
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1"],
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
        "--no-history",
        action="store_true",
        help="Disable cross-evaluation planning retrieval and indexing.",
    )
    return parser.parse_args(argv)


def validate_and_normalize_agent_args(
    args: argparse.Namespace,
) -> bool:
    """Validate cross-option contracts for the sole production Plan Agent.

    The returned flag identifies the providerless, bound-task ``--plan-only``
    form.  This function performs no provider, simulator, or filesystem work;
    it only preserves the CLI's established argument semantics.
    """

    if args.smolvla_port < 1 or args.smolvla_port > 65535:
        raise SystemExit("--smolvla-port must be in [1, 65535]")
    if args.hyvla_port < 1 or args.hyvla_port > 65535:
        raise SystemExit("--hyvla-port must be in [1, 65535]")

    bound_plan_only = bool(
        args.plan_only
        and args.bound_task_name is not None
        and not args.auto_route
    )
    if not bound_plan_only:
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
    return bound_plan_only


def resolve_plan_agent_control_required(
    *,
    candidate_resolution: Mapping[str, Any] | None = None,
) -> bool:
    """Use a typed official experiment directly; otherwise keep a control."""

    return not (
        isinstance(candidate_resolution, Mapping)
        and candidate_resolution.get("resolution")
        == "official_execution_from_typed_needs"
        and candidate_resolution.get("execution_authorized") is True
    )


def resolve_plan_agent_candidate_budget(
    max_agent_rounds: int | None,
    *,
    candidate_resolution: Mapping[str, Any] | None = None,
) -> int | None:
    """Return candidate rounds after charging only a required control."""

    if max_agent_rounds is None:
        return None
    return int(max_agent_rounds) - int(
        resolve_plan_agent_control_required(
            candidate_resolution=candidate_resolution,
        )
    )
