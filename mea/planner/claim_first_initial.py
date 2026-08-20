"""Direct initial-plan construction for the production Plan Agent runtime.

This module has no catalog or task-specific planner delegation. The caller has
already frozen a task/checkpoint target; the only remaining initial planning
decision is whether runtime limits require an unchanged official control before
Query-derived candidates.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mea.toolgen import official_success_tool_request

from .query_interpretation import OFFICIAL_CONTROL_TEMPLATE_ID
from .policy_task_binding import (
    PolicyTaskBindingError,
    policy_task_binding_from_target,
)
from .runtime_limits import validate_plan_runtime_limits


_EXECUTION_BACKENDS = frozenset({"expert", "act", "both"})
_CONTROL_GATES = ("render", "rule")
_POST_EXECUTION_GATES = ("toolkit", "planned_tool", "aggregate")


class PlanAgentInitialPlanError(ValueError):
    """Raised when a direct Plan Agent initial plan cannot be constructed."""


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanAgentInitialPlanError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlanAgentInitialPlanError(f"{field} must be a positive integer")
    return value


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanAgentInitialPlanError(f"{field} must be an object")
    return deepcopy(dict(value))


def _target_binding(target: Mapping[str, Any]) -> dict[str, Any]:
    """Read the production binding while preserving old fixture compatibility."""

    if "policy_task_binding" in target:
        try:
            return policy_task_binding_from_target(target)
        except (PolicyTaskBindingError, TypeError) as exc:
            raise PlanAgentInitialPlanError(str(exc)) from exc
    return {
        "task_name": _text(target.get("task_name"), field="target.task_name"),
        "task_module": f"envs.{target.get('task_name')}",
        "policy": _mapping(target.get("policy"), field="target.policy"),
        "checkpoint": _mapping(
            target.get("checkpoint"), field="target.checkpoint"
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_plan_agent_execution_binding(
    *,
    start_seed: int,
    num_episodes: int,
    execution_backend: str = "act",
) -> dict[str, Any]:
    """Build the frozen execution transport shared by control/candidate round 1."""

    seed = start_seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PlanAgentInitialPlanError(
            "start_seed must be a non-negative integer"
        )
    count = _positive_int(num_episodes, field="num_episodes")
    backend = _text(execution_backend, field="execution_backend").casefold()
    if backend not in _EXECUTION_BACKENDS:
        raise PlanAgentInitialPlanError(
            "execution_backend must be one of expert, act, both"
        )
    seeds = [seed + index for index in range(count)]
    gates = [
        *_CONTROL_GATES,
        *(["expert"] if backend in {"expert", "both"} else []),
        *(["act"] if backend in {"act", "both"} else []),
        *_POST_EXECUTION_GATES,
    ]
    return {
        "backend": backend,
        "seeds": seeds,
        "num_episodes": len(seeds),
        "gates": gates,
    }


def build_plan_agent_control_round(
    target: Mapping[str, Any],
    user_request: str,
    *,
    execution_binding: Mapping[str, Any],
    task_module: str | None = None,
    telemetry_profile: str = "balanced_v1",
) -> dict[str, Any]:
    """Build one unchanged official control without a task-specific planner."""

    bound_target = _mapping(target, field="target")
    binding = _target_binding(bound_target)
    task_name = binding["task_name"]
    query = _text(user_request, field="user_request")
    execution = _mapping(execution_binding, field="execution_binding")
    module = binding["task_module"]
    if (
        task_module is not None
        and _text(task_module, field="task_module") != module
    ):
        raise PlanAgentInitialPlanError(
            "official control task_module differs from PolicyTaskBinding"
        )
    return {
        "round_id": "round_1",
        "template_id": OFFICIAL_CONTROL_TEMPLATE_ID,
        "sub_aspect": OFFICIAL_CONTROL_TEMPLATE_ID,
        "rationale": (
            "Establish the unchanged official-task control required by the "
            "runtime limits before attributing evidence to a generated candidate."
        ),
        "task_instruction": query,
        "task_name": task_name,
        "task_module": module,
        "telemetry_profile": _text(
            telemetry_profile, field="telemetry_profile"
        ),
        "route": "official",
        "variant_hint": {},
        "execution": execution,
        "observations": [
            "scene_alignment",
            "expert_solvable",
            "trusted_tools",
            "aggregate",
        ],
        "tool_request": official_success_tool_request(task_name),
    }


class PlanAgentInitialPlanBuilder:
    """Persist a Plan Agent initial manifest directly from a frozen target."""

    planner_kind = "plan_agent_direct_initial_v1"

    def __init__(
        self,
        repo_root: str | Path,
        *,
        target: Mapping[str, Any],
        max_rounds: int,
        start_seed: int,
        num_episodes: int = 5,
        execution_backend: str = "act",
        task_module: str | None = None,
        telemetry_profile: str = "balanced_v1",
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.target = _mapping(target, field="target")
        binding = _target_binding(self.target)
        self.task_name = binding["task_name"]
        self.policy = _mapping(binding.get("policy"), field="binding.policy")
        self.checkpoint = _mapping(
            binding.get("checkpoint"), field="binding.checkpoint"
        )
        self.max_rounds = _positive_int(max_rounds, field="max_rounds")
        if (
            task_module is not None
            and _text(task_module, field="task_module")
            != binding["task_module"]
        ):
            raise PlanAgentInitialPlanError(
                "task_module differs from the frozen PolicyTaskBinding"
            )
        self.task_module = binding["task_module"]
        self.telemetry_profile = _text(
            telemetry_profile, field="telemetry_profile"
        )
        self.execution_binding = build_plan_agent_execution_binding(
            start_seed=start_seed,
            num_episodes=num_episodes,
            execution_backend=execution_backend,
        )
        self.control_template = OFFICIAL_CONTROL_TEMPLATE_ID

    def plan(
        self,
        user_request: str,
        *,
        evaluation_id: str,
        control_required: bool,
        runtime_limits: Mapping[str, Any] | None = None,
        history_context: list[dict[str, Any]] | None = None,
        history_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create the initial manifest without catalog/task-specific planning.

        A no-control plan intentionally starts with an empty round list.  The
        caller should materialize its already discovered typed Proposal
        with ``manifest["initial_execution_binding"]`` and then normalize it
        through the production ``PlanAgentSession``.
        """

        query = _text(user_request, field="user_request")
        resolved_id = _text(evaluation_id, field="evaluation_id")
        if re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id) is None:
            raise PlanAgentInitialPlanError(
                "evaluation_id must begin with 'eval_'"
            )
        if not isinstance(control_required, bool):
            raise PlanAgentInitialPlanError("control_required must be bool")
        contract = (
            validate_plan_runtime_limits(runtime_limits)
            if runtime_limits is not None
            else None
        )
        if contract is not None:
            contract_requires_control = contract["control_requirement"] == "required"
            if contract_requires_control != control_required:
                raise PlanAgentInitialPlanError(
                    "control_required conflicts with runtime limits"
                )
            required_rounds = int(contract["round_budget"]) + int(
                control_required
            )
            if required_rounds > self.max_rounds:
                raise PlanAgentInitialPlanError(
                    "runtime limits exceed the initial Plan Agent round budget"
                )
        elif control_required and self.max_rounds < 2:
            raise PlanAgentInitialPlanError(
                "control-required Plan Agent plan needs one candidate round"
            )

        evaluation_dir = self.repo_root / "mea/evaluation_runs" / resolved_id
        if evaluation_dir.exists():
            raise PlanAgentInitialPlanError(
                f"evaluation directory already exists: {evaluation_dir}"
            )
        for child in ("plan", "execution", "summary"):
            (evaluation_dir / child).mkdir(parents=True, exist_ok=False)

        compact_history = [
            {
                key: item.get(key)
                for key in (
                    "evaluation_id",
                    "user_request",
                    "task_name",
                    "similarity",
                )
            }
            for item in (history_context or [])
            if isinstance(item, dict)
        ]
        history_retrieval = {
            "schema_version": 1,
            "status": "passed" if compact_history else "empty",
            "match_count": len(compact_history),
            "matches": compact_history,
            **deepcopy(dict(history_metadata or {})),
        }
        rounds = (
            [
                build_plan_agent_control_round(
                    self.target,
                    query,
                    execution_binding=self.execution_binding,
                    task_module=self.task_module,
                    telemetry_profile=self.telemetry_profile,
                )
            ]
            if control_required
            else []
        )
        plan: dict[str, Any] = {
            "schema_version": 5,
            "task_name": self.task_name,
            "policy": deepcopy(self.policy),
            "checkpoint": deepcopy(self.checkpoint),
            "checkpoint_id": self.checkpoint.get("checkpoint_id"),
            "evaluation_goal": f"answer_open_query_with_evidence: {query}",
            "requested_template_ids": (
                [self.control_template] if control_required else []
            ),
            "rounds": rounds,
            "round_decisions": [],
            "max_rounds": self.max_rounds,
            "planning_state": (
                "awaiting_round_1_observation"
                if control_required
                else "awaiting_initial_query_candidate_materialization"
            ),
        }
        if contract is not None:
            plan["runtime_limits"] = deepcopy(contract)
        manifest = {
            "schema_version": 5,
            "evaluation_id": resolved_id,
            "status": (
                "planned_round_1"
                if control_required
                else "awaiting_initial_query_candidate_materialization"
            ),
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": query,
            "planner": {
                "kind": self.planner_kind,
                "model_requested": None,
                "provider_called": False,
                "proposal_source": (
                    "runtime_control_requirement"
                    if control_required
                    else "runtime_plan_agent_proposal_pending"
                ),
                "task_specific_planner_used": False,
            },
            "plan_path": "plan/evaluation_plan.json",
            "history_retrieval_path": "plan/history_retrieval.json",
            "history_retrieval": history_retrieval,
            "initial_execution_binding": deepcopy(self.execution_binding),
            "plan": plan,
        }
        _write_json(evaluation_dir / "request.json", {"user_request": query})
        _write_json(
            evaluation_dir / "plan/history_retrieval.json",
            history_retrieval,
        )
        _write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
        _write_json(evaluation_dir / "manifest.json", manifest)
        return manifest


__all__ = [
    "PlanAgentInitialPlanBuilder",
    "PlanAgentInitialPlanError",
    "build_plan_agent_control_round",
    "build_plan_agent_execution_binding",
]
