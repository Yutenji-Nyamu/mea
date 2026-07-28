"""Direct initial-plan construction for the production ClaimFirst runtime.

This module deliberately does not import ``CatalogPlanAgent`` or any
task-specific planner.  The caller has already frozen a task/checkpoint target;
the only remaining initial planning decision is whether the QueryContract
requires an unchanged official control before Query-derived candidates.
"""

from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mea.toolgen import official_success_tool_request

from .claim_first_runtime import control_template_id
from .query_contract import validate_query_sufficiency_contract


_EXECUTION_BACKENDS = frozenset({"expert", "act", "both"})
_CONTROL_GATES = ("render", "rule")
_POST_EXECUTION_GATES = ("toolkit", "planned_tool", "aggregate")


class ClaimFirstInitialPlanError(ValueError):
    """Raised when a direct ClaimFirst initial plan cannot be constructed."""


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimFirstInitialPlanError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClaimFirstInitialPlanError(f"{field} must be a positive integer")
    return value


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaimFirstInitialPlanError(f"{field} must be an object")
    return deepcopy(dict(value))


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_claim_first_execution_binding(
    *,
    start_seed: int,
    num_episodes: int,
    execution_backend: str = "act",
) -> dict[str, Any]:
    """Build the frozen execution transport shared by control/candidate round 1."""

    seed = start_seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ClaimFirstInitialPlanError(
            "start_seed must be a non-negative integer"
        )
    count = _positive_int(num_episodes, field="num_episodes")
    backend = _text(execution_backend, field="execution_backend").casefold()
    if backend not in _EXECUTION_BACKENDS:
        raise ClaimFirstInitialPlanError(
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


def build_claim_first_control_round(
    target: Mapping[str, Any],
    user_request: str,
    *,
    execution_binding: Mapping[str, Any],
    task_module: str | None = None,
    telemetry_profile: str = "balanced_v1",
) -> dict[str, Any]:
    """Build one unchanged official control without a task-specific planner."""

    bound_target = _mapping(target, field="target")
    task_name = _text(bound_target.get("task_name"), field="target.task_name")
    query = _text(user_request, field="user_request")
    execution = _mapping(execution_binding, field="execution_binding")
    module = (
        _text(task_module, field="task_module")
        if task_module is not None
        else f"envs.{task_name}"
    )
    return {
        "round_id": "round_1",
        "template_id": control_template_id(bound_target),
        "sub_aspect": "task_execution.official_baseline",
        "rationale": (
            "Establish the unchanged official-task control required by the "
            "QueryContract before attributing evidence to a generated candidate."
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


class ClaimFirstInitialPlanBuilder:
    """Persist a ClaimFirst initial manifest directly from a frozen target."""

    planner_kind = "claim_first_direct_initial_v1"

    def __init__(
        self,
        repo_root: str | Path,
        *,
        target: Mapping[str, Any],
        max_rounds: int,
        start_seed: int,
        num_episodes: int = 1,
        execution_backend: str = "act",
        task_module: str | None = None,
        telemetry_profile: str = "balanced_v1",
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.target = _mapping(target, field="target")
        self.task_name = _text(
            self.target.get("task_name"), field="target.task_name"
        )
        self.policy = _mapping(self.target.get("policy"), field="target.policy")
        self.checkpoint = _mapping(
            self.target.get("checkpoint"), field="target.checkpoint"
        )
        self.max_rounds = _positive_int(max_rounds, field="max_rounds")
        self.task_module = task_module
        self.telemetry_profile = _text(
            telemetry_profile, field="telemetry_profile"
        )
        self.execution_binding = build_claim_first_execution_binding(
            start_seed=start_seed,
            num_episodes=num_episodes,
            execution_backend=execution_backend,
        )
        # Resolve the neutral control once at construction, even for a
        # no-control Query.  It validates that the bound target can transport a
        # later runtime candidate without selecting a task-specific itinerary.
        self.control_template = control_template_id(self.target)

    def plan(
        self,
        user_request: str,
        *,
        evaluation_id: str,
        control_required: bool,
        query_contract: Mapping[str, Any] | None = None,
        history_context: list[dict[str, Any]] | None = None,
        history_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create the initial manifest without catalog/task-specific planning.

        A no-control plan intentionally starts with an empty round list.  The
        caller should materialize its already discovered ``ExperimentCandidate``
        with ``manifest["initial_execution_binding"]`` and then normalize it
        through ``OpenWorldPlanSession``.
        """

        query = _text(user_request, field="user_request")
        resolved_id = _text(evaluation_id, field="evaluation_id")
        if re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id) is None:
            raise ClaimFirstInitialPlanError(
                "evaluation_id must begin with 'eval_'"
            )
        if not isinstance(control_required, bool):
            raise ClaimFirstInitialPlanError("control_required must be bool")
        contract = (
            validate_query_sufficiency_contract(query_contract)
            if query_contract is not None
            else None
        )
        if contract is not None:
            contract_requires_control = (
                contract["control_requirement"] == "required"
            )
            if contract_requires_control != control_required:
                raise ClaimFirstInitialPlanError(
                    "control_required conflicts with QueryContract"
                )
            required_rounds = int(contract["round_budget"]) + int(
                control_required
            )
            if required_rounds > self.max_rounds:
                raise ClaimFirstInitialPlanError(
                    "QueryContract exceeds the initial ClaimFirst round budget"
                )
        elif control_required and self.max_rounds < 2:
            raise ClaimFirstInitialPlanError(
                "control-required ClaimFirst plan needs one candidate round"
            )

        evaluation_dir = self.repo_root / "mea/evaluation_runs" / resolved_id
        if evaluation_dir.exists():
            raise ClaimFirstInitialPlanError(
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
                build_claim_first_control_round(
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
            plan["query_contract"] = deepcopy(contract)
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
            "base_commit": _git_head(self.repo_root),
            "planner": {
                "kind": self.planner_kind,
                "model_requested": None,
                "provider_called": False,
                "proposal_source": (
                    "runtime_query_contract_control"
                    if control_required
                    else "runtime_free_concern_candidate_pending"
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
    "ClaimFirstInitialPlanBuilder",
    "ClaimFirstInitialPlanError",
    "build_claim_first_control_round",
    "build_claim_first_execution_binding",
]
