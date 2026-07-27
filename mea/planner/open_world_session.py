"""Open-world, single-checkpoint planning session.

The legacy :class:`BoundTaskPlanSession` is intentionally a finite catalog
protocol.  This module is the production ClaimFirst counterpart: the catalog
retrieves one ACT-ready base task and its official control, while later rounds
carry Query-derived :class:`ExperimentCandidate` objects rather than requiring
an aspect or template registration.

The session freezes task, policy, checkpoint, control round, and total rollout
budget.  It does *not* treat the base task's catalog ``max_rounds`` as a
generation limit; an official-only task can therefore run a control followed
by bounded runtime-generated experiments without changing its checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from mea.capability_adapter import resolve_task_adapter

from .catalog import catalog_task, validate_act_catalog
from .context import build_planning_context
from .experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from .query_contract import (
    QuerySufficiencyError,
    assess_query_sufficiency,
    extend_query_candidate_universe,
    validate_query_sufficiency_contract,
)


class OpenWorldSessionError(ValueError):
    """Raised when an open-world plan leaves its frozen evaluation boundary."""


_TARGET_KEYS = {
    "schema_version",
    "binding_mode",
    "task_name",
    "task_family",
    "task_profile",
    "planner_kind",
    "policy",
    "checkpoint",
    "max_rounds",
    "catalog_max_rounds",
    "control_template_id",
    "aspects",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenWorldSessionError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OpenWorldSessionError(f"{field} must be a positive integer")
    return value


def _catalog_templates(task: Mapping[str, Any]) -> set[str]:
    return {
        str(template_id)
        for aspect in task.get("aspects", [])
        if isinstance(aspect, Mapping)
        for template_id in aspect.get("template_ids", [])
    }


def build_open_world_evaluation_target(
    catalog: Mapping[str, Any],
    task_name: str,
    *,
    max_rounds: int,
) -> dict[str, Any]:
    """Freeze one ACT task/checkpoint while allowing a larger runtime budget."""

    trusted_catalog = validate_act_catalog(catalog)
    task = catalog_task(trusted_catalog, _text(task_name, "task_name"))
    budget = _positive_int(max_rounds, "max_rounds")
    adapter = resolve_task_adapter(task["task_name"])
    control_template = _text(
        adapter.get("control_template_id"),
        "TaskAdapter.control_template_id",
    )
    if control_template not in _catalog_templates(task):
        raise OpenWorldSessionError(
            "the TaskAdapter control template is absent from the retrieval catalog"
        )
    return {
        "schema_version": 2,
        "binding_mode": "single_task_single_checkpoint_open_world",
        "task_name": task["task_name"],
        "task_family": task["task_family"],
        "task_profile": task["task_profile"],
        # Retained only so the trusted planning context can retrieve existing
        # artifacts.  It does not select the production planner.
        "planner_kind": task["planner_kind"],
        "policy": deepcopy(trusted_catalog["policy"]),
        "checkpoint": deepcopy(task["checkpoint"]),
        "max_rounds": budget,
        "catalog_max_rounds": int(task["max_rounds"]),
        "control_template_id": control_template,
        "aspects": deepcopy(task["aspects"]),
    }


def validate_open_world_evaluation_target(
    value: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an open target against catalog identity, not catalog budget."""

    if not isinstance(value, Mapping) or set(value) != _TARGET_KEYS:
        raise OpenWorldSessionError(
            f"OpenWorldEvaluationTarget fields must be exactly {sorted(_TARGET_KEYS)}"
        )
    target = deepcopy(dict(value))
    if target.get("schema_version") != 2:
        raise OpenWorldSessionError(
            "OpenWorldEvaluationTarget.schema_version must be 2"
        )
    if target.get("binding_mode") != "single_task_single_checkpoint_open_world":
        raise OpenWorldSessionError(
            "OpenWorldEvaluationTarget.binding_mode is invalid"
        )
    expected = build_open_world_evaluation_target(
        catalog,
        _text(target.get("task_name"), "target.task_name"),
        max_rounds=_positive_int(target.get("max_rounds"), "target.max_rounds"),
    )
    if target != expected:
        raise OpenWorldSessionError(
            "OpenWorldEvaluationTarget differs from the ready ACT catalog"
        )
    return target


class OpenWorldPlanSession:
    """ClaimFirst session with runtime candidates and a frozen ACT binding."""

    def __init__(
        self,
        catalog: Mapping[str, Any],
        target: Mapping[str, Any],
        *,
        control_round: Mapping[str, Any] | None = None,
        query_contract: Mapping[str, Any] | None = None,
    ):
        self.catalog = validate_act_catalog(catalog)
        self.target = validate_open_world_evaluation_target(
            target, self.catalog
        )
        self._control_round: dict[str, Any] | None = None
        self._query_contract: dict[str, Any] | None = None
        if control_round is not None:
            self._control_round = self._validate_control_round(control_round)
        if query_contract is not None:
            self._query_contract = self._validate_query_contract(query_contract)

    @classmethod
    def from_catalog(
        cls,
        catalog: Mapping[str, Any],
        task_name: str,
        *,
        max_rounds: int,
        control_round: Mapping[str, Any] | None = None,
        query_contract: Mapping[str, Any] | None = None,
    ) -> "OpenWorldPlanSession":
        return cls(
            catalog,
            build_open_world_evaluation_target(
                catalog, task_name, max_rounds=max_rounds
            ),
            control_round=control_round,
            query_contract=query_contract,
        )

    def _validate_optional_binding(
        self,
        value: Mapping[str, Any],
        *,
        location: str,
        allow_legacy_max_rounds: bool = False,
    ) -> None:
        expected = {
            "task_name": self.target["task_name"],
            "policy": self.target["policy"],
            "checkpoint": self.target["checkpoint"],
            "checkpoint_id": self.target["checkpoint"].get("checkpoint_id"),
        }
        for field, trusted in expected.items():
            if field in value and value[field] != trusted:
                raise OpenWorldSessionError(
                    f"{location} cannot change bound {field}"
                )
        if "max_rounds" in value:
            supplied = _positive_int(
                value["max_rounds"], f"{location}.max_rounds"
            )
            allowed = {self.target["max_rounds"]}
            if allow_legacy_max_rounds:
                allowed.add(self.target["catalog_max_rounds"])
            if supplied not in allowed:
                raise OpenWorldSessionError(
                    f"{location} cannot change bound max_rounds"
                )

    def _validate_control_round(
        self, round_plan: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(round_plan, Mapping):
            raise OpenWorldSessionError("control round must be an object")
        control = deepcopy(dict(round_plan))
        self._validate_optional_binding(
            control, location="control_round", allow_legacy_max_rounds=True
        )
        execution = control.get("execution")
        if isinstance(execution, Mapping):
            self._validate_optional_binding(
                execution, location="control_round.execution"
            )
        if (
            _text(control.get("template_id"), "control_round.template_id")
            != self.target["control_template_id"]
        ):
            raise OpenWorldSessionError(
                "the first round must use the frozen official control template"
            )
        _text(control.get("round_id"), "control_round.round_id")
        if control.get("experiment_candidate") is not None:
            raise OpenWorldSessionError(
                "the official control round cannot carry an ExperimentCandidate"
            )
        return control

    def _bind_control_round(
        self, round_plan: Mapping[str, Any]
    ) -> dict[str, Any]:
        control = self._validate_control_round(round_plan)
        if self._control_round is None:
            self._control_round = deepcopy(control)
        elif control != self._control_round:
            raise OpenWorldSessionError(
                "plan cannot rewrite the frozen official control round"
            )
        return control

    def _validate_query_contract(
        self, contract: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            normalized = validate_query_sufficiency_contract(contract)
        except QuerySufficiencyError as exc:
            raise OpenWorldSessionError(
                f"invalid open-world QueryContract: {exc}"
            ) from exc
        if normalized["schema_version"] != 2:
            raise OpenWorldSessionError(
                "open-world session requires QueryContract schema_version=2"
            )
        candidate_budget = self.target["max_rounds"] - 1
        if normalized["round_budget"] > candidate_budget:
            raise OpenWorldSessionError(
                "QueryContract spends rounds reserved for the official control"
            )
        if self._query_contract is not None:
            old = self._query_contract
            if normalized["claim_type"] != old["claim_type"]:
                raise OpenWorldSessionError(
                    "QueryContract cannot change claim_type during a session"
                )
            old_universe = list(old["candidate_universe"])
            new_universe = list(normalized["candidate_universe"])
            if new_universe[: len(old_universe)] != old_universe:
                raise OpenWorldSessionError(
                    "QueryContract cannot remove or reorder discovered candidates"
                )
        return normalized

    def _candidate_from_round(
        self,
        round_plan: Mapping[str, Any],
        *,
        location: str,
    ) -> dict[str, Any]:
        if not isinstance(round_plan, Mapping):
            raise OpenWorldSessionError(f"{location} must be an object")
        self._validate_optional_binding(round_plan, location=location)
        execution = round_plan.get("execution")
        if isinstance(execution, Mapping):
            self._validate_optional_binding(
                execution, location=f"{location}.execution"
            )
        try:
            candidate = validate_experiment_candidate(
                round_plan.get("experiment_candidate")
            )
        except (ExperimentCandidateError, TypeError) as exc:
            raise OpenWorldSessionError(
                f"{location} needs a valid ExperimentCandidate: {exc}"
            ) from exc
        if candidate["base_task"] != self.target["task_name"]:
            raise OpenWorldSessionError(
                f"{location} ExperimentCandidate cannot switch the base task"
            )
        round_candidate_id = round_plan.get(
            "candidate_id", candidate["candidate_id"]
        )
        if round_candidate_id != candidate["candidate_id"]:
            raise OpenWorldSessionError(
                f"{location}.candidate_id conflicts with ExperimentCandidate"
            )
        if round_plan.get("template_id") not in {None, ""}:
            raise OpenWorldSessionError(
                f"{location} runtime candidate cannot require a catalog template"
            )
        _text(round_plan.get("round_id"), f"{location}.round_id")
        return candidate

    def _normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, Mapping):
            raise OpenWorldSessionError("plan must be an object")
        normalized = deepcopy(dict(plan))
        self._validate_optional_binding(
            normalized, location="plan", allow_legacy_max_rounds=True
        )
        task_name = str(
            normalized.get("task_name") or self.target["task_name"]
        )
        if task_name != self.target["task_name"]:
            raise OpenWorldSessionError("plan cannot switch the bound task")
        rounds = normalized.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise OpenWorldSessionError(
                "open-world plan must contain the official control round"
            )
        if len(rounds) > self.target["max_rounds"]:
            raise OpenWorldSessionError(
                "materialized rounds exceed the open-world round budget"
            )

        normalized_rounds = [self._bind_control_round(rounds[0])]
        candidates: list[dict[str, Any]] = []
        candidate_ids: list[str] = []
        round_ids = {
            _text(normalized_rounds[0].get("round_id"), "rounds[0].round_id")
        }
        for index, raw_round in enumerate(rounds[1:], start=1):
            candidate = self._candidate_from_round(
                raw_round, location=f"rounds[{index}]"
            )
            candidate_id = candidate["candidate_id"]
            if candidate_id in candidate_ids:
                raise OpenWorldSessionError(
                    f"candidate was already materialized: {candidate_id}"
                )
            round_id = _text(
                raw_round.get("round_id"), f"rounds[{index}].round_id"
            )
            if round_id in round_ids:
                raise OpenWorldSessionError(
                    f"duplicate round_id in open-world plan: {round_id!r}"
                )
            round_ids.add(round_id)
            enriched = deepcopy(dict(raw_round))
            enriched["task_name"] = self.target["task_name"]
            enriched["candidate_id"] = candidate_id
            enriched["experiment_candidate"] = candidate
            normalized_rounds.append(enriched)
            candidates.append(candidate)
            candidate_ids.append(candidate_id)

        contract_value = normalized.get("query_contract")
        contract = (
            self._validate_query_contract(contract_value)
            if contract_value is not None
            else deepcopy(self._query_contract)
        )
        if contract is not None:
            missing = [
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id not in contract["candidate_universe"]
            ]
            if missing:
                if contract["candidate_universe_closed"]:
                    raise OpenWorldSessionError(
                        "closed QueryContract cannot accept a runtime candidate"
                    )
                try:
                    contract = extend_query_candidate_universe(
                        contract,
                        missing,
                        candidate_universe_closed=False,
                    )
                except QuerySufficiencyError as exc:
                    raise OpenWorldSessionError(
                        f"cannot extend QueryContract: {exc}"
                    ) from exc
            self._query_contract = self._validate_query_contract(contract)

        normalized["task_name"] = self.target["task_name"]
        normalized["policy"] = deepcopy(self.target["policy"])
        normalized["checkpoint"] = deepcopy(self.target["checkpoint"])
        normalized["checkpoint_id"] = self.target["checkpoint"].get(
            "checkpoint_id"
        )
        normalized["max_rounds"] = self.target["max_rounds"]
        normalized["rounds"] = normalized_rounds
        normalized["experiment_candidates"] = candidates
        normalized["requested_candidate_ids"] = candidate_ids
        if contract is not None:
            normalized["query_contract"] = deepcopy(contract)
        return normalized

    def normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one plan while keeping catalog and runtime concerns apart."""

        return self._normalize_plan(plan)

    def planning_context(self, repo_root: str | Path) -> dict[str, Any]:
        """Return trusted retrieval context for the frozen base task.

        The returned adapter templates are retrieval hints.  Open-world round
        authorization is owned by ``ExperimentCandidate`` validation here, not
        by membership in that template list.
        """

        return build_planning_context(repo_root, self.target)

    def assess_query_sufficiency(
        self,
        plan: Mapping[str, Any],
        candidate_evidence: Iterable[Mapping[str, Any]],
        *,
        completed_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate the plan's dynamically extended QueryContract."""

        normalized = self._normalize_plan(plan)
        contract = normalized.get("query_contract")
        if not isinstance(contract, Mapping):
            raise OpenWorldSessionError(
                "plan has no open-world QueryContract"
            )
        evidence = [deepcopy(dict(item)) for item in candidate_evidence]
        try:
            return assess_query_sufficiency(
                contract,
                evidence,
                completed_rounds=completed_rounds,
            )
        except (QuerySufficiencyError, TypeError, ValueError) as exc:
            raise OpenWorldSessionError(
                f"invalid candidate evidence: {exc}"
            ) from exc

    @staticmethod
    def _candidate_evidence_from_history(
        observation_history: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for observation in observation_history:
            if not isinstance(observation, Mapping):
                raise OpenWorldSessionError(
                    "observation history items must be objects"
                )
            value = observation.get("candidate_evidence")
            if isinstance(value, Mapping):
                result.append(deepcopy(dict(value)))
        return result

    def apply_plan_step(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        proposal: Mapping[str, Any],
        *,
        materialized_round: Mapping[str, Any] | None = None,
        source: str = "provider",
        query_contract: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Append one runtime candidate or stop under QueryContract evidence."""

        current = self._normalize_plan(plan)
        if len(observation_history) != len(current["rounds"]):
            raise OpenWorldSessionError(
                "every materialized round needs aligned evidence before planning"
            )
        if not isinstance(proposal, Mapping):
            raise OpenWorldSessionError("PlanStepProposal must be an object")
        step = deepcopy(dict(proposal))
        action = step.get("action")
        if action not in {"propose", "refine", "stop"}:
            raise OpenWorldSessionError(
                "PlanStepProposal.action must be propose, refine, or stop"
            )
        if query_contract is not None:
            current["query_contract"] = self._validate_query_contract(
                query_contract
            )
        contract = current.get("query_contract")
        assessment: dict[str, Any] | None = None
        if isinstance(contract, Mapping):
            evidence = self._candidate_evidence_from_history(
                observation_history
            )
            assessment = self.assess_query_sufficiency(
                current,
                evidence,
                completed_rounds=max(len(current["rounds"]) - 1, 0),
            )

        updated = deepcopy(current)
        next_round = None
        candidate = None
        if action == "stop":
            if materialized_round is not None:
                raise OpenWorldSessionError(
                    "stop PlanStepProposal cannot contain a materialized round"
                )
            if step.get("answered_query") is True and (
                assessment is None
                or assessment.get("evidence_sufficient") is not True
            ):
                raise OpenWorldSessionError(
                    "answered_query=true requires sufficient QueryContract evidence"
                )
            if assessment is not None and not assessment["should_stop"]:
                raise OpenWorldSessionError(
                    "QueryContract evidence requires another candidate round"
                )
        else:
            if len(current["rounds"]) >= self.target["max_rounds"]:
                raise OpenWorldSessionError(
                    "open-world round budget is exhausted"
                )
            try:
                candidate = validate_experiment_candidate(
                    step.get("experiment_candidate")
                )
            except (ExperimentCandidateError, TypeError) as exc:
                raise OpenWorldSessionError(
                    f"continuing PlanStepProposal needs ExperimentCandidate: {exc}"
                ) from exc
            if candidate["base_task"] != self.target["task_name"]:
                raise OpenWorldSessionError(
                    "PlanStepProposal cannot switch the base task"
                )
            if (
                step.get("candidate_id", candidate["candidate_id"])
                != candidate["candidate_id"]
            ):
                raise OpenWorldSessionError(
                    "PlanStepProposal candidate_id conflicts with ExperimentCandidate"
                )
            if candidate["candidate_id"] in current[
                "requested_candidate_ids"
            ]:
                raise OpenWorldSessionError(
                    "PlanStepProposal repeats an executed candidate"
                )
            if not isinstance(materialized_round, Mapping):
                raise OpenWorldSessionError(
                    "continuing PlanStepProposal needs a materialized round"
                )
            next_round = deepcopy(dict(materialized_round))
            next_round.setdefault("experiment_candidate", candidate)
            next_round.setdefault("candidate_id", candidate["candidate_id"])
            self._candidate_from_round(
                next_round, location="PlanStepProposal.materialized_round"
            )
            if (
                next_round["experiment_candidate"] != candidate
                or next_round["candidate_id"] != candidate["candidate_id"]
            ):
                raise OpenWorldSessionError(
                    "materialized round conflicts with PlanStepProposal candidate"
                )
            updated["rounds"].append(next_round)
            if isinstance(contract, Mapping):
                if contract["candidate_universe_closed"]:
                    raise OpenWorldSessionError(
                        "closed QueryContract cannot accept a runtime candidate"
                    )
                updated["query_contract"] = extend_query_candidate_universe(
                    contract,
                    [candidate["candidate_id"]],
                    candidate_universe_closed=False,
                )

        transition = {
            "propose": "switch_concern",
            "refine": "refine_concern",
            "stop": "stop",
        }[action]
        decision = {
            "schema_version": 2,
            "action": "stop" if action == "stop" else "continue",
            "transition": transition,
            "candidate_id": (
                candidate["candidate_id"] if candidate is not None else None
            ),
            "observation_summary": str(step.get("rationale") or "").strip(),
            "decision_reason": (
                "provider_authored_open_world_step"
                if source == "provider" or source.startswith("provider_")
                else "deterministic_fallback_after_provider_failure"
            ),
            "answered_query": bool(step.get("answered_query", False)),
            "plan_step_source": str(source),
            "plan_step_proposal": step,
            "round_budget_before_decision": (
                self.target["max_rounds"] - len(current["rounds"])
            ),
            "query_assessment": deepcopy(assessment),
            "next_round": deepcopy(next_round),
        }
        updated.setdefault("round_decisions", []).append(decision)
        updated.setdefault("plan_step_history", []).append(step)
        updated["planning_state"] = (
            f"stopped_after_round_{len(current['rounds'])}"
            if action == "stop"
            else f"awaiting_round_{len(updated['rounds'])}_observation"
        )
        return self._normalize_plan(updated), decision, {
            "schema_version": 1,
            "session_kind": "open_world_claim_first",
            "query_assessment": deepcopy(assessment),
            "round_budget_remaining": max(
                self.target["max_rounds"] - len(current["rounds"]), 0
            ),
        }

    def snapshot(
        self,
        user_query: str,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a compact auditable view of the runtime-generated session."""

        query = _text(user_query, "user_query")
        normalized = self._normalize_plan(plan)
        history = list(observation_history or [])
        if len(history) > len(normalized["rounds"]):
            raise OpenWorldSessionError(
                "observation history exceeds materialized rounds"
            )
        assessment = None
        if isinstance(normalized.get("query_contract"), Mapping):
            assessment = self.assess_query_sufficiency(
                normalized,
                self._candidate_evidence_from_history(history),
                completed_rounds=max(len(history) - 1, 0),
            )
        return {
            "schema_version": 2,
            "session_kind": "open_world_single_task_adaptive_evaluation",
            "user_query": query,
            "target": deepcopy(self.target),
            "control_round": deepcopy(normalized["rounds"][0]),
            "experiment_candidates": deepcopy(
                normalized["experiment_candidates"]
            ),
            "planning_state": normalized.get("planning_state"),
            "round_budget": normalized["max_rounds"],
            "completed_rounds": len(history),
            "rounds": deepcopy(normalized["rounds"]),
            "decisions": deepcopy(
                list(normalized.get("round_decisions") or [])
            ),
            "query_contract": deepcopy(normalized.get("query_contract")),
            "query_assessment": assessment,
        }


__all__ = [
    "OpenWorldPlanSession",
    "OpenWorldSessionError",
    "build_open_world_evaluation_target",
    "validate_open_world_evaluation_target",
]
