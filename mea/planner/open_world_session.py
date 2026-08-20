"""Frozen execution transport for one Plan Agent policy binding.

The public production owner is :class:`PlanAgentSession`; this module only
validates and transports its executable plan state after runtime binding has
selected one policy-ready base task. Later rounds carry Query-derived
Proposals rather than requiring an aspect or template registration.

The transport freezes task, policy, checkpoint, and total rollout budget.  Its
Runtime limits record whether an official control round is required.  They do
*not* make semantic Plan Agent decisions and is intentionally absent from the
package-level public API.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from .context import build_planning_context
from .experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from .runtime_limits import (
    PlanRuntimeError,
    summarize_plan_evidence,
    validate_agent_stop,
    validate_plan_runtime_limits,
)
from .policy_task_binding import (
    PolicyTaskBindingError,
    policy_task_binding_from_target,
)
from .query_interpretation import OFFICIAL_CONTROL_TEMPLATE_ID


class OpenWorldSessionError(ValueError):
    """Raised when an open-world plan leaves its frozen evaluation boundary."""


_TARGET_KEYS = {
    "schema_version",
    "binding_mode",
    "policy_task_binding",
    "max_rounds",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenWorldSessionError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OpenWorldSessionError(f"{field} must be a positive integer")
    return value


def validate_open_world_evaluation_target(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen runtime target.

    Task/checkpoint discovery is owned by ``runtime_task_binding``. Runtime
    candidates are never admitted by catalog membership.
    """

    if not isinstance(value, Mapping):
        raise OpenWorldSessionError(
            "OpenWorldEvaluationTarget must be an object"
        )
    raw = deepcopy(dict(value))
    if set(raw) != _TARGET_KEYS:
        raise OpenWorldSessionError(
            f"OpenWorldEvaluationTarget fields must be exactly {sorted(_TARGET_KEYS)}"
        )
    target = raw
    if target.get("schema_version") != 3:
        raise OpenWorldSessionError(
            "OpenWorldEvaluationTarget.schema_version must be 3"
        )
    if target.get("binding_mode") != "single_task_single_checkpoint_open_world":
        raise OpenWorldSessionError(
            "OpenWorldEvaluationTarget.binding_mode is invalid"
        )
    try:
        binding = policy_task_binding_from_target(target)
    except (PolicyTaskBindingError, TypeError) as exc:
        raise OpenWorldSessionError(str(exc)) from exc
    target["policy_task_binding"] = binding
    target["max_rounds"] = _positive_int(
        target.get("max_rounds"), "target.max_rounds"
    )
    return target


class _FrozenExecutionTransport:
    """Internal executable-plan validator for one frozen policy binding."""

    def __init__(
        self,
        target: Mapping[str, Any],
        *,
        control_round: Mapping[str, Any] | None = None,
        runtime_limits: Mapping[str, Any] | None = None,
    ):
        self.target = validate_open_world_evaluation_target(target)
        self.binding = policy_task_binding_from_target(self.target)
        self.task_name = self.binding["task_name"]
        self.policy = self.binding["policy"]
        self.checkpoint = self.binding["checkpoint"]
        self.control_template_id = OFFICIAL_CONTROL_TEMPLATE_ID
        self._control_round: dict[str, Any] | None = None
        self._runtime_limits: dict[str, Any] | None = None
        if runtime_limits is not None:
            self._runtime_limits = self._validate_runtime_limits(runtime_limits)
        if control_round is not None:
            if not self._control_required(self._runtime_limits):
                raise OpenWorldSessionError(
                    "runtime limits do not require an official control round"
                )
            self._control_round = self._validate_control_round(control_round)

    @classmethod
    def from_target(
        cls,
        target: Mapping[str, Any],
        *,
        control_round: Mapping[str, Any] | None = None,
        runtime_limits: Mapping[str, Any] | None = None,
    ) -> "_FrozenExecutionTransport":
        """Start from an already frozen runtime binding.

        This is the production constructor.  It keeps task/checkpoint discovery
        outside the Plan session; later candidates are authorized by their
        typed Proposal rather than a predeclared task menu.
        """

        return cls(
            target,
            control_round=control_round,
            runtime_limits=runtime_limits,
        )

    def _validate_optional_binding(
        self,
        value: Mapping[str, Any],
        *,
        location: str,
    ) -> None:
        expected = {
            "task_name": self.task_name,
            "policy": self.policy,
            "checkpoint": self.checkpoint,
            "checkpoint_id": self.checkpoint.get("checkpoint_id"),
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
            if supplied != self.target["max_rounds"]:
                raise OpenWorldSessionError(
                    f"{location} cannot change bound max_rounds"
                )

    def _validate_control_round(
        self, round_plan: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(round_plan, Mapping):
            raise OpenWorldSessionError("control round must be an object")
        control = deepcopy(dict(round_plan))
        self._validate_optional_binding(control, location="control_round")
        execution = control.get("execution")
        if isinstance(execution, Mapping):
            self._validate_optional_binding(
                execution, location="control_round.execution"
            )
        if (
            _text(control.get("template_id"), "control_round.template_id")
            != self.control_template_id
        ):
            raise OpenWorldSessionError(
                "the first round must use the frozen official control template"
            )
        if (
            "task_module" in control
            and control.get("task_module") != self.binding["task_module"]
        ):
            raise OpenWorldSessionError(
                "control_round cannot change bound task_module"
            )
        _text(control.get("round_id"), "control_round.round_id")
        if (
            control.get("proposal") is not None
            or control.get("experiment_candidate") is not None
        ):
            raise OpenWorldSessionError(
                "the official control round cannot carry a Proposal"
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

    def _validate_runtime_limits(
        self, contract: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            normalized = validate_plan_runtime_limits(contract)
        except PlanRuntimeError as exc:
            raise OpenWorldSessionError(
                f"invalid open-world runtime limits: {exc}"
            ) from exc
        candidate_budget = self.target["max_rounds"] - (
            1 if self._control_required(normalized) else 0
        )
        if normalized["round_budget"] > candidate_budget:
            raise OpenWorldSessionError(
                "runtime limits exceed the candidate-round budget after the "
                "control requirement"
            )
        if (
            not self._control_required(normalized)
            and self._control_round is not None
        ):
            raise OpenWorldSessionError(
                "runtime limits cannot disable an already bound control round"
            )
        if self._runtime_limits is not None:
            old = self._runtime_limits
            if (
                normalized["control_requirement"]
                != old["control_requirement"]
            ):
                raise OpenWorldSessionError(
                    "runtime limits cannot change control_requirement during a "
                    "session"
                )
        return normalized

    @staticmethod
    def _control_required(contract: Mapping[str, Any] | None) -> bool:
        return (
            contract is None
            or contract.get("control_requirement", "required") == "required"
        )

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
            raw_proposal = round_plan.get("proposal")
            if raw_proposal is None:
                raw_proposal = round_plan.get("experiment_candidate")
            candidate = validate_experiment_candidate(
                raw_proposal
            )
        except (ExperimentCandidateError, TypeError) as exc:
            raise OpenWorldSessionError(
                f"{location} needs a valid Proposal: {exc}"
            ) from exc
        if candidate["base_task"] != self.task_name:
            raise OpenWorldSessionError(
                f"{location} Proposal cannot switch the base task"
            )
        round_candidate_id = round_plan.get(
            "candidate_id", candidate["candidate_id"]
        )
        if round_candidate_id != candidate["candidate_id"]:
            raise OpenWorldSessionError(
                f"{location}.candidate_id conflicts with Proposal"
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
        self._validate_optional_binding(normalized, location="plan")
        task_name = str(
            normalized.get("task_name") or self.task_name
        )
        if task_name != self.task_name:
            raise OpenWorldSessionError("plan cannot switch the bound task")
        contract_value = normalized.get("runtime_limits")
        contract = (
            self._validate_runtime_limits(contract_value)
            if contract_value is not None
            else deepcopy(self._runtime_limits)
        )
        control_required = self._control_required(contract)
        rounds = normalized.get("rounds")
        if not isinstance(rounds, list):
            raise OpenWorldSessionError(
                "open-world plan rounds must be a list"
            )
        if control_required and not rounds:
            raise OpenWorldSessionError(
                "open-world plan must contain the official control round"
            )
        if len(rounds) > self.target["max_rounds"]:
            raise OpenWorldSessionError(
                "materialized rounds exceed the open-world round budget"
            )

        normalized_rounds: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        candidate_ids: list[str] = []
        round_ids: set[str] = set()
        candidate_start = 0
        if control_required:
            control = self._bind_control_round(rounds[0])
            normalized_rounds.append(control)
            round_ids.add(
                _text(control.get("round_id"), "rounds[0].round_id")
            )
            candidate_start = 1
        for index, raw_round in enumerate(
            rounds[candidate_start:],
            start=candidate_start,
        ):
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
            enriched.pop("experiment_candidate", None)
            enriched["task_name"] = self.task_name
            enriched["candidate_id"] = candidate_id
            enriched["proposal"] = candidate
            normalized_rounds.append(enriched)
            candidates.append(candidate)
            candidate_ids.append(candidate_id)

        if contract is not None:
            self._runtime_limits = self._validate_runtime_limits(contract)

        normalized["task_name"] = self.task_name
        normalized["policy"] = deepcopy(self.policy)
        normalized["checkpoint"] = deepcopy(self.checkpoint)
        normalized["checkpoint_id"] = self.checkpoint.get("checkpoint_id")
        normalized["max_rounds"] = self.target["max_rounds"]
        normalized["rounds"] = normalized_rounds
        normalized.pop("experiment_candidates", None)
        normalized["proposals"] = candidates
        normalized["requested_candidate_ids"] = candidate_ids
        if contract is not None:
            normalized["runtime_limits"] = deepcopy(contract)
        return normalized

    def normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one plan inside the frozen runtime binding."""

        return self._normalize_plan(plan)

    def planning_context(self, repo_root: str | Path) -> dict[str, Any]:
        """Return trusted task/policy context for the frozen base task."""

        return build_planning_context(repo_root, self.target)

    def summarize_evidence(
        self,
        plan: Mapping[str, Any],
        candidate_evidence: Iterable[Mapping[str, Any]],
        *,
        completed_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Summarize completed evidence within the external runtime limits."""

        normalized = self._normalize_plan(plan)
        contract = normalized.get("runtime_limits")
        if not isinstance(contract, Mapping):
            raise OpenWorldSessionError(
                "plan has no open-world runtime limits"
            )
        evidence = [deepcopy(dict(item)) for item in candidate_evidence]
        try:
            return summarize_plan_evidence(
                contract,
                evidence,
                completed_rounds=completed_rounds,
            )
        except (PlanRuntimeError, TypeError, ValueError) as exc:
            raise OpenWorldSessionError(
                f"invalid candidate evidence: {exc}"
            ) from exc

    @staticmethod
    def _is_unchanged_official_retry(
        round_plan: Mapping[str, Any],
    ) -> bool:
        if round_plan.get("route") != "official":
            return False
        proposal = round_plan.get("proposal") or round_plan.get(
            "experiment_candidate"
        )
        return bool(
            isinstance(proposal, Mapping)
            and all(
                proposal.get(field) is None
                for field in (
                    "scene_need",
                    "checker_need",
                    "rule_tool_need",
                    "vqa_tool_need",
                    "tool_need",
                )
            )
        )

    @classmethod
    def _policy_candidate_evidence_from_history(
        cls,
        round_plans: Iterable[Mapping[str, Any]],
        observation_history: Iterable[Mapping[str, Any]],
        *,
        control_required: bool,
    ) -> list[dict[str, Any]]:
        """Return evidence charged to the post-control candidate budget."""

        plans = list(round_plans)
        observations_history = list(observation_history)
        if len(plans) != len(observations_history):
            raise OpenWorldSessionError(
                "round plans and observation history must be aligned"
            )
        result: list[dict[str, Any]] = []
        for index, (round_plan, observation) in enumerate(
            zip(plans, observations_history)
        ):
            if not isinstance(observation, Mapping):
                raise OpenWorldSessionError(
                    "observation history items must be objects"
                )
            if control_required and index == 0:
                continue
            if cls._is_unchanged_official_retry(round_plan):
                continue
            observations = observation.get("observations")
            if (
                isinstance(observations, Mapping)
                and isinstance(
                    observations.get("planning_observation"), Mapping
                )
            ):
                continue
            value = observation.get("candidate_evidence")
            if isinstance(value, Mapping):
                result.append(deepcopy(dict(value)))
        return result

    @classmethod
    def _completed_policy_candidate_rounds(
        cls,
        round_plans: Iterable[Mapping[str, Any]],
        observation_history: Iterable[Mapping[str, Any]],
        *,
        control_required: bool,
    ) -> int:
        plans = list(round_plans)
        observations_history = list(observation_history)
        if len(plans) != len(observations_history):
            raise OpenWorldSessionError(
                "round plans and observation history must be aligned"
            )
        completed = 0
        for index, (round_plan, observation) in enumerate(
            zip(plans, observations_history)
        ):
            if not isinstance(observation, Mapping):
                raise OpenWorldSessionError(
                    "observation history items must be objects"
                )
            if control_required and index == 0:
                continue
            if cls._is_unchanged_official_retry(round_plan):
                completed += 1
                continue
            observations = observation.get("observations")
            if (
                isinstance(observations, Mapping)
                and isinstance(
                    observations.get("planning_observation"), Mapping
                )
            ):
                continue
            completed += 1
        return completed

    def apply_plan_step(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        proposal: Mapping[str, Any],
        *,
        materialized_round: Mapping[str, Any] | None = None,
        source: str = "provider",
        runtime_limits: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Append one runtime candidate or an Agent-authored stop."""

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
        if runtime_limits is not None:
            current["runtime_limits"] = self._validate_runtime_limits(
                runtime_limits
            )
        contract = current.get("runtime_limits")
        assessment: dict[str, Any] | None = None
        if isinstance(contract, Mapping):
            control_required = self._control_required(contract)
            evidence = self._policy_candidate_evidence_from_history(
                current["rounds"],
                observation_history,
                control_required=control_required,
            )
            completed_candidate_rounds = (
                self._completed_policy_candidate_rounds(
                    current["rounds"],
                    observation_history,
                    control_required=control_required,
                )
            )
            assessment = self.summarize_evidence(
                current,
                evidence,
                completed_rounds=completed_candidate_rounds,
            )

        updated = deepcopy(current)
        next_round = None
        candidate = None
        if action == "stop":
            if materialized_round is not None:
                raise OpenWorldSessionError(
                    "stop PlanStepProposal cannot contain a materialized round"
                )
            if assessment is not None:
                try:
                    answered = bool(step.get("answered_query", False))
                    assessment = validate_agent_stop(
                        assessment,
                        rationale=str(step.get("rationale") or ""),
                        answer=(
                            str(step.get("answer") or "").strip() or None
                        ),
                        claim_verdict=str(
                            step.get("claim_verdict")
                            or ("inconclusive" if not answered else "")
                        ),
                        evidence_sufficient=answered,
                    )
                except PlanRuntimeError as exc:
                    raise OpenWorldSessionError(str(exc)) from exc
        else:
            if len(current["rounds"]) >= self.target["max_rounds"]:
                raise OpenWorldSessionError(
                    "open-world round budget is exhausted"
                )
            try:
                raw_proposal = step.get("proposal")
                if raw_proposal is None:
                    raw_proposal = step.get("experiment_candidate")
                candidate = validate_experiment_candidate(
                    raw_proposal
                )
            except (ExperimentCandidateError, TypeError) as exc:
                raise OpenWorldSessionError(
                    f"continuing PlanStepProposal needs Proposal: {exc}"
                ) from exc
            if candidate["base_task"] != self.task_name:
                raise OpenWorldSessionError(
                    "PlanStepProposal cannot switch the base task"
                )
            if (
                step.get("candidate_id", candidate["candidate_id"])
                != candidate["candidate_id"]
            ):
                raise OpenWorldSessionError(
                    "PlanStepProposal candidate_id conflicts with Proposal"
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
            next_round.pop("experiment_candidate", None)
            next_round.setdefault("proposal", candidate)
            next_round.setdefault("candidate_id", candidate["candidate_id"])
            self._candidate_from_round(
                next_round, location="PlanStepProposal.materialized_round"
            )
            if (
                next_round["proposal"] != candidate
                or next_round["candidate_id"] != candidate["candidate_id"]
            ):
                raise OpenWorldSessionError(
                    "materialized round conflicts with PlanStepProposal candidate"
                )
            updated["rounds"].append(next_round)

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
            "session_kind": "plan_agent_session",
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
        if isinstance(normalized.get("runtime_limits"), Mapping):
            control_required = self._control_required(
                normalized["runtime_limits"]
            )
            candidate_evidence = self._policy_candidate_evidence_from_history(
                normalized["rounds"][: len(history)],
                history,
                control_required=control_required,
            )
            completed_candidate_rounds = (
                self._completed_policy_candidate_rounds(
                    normalized["rounds"][: len(history)],
                    history,
                    control_required=control_required,
                )
            )
            assessment = self.summarize_evidence(
                normalized,
                candidate_evidence,
                completed_rounds=completed_candidate_rounds,
            )
        control_required = self._control_required(
            normalized.get("runtime_limits")
            if isinstance(normalized.get("runtime_limits"), Mapping)
            else None
        )
        return {
            "schema_version": 2,
            "session_kind": "open_world_single_task_adaptive_evaluation",
            "user_query": query,
            "target": deepcopy(self.target),
            "control_round": (
                deepcopy(normalized["rounds"][0])
                if control_required and normalized["rounds"]
                else None
            ),
            "proposals": deepcopy(
                normalized["proposals"]
            ),
            "planning_state": normalized.get("planning_state"),
            "round_budget": normalized["max_rounds"],
            "completed_rounds": len(history),
            "rounds": deepcopy(normalized["rounds"]),
            "decisions": deepcopy(
                list(normalized.get("round_decisions") or [])
            ),
            "runtime_limits": deepcopy(normalized.get("runtime_limits")),
            "query_assessment": assessment,
        }


# Compatibility imports for historical tests and external callers.  Production
# code owns this transport through ``PlanAgentSession``; these aliases are not
# re-exported from ``mea.planner``.
PlanAgentExecutionSession = _FrozenExecutionTransport
OpenWorldPlanSession = _FrozenExecutionTransport


__all__ = [
    "OpenWorldSessionError",
    "validate_open_world_evaluation_target",
]
