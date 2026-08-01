"""Frozen execution transport for one Plan Agent policy binding.

The legacy :class:`BoundTaskPlanSession` is intentionally a finite catalog
protocol.  The public production owner is :class:`PlanAgentSession`; this
module only validates and transports its executable plan state after runtime
binding has selected one policy-ready base task.  Later rounds carry
Query-derived Proposals rather than requiring an aspect or template
registration.

The transport freezes task, policy, checkpoint, and total rollout budget.  Its
QueryContract decides whether an official control round is required.  It does
*not* make semantic Plan Agent decisions and is intentionally absent from the
package-level public API.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from mea.capability_adapter import resolve_task_retrieval_index

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
from .policy_task_binding import (
    PolicyTaskBindingError,
    build_policy_task_binding,
    policy_task_binding_from_target,
)


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


def _decision_planning_lineage(
    step: Mapping[str, Any],
    observation_history: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate round ordering for an auditable Plan Agent decision."""

    raw = step.get("planning_lineage")
    if raw is None:
        if step.get("action") in {"propose", "refine"}:
            raise OpenWorldSessionError(
                "continuing open-world PlanStepProposal requires "
                "planning_lineage"
            )
        return None
    if not isinstance(raw, Mapping):
        raise OpenWorldSessionError(
            "PlanStepProposal.planning_lineage must be an object"
        )
    lineage = deepcopy(dict(raw))
    required = {
        "schema_version",
        "decision_kind",
        "evidence_conditioned",
        "completed_round_ids",
        "completed_round_count",
        "input_digest",
    }
    if set(lineage) != required or lineage.get("schema_version") != 1:
        raise OpenWorldSessionError(
            "PlanStepProposal.planning_lineage has an invalid schema"
        )
    raw_ids = lineage.get("completed_round_ids")
    if not isinstance(raw_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_ids
    ):
        raise OpenWorldSessionError(
            "planning_lineage.completed_round_ids must contain round ids"
        )
    observations = list(observation_history)
    observed_ids = [
        _text(item.get("round_id"), "observation_history[].round_id")
        for item in observations
        if isinstance(item, Mapping)
    ]
    if len(observed_ids) != len(observations):
        raise OpenWorldSessionError(
            "observation history items must be objects"
        )
    if lineage.get("completed_round_count") != len(raw_ids):
        raise OpenWorldSessionError(
            "planning_lineage completed-round count is inconsistent"
        )
    decision_kind = lineage.get("decision_kind")
    if decision_kind == "evidence_conditioned_refinement":
        if lineage.get("evidence_conditioned") is not True:
            raise OpenWorldSessionError(
                "evidence-conditioned refinement must set "
                "evidence_conditioned=true"
            )
        if raw_ids != observed_ids or not raw_ids:
            raise OpenWorldSessionError(
                "evidence-conditioned refinement must name every completed "
                "round in order"
            )
        digest = lineage.get("input_digest")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OpenWorldSessionError(
                "evidence-conditioned refinement needs a sha256 input digest"
            )
    elif decision_kind == "query_initial_candidate":
        if (
            lineage.get("evidence_conditioned") is not False
            or raw_ids
            or observed_ids
        ):
            raise OpenWorldSessionError(
                "query-initial planning is valid only before any round "
                "evidence exists"
            )
    elif decision_kind == "pre_evidence_query_candidate":
        if (
            lineage.get("evidence_conditioned") is not False
            or raw_ids
            or observed_ids
        ):
            raise OpenWorldSessionError(
                "pre-evidence planning is valid only before any round "
                "evidence exists"
            )
    else:
        raise OpenWorldSessionError(
            "planning_lineage.decision_kind is not recognized"
        )
    return lineage


def _catalog_templates(task: Mapping[str, Any]) -> set[str]:
    return {
        str(template_id)
        for aspect in task.get("aspects", [])
        if isinstance(aspect, Mapping)
        for template_id in aspect.get("template_ids", [])
    }


def _retrieval_aspects(task_name: str) -> list[dict[str, Any]]:
    """Project legacy capability contracts into non-authoritative hints."""

    retrieval_index = resolve_task_retrieval_index(
        task_name,
        allow_unregistered=True,
    )
    grouped: dict[str, dict[str, Any]] = {}
    for contract in retrieval_index["entries"]:
        aspect = contract["aspect"]
        aspect_id = str(aspect["aspect_id"])
        entry = grouped.setdefault(
            aspect_id,
            {
                "aspect_id": aspect_id,
                "description": (
                    "Retrieval hint for "
                    f"{aspect['semantic_scope']} / {aspect['target_role']}."
                ),
                "template_ids": [],
            },
        )
        entry["template_ids"].append(str(contract["template_id"]))
    return [deepcopy(grouped[key]) for key in sorted(grouped)]


def build_open_world_evaluation_target(
    catalog: Mapping[str, Any],
    task_name: str,
    *,
    max_rounds: int,
    task_module: str | None = None,
) -> dict[str, Any]:
    """Freeze only the policy/task execution boundary and runtime budget.

    The catalog is a compatibility discovery input here.  Planner kinds,
    profiles, catalog round caps, aspects, and templates are intentionally not
    copied into the production target.
    """

    trusted_catalog = validate_act_catalog(catalog)
    task = catalog_task(trusted_catalog, _text(task_name, "task_name"))
    budget = _positive_int(max_rounds, "max_rounds")
    retrieval_index = resolve_task_retrieval_index(
        task["task_name"],
        allow_unregistered=True,
    )
    control_template = _text(
        retrieval_index.get("control_template_id"),
        "TaskRetrievalIndex.control_template_id",
    )
    if control_template not in _catalog_templates(task):
        raise OpenWorldSessionError(
            "the TaskAdapter control template is absent from the retrieval catalog"
        )
    return {
        "schema_version": 3,
        "binding_mode": "single_task_single_checkpoint_open_world",
        "policy_task_binding": build_policy_task_binding(
            task_name=task["task_name"],
            task_family=task["task_family"],
            policy=trusted_catalog["policy"],
            checkpoint=task["checkpoint"],
            task_module=task_module,
        ),
        "max_rounds": budget,
    }


def validate_open_world_evaluation_target(
    value: Mapping[str, Any],
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a frozen runtime target.

    The production Plan Agent path validates the bound target directly;
    the catalog is only needed while task/checkpoint discovery builds the
    frozen target. Runtime candidates are never admitted by catalog membership.
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
    task_name = binding["task_name"]
    target["policy_task_binding"] = binding
    target["max_rounds"] = _positive_int(
        target.get("max_rounds"), "target.max_rounds"
    )
    if catalog is None:
        return target
    expected = build_open_world_evaluation_target(
        catalog,
        task_name,
        max_rounds=target["max_rounds"],
        task_module=binding["task_module"],
    )
    if target != expected:
        raise OpenWorldSessionError(
            "OpenWorldEvaluationTarget differs from the ready ACT catalog"
        )
    return target


class _FrozenExecutionTransport:
    """Internal executable-plan validator for one frozen policy binding."""

    def __init__(
        self,
        catalog: Mapping[str, Any] | None,
        target: Mapping[str, Any],
        *,
        control_round: Mapping[str, Any] | None = None,
        query_contract: Mapping[str, Any] | None = None,
    ):
        self.catalog = (
            validate_act_catalog(catalog) if catalog is not None else None
        )
        self.target = validate_open_world_evaluation_target(
            target, self.catalog
        )
        self.binding = policy_task_binding_from_target(self.target)
        self.task_name = self.binding["task_name"]
        self.policy = self.binding["policy"]
        self.checkpoint = self.binding["checkpoint"]
        self.control_template_id = resolve_task_retrieval_index(
            self.task_name,
            allow_unregistered=True,
        )["control_template_id"]
        self.retrieval_aspects = _retrieval_aspects(self.task_name)
        self._control_round: dict[str, Any] | None = None
        self._query_contract: dict[str, Any] | None = None
        if query_contract is not None:
            self._query_contract = self._validate_query_contract(query_contract)
        if control_round is not None:
            if not self._control_required(self._query_contract):
                raise OpenWorldSessionError(
                    "QueryContract does not require an official control round"
                )
            self._control_round = self._validate_control_round(control_round)

    @classmethod
    def from_target(
        cls,
        target: Mapping[str, Any],
        *,
        control_round: Mapping[str, Any] | None = None,
        query_contract: Mapping[str, Any] | None = None,
    ) -> "_FrozenExecutionTransport":
        """Start from an already frozen runtime binding.

        This is the production constructor.  It keeps task/checkpoint discovery
        outside the Plan session and makes the catalog a retrieval concern
        rather than a planning or execution authorization boundary.
        """

        return cls(
            None,
            target,
            control_round=control_round,
            query_contract=query_contract,
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

    def _validate_query_contract(
        self, contract: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            normalized = validate_query_sufficiency_contract(contract)
        except QuerySufficiencyError as exc:
            raise OpenWorldSessionError(
                f"invalid open-world QueryContract: {exc}"
            ) from exc
        if normalized["schema_version"] != 3:
            raise OpenWorldSessionError(
                "open-world session requires normalized QueryContract "
                "schema_version=3"
            )
        candidate_budget = self.target["max_rounds"] - (
            1 if self._control_required(normalized) else 0
        )
        if normalized["round_budget"] > candidate_budget:
            raise OpenWorldSessionError(
                "QueryContract exceeds the candidate-round budget after its "
                "control requirement"
            )
        if (
            not self._control_required(normalized)
            and self._control_round is not None
        ):
            raise OpenWorldSessionError(
                "QueryContract cannot disable an already bound control round"
            )
        if self._query_contract is not None:
            old = self._query_contract
            if normalized["claim_type"] != old["claim_type"]:
                raise OpenWorldSessionError(
                    "QueryContract cannot change claim_type during a session"
                )
            if (
                normalized["control_requirement"]
                != old["control_requirement"]
            ):
                raise OpenWorldSessionError(
                    "QueryContract cannot change control_requirement during a "
                    "session"
                )
            old_universe = list(old["candidate_universe"])
            new_universe = list(normalized["candidate_universe"])
            if new_universe[: len(old_universe)] != old_universe:
                raise OpenWorldSessionError(
                    "QueryContract cannot remove or reorder discovered candidates"
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
        contract_value = normalized.get("query_contract")
        contract = (
            self._validate_query_contract(contract_value)
            if contract_value is not None
            else deepcopy(self._query_contract)
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
            normalized["query_contract"] = deepcopy(contract)
        return normalized

    def normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one plan while keeping catalog and runtime concerns apart."""

        return self._normalize_plan(plan)

    def planning_context(self, repo_root: str | Path) -> dict[str, Any]:
        """Return trusted retrieval context for the frozen base task.

        The returned adapter templates are retrieval hints.  Open-world round
        authorization is owned by typed Proposal validation here, not
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

    @staticmethod
    def _completed_policy_candidate_rounds(
        observation_history: Iterable[Mapping[str, Any]],
        *,
        control_required: bool,
    ) -> int:
        completed = 0
        for index, observation in enumerate(observation_history):
            if not isinstance(observation, Mapping):
                raise OpenWorldSessionError(
                    "observation history items must be objects"
                )
            if control_required and index == 0:
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
        planning_lineage = _decision_planning_lineage(
            step,
            observation_history,
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
            completed_candidate_rounds = (
                self._completed_policy_candidate_rounds(
                    observation_history,
                    control_required=self._control_required(contract),
                )
            )
            assessment = self.assess_query_sufficiency(
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
            "planning_lineage": deepcopy(planning_lineage),
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
        if isinstance(normalized.get("query_contract"), Mapping):
            candidate_evidence = self._candidate_evidence_from_history(history)
            completed_candidate_rounds = (
                self._completed_policy_candidate_rounds(
                    history,
                    control_required=self._control_required(
                        normalized["query_contract"]
                    ),
                )
            )
            assessment = self.assess_query_sufficiency(
                normalized,
                candidate_evidence,
                completed_rounds=completed_candidate_rounds,
            )
        control_required = self._control_required(
            normalized.get("query_contract")
            if isinstance(normalized.get("query_contract"), Mapping)
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
            "query_contract": deepcopy(normalized.get("query_contract")),
            "query_assessment": assessment,
        }


# Compatibility imports for historical tests and external callers.  Production
# code owns this transport through ``PlanAgentSession``; these aliases are not
# re-exported from ``mea.planner``.
PlanAgentExecutionSession = _FrozenExecutionTransport
OpenWorldPlanSession = _FrozenExecutionTransport


__all__ = [
    "OpenWorldSessionError",
    "build_open_world_evaluation_target",
    "validate_open_world_evaluation_target",
]
