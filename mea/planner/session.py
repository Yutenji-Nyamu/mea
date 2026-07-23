"""Single-task, task-agnostic PlanSession contract.

One evaluation binds exactly one policy checkpoint and one RoboTwin task.  The
Plan Agent may adapt across sub-aspects and variants of that task, but changing
the task or checkpoint requires a new evaluation.  Task-specific planners are
adapters behind this common state boundary.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.proposals import (
    ProposalError,
    attach_round_proposals,
    validate_task_proposal,
    validate_tool_proposal,
)

from .catalog import catalog_task, validate_act_catalog
from .context import build_planning_context
from .evidence_policy import assess_conditional_transition
from .adaptive_step import validate_plan_step_proposal
from .query_contract import (
    assess_query_sufficiency as assess_query_contract,
    validate_query_sufficiency_contract,
)


class PlanSessionError(ValueError):
    """Raised when a plan attempts to leave its bound evaluation target."""


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
    "aspects",
}


def validate_adaptive_choice(
    assessment: Mapping[str, Any],
    choice: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a model-selected transition inside the evidence boundary.

    Evidence still owns whether execution may continue and which transition
    classes/aspects/templates are safe.  The optional choice may select any
    member of those sets.  Missing fields use the deterministic first-item
    fallback, preserving legacy replay and task-adapter behavior.
    """

    if not isinstance(assessment, Mapping):
        raise PlanSessionError("conditional assessment must be an object")
    if choice is None:
        supplied: dict[str, Any] = {}
    elif isinstance(choice, Mapping):
        supplied = dict(choice)
    else:
        raise PlanSessionError("adaptive choice must be an object")
    required_action = assessment.get("required_action")
    action = supplied.get("action", required_action)
    fallback_transition = assessment.get("required_transition")
    transition = supplied.get("transition", fallback_transition)
    remaining = assessment.get("remaining_template_ids_by_aspect")
    available = assessment.get("available_transitions")
    if required_action not in {"continue", "stop"}:
        raise PlanSessionError(
            f"unsupported required adaptive action: {required_action!r}"
        )
    if action != required_action:
        raise PlanSessionError(
            f"adaptive action {action!r} conflicts with evidence action "
            f"{required_action!r}"
        )
    if not isinstance(remaining, Mapping):
        raise PlanSessionError("conditional assessment is missing remaining templates")
    if not isinstance(available, Mapping):
        raise PlanSessionError("conditional assessment is missing available transitions")

    next_aspect = supplied.get("next_aspect_id")
    supplied_template = supplied.get("next_template_id")
    next_template = None
    if action == "stop":
        if (
            transition != "stop"
            or next_aspect is not None
            or supplied_template is not None
        ):
            raise PlanSessionError(
                "stop choice must use transition=stop and no next target"
            )
    else:
        if transition not in {"drill_down", "switch_aspect"}:
            raise PlanSessionError(
                "continue assessment must select an adaptive transition"
            )
        allowed_aspects = available.get(transition)
        if not isinstance(allowed_aspects, list) or not allowed_aspects:
            raise PlanSessionError(
                f"evidence does not allow transition {transition!r}"
            )
        if next_aspect is None and supplied_template is not None:
            matching_aspects = [
                str(aspect_id)
                for aspect_id in allowed_aspects
                if supplied_template in (remaining.get(aspect_id) or [])
            ]
            if len(matching_aspects) == 1:
                next_aspect = matching_aspects[0]
        if next_aspect is None:
            next_aspect = str(allowed_aspects[0])
        if next_aspect not in allowed_aspects:
            raise PlanSessionError(
                f"adaptive aspect {next_aspect!r} is outside allowed "
                f"{transition} candidates"
            )
        candidates = remaining.get(next_aspect)
        if not isinstance(candidates, list) or not candidates:
            raise PlanSessionError(
                "continue assessment has no remaining template for its aspect"
            )
        next_template = (
            str(candidates[0])
            if supplied_template is None
            else str(supplied_template)
        )
        if next_template not in candidates:
            raise PlanSessionError(
                f"adaptive template {next_template!r} is outside the allowed "
                f"candidates for {next_aspect!r}"
            )

    return {
        "schema_version": 1,
        "action": action,
        "transition": transition,
        "next_aspect_id": next_aspect,
        "next_template_id": next_template,
        "round_budget_remaining": assessment.get("round_budget_remaining"),
        "evidence_assessment": deepcopy(dict(assessment)),
    }


def build_adaptive_directive(
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deterministic fallback directive used by legacy callers."""

    return validate_adaptive_choice(assessment)


def build_evaluation_target(
    catalog: Mapping[str, Any],
    task_name: str,
    *,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    """Freeze one ready task/checkpoint before sub-aspect planning begins."""

    trusted_catalog = validate_act_catalog(catalog)
    task = catalog_task(trusted_catalog, str(task_name))
    task_max = int(task["max_rounds"])
    if max_rounds is None:
        resolved_max = task_max
    else:
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
            raise PlanSessionError("max_rounds must be an integer")
        resolved_max = max_rounds
    if resolved_max < 1 or resolved_max > task_max:
        raise PlanSessionError(
            f"max_rounds must be in [1, {task_max}] for {task_name!r}"
        )
    return {
        "schema_version": 1,
        "binding_mode": "single_task_single_checkpoint",
        "task_name": task["task_name"],
        "task_family": task["task_family"],
        "task_profile": task["task_profile"],
        "planner_kind": task["planner_kind"],
        "policy": deepcopy(trusted_catalog["policy"]),
        "checkpoint": deepcopy(task["checkpoint"]),
        "max_rounds": resolved_max,
        "aspects": deepcopy(task["aspects"]),
    }


def validate_evaluation_target(
    value: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TARGET_KEYS:
        raise PlanSessionError(
            f"EvaluationTarget fields must be exactly {sorted(_TARGET_KEYS)}"
        )
    target = deepcopy(dict(value))
    expected = build_evaluation_target(
        catalog,
        str(target.get("task_name") or ""),
        max_rounds=target.get("max_rounds"),
    )
    if target != expected:
        raise PlanSessionError("EvaluationTarget differs from the ready ACT catalog")
    return target


class BoundTaskPlanSession:
    """Normalize task-adapter plans into one paper-level session state."""

    def __init__(self, catalog: Mapping[str, Any], target: Mapping[str, Any]):
        self.catalog = validate_act_catalog(catalog)
        self.target = validate_evaluation_target(target, self.catalog)
        self.aspect_catalog = {
            str(item["aspect_id"]): deepcopy(item) for item in self.target["aspects"]
        }
        self.template_to_aspect = {
            str(template_id): aspect_id
            for aspect_id, aspect in self.aspect_catalog.items()
            for template_id in aspect["template_ids"]
        }

    @classmethod
    def from_catalog(
        cls,
        catalog: Mapping[str, Any],
        task_name: str,
        *,
        max_rounds: int | None = None,
    ) -> "BoundTaskPlanSession":
        return cls(
            catalog,
            build_evaluation_target(catalog, task_name, max_rounds=max_rounds),
        )

    def _selected_aspects(self, plan: Mapping[str, Any]) -> list[str]:
        raw = plan.get("requested_aspect_ids")
        if isinstance(raw, list) and raw:
            selected = [str(item) for item in raw]
        else:
            templates = plan.get("requested_template_ids")
            if not isinstance(templates, list) or not templates:
                raise PlanSessionError(
                    "plan must select requested_aspect_ids or requested_template_ids"
                )
            selected = []
            for template in templates:
                aspect = self.template_to_aspect.get(str(template))
                if aspect is None:
                    raise PlanSessionError(
                        f"template {template!r} is outside the bound task"
                    )
                if aspect not in selected:
                    selected.append(aspect)
        unknown = [item for item in selected if item not in self.aspect_catalog]
        if unknown:
            raise PlanSessionError(f"plan selected unsupported aspects: {unknown}")
        return selected

    def assess_query_sufficiency(
        self,
        contract: Mapping[str, Any],
        candidate_evidence: list[Mapping[str, Any]],
        *,
        completed_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Assess query truth conditions inside this session's frozen variants.

        The method is intentionally independent from ``navigation_options``:
        evidence may make a quantified query answerable before every routed
        aspect is covered, while an ambiguous diagnostic query may remain
        unanswered after its rollout budget ends.
        """

        normalized = validate_query_sufficiency_contract(contract)
        known_templates = set(self.template_to_aspect)
        unknown = sorted(
            set(normalized["candidate_universe"]) - known_templates
        )
        if unknown:
            raise PlanSessionError(
                f"query candidate universe leaves the bound task: {unknown}"
            )
        if normalized["round_budget"] > self.target["max_rounds"]:
            raise PlanSessionError(
                "query sufficiency contract exceeds the bound round budget"
            )
        return assess_query_contract(
            normalized,
            candidate_evidence,
            completed_rounds=completed_rounds,
        )

    def _normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, Mapping):
            raise PlanSessionError("plan must be an object")
        normalized = deepcopy(dict(plan))
        task_name = str(normalized.get("task_name") or self.target["task_name"])
        if task_name != self.target["task_name"]:
            raise PlanSessionError(
                f"plan cannot switch bound task {self.target['task_name']!r} to "
                f"{task_name!r}"
            )
        policy = normalized.get("policy")
        if policy is not None and policy != self.target["policy"]:
            raise PlanSessionError("plan cannot change the bound ACT policy contract")
        checkpoint = normalized.get("checkpoint")
        if checkpoint is not None and checkpoint != self.target["checkpoint"]:
            raise PlanSessionError("plan cannot change the bound ACT checkpoint")
        checkpoint_id = normalized.get("checkpoint_id")
        expected_checkpoint_id = self.target["checkpoint"].get("checkpoint_id")
        if checkpoint_id is not None and checkpoint_id != expected_checkpoint_id:
            raise PlanSessionError("plan cannot change the bound ACT checkpoint")
        raw_max_rounds = (
            normalized["max_rounds"]
            if "max_rounds" in normalized
            else self.target["max_rounds"]
        )
        if isinstance(raw_max_rounds, bool) or not isinstance(raw_max_rounds, int):
            raise PlanSessionError("plan max_rounds must be an integer")
        max_rounds = raw_max_rounds
        if max_rounds > self.target["max_rounds"] or max_rounds < 1:
            raise PlanSessionError("plan exceeds the bound round budget")
        rounds = normalized.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise PlanSessionError("plan must contain at least one round")
        if len(rounds) > max_rounds:
            raise PlanSessionError("materialized rounds exceed the plan budget")
        normalized["task_name"] = task_name
        normalized["max_rounds"] = max_rounds
        normalized["requested_aspect_ids"] = self._selected_aspects(normalized)
        normalized_rounds: list[dict[str, Any]] = []
        for round_plan in rounds:
            if not isinstance(round_plan, Mapping):
                raise PlanSessionError("every round must be an object")
            round_task = str(round_plan.get("task_name") or task_name)
            if round_task != task_name:
                raise PlanSessionError("round cannot switch the bound task")
            try:
                enriched = (
                    attach_round_proposals(round_plan)
                    if "task_proposal" not in round_plan
                    else deepcopy(dict(round_plan))
                )
                task_proposal = validate_task_proposal(
                    enriched["task_proposal"], expected_task_name=task_name
                )
                tool_proposal = validate_tool_proposal(
                    enriched["tool_proposal"],
                    expected_task_name=task_name,
                    expected_aspect_id=task_proposal["aspect_id"],
                )
            except (KeyError, ProposalError) as exc:
                raise PlanSessionError(f"invalid round proposal: {exc}") from exc
            enriched["task_name"] = task_name
            enriched["task_proposal"] = task_proposal
            enriched["tool_proposal"] = tool_proposal
            normalized_rounds.append(enriched)
        normalized["rounds"] = normalized_rounds
        return normalized

    def coverage(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Summarize query scope, dynamically discovered aspects, and evidence.

        A materialized next round is not covered until a matching observation
        exists.  This distinction prevents a planned template from being
        reported as evidence and gives final Feedback a compact answer/coverage
        contract without requiring a cross-task EvaluationGraph.
        """

        normalized = self._normalize_plan(plan)
        history = list(observation_history or [])
        if len(history) > len(normalized["rounds"]):
            raise PlanSessionError("observation history exceeds materialized rounds")
        for round_plan, observation in zip(normalized["rounds"], history):
            if not isinstance(observation, Mapping):
                raise PlanSessionError("each coverage observation must be an object")
            if observation.get("round_id") != round_plan.get("round_id"):
                raise PlanSessionError("coverage observation round_id does not match plan")

        required = list(normalized["requested_aspect_ids"])
        planned: list[str] = []
        covered: list[str] = []
        for index, round_plan in enumerate(normalized["rounds"]):
            aspect_id = str(round_plan["task_proposal"]["aspect_id"])
            if aspect_id not in planned:
                planned.append(aspect_id)
            if index < len(history) and aspect_id not in covered:
                covered.append(aspect_id)
        initially_requested = list(plan.get("initial_requested_aspect_ids") or required)
        discovered = [item for item in planned if item not in initially_requested]
        uncovered = [item for item in required if item not in covered]
        return {
            "schema_version": 1,
            "initial_requested_aspect_ids": initially_requested,
            "required_aspect_ids": required,
            "planned_aspect_ids": planned,
            "covered_aspect_ids": covered,
            "discovered_aspect_ids": discovered,
            "uncovered_required_aspect_ids": uncovered,
            "completed_rounds": len(history),
            "coverage_status": (
                "complete" if not uncovered else "partial" if covered else "not_started"
            ),
        }

    def navigation_options(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        *,
        allowed_template_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Expose all evidence-legal next steps in the frozen task catalog.

        Unlike the legacy adapter policy, this method does not freeze the
        sub-aspect set before round one.  It temporarily projects every trusted
        aspect/template of the already-bound task into the navigation catalog;
        the provider may then discover a new aspect, refine the current one, or
        stop.  A preregistered comparison may supply ``allowed_template_ids``
        to keep that discovery inside its hash-pinned candidate universe.
        Executable fields remain outside model control.
        """

        current = self._normalize_plan(plan)
        if allowed_template_ids is None:
            allowed = {
                str(template_id)
                for aspect in self.aspect_catalog.values()
                for template_id in aspect["template_ids"]
            }
        else:
            allowed_list = [str(template_id) for template_id in allowed_template_ids]
            if not allowed_list or len(allowed_list) != len(set(allowed_list)):
                raise PlanSessionError(
                    "allowed_template_ids must be a non-empty unique template list"
                )
            catalog_templates = {
                str(template_id)
                for aspect in self.aspect_catalog.values()
                for template_id in aspect["template_ids"]
            }
            unknown = sorted(set(allowed_list) - catalog_templates)
            if unknown:
                raise PlanSessionError(
                    f"allowed_template_ids contains unknown templates: {unknown}"
                )
            current_templates = {
                str(template_id)
                for template_id in current.get("requested_template_ids", [])
            }
            if not current_templates.issubset(set(allowed_list)):
                raise PlanSessionError(
                    "the current plan leaves the allowed template universe"
                )
            allowed = set(allowed_list)

        scoped_catalog = {
            aspect_id: {
                **deepcopy(aspect),
                "template_ids": [
                    str(template_id)
                    for template_id in aspect["template_ids"]
                    if str(template_id) in allowed
                ],
            }
            for aspect_id, aspect in self.aspect_catalog.items()
            if any(str(template_id) in allowed for template_id in aspect["template_ids"])
        }
        expanded = deepcopy(current)
        expanded["requested_aspect_ids"] = list(scoped_catalog)
        expanded["requested_template_ids"] = [
            str(template_id)
            for aspect in scoped_catalog.values()
            for template_id in aspect["template_ids"]
        ]
        assessment = assess_conditional_transition(
            expanded,
            observation_history,
            aspect_catalog=scoped_catalog,
        )
        current_aspect = str(assessment["current_aspect_id"])
        remaining = assessment["remaining_template_ids_by_aspect"]
        aspect_coverage = self.coverage(current, observation_history)
        initial_required = list(
            aspect_coverage["initial_requested_aspect_ids"]
        )
        covered = set(aspect_coverage["covered_aspect_ids"])
        initial_uncovered = [
            aspect_id for aspect_id in initial_required if aspect_id not in covered
        ]
        refine = []
        if remaining.get(current_aspect):
            refine.append(
                {
                    "aspect_id": current_aspect,
                    "template_ids": list(remaining[current_aspect]),
                }
            )
        propose = [
            {
                "aspect_id": aspect_id,
                "template_ids": list(template_ids),
                "initially_required": aspect_id in initial_required,
            }
            for aspect_id, template_ids in remaining.items()
            if aspect_id != current_aspect and template_ids
        ]
        propose.sort(
            key=lambda item: (
                item["aspect_id"] not in initial_uncovered,
                item["aspect_id"],
            )
        )

        unresolved = assessment["state"] in {
            "evidence_conflict",
            "aggregate_uncertain",
        }
        policy_success = assessment.get("policy_success")
        # Conflicting evidence, or a measured failure with an available
        # counterfactual, must first refine the current aspect.  Once that
        # aspect has no remaining template, switching remains legal so the
        # bounded evaluation cannot dead-end unnecessarily.
        if refine and (
            unresolved
            or (policy_success is not None and float(policy_success) < 1.0)
        ):
            propose = []

        measured_failure_with_counterfactual = bool(
            refine
            and policy_success is not None
            and float(policy_success) < 1.0
        )
        forced_stop = (
            assessment["state"] == "pipeline_failure"
            or int(assessment["round_budget_remaining"]) <= 0
            or (not refine and not propose)
        )
        stop_allowed = forced_stop or (
            not unresolved
            and not initial_uncovered
            and not measured_failure_with_counterfactual
        )
        if forced_stop:
            fallback = {
                "schema_version": 1,
                "action": "stop",
                "aspect_id": None,
                "template_id": None,
                "rationale": "Execution cannot safely continue inside the bound budget.",
                "answered_query": False,
            }
        else:
            transition = assessment.get("required_transition")
            required_proposals = [
                item
                for item in propose
                if item["aspect_id"] in initial_uncovered
            ]
            if transition == "drill_down" and refine:
                candidates = refine
            elif required_proposals:
                candidates = required_proposals
            elif stop_allowed:
                candidates = []
            else:
                candidates = refine
            if candidates:
                candidate = candidates[0]
                fallback = {
                    "schema_version": 1,
                    "action": "refine" if candidate in refine else "propose",
                    "aspect_id": candidate["aspect_id"],
                    "template_id": candidate["template_ids"][0],
                    "rationale": "Deterministic fallback selected the first legal evidence-conditioned step.",
                    "answered_query": False,
                }
            else:
                fallback = {
                    "schema_version": 1,
                    "action": "stop",
                    "aspect_id": None,
                    "template_id": None,
                    "rationale": (
                        "The initially required query aspects are covered; "
                        "provider failure must not spend rollout budget on an "
                        "unrequested discovery."
                    ),
                    "answered_query": True,
                }
        return {
            "schema_version": 1,
            "task_name": self.target["task_name"],
            "checkpoint_id": self.target["checkpoint"].get("checkpoint_id"),
            "current_aspect_id": current_aspect,
            "round_budget_remaining": assessment["round_budget_remaining"],
            "evidence_state": assessment["state"],
            "evidence_packet": deepcopy(assessment["evidence_packet"]),
            "initial_required_aspect_ids": initial_required,
            "covered_aspect_ids": sorted(covered),
            "uncovered_initial_required_aspect_ids": initial_uncovered,
            "discoverable_aspect_ids": [
                item["aspect_id"]
                for item in propose
                if item["aspect_id"] not in initial_required
            ],
            "available_steps": {
                "refine": refine,
                "propose": propose,
                "stop": stop_allowed,
            },
            "stop_requires_answered_query": stop_allowed and not forced_stop,
            "forced_stop": forced_stop,
            "fallback_step": fallback,
        }

    def apply_plan_step(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        proposal: Mapping[str, Any],
        *,
        materialized_round: Mapping[str, Any] | None = None,
        source: str = "provider",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Append one discovered/refined round, or stop, without leaving bounds."""

        current = self._normalize_plan(plan)
        options = self.navigation_options(current, observation_history)
        step = validate_plan_step_proposal(proposal, options)
        updated = deepcopy(current)
        action = step["action"]
        next_round = None
        if action != "stop":
            if not isinstance(materialized_round, Mapping):
                raise PlanSessionError("continuing PlanStepProposal needs a materialized round")
            self._validate_materialized_round(
                materialized_round,
                expected_aspect_id=str(step["aspect_id"]),
                expected_template_id=str(step["template_id"]),
                location="PlanStepProposal.materialized_round",
            )
            next_round = deepcopy(dict(materialized_round))
            if step["aspect_id"] not in updated["requested_aspect_ids"]:
                updated["requested_aspect_ids"].append(step["aspect_id"])
            requested_templates = list(updated.get("requested_template_ids") or [])
            if step["template_id"] not in requested_templates:
                requested_templates.append(step["template_id"])
            updated["requested_template_ids"] = requested_templates
            updated["rounds"].append(next_round)

        transition = {
            "propose": "switch_aspect",
            "refine": "drill_down",
            "stop": "stop",
        }[action]
        decision = {
            "schema_version": 3,
            "action": "stop" if action == "stop" else "continue",
            "transition": transition,
            "next_aspect_id": step["aspect_id"],
            "next_template_id": step["template_id"],
            "observation_summary": step["rationale"],
            "decision_reason": (
                "provider_authored_plan_step"
                if source == "provider" or source.startswith("provider_")
                else "deterministic_fallback_after_provider_failure"
            ),
            "answered_query": step["answered_query"],
            "plan_step_source": str(source),
            "plan_step_proposal": step,
            "round_budget_before_decision": options["round_budget_remaining"],
            "evidence_assessment": options,
            "next_round": next_round,
        }
        updated.setdefault("round_decisions", []).append(decision)
        updated.setdefault("plan_step_history", []).append(step)
        updated.setdefault(
            "initial_requested_aspect_ids", list(current["requested_aspect_ids"])
        )
        updated["planning_state"] = (
            f"stopped_after_round_{len(current['rounds'])}"
            if action == "stop"
            else f"awaiting_round_{len(updated['rounds'])}_observation"
        )
        return self._normalize_plan(updated), decision, options

    def snapshot(
        self,
        user_query: str,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a compact state used by reports and replay validation."""

        query = str(user_query).strip()
        if not query:
            raise PlanSessionError("user_query must be non-empty")
        normalized = self._normalize_plan(plan)
        return {
            "schema_version": 1,
            "session_kind": "bound_single_task_adaptive_evaluation",
            "user_query": query,
            "target": deepcopy(self.target),
            "selected_aspect_ids": list(normalized["requested_aspect_ids"]),
            "planning_state": normalized.get("planning_state"),
            "round_budget": normalized["max_rounds"],
            "rounds": [
                {
                    "round_id": item.get("round_id"),
                    "aspect_id": item["task_proposal"]["aspect_id"],
                    "template_id": item.get("template_id"),
                    "task_proposal": deepcopy(item["task_proposal"]),
                    "tool_proposal": deepcopy(item["tool_proposal"]),
                }
                for item in normalized["rounds"]
            ],
            "decisions": deepcopy(list(normalized.get("round_decisions") or [])),
            "aspect_coverage": self.coverage(normalized, observation_history),
        }

    def normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Return the executable bound-task plan after contract enrichment.

        Task-specific planners may still emit their legacy materialized shape.
        The PlanSession turns that shape into the common proposal contract that
        the execution loop consumes, while enforcing the frozen task, policy,
        checkpoint-derived capability set, and round budget.
        """

        return self._normalize_plan(plan)

    def planning_context(self, repo_root: str | Path) -> dict[str, Any]:
        """Project this frozen target into model-facing policy/simulator cards."""

        return build_planning_context(repo_root, self.target)

    def assess(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replay the same task-agnostic evidence transition used at runtime."""

        normalized = self._normalize_plan(plan)
        return assess_conditional_transition(
            normalized,
            observation_history,
            aspect_catalog=self.aspect_catalog,
        )

    def directive(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        *,
        candidate_decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a validated candidate or the deterministic legacy fallback."""

        assessment = self.assess(plan, observation_history)
        return validate_adaptive_choice(assessment, candidate_decision)

    def _validate_optional_binding(
        self,
        value: Mapping[str, Any],
        *,
        location: str,
    ) -> None:
        """Reject optional adapter metadata that leaves the frozen target."""

        expected = {
            "task_name": self.target["task_name"],
            "policy": self.target["policy"],
            "checkpoint": self.target["checkpoint"],
            "checkpoint_id": self.target["checkpoint"].get("checkpoint_id"),
            "max_rounds": self.target["max_rounds"],
        }
        for field, trusted in expected.items():
            if field in value and value[field] != trusted:
                raise PlanSessionError(f"{location} cannot change bound {field}")

    def _validate_materialized_round(
        self,
        round_plan: Mapping[str, Any],
        *,
        expected_aspect_id: str,
        expected_template_id: str,
        location: str,
    ) -> None:
        self._validate_optional_binding(round_plan, location=location)
        execution = round_plan.get("execution")
        if isinstance(execution, Mapping):
            self._validate_optional_binding(execution, location=f"{location}.execution")
        actual_template = str(round_plan.get("template_id") or "")
        if actual_template != expected_template_id:
            raise PlanSessionError(
                f"{location} template {actual_template!r} conflicts with "
                f"directive template {expected_template_id!r}"
            )
        trusted_aspect = self.template_to_aspect.get(actual_template)
        if trusted_aspect != expected_aspect_id:
            raise PlanSessionError(
                f"{location} template is not registered for directive aspect "
                f"{expected_aspect_id!r}"
            )
        proposal = round_plan.get("task_proposal")
        proposal_aspect = (
            proposal.get("aspect_id")
            if isinstance(proposal, Mapping)
            else round_plan.get("aspect_id") or round_plan.get("sub_aspect")
        )
        if proposal_aspect != expected_aspect_id:
            raise PlanSessionError(
                f"{location} aspect {proposal_aspect!r} conflicts with "
                f"directive aspect {expected_aspect_id!r}"
            )

    def adjudicate(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        *,
        candidate_plan: Mapping[str, Any],
        candidate_decision: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate an adapter-materialized transition and make it canonical.

        Existing task adapters still own task-specific round materialization.
        This final boundary owns adaptive navigation: it rejects any candidate
        that changes the frozen target, request, budget, prior rounds, or the
        evidence-selected aspect/template.  Legacy decisions may omit the new
        ``transition`` and ``next_aspect_id`` fields; they are injected after
        all supplied control fields have been checked.
        """

        if not isinstance(candidate_plan, Mapping):
            raise PlanSessionError("candidate_plan must be an object")
        if not isinstance(candidate_decision, Mapping):
            raise PlanSessionError("candidate_decision must be an object")
        current = self._normalize_plan(plan)
        candidate = self._normalize_plan(candidate_plan)
        supplied = dict(candidate_decision)
        directive = self.directive(
            current,
            observation_history,
            candidate_decision=supplied,
        )

        self._validate_optional_binding(
            candidate_decision, location="candidate_decision"
        )
        for field in ("requested_aspect_ids", "requested_template_ids"):
            if candidate.get(field) != current.get(field):
                raise PlanSessionError(f"candidate_plan cannot change {field}")
        if candidate["max_rounds"] != current["max_rounds"]:
            raise PlanSessionError("candidate_plan cannot change the round budget")

        current_rounds = current["rounds"]
        candidate_rounds = candidate["rounds"]
        if candidate_rounds[: len(current_rounds)] != current_rounds:
            raise PlanSessionError("candidate_plan cannot rewrite prior rounds")
        expected_round_count = len(current_rounds) + (
            1 if directive["action"] == "continue" else 0
        )
        if len(candidate_rounds) != expected_round_count:
            raise PlanSessionError(
                "candidate_plan round count conflicts with adaptive directive"
            )

        required_controls = {
            "action": directive["action"],
            "transition": directive["transition"],
            "next_aspect_id": directive["next_aspect_id"],
            "next_template_id": directive["next_template_id"],
        }
        if "action" not in supplied or "next_template_id" not in supplied:
            raise PlanSessionError(
                "candidate_decision must contain action and next_template_id"
            )
        for field, trusted in required_controls.items():
            if field in supplied and supplied[field] != trusted:
                raise PlanSessionError(
                    f"candidate_decision {field} conflicts with adaptive directive"
                )

        next_round = None
        if directive["action"] == "continue":
            next_round = candidate_rounds[-1]
            self._validate_materialized_round(
                next_round,
                expected_aspect_id=str(directive["next_aspect_id"]),
                expected_template_id=str(directive["next_template_id"]),
                location="candidate_plan.next_round",
            )
            supplied_next_round = supplied.get("next_round")
            if isinstance(supplied_next_round, Mapping):
                self._validate_materialized_round(
                    supplied_next_round,
                    expected_aspect_id=str(directive["next_aspect_id"]),
                    expected_template_id=str(directive["next_template_id"]),
                    location="candidate_decision.next_round",
                )
        elif supplied.get("next_round") is not None:
            raise PlanSessionError("stop decision cannot contain a next round")

        expected_state = (
            f"awaiting_round_{expected_round_count}_observation"
            if directive["action"] == "continue"
            else f"stopped_after_round_{len(current_rounds)}"
        )
        if candidate.get("planning_state") != expected_state:
            raise PlanSessionError(
                "candidate_plan planning_state conflicts with adaptive directive"
            )

        current_decisions = list(current.get("round_decisions") or [])
        candidate_decisions = list(candidate.get("round_decisions") or [])
        if candidate_decisions[: len(current_decisions)] != current_decisions:
            raise PlanSessionError("candidate_plan cannot rewrite prior decisions")
        if len(candidate_decisions) != len(current_decisions) + 1:
            raise PlanSessionError(
                "candidate_plan must append exactly one adapter decision"
            )
        if candidate_decisions[-1] != supplied:
            raise PlanSessionError(
                "candidate_plan decision does not match candidate_decision"
            )

        canonical_decision = deepcopy(supplied)
        canonical_decision.update(required_controls)
        canonical_decision["round_budget_before_decision"] = current[
            "max_rounds"
        ] - len(current_rounds)
        canonical_decision["evidence_assessment"] = deepcopy(
            directive["evidence_assessment"]
        )
        canonical_decision["next_round"] = deepcopy(next_round)
        candidate["round_decisions"][-1] = canonical_decision
        return candidate, canonical_decision


__all__ = [
    "BoundTaskPlanSession",
    "PlanSessionError",
    "build_adaptive_directive",
    "build_evaluation_target",
    "validate_adaptive_choice",
    "validate_evaluation_target",
]
