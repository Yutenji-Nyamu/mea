"""Single-task, task-agnostic PlanSession contract.

One evaluation binds exactly one policy checkpoint and one RoboTwin task.  The
Plan Agent may adapt across sub-aspects and variants of that task, but changing
the task or checkpoint requires a new evaluation.  Task-specific planners are
adapters behind this common state boundary.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from mea.proposals import (
    ProposalError,
    attach_round_proposals,
    validate_task_proposal,
    validate_tool_proposal,
)

from .catalog import catalog_task, validate_act_catalog
from .evidence_policy import assess_conditional_transition


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
            str(item["aspect_id"]): deepcopy(item)
            for item in self.target["aspects"]
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
            build_evaluation_target(
                catalog, task_name, max_rounds=max_rounds
            ),
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

    def snapshot(self, user_query: str, plan: Mapping[str, Any]) -> dict[str, Any]:
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
        }

    def normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Return the executable bound-task plan after contract enrichment.

        Task-specific planners may still emit their legacy materialized shape.
        The PlanSession turns that shape into the common proposal contract that
        the execution loop consumes, while enforcing the frozen task, policy,
        checkpoint-derived capability set, and round budget.
        """

        return self._normalize_plan(plan)

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


__all__ = [
    "BoundTaskPlanSession",
    "PlanSessionError",
    "build_evaluation_target",
    "validate_evaluation_target",
]
