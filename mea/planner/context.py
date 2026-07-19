"""Trusted model-facing context for one bound MEA planning session.

The paper-level Plan Agent needs policy metadata, simulator constraints, and
the task capabilities that it may request.  Those facts already exist in the
EvaluationTarget, TaskSchema, and declarative capability registry.  This
module projects them into one compact JSON contract without exposing local
paths or inventing policy metadata that is not available in the repository.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.capability_adapter import registered_capability_contracts
from mea.toolkit import load_task_schema


class PlanningContextError(ValueError):
    """Raised when trusted planning context is incomplete or has been changed."""


_CONTEXT_KEYS = {
    "schema_version",
    "policy_card",
    "simulator_card",
    "adapter_view",
}
_POLICY_KEYS = {
    "schema_version",
    "policy_name",
    "checkpoint_id",
    "checkpoint_setting",
    "expert_data_num",
    "language_conditioned",
    "single_task_checkpoint",
    "task_name",
    "action_dimension",
    "checkpoint_ready",
    "unknown_metadata",
}
_SIMULATOR_KEYS = {
    "schema_version",
    "simulator_name",
    "task_name",
    "task_family",
    "physics_timestep_seconds",
    "action_dimension",
    "tracked_actors",
    "probe_task_attributes",
    "semantic_roles",
    "success_contract",
    "available_aspect_ids",
}
_ADAPTER_KEYS = {"schema_version", "task_name", "planner_kind", "templates"}
_TEMPLATE_KEYS = {
    "template_id",
    "aspect_id",
    "semantic_scope",
    "target_role",
    "taskgen_operation",
    "capability_id",
    "controlled_axis",
    "generation_mode",
    "allowed_change_roots",
    "tool_metric",
    "vqa_phenomenon_ids",
    "required_gates",
}
_UNKNOWN_POLICY_METADATA = [
    "action_scaling",
    "camera_names",
    "observation_keys",
]


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanningContextError(f"{field} must be an object")
    return value


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanningContextError(f"{field} must be a non-empty string")
    return value.strip()


def _build_planning_context(
    repo_root: str | Path,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    trusted_target = _require_mapping(target, field="EvaluationTarget")
    task_name = _require_text(
        trusted_target.get("task_name"), field="EvaluationTarget.task_name"
    )
    task_family = _require_text(
        trusted_target.get("task_family"), field="EvaluationTarget.task_family"
    )
    planner_kind = _require_text(
        trusted_target.get("planner_kind"), field="EvaluationTarget.planner_kind"
    )
    policy = _require_mapping(
        trusted_target.get("policy"), field="EvaluationTarget.policy"
    )
    checkpoint = _require_mapping(
        trusted_target.get("checkpoint"), field="EvaluationTarget.checkpoint"
    )
    raw_aspects = trusted_target.get("aspects")
    if not isinstance(raw_aspects, list) or not raw_aspects:
        raise PlanningContextError("EvaluationTarget.aspects must be a non-empty list")

    try:
        schema = load_task_schema(repo_root, task_name)
    except Exception as exc:
        raise PlanningContextError(f"cannot load trusted TaskSchema: {exc}") from exc
    if schema.get("task_family") != task_family:
        raise PlanningContextError(
            "EvaluationTarget task_family differs from the trusted TaskSchema"
        )

    registered = {
        str(contract["template_id"]): contract
        for contract in registered_capability_contracts(task_name)
    }
    templates: list[dict[str, Any]] = []
    aspect_ids: list[str] = []
    seen_templates: set[str] = set()
    for raw_aspect in raw_aspects:
        aspect = _require_mapping(raw_aspect, field="EvaluationTarget.aspect")
        aspect_id = _require_text(
            aspect.get("aspect_id"), field="EvaluationTarget.aspect.aspect_id"
        )
        if aspect_id in aspect_ids:
            raise PlanningContextError(f"duplicate target aspect: {aspect_id!r}")
        aspect_ids.append(aspect_id)
        template_ids = aspect.get("template_ids")
        if not isinstance(template_ids, list) or not template_ids:
            raise PlanningContextError(f"aspect {aspect_id!r} has no templates")
        for raw_template_id in template_ids:
            template_id = _require_text(
                raw_template_id, field=f"{aspect_id}.template_id"
            )
            if template_id in seen_templates:
                raise PlanningContextError(f"duplicate target template: {template_id!r}")
            seen_templates.add(template_id)
            try:
                contract = registered[template_id]
            except KeyError as exc:
                raise PlanningContextError(
                    f"target template is absent from capability registry: {template_id!r}"
                ) from exc
            if contract["aspect"]["aspect_id"] != aspect_id:
                raise PlanningContextError(
                    f"target aspect differs from capability contract: {template_id!r}"
                )
            taskgen = contract["taskgen"]
            templates.append(
                {
                    "template_id": template_id,
                    "aspect_id": aspect_id,
                    "semantic_scope": contract["aspect"]["semantic_scope"],
                    "target_role": contract["aspect"]["target_role"],
                    "taskgen_operation": taskgen["operation"],
                    "capability_id": taskgen["capability_id"],
                    "controlled_axis": taskgen["controlled_axis"],
                    "generation_mode": taskgen["generation_mode"],
                    "allowed_change_roots": list(taskgen["allowed_change_roots"]),
                    "tool_metric": contract["tool"]["metric"],
                    "vqa_phenomenon_ids": list(
                        contract["vqa"]["phenomenon_ids"]
                    ),
                    "required_gates": list(contract["required_gates"]),
                }
            )

    action_dimension = schema.get("action_dimension")
    if (
        isinstance(action_dimension, bool)
        or not isinstance(action_dimension, int)
        or action_dimension < 0
    ):
        raise PlanningContextError("TaskSchema action_dimension is invalid")
    policy_card = {
        "schema_version": 1,
        "policy_name": _require_text(policy.get("name"), field="policy.name"),
        "checkpoint_id": _require_text(
            checkpoint.get("checkpoint_id"), field="checkpoint.checkpoint_id"
        ),
        "checkpoint_setting": _require_text(
            checkpoint.get("checkpoint_setting"),
            field="checkpoint.checkpoint_setting",
        ),
        "expert_data_num": checkpoint.get("expert_data_num"),
        "language_conditioned": policy.get("language_conditioned"),
        "single_task_checkpoint": True,
        "task_name": task_name,
        "action_dimension": action_dimension,
        "checkpoint_ready": checkpoint.get("ready"),
        "unknown_metadata": list(_UNKNOWN_POLICY_METADATA),
    }
    if (
        isinstance(policy_card["expert_data_num"], bool)
        or not isinstance(policy_card["expert_data_num"], int)
        or policy_card["expert_data_num"] < 0
    ):
        raise PlanningContextError("checkpoint.expert_data_num must be non-negative")
    if not isinstance(policy_card["language_conditioned"], bool):
        raise PlanningContextError("policy.language_conditioned must be boolean")
    if policy_card["checkpoint_ready"] is not True:
        raise PlanningContextError("planning context requires a ready checkpoint")

    simulator_card = {
        "schema_version": 1,
        "simulator_name": "RoboTwin",
        "task_name": task_name,
        "task_family": task_family,
        "physics_timestep_seconds": schema["physics_timestep_seconds"],
        "action_dimension": action_dimension,
        "tracked_actors": deepcopy(schema["tracked_actors"]),
        "probe_task_attributes": list(schema.get("probe_task_attributes") or []),
        "semantic_roles": deepcopy(schema.get("semantic_roles") or {}),
        "success_contract": deepcopy(schema.get("success_contract") or {}),
        "available_aspect_ids": aspect_ids,
    }
    adapter_view = {
        "schema_version": 1,
        "task_name": task_name,
        "planner_kind": planner_kind,
        "templates": templates,
    }
    return {
        "schema_version": 1,
        "policy_card": policy_card,
        "simulator_card": simulator_card,
        "adapter_view": adapter_view,
    }


def build_planning_context(
    repo_root: str | Path,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a strict context card from trusted repository-owned sources."""

    context = _build_planning_context(repo_root, target)
    return validate_planning_context(context, repo_root=repo_root, target=target)


def validate_planning_context(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject extra fields or any divergence from the trusted source cards."""

    if not isinstance(value, Mapping) or set(value) != _CONTEXT_KEYS:
        raise PlanningContextError(
            f"PlanningContext fields must be exactly {sorted(_CONTEXT_KEYS)}"
        )
    context = deepcopy(dict(value))
    if context.get("schema_version") != 1:
        raise PlanningContextError("PlanningContext.schema_version must be 1")
    nested_contracts = (
        ("policy_card", _POLICY_KEYS),
        ("simulator_card", _SIMULATOR_KEYS),
        ("adapter_view", _ADAPTER_KEYS),
    )
    for name, keys in nested_contracts:
        nested = context.get(name)
        if not isinstance(nested, Mapping) or set(nested) != keys:
            raise PlanningContextError(
                f"{name} fields must be exactly {sorted(keys)}"
            )
    templates = context["adapter_view"].get("templates")
    if not isinstance(templates, list) or not templates:
        raise PlanningContextError("adapter_view.templates must be non-empty")
    for template in templates:
        if not isinstance(template, Mapping) or set(template) != _TEMPLATE_KEYS:
            raise PlanningContextError(
                f"adapter template fields must be exactly {sorted(_TEMPLATE_KEYS)}"
            )
    expected = _build_planning_context(repo_root, target)
    if context != expected:
        raise PlanningContextError("PlanningContext differs from trusted sources")
    return context


__all__ = [
    "PlanningContextError",
    "build_planning_context",
    "validate_planning_context",
]
