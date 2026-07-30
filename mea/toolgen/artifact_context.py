"""Small shared context for Rule Tool and VQA generation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.toolkit.schema import load_task_schema, validate_task_schema


class ToolArtifactContextError(ValueError):
    pass


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolArtifactContextError(f"{field} must be a non-empty string")
    return value.strip()


def _need(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ToolArtifactContextError(f"{field} must be a mapping or null")
    need = deepcopy(dict(value))
    need["description"] = _text(need.get("description"), f"{field}.description")
    if not isinstance(need.get("reuse_first"), bool):
        raise ToolArtifactContextError(f"{field}.reuse_first must be bool")
    return need


def _proposal(value: Mapping[str, Any] | None, task: str) -> dict[str, Any]:
    empty = {
        "status": "unavailable",
        "candidate_id": None,
        "source_query": None,
        "base_task": task,
        "semantic_concern": None,
        "scene_need": None,
        "checker_need": None,
        "rule_tool_need": None,
        "vqa_tool_need": None,
    }
    if value is None:
        return empty
    if not isinstance(value, Mapping):
        raise ToolArtifactContextError("proposal must be a mapping or null")
    if _text(value.get("base_task"), "proposal.base_task") != task:
        raise ToolArtifactContextError(
            "proposal.base_task differs from the bound runtime task"
        )

    rule_need = value.get("rule_tool_need")
    vqa_need = value.get("vqa_tool_need")
    legacy_need = value.get("tool_need")
    if (
        rule_need is None
        and vqa_need is None
        and isinstance(legacy_need, Mapping)
    ):
        if legacy_need.get("kind") == "vqa":
            vqa_need = legacy_need
        else:
            rule_need = legacy_need

    return {
        "status": "available",
        "candidate_id": _text(value.get("candidate_id"), "proposal.candidate_id"),
        "source_query": _text(value.get("source_query"), "proposal.source_query"),
        "base_task": task,
        "semantic_concern": _text(
            value.get("semantic_concern"),
            "proposal.semantic_concern",
        ),
        "scene_need": _need(value.get("scene_need"), "proposal.scene_need"),
        "checker_need": _need(value.get("checker_need"), "proposal.checker_need"),
        "rule_tool_need": _need(rule_need, "proposal.rule_tool_need"),
        "vqa_tool_need": _need(vqa_need, "proposal.vqa_tool_need"),
    }


_TASK_FIELDS = (
    "scene_origin",
    "success_origin",
    "success_semantics_preserved",
    "success_official_equivalent",
    "success_compiler_eligible",
    "success_act_eligible",
    "success_execution_scope",
    "success_outcome_label",
)


def _task_artifact(
    value: Mapping[str, Any] | None, task: str
) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise ToolArtifactContextError(
            "task_artifact_summary must be a mapping or null"
        )
    source = value or {}
    if source.get("task_name") not in {None, task}:
        raise ToolArtifactContextError(
            "task_artifact_summary.task_name differs from the bound task"
        )
    return {
        "status": "unavailable" if value is None else "available",
        "task_name": task,
        **{field: deepcopy(source.get(field)) for field in _TASK_FIELDS},
    }


def _oracle(
    broker: Mapping[str, Any] | None, legacy_available: bool
) -> dict[str, Any]:
    if broker is not None:
        if not isinstance(broker, Mapping):
            raise ToolArtifactContextError("oracle broker must be a mapping")
        fixture_count = broker.get("fixture_episode_count")
        if (
            broker.get("independent") is not True
            or not isinstance(fixture_count, int)
            or isinstance(fixture_count, bool)
            or fixture_count < 1
        ):
            raise ToolArtifactContextError(
                "derived observable broker must be independent and own fixtures"
            )
        return {
            "status": "available",
            "reason_code": None,
            "reason": None,
            "broker_id": _text(
                broker.get("broker_id"), "oracle_broker.broker_id"
            ),
            "independent": True,
            "fixture_episode_count": fixture_count,
            "source": "independent_oracle_broker",
        }

    if legacy_available:
        return {
            "status": "available",
            "reason_code": None,
            "reason": None,
            "broker_id": "legacy_caller_attestation",
            "independent": True,
            "fixture_episode_count": None,
            "source": "legacy_boolean_compatibility",
        }

    return {
        "status": "unsupported",
        "reason_code": "independent_oracle_broker_unavailable",
        "reason": (
            "A derived observable is unavailable without caller-owned fixture "
            "episodes and an independent oracle broker."
        ),
        "broker_id": None,
        "independent": False,
        "fixture_episode_count": None,
        "source": "runtime_capability",
    }


def build_tool_artifact_context(
    repo_root: str | Path,
    *,
    task_name: str,
    proposal: Mapping[str, Any] | None = None,
    task_artifact_summary: Mapping[str, Any] | None = None,
    runtime_schema: Mapping[str, Any] | None = None,
    reusable_rule_tools: list[Mapping[str, Any]] | None = None,
    reusable_vqa_questions: list[Mapping[str, Any]] | None = None,
    derived_observable_oracle_broker: Mapping[str, Any] | None = None,
    legacy_derived_observable_oracle_available: bool = False,
) -> dict[str, Any]:
    task = _text(task_name, "task_name")
    schema = (
        validate_task_schema(
            dict(runtime_schema),
            expected_task_name=task,
        )
        if isinstance(runtime_schema, Mapping)
        else load_task_schema(Path(repo_root).expanduser().resolve(), task)
    )
    context = {
        "schema_version": 1,
        "kind": "tool_artifact_context",
        "task_name": task,
        "proposal": _proposal(proposal, task),
        "task_artifact": _task_artifact(task_artifact_summary, task),
        "runtime_schema": {
            "task_name": task,
            "task_schema_version": schema["schema_version"],
            "tracked_actors": deepcopy(schema["tracked_actors"]),
            "semantic_fields": deepcopy(schema["semantic_fields"]),
            "common_events": ["contact_interval", "success_transition"],
        },
        "reusable_artifacts": {
            "rule_tools": deepcopy(reusable_rule_tools or []),
            "vqa_questions": deepcopy(reusable_vqa_questions or []),
        },
        "oracle_broker": {
            "derived_observable": _oracle(
                derived_observable_oracle_broker,
                legacy_derived_observable_oracle_available,
            )
        },
    }
    return validate_tool_artifact_context(context)


def validate_tool_artifact_context(value: Any) -> dict[str, Any]:
    """Validate only shared method boundaries, not a duplicate TaskSchema."""

    if not isinstance(value, Mapping):
        raise ToolArtifactContextError("ToolArtifactContext must be a mapping")
    context = deepcopy(dict(value))
    task = _text(context.get("task_name"), "ToolArtifactContext.task_name")
    bound_tasks = (
        context.get("proposal", {}).get("base_task"),
        context.get("task_artifact", {}).get("task_name"),
        context.get("runtime_schema", {}).get("task_name"),
    )
    if (
        context.get("schema_version") != 1
        or context.get("kind") != "tool_artifact_context"
        or any(item != task for item in bound_tasks)
    ):
        raise ToolArtifactContextError("ToolArtifactContext task binding is invalid")

    reusable = context.get("reusable_artifacts", {})
    if not all(
        isinstance(reusable.get(key), list)
        for key in ("rule_tools", "vqa_questions")
    ):
        raise ToolArtifactContextError("reusable artifacts must be lists")
    oracle = context.get("oracle_broker", {}).get("derived_observable", {})
    if oracle.get("status") not in {"available", "unsupported"}:
        raise ToolArtifactContextError(
            "derived observable oracle status is invalid"
        )
    try:
        json.dumps(context, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ToolArtifactContextError(
            "ToolArtifactContext must be finite JSON"
        ) from exc
    return context


__all__ = [
    "ToolArtifactContextError",
    "build_tool_artifact_context",
    "validate_tool_artifact_context",
]
