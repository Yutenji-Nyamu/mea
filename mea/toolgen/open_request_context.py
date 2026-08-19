"""Runtime telemetry and reusable-artifact context for ToolGen."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.toolkit.schema import load_task_schema, validate_task_schema
from mea.task_guide import load_task_guide

from .artifact_context import build_tool_artifact_context, validate_tool_artifact_context
from .open_request_contract import _text
from .router import catalog_snapshot


def tool_generation_context(
    repo_root: str | Path,
    *,
    task_name: str,
    generated_checker_semantics: bool = False,
    runtime_schema: Mapping[str, Any] | None = None,
    reusable_tool_requests: list[Mapping[str, Any]] | None = None,
    reusable_vqa_questions: list[Mapping[str, Any]] | None = None,
    forbidden_metric_ids: set[str] | None = None,
    derived_observable_oracle_available: bool = False,
    derived_observable_oracle_broker: Mapping[str, Any] | None = None,
    proposal: Mapping[str, Any] | None = None,
    task_artifact_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the task telemetry and executable Tool surface, without routing."""

    root = Path(repo_root).expanduser().resolve()
    task = _text(task_name, "task_name")
    schema = (
        validate_task_schema(
            dict(runtime_schema),
            expected_task_name=task,
        )
        if isinstance(runtime_schema, Mapping)
        else load_task_schema(root, task)
    )
    artifact_context = build_tool_artifact_context(
        root,
        task_name=task,
        proposal=proposal,
        task_artifact_summary=task_artifact_summary,
        runtime_schema=schema,
        reusable_rule_tools=reusable_tool_requests,
        reusable_vqa_questions=reusable_vqa_questions,
        derived_observable_oracle_broker=(
            derived_observable_oracle_broker
        ),
        legacy_derived_observable_oracle_available=(
            derived_observable_oracle_available
        ),
    )
    derived_available = (
        artifact_context["oracle_broker"]["derived_observable"]["status"]
        == "available"
    )
    tool_registry = catalog_snapshot()
    if not derived_available:
        typed_registry = tool_registry["typed_metric_spec"]
        typed_registry["schema_versions"] = [1]
        typed_registry["operations"] = [
            operation
            for operation in typed_registry["operations"]
            if operation != "derived_observable"
        ]
        typed_registry["execution"] = "compile_validate_register"
    forbidden = {
        _text(metric, "forbidden_metric_id")
        for metric in (forbidden_metric_ids or set())
    }
    if generated_checker_semantics:
        forbidden.update(
            {
                "official_check_success",
                "time_to_success",
            }
        )
    if forbidden:
        tool_registry["trusted_tools"] = [
            item
            for item in tool_registry["trusted_tools"]
            if item["name"] not in forbidden
        ]
        tool_registry["composite_targets"] = [
            item
            for item in tool_registry["composite_targets"]
            if item["metric"] not in forbidden
        ]
    reusable = []
    for item in reusable_tool_requests or []:
        if not isinstance(item, Mapping):
            continue
        reusable_item = deepcopy(dict(item))
        request = reusable_item.get("request")
        request = request if isinstance(request, Mapping) else reusable_item
        metric_spec = request.get("metric_spec")
        if (
            not derived_available
            and isinstance(metric_spec, Mapping)
            and metric_spec.get("operation") == "derived_observable"
        ):
            continue
        if request.get("metric") in forbidden:
            continue
        reusable.append(reusable_item)
    artifact_context["reusable_artifacts"]["rule_tools"] = deepcopy(reusable)
    artifact_context = validate_tool_artifact_context(artifact_context)
    return {
        "schema_version": 1,
        "task_name": task,
        "task_implementation_guide": load_task_guide(root, task) or None,
        "telemetry_schema": {
            "tracked_actors": [
                {
                    "id": actor["id"],
                    "scene_name": actor["scene_name"],
                    "functional_points": deepcopy(actor.get("functional_points", [])),
                    "contact_points": deepcopy(actor.get("contact_points", [])),
                }
                for actor in schema["tracked_actors"]
            ],
            "semantic_fields": [
                {
                    "name": field["name"],
                    "source": field["source"],
                    "actor_id": field.get("actor_id"),
                    "point_id": field.get("point_id"),
                    "side": field.get("side"),
                }
                for field in schema["semantic_fields"]
            ],
            "common_events": [
                "contact_interval",
                "success_transition",
            ],
        },
        "typed_operator_contracts": {
            "minimum_distance": {
                "schema_version": 1,
                "operation": "minimum_distance",
                "left_signal": "<advertised_semantic_field_name>",
                "right_signal": "<different_advertised_semantic_field_name>",
                "dimensions": ["x", "y", "z"],
                "unit": "m",
                "null_semantics": "null_if_no_finite_sample",
            },
            "terminal_minimum_distance": {
                "schema_version": 1,
                "operation": "terminal_minimum_distance",
                "left_signals": [
                    "<one_or_more_advertised_robot_position_fields>"
                ],
                "right_signal": "<advertised_target_position_field>",
                "dimensions": ["x", "y", "z"],
                "unit": "m",
                "null_semantics": "null_if_terminal_not_finite",
            },
            "event_count": {
                "schema_version": 1,
                "operation": "event_count",
                "event": {
                    "event_type": "contact_interval",
                    "actors": (
                        "<null_or_two_advertised_actor_ids>"
                    ),
                    "physical_only": True,
                },
                "unit": "count",
                "null_semantics": "zero_if_absent",
            },
            "time_between_events": {
                "schema_version": 1,
                "operation": "time_between_events",
                "start_event": {
                    "event_type": "contact_interval",
                    "actors": (
                        "<null_or_two_advertised_actor_ids>"
                    ),
                    "physical_only": True,
                },
                "end_event": {
                    "event_type": "success_transition",
                    "actors": None,
                    "physical_only": False,
                },
                "unit": "s",
                "null_semantics": "null_if_missing_or_reversed",
            },
            "terminal_signal_component": {
                "schema_version": 1,
                "operation": "terminal_signal_component",
                "signal": "<advertised_semantic_field_name>",
                "component": "<x_or_y_or_z>",
                "absolute": False,
                "unit": "m",
                "null_semantics": "null_if_terminal_not_finite",
            },
            "terminal_signal_difference": {
                "schema_version": 1,
                "operation": "terminal_signal_difference",
                "left_signal": "<advertised_semantic_field_name>",
                "right_signal": "<different_advertised_semantic_field_name>",
                "component": "<x_or_y_or_z>",
                "absolute": False,
                "unit": "m",
                "null_semantics": "null_if_terminal_not_finite",
            },
        },
        "outcome_semantics": (
            "generated_checker_experimental"
            if generated_checker_semantics
            else "official_task"
        ),
        "telemetry_schema_source": (
            "executed_episode_schema"
            if runtime_schema is not None
            else "official_task_schema"
        ),
        "forbidden_metric_ids": sorted(forbidden),
        "derived_observable_validation_available": derived_available,
        "tool_registry": tool_registry,
        "validated_generated_tools": reusable,
        "artifact_context": artifact_context,
    }
