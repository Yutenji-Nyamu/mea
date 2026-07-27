"""Query-induced Tool requests for the open-world evaluation path.

The Plan Agent names an evidence need.  This adapter exposes only the
task's recorded telemetry schema and the executable Tool registry, then asks
the provider for either an exact reusable metric or a typed MetricSpec.  It
does not expose task/aspect templates or a preferred metric.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.providers.json_response import extract_json_response
from mea.toolkit.schema import load_task_schema, validate_task_schema

from .metric_spec import MetricSpecError, metric_spec_tool_spec
from .router import (
    ToolRouterError,
    catalog_snapshot,
    route_tool_request,
    validate_tool_request,
)


class OpenToolRequestError(ValueError):
    """Raised when a Query-induced Tool request is malformed."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenToolRequestError(f"{field} must be a non-empty string")
    return value.strip()


def tool_generation_context(
    repo_root: str | Path,
    *,
    task_name: str,
    generated_checker_semantics: bool = False,
    runtime_schema: Mapping[str, Any] | None = None,
    reusable_tool_requests: list[Mapping[str, Any]] | None = None,
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
    tool_registry = catalog_snapshot()
    forbidden_metric_ids: list[str] = []
    if generated_checker_semantics:
        forbidden_metric_ids = [
            "official_check_success",
            "time_to_success",
        ]
        tool_registry["trusted_tools"] = [
            item
            for item in tool_registry["trusted_tools"]
            if item["name"] not in forbidden_metric_ids
        ]
        unhashed = dict(tool_registry)
        unhashed.pop("snapshot_sha256", None)
        tool_registry["snapshot_sha256"] = hashlib.sha256(
            json.dumps(
                unhashed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    reusable = [
        deepcopy(dict(item))
        for item in (reusable_tool_requests or [])
        if isinstance(item, Mapping)
    ]
    return {
        "schema_version": 1,
        "task_name": task,
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
        "forbidden_metric_ids": forbidden_metric_ids,
        "tool_registry": tool_registry,
        "validated_generated_tools": reusable,
    }


def validate_open_tool_request(
    value: Mapping[str, Any],
    *,
    task_name: str,
    available_signal_names: set[str] | None = None,
    available_signal_sides: Mapping[str, str] | None = None,
    available_actor_ids: set[str] | None = None,
    forbidden_metric_ids: set[str] | None = None,
    measurement_need: str | None = None,
) -> dict[str, Any]:
    """Require an executable exact reuse or typed MetricSpec request."""

    try:
        request = validate_tool_request(dict(value))
        decision = route_tool_request(request)["route_decision"]
    except (ToolRouterError, TypeError, ValueError) as exc:
        raise OpenToolRequestError(str(exc)) from exc
    task = _text(task_name, "task_name")
    if request["task_name"] != task:
        raise OpenToolRequestError(
            "provider Tool request changed the bound task"
        )
    if (
        forbidden_metric_ids is not None
        and request["metric"] in forbidden_metric_ids
    ):
        raise OpenToolRequestError(
            "metric is incompatible with the generated-checker outcome "
            f"semantics: {request['metric']}"
        )
    if decision["status"] != "resolved":
        raise OpenToolRequestError(
            "a novel metric must include a valid typed MetricSpec"
        )
    if decision["resolved_route"] not in {
        "reuse",
        "force_codegen",
        "typed_metric_spec_compile",
    }:
        raise OpenToolRequestError(
            "open Tool request must reuse, generate a registered composite "
            "target, or compile a typed MetricSpec"
        )
    metric_spec = request.get("metric_spec")
    if isinstance(metric_spec, Mapping):
        try:
            metric_spec_tool_spec(
                task_name=request["task_name"],
                metric=request["metric"],
                question=request["question"],
                metric_spec=metric_spec,
            )
        except MetricSpecError as exc:
            raise OpenToolRequestError(str(exc)) from exc
        operation = metric_spec["operation"]
        if operation == "minimum_distance" and available_signal_names is not None:
            requested_signals = {
                str(metric_spec["left_signal"]),
                str(metric_spec["right_signal"]),
            }
            missing = sorted(requested_signals - available_signal_names)
            if missing:
                raise OpenToolRequestError(
                    f"MetricSpec uses unavailable telemetry signals: {missing}"
                )
            need = (measurement_need or "").casefold()
            active_side_requested = any(
                phrase in need
                for phrase in (
                    "active arm",
                    "active-arm",
                    "active gripper",
                    "active tcp",
                    "主动臂",
                    "活动臂",
                )
            )
            sided_signals = {
                str(signal): str(side)
                for signal, side in (available_signal_sides or {}).items()
                if isinstance(side, str) and side.strip()
            }
            available_sides = set(sided_signals.values())
            if (
                active_side_requested
                and len(available_sides) > 1
                and requested_signals.intersection(sided_signals)
            ):
                raise OpenToolRequestError(
                    "a fixed-side minimum_distance MetricSpec cannot satisfy "
                    "an active-arm measurement need; reuse or generate a "
                    "registered metric that selects the active side at runtime"
                )
        if operation in {"event_count", "time_between_events"}:
            selectors = (
                [metric_spec["event"]]
                if operation == "event_count"
                else [metric_spec["start_event"], metric_spec["end_event"]]
            )
            referenced_actors = {
                str(actor)
                for selector in selectors
                for actor in (selector.get("actors") or [])
            }
            if available_actor_ids is not None:
                missing_actors = sorted(referenced_actors - available_actor_ids)
                if missing_actors:
                    raise OpenToolRequestError(
                        "MetricSpec uses unavailable telemetry actors: "
                        f"{missing_actors}"
                    )
    return request


class OpenToolRequestAgent:
    """Generate one executable Tool request from a semantic evidence need."""

    def __init__(
        self,
        repo_root: str | Path,
        provider: Any,
        *,
        model: str,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = _text(model, "model")
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _prompt(
        *,
        source_query: str,
        semantic_concern: str,
        tool_need: str,
        context: Mapping[str, Any],
    ) -> str:
        example = {
            "schema_version": 2,
            "task_name": context["task_name"],
            "metric": "query_derived_metric",
            "question": "The precise question answered by the metric.",
            "metric_spec": {
                "schema_version": 1,
                "operation": "minimum_distance",
                "left_signal": "actor_a_position",
                "right_signal": "actor_b_position",
                "dimensions": ["x", "y", "z"],
                "unit": "m",
                "null_semantics": "null_if_no_finite_sample",
            },
        }
        return (
            "You are ToolGen in ManipEvalAgent. Derive the smallest executable "
            "measurement needed by the open Query. First inspect both the "
            "trusted static registry and validated_generated_tools. For an "
            "exact static match, return schema_version=1 with its metric id. "
            "For an exact generated match, copy that entry's schema_version=2 "
            "request and MetricSpec exactly. Otherwise return "
            "schema_version=2 and a MetricSpec using only the advertised typed "
            "operator contracts and telemetry names. Replace angle-bracket "
            "placeholders with real advertised names or null. A registered "
            "composite target is an exact static match and may be selected by "
            "its schema_version=1 metric id; it will be generated and validated "
            "when no compatible registration exists. A fixed left/right signal "
            "does not satisfy an active-arm or active-gripper need when both "
            "sides are advertised. Do not invent an unavailable signal, task "
            "name, template, or aspect. Return strict JSON only.\n\n"
            f"ORIGINAL QUERY:\n{source_query}\n\n"
            f"SEMANTIC CONCERN:\n{semantic_concern}\n\n"
            f"MEASUREMENT NEED:\n{tool_need}\n\n"
            "TELEMETRY AND TOOL CONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n\nOUTPUT SHAPE EXAMPLE (choose fields according to schema version):\n"
            + json.dumps(example, ensure_ascii=False, indent=2)
        )

    def propose(
        self,
        *,
        source_query: str,
        semantic_concern: str,
        tool_need: str,
        task_name: str,
        generated_checker_semantics: bool = False,
        runtime_schema: Mapping[str, Any] | None = None,
        reusable_tool_requests: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        context = tool_generation_context(
            self.repo_root,
            task_name=task_name,
            generated_checker_semantics=generated_checker_semantics,
            runtime_schema=runtime_schema,
            reusable_tool_requests=reusable_tool_requests,
        )
        prompt = self._prompt(
            source_query=_text(source_query, "source_query"),
            semantic_concern=_text(semantic_concern, "semantic_concern"),
            tool_need=_text(tool_need, "tool_need"),
            context=context,
        )
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []
        request: dict[str, Any] | None = None
        for _attempt in range(2):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\n\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + "\nReturn one corrected complete JSON object."
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict ToolRequest JSON.",
                    max_tokens=900,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                request = validate_open_tool_request(
                    extract_json_response(response),
                    task_name=str(context["task_name"]),
                    available_signal_names={
                        str(item["name"])
                        for item in context["telemetry_schema"]["semantic_fields"]
                    },
                    available_signal_sides={
                        str(item["name"]): str(item["side"])
                        for item in context["telemetry_schema"]["semantic_fields"]
                        if isinstance(item.get("side"), str)
                        and str(item["side"]).strip()
                    },
                    available_actor_ids={
                        str(item["id"])
                        for item in context["telemetry_schema"]["tracked_actors"]
                    },
                    forbidden_metric_ids=set(
                        context["forbidden_metric_ids"]
                    ),
                    measurement_need=tool_need,
                )
                break
            except Exception as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        if request is None:
            raise OpenToolRequestError(
                "provider failed two open Tool request attempts: "
                + " | ".join(self.last_errors)
            )
        return {
            "schema_version": 1,
            "source": "provider_query_induced_tool_request",
            "tool_request": request,
            "context": context,
            "provider": {
                "model_requested": self.model,
                "called": True,
                "attempt_count": len(self.last_responses),
                "errors": list(self.last_errors),
                "last_metadata": deepcopy(
                    dict(getattr(self.provider, "last_metadata", {}) or {})
                ),
            },
        }


__all__ = [
    "OpenToolRequestAgent",
    "OpenToolRequestError",
    "tool_generation_context",
    "validate_open_tool_request",
]
