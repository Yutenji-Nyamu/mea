"""Query-induced Tool requests for the open-world evaluation path.

The Plan Agent names an evidence need.  This adapter exposes only the task's
recorded telemetry schema and executable Tool registry, then asks the provider
for either exact reuse or a typed semantic oracle.  On a miss, ToolGen writes
the Python implementation and validates it against that independent oracle.
No task/aspect template or preferred metric is exposed.
"""

from __future__ import annotations

import hashlib
import json
import re
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
    forbidden_metric_ids: set[str] | None = None,
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
        and item.get("metric") not in forbidden
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
    """Require exact reuse or an oracle-backed Python Tool request."""

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
            "metric is already measured by the base Toolkit or incompatible "
            f"with this outcome contract: {request['metric']}"
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
    if available_signal_names is not None and measurement_need:
        _validate_terminal_signal_alignment(
            metric_spec if isinstance(metric_spec, Mapping) else None,
            measurement_need=measurement_need,
            available_signal_names=available_signal_names,
        )
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
        if available_signal_names is not None and operation in {
            "minimum_distance",
            "terminal_signal_component",
            "terminal_signal_difference",
        }:
            requested_signals = (
                {str(metric_spec["signal"])}
                if operation == "terminal_signal_component"
                else {
                    str(metric_spec["left_signal"]),
                    str(metric_spec["right_signal"]),
                }
            )
            missing = sorted(requested_signals - available_signal_names)
            if missing:
                raise OpenToolRequestError(
                    f"MetricSpec uses unavailable telemetry signals: {missing}"
                )
        if operation == "minimum_distance" and available_signal_names is not None:
            requested_signals = {
                str(metric_spec["left_signal"]),
                str(metric_spec["right_signal"]),
            }
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
            robot_target_distance_requested = any(
                phrase in need
                for phrase in (
                    "click-point accuracy",
                    "click point accuracy",
                    "tcp accuracy",
                    "tcp distance",
                    "end-effector accuracy",
                    "end effector accuracy",
                    "gripper accuracy",
                    "gripper distance",
                )
            )
            requested_sided = requested_signals.intersection(sided_signals)
            if robot_target_distance_requested and (
                not requested_sided
                or requested_sided == requested_signals
            ):
                raise OpenToolRequestError(
                    "a robot-target accuracy need requires minimum_distance "
                    "between one advertised robot TCP/gripper signal and one "
                    "target signal; target-target or robot-robot distance is "
                    "not aligned"
                )
            if (
                active_side_requested
                and len(available_sides) > 1
                and requested_sided
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


def _normalized_words(value: str) -> set[str]:
    return {
        item
        for item in re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
        if item
    }


def _validate_terminal_signal_alignment(
    metric_spec: Mapping[str, Any] | None,
    *,
    measurement_need: str,
    available_signal_names: set[str],
) -> None:
    """Bind explicit terminal component/difference needs to advertised fields.

    This is deliberately schema-driven: task names and semantic roles are not
    enumerated. It activates only when the need contains a terminal cue, an
    x/y/z component cue, and enough words to identify advertised signals.
    """

    need = measurement_need.casefold()
    terminal_requested = any(
        cue in need
        for cue in (
            "final",
            "terminal",
            "end-of-rollout",
            "end of rollout",
            "最终",
            "终端",
            "结束时",
        )
    )
    component_cues = {
        "x": (" x ", "x-axis", "x axis", "x-coordinate", "x coordinate", "x分量", "x轴"),
        "y": (" y ", "y-axis", "y axis", "y-coordinate", "y coordinate", "y分量", "y轴"),
        "z": (" z ", "z-axis", "z axis", "z-coordinate", "z coordinate", "height", "高度", "z分量", "z轴"),
    }
    padded_need = f" {need} "
    requested_components = {
        component
        for component, cues in component_cues.items()
        if any(cue in padded_need for cue in cues)
    }
    # In manipulation evaluation, an unqualified "lift height difference"
    # asks for the achieved end-state separation of two lifted objects. This
    # is the generic terminal interpretation unless the need names another
    # temporal reduction.
    lift_height_difference_requested = any(
        cue in padded_need
        for cue in (
            " lift height difference ",
            " lift-height difference ",
        )
    )
    if (
        not (terminal_requested or lift_height_difference_requested)
        or not requested_components
    ):
        return

    difference_requested = any(
        cue in padded_need
        for cue in (
            " difference ",
            " delta ",
            " minus ",
            " relative to ",
            " compared to ",
            " compared with ",
            " versus ",
            " vs ",
            " height gap ",
            " vertical gap ",
            " z gap ",
        )
    )
    if difference_requested:
        if not isinstance(metric_spec, Mapping):
            raise OpenToolRequestError(
                "an explicit terminal two-signal difference need requires a "
                "schema_version=2 terminal_signal_difference contract; an "
                "unrelated schema_version=1 Tool reuse is not aligned"
            )
        if metric_spec.get("operation") != "terminal_signal_difference":
            raise OpenToolRequestError(
                "an explicit terminal two-signal difference need requires "
                "terminal_signal_difference rather than a component, event, "
                "or distance metric"
            )
        requested_component = metric_spec.get("component")
        if requested_component not in requested_components:
            raise OpenToolRequestError(
                "terminal_signal_difference component does not match the "
                f"explicit measurement need: {sorted(requested_components)}"
            )
        absolute_difference_requested = any(
            cue in padded_need
            for cue in (
                " absolute difference ",
                " absolute height difference ",
                " difference magnitude ",
                " unsigned difference ",
            )
        )
        if (
            absolute_difference_requested
            and metric_spec.get("absolute") is not True
        ):
            raise OpenToolRequestError(
                "terminal_signal_difference must set absolute=true for an "
                "explicit absolute-difference measurement need"
            )
        return

    need_words = _normalized_words(need)
    scored: list[tuple[int, str]] = []
    for signal in sorted(available_signal_names):
        signal_words = _normalized_words(signal)
        distinctive = signal_words - {"position", "pose", "point"}
        score = len(distinctive & need_words)
        if signal.replace("_", " ").casefold() in need:
            score += len(signal_words) + 1
        if score:
            scored.append((score, signal))
    if not scored:
        return

    best_score = max(score for score, _signal in scored)
    matched_signals = {
        signal for score, signal in scored if score == best_score
    }
    if not isinstance(metric_spec, Mapping):
        raise OpenToolRequestError(
            "an explicit terminal semantic-field/component need requires a "
            "schema_version=2 terminal_signal_component contract; an unrelated "
            "schema_version=1 Tool reuse is not aligned"
        )
    if metric_spec.get("operation") != "terminal_signal_component":
        raise OpenToolRequestError(
            "an explicit terminal semantic-field/component need requires "
            "terminal_signal_component rather than an event or distance metric"
        )
    if metric_spec.get("signal") not in matched_signals:
        raise OpenToolRequestError(
            "terminal_signal_component must consume the advertised semantic "
            f"field referenced by the measurement need: {sorted(matched_signals)}"
        )
    requested_component = metric_spec.get("component")
    if requested_component not in requested_components:
        raise OpenToolRequestError(
            "terminal_signal_component component does not match the explicit "
            f"measurement need: {sorted(requested_components)}"
        )
    absolute_requested = any(
        cue in padded_need
        for cue in (
            f"absolute {requested_component}",
            f"absolute-{requested_component}",
            f"{requested_component} magnitude",
            f"{requested_component}-magnitude",
            f"{requested_component} 绝对",
            f"{requested_component}绝对",
            f"绝对 {requested_component}",
            f"绝对{requested_component}",
        )
    )
    if absolute_requested and metric_spec.get("absolute") is not True:
        raise OpenToolRequestError(
            "terminal_signal_component must set absolute=true for an explicit "
            "absolute-component measurement need"
        )


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
            "request and MetricSpec exactly. Otherwise return schema_version=2 "
            "and the smallest MetricSpec semantic oracle using only the "
            "advertised typed operator contracts and telemetry names. ToolGen "
            "will then generate Python rather than compiling this oracle. "
            "Replace angle-bracket "
            "placeholders with real advertised names or null. A registered "
            "composite target is an exact static match and may be selected by "
            "its schema_version=1 metric id; it will be generated and validated "
            "when no compatible registration exists. A fixed left/right signal "
            "does not satisfy an active-arm or active-gripper need when both "
            "sides are advertised. Do not invent an unavailable signal, task "
            "name, template, or aspect. Never select a metric listed in "
            "forbidden_metric_ids: those values are already present in the "
            "base Toolkit evidence or are semantically incompatible, so this "
            "Tool must add the missing Query-specific measurement. A click-point, "
            "TCP, end-effector, or gripper accuracy need must compare one "
            "advertised robot signal with one target signal; target-target "
            "distance is not robot accuracy. When "
            "MEASUREMENT NEED explicitly asks "
            "for a final/terminal x, y, z, height, or absolute component of an "
            "advertised semantic field, use terminal_signal_component with "
            "that exact signal and component. When it asks for the final or "
            "terminal difference, delta, or relative height/component between "
            "two advertised semantic fields, use terminal_signal_difference "
            "with those exact left/right signals and component. Do not replace "
            "a terminal two-signal difference with minimum_distance or a "
            "single terminal_signal_component. Treat an unqualified lift "
            "height difference between two objects as their terminal z "
            "difference; an event metric is not aligned. "
            "Return strict JSON only.\n\n"
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
        forbidden_metric_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        context = tool_generation_context(
            self.repo_root,
            task_name=task_name,
            generated_checker_semantics=generated_checker_semantics,
            runtime_schema=runtime_schema,
            reusable_tool_requests=reusable_tool_requests,
            forbidden_metric_ids=forbidden_metric_ids,
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
        filled_bound_fields: list[str] = []
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
                raw_request = extract_json_response(response)
                if not isinstance(raw_request, dict):
                    raise OpenToolRequestError(
                        "provider Tool request must be a JSON object"
                    )
                for field, value in (
                    ("task_name", str(context["task_name"])),
                    ("question", _text(tool_need, "tool_need")),
                ):
                    if field not in raw_request:
                        raw_request[field] = value
                        if field not in filled_bound_fields:
                            filled_bound_fields.append(field)
                request = validate_open_tool_request(
                    raw_request,
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
                "bound_fields_filled": filled_bound_fields,
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
