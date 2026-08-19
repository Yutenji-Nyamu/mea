"""MetricSpec schemas and executable ToolSpec contracts."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

class MetricSpecError(RuntimeError):
    """Raised when a typed metric request or its validation evidence is invalid."""


_MINIMUM_DISTANCE_KEYS = {
    "schema_version",
    "operation",
    "left_signal",
    "right_signal",
    "dimensions",
    "unit",
    "null_semantics",
}
_TERMINAL_MINIMUM_DISTANCE_KEYS = {
    "schema_version",
    "operation",
    "left_signals",
    "right_signal",
    "dimensions",
    "unit",
    "null_semantics",
}
_EVENT_COUNT_KEYS = {
    "schema_version",
    "operation",
    "event",
    "unit",
    "null_semantics",
}
_TIME_BETWEEN_EVENTS_KEYS = {
    "schema_version",
    "operation",
    "start_event",
    "end_event",
    "unit",
    "null_semantics",
}
_TERMINAL_SIGNAL_COMPONENT_REQUIRED_KEYS = {
    "schema_version",
    "operation",
    "signal",
    "component",
    "unit",
    "null_semantics",
}
_TERMINAL_SIGNAL_COMPONENT_KEYS = (
    _TERMINAL_SIGNAL_COMPONENT_REQUIRED_KEYS | {"absolute"}
)
_TERMINAL_SIGNAL_DIFFERENCE_REQUIRED_KEYS = {
    "schema_version",
    "operation",
    "left_signal",
    "right_signal",
    "component",
    "unit",
    "null_semantics",
}
_TERMINAL_SIGNAL_DIFFERENCE_KEYS = (
    _TERMINAL_SIGNAL_DIFFERENCE_REQUIRED_KEYS | {"absolute"}
)
_DERIVED_OBSERVABLE_KEYS = {
    "schema_version",
    "operation",
    "observable_id",
    "description",
    "required_signals",
    "unit",
    "null_semantics",
}
_EVENT_SELECTOR_KEYS = {"event_type", "actors", "physical_only"}
_V1_OPERATIONS = {
    "minimum_distance",
    "terminal_minimum_distance",
    "event_count",
    "time_between_events",
    "terminal_signal_component",
    "terminal_signal_difference",
}
_EVENT_TYPES = {"contact_interval", "success_transition"}
_SIGNAL = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
_METRIC = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_ACTOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
_UNIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_/*.^-]{0,31}$")
_DIMENSION_INDEX = {"x": 0, "y": 1, "z": 2}
_CORE_ARTIFACTS = (
    "episode.json",
    "schema.json",
    "states.csv",
    "semantic_trace.npz",
    "events.jsonl",
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_event_selector(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EVENT_SELECTOR_KEYS:
        raise MetricSpecError(
            f"MetricSpec.{field} fields must be exactly "
            f"{sorted(_EVENT_SELECTOR_KEYS)}"
        )
    selector = deepcopy(dict(value))
    event_type = selector.get("event_type")
    if not isinstance(event_type, str) or event_type not in _EVENT_TYPES:
        raise MetricSpecError(
            f"MetricSpec.{field}.event_type must be contact_interval or "
            "success_transition"
        )
    physical_only = selector.get("physical_only")
    if type(physical_only) is not bool:
        raise MetricSpecError(
            f"MetricSpec.{field}.physical_only must be a boolean"
        )
    actors = selector.get("actors")
    if event_type == "success_transition":
        if actors is not None or physical_only:
            raise MetricSpecError(
                f"MetricSpec.{field} success_transition requires "
                "actors=null and physical_only=false"
            )
    else:
        if actors is not None and (
            not isinstance(actors, list)
            or len(actors) != 2
            or any(
                not isinstance(item, str) or not _ACTOR.fullmatch(item)
                for item in actors
            )
            or len(set(actors)) != 2
        ):
            raise MetricSpecError(
                f"MetricSpec.{field}.actors must be null or two distinct "
                "safe actor ids"
            )
        if actors is not None:
            selector["actors"] = sorted(actors)
    return selector


def _validate_derived_observable(spec: dict[str, Any]) -> dict[str, Any]:
    if set(spec) != _DERIVED_OBSERVABLE_KEYS:
        raise MetricSpecError(
            "MetricSpec fields for derived_observable must be exactly "
            f"{sorted(_DERIVED_OBSERVABLE_KEYS)}"
        )
    observable_id = spec.get("observable_id")
    if not isinstance(observable_id, str) or not _METRIC.fullmatch(observable_id):
        raise MetricSpecError(
            "derived_observable.observable_id must be lower_snake_case"
        )
    description = spec.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description.strip()) > 240
    ):
        raise MetricSpecError(
            "derived_observable.description must contain 1-240 characters"
        )
    raw_signals = spec.get("required_signals")
    if (
        not isinstance(raw_signals, list)
        or not raw_signals
        or len(raw_signals) > 8
        or any(
            not isinstance(signal, str) or not _SIGNAL.fullmatch(signal)
            for signal in raw_signals
        )
        or len(set(raw_signals)) != len(raw_signals)
    ):
        raise MetricSpecError(
            "derived_observable.required_signals must contain 1-8 unique "
            "safe trace names"
        )
    unit = spec.get("unit")
    if not isinstance(unit, str) or not _UNIT.fullmatch(unit):
        raise MetricSpecError(
            "derived_observable.unit must be a compact physical unit"
        )
    if spec.get("null_semantics") != "null_if_no_finite_sample":
        raise MetricSpecError(
            "derived_observable requires "
            "null_semantics=null_if_no_finite_sample"
        )
    return {
        **spec,
        "observable_id": observable_id,
        "description": description.strip(),
        "required_signals": list(raw_signals),
        "unit": unit,
    }


def validate_metric_spec(value: Any) -> dict[str, Any]:
    """Validate a legacy metric or a provider-proposed derived observable."""

    if not isinstance(value, Mapping):
        raise MetricSpecError("MetricSpec must be an object")
    spec = deepcopy(dict(value))
    schema_version = spec.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise MetricSpecError("MetricSpec.schema_version must be 1 or 2")
    operation = spec.get("operation")
    if schema_version == 2:
        if operation != "derived_observable":
            raise MetricSpecError(
                "MetricSpec schema_version=2 requires "
                "operation=derived_observable"
            )
        return _validate_derived_observable(spec)
    if not isinstance(operation, str) or operation not in _V1_OPERATIONS:
        raise MetricSpecError(
            "MetricSpec.operation must be one of "
            f"{sorted(_V1_OPERATIONS)}"
        )
    if operation == "terminal_minimum_distance":
        if set(spec) != _TERMINAL_MINIMUM_DISTANCE_KEYS:
            raise MetricSpecError(
                "MetricSpec fields for terminal_minimum_distance must be "
                f"exactly {sorted(_TERMINAL_MINIMUM_DISTANCE_KEYS)}"
            )
        left_signals = spec.get("left_signals")
        if (
            not isinstance(left_signals, list)
            or not 1 <= len(left_signals) <= 8
            or any(
                not isinstance(signal, str) or not _SIGNAL.fullmatch(signal)
                for signal in left_signals
            )
            or len(set(left_signals)) != len(left_signals)
        ):
            raise MetricSpecError(
                "terminal_minimum_distance.left_signals must contain 1-8 "
                "unique safe trace signals"
            )
        right_signal = spec.get("right_signal")
        if not isinstance(right_signal, str) or not _SIGNAL.fullmatch(right_signal):
            raise MetricSpecError(
                "terminal_minimum_distance.right_signal is not a safe trace signal"
            )
        if right_signal in left_signals:
            raise MetricSpecError(
                "terminal_minimum_distance right_signal must differ from left_signals"
            )
        dimensions = spec.get("dimensions")
        if dimensions not in (["x", "y"], ["x", "y", "z"]):
            raise MetricSpecError(
                "terminal_minimum_distance.dimensions must be [x,y] or [x,y,z]"
            )
        if spec.get("unit") != "m":
            raise MetricSpecError(
                "terminal_minimum_distance currently requires unit=m"
            )
        if spec.get("null_semantics") != "null_if_terminal_not_finite":
            raise MetricSpecError(
                "terminal_minimum_distance requires "
                "null_semantics=null_if_terminal_not_finite"
            )
        spec["left_signals"] = list(left_signals)
        spec["right_signal"] = right_signal
        return spec
    if operation == "terminal_signal_component":
        keys = set(spec)
        if keys not in {
            frozenset(_TERMINAL_SIGNAL_COMPONENT_REQUIRED_KEYS),
            frozenset(_TERMINAL_SIGNAL_COMPONENT_KEYS),
        }:
            raise MetricSpecError(
                "MetricSpec fields for terminal_signal_component must be exactly "
                f"{sorted(_TERMINAL_SIGNAL_COMPONENT_REQUIRED_KEYS)} with "
                "optional absolute"
            )
        signal = spec.get("signal")
        if not isinstance(signal, str) or not _SIGNAL.fullmatch(signal):
            raise MetricSpecError(
                "MetricSpec.signal is not a safe trace signal"
            )
        component = spec.get("component")
        if component not in _DIMENSION_INDEX:
            raise MetricSpecError(
                "terminal_signal_component requires component=x, y, or z"
            )
        absolute = spec.get("absolute", False)
        if type(absolute) is not bool:
            raise MetricSpecError(
                "MetricSpec.absolute must be a boolean when provided"
            )
        if spec.get("unit") != "m":
            raise MetricSpecError(
                "terminal_signal_component currently requires unit=m"
            )
        if spec.get("null_semantics") != "null_if_terminal_not_finite":
            raise MetricSpecError(
                "terminal_signal_component requires "
                "null_semantics=null_if_terminal_not_finite"
            )
        spec["signal"] = signal
        spec["component"] = component
        spec["absolute"] = absolute
        return spec
    if operation == "terminal_signal_difference":
        keys = set(spec)
        if keys not in {
            frozenset(_TERMINAL_SIGNAL_DIFFERENCE_REQUIRED_KEYS),
            frozenset(_TERMINAL_SIGNAL_DIFFERENCE_KEYS),
        }:
            raise MetricSpecError(
                "MetricSpec fields for terminal_signal_difference must be "
                f"exactly {sorted(_TERMINAL_SIGNAL_DIFFERENCE_REQUIRED_KEYS)} "
                "with optional absolute"
            )
        for field in ("left_signal", "right_signal"):
            signal = spec.get(field)
            if not isinstance(signal, str) or not _SIGNAL.fullmatch(signal):
                raise MetricSpecError(
                    f"MetricSpec.{field} is not a safe trace signal"
                )
            spec[field] = signal
        if spec["left_signal"] == spec["right_signal"]:
            raise MetricSpecError(
                "terminal_signal_difference signals must be distinct"
            )
        component = spec.get("component")
        if component not in _DIMENSION_INDEX:
            raise MetricSpecError(
                "terminal_signal_difference requires component=x, y, or z"
            )
        absolute = spec.get("absolute", False)
        if type(absolute) is not bool:
            raise MetricSpecError(
                "MetricSpec.absolute must be a boolean when provided"
            )
        if spec.get("unit") != "m":
            raise MetricSpecError(
                "terminal_signal_difference currently requires unit=m"
            )
        if spec.get("null_semantics") != "null_if_terminal_not_finite":
            raise MetricSpecError(
                "terminal_signal_difference requires "
                "null_semantics=null_if_terminal_not_finite"
            )
        spec["component"] = component
        spec["absolute"] = absolute
        return spec
    expected_keys = {
        "minimum_distance": _MINIMUM_DISTANCE_KEYS,
        "event_count": _EVENT_COUNT_KEYS,
        "time_between_events": _TIME_BETWEEN_EVENTS_KEYS,
    }[operation]
    if set(spec) != expected_keys:
        raise MetricSpecError(
            f"MetricSpec fields for {operation} must be exactly "
            f"{sorted(expected_keys)}"
        )
    if operation == "event_count":
        spec["event"] = _validate_event_selector(spec.get("event"), field="event")
        if spec.get("unit") != "count":
            raise MetricSpecError("event_count requires unit=count")
        if spec.get("null_semantics") != "zero_if_absent":
            raise MetricSpecError(
                "event_count requires null_semantics=zero_if_absent"
            )
        return spec
    if operation == "time_between_events":
        spec["start_event"] = _validate_event_selector(
            spec.get("start_event"), field="start_event"
        )
        spec["end_event"] = _validate_event_selector(
            spec.get("end_event"), field="end_event"
        )
        if spec.get("unit") != "s":
            raise MetricSpecError("time_between_events requires unit=s")
        if spec.get("null_semantics") != "null_if_missing_or_reversed":
            raise MetricSpecError(
                "time_between_events requires "
                "null_semantics=null_if_missing_or_reversed"
            )
        return spec
    for field in ("left_signal", "right_signal"):
        signal = spec.get(field)
        if not isinstance(signal, str) or not _SIGNAL.fullmatch(signal):
            raise MetricSpecError(f"MetricSpec.{field} is not a safe trace signal")
        spec[field] = signal
    if spec["left_signal"] == spec["right_signal"]:
        raise MetricSpecError("MetricSpec signals must be distinct")
    dimensions = spec.get("dimensions")
    if dimensions not in (["x", "y"], ["x", "y", "z"]):
        raise MetricSpecError("MetricSpec.dimensions must be [x,y] or [x,y,z]")
    if spec.get("unit") != "m":
        raise MetricSpecError("minimum_distance currently requires unit=m")
    if spec.get("null_semantics") != "null_if_no_finite_sample":
        raise MetricSpecError(
            "MetricSpec.null_semantics must be null_if_no_finite_sample"
        )
    return spec


def metric_spec_tool_spec(
    *,
    task_name: str,
    metric: str,
    question: str,
    metric_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one MetricSpec into the existing routeful ToolSpec envelope."""

    task = str(task_name).strip()
    metric_id = str(metric).strip()
    prompt = str(question).strip()
    if not task:
        raise MetricSpecError("task_name must be non-empty")
    if not _METRIC.fullmatch(metric_id):
        raise MetricSpecError("metric must be a safe lower_snake_case identifier")
    if not prompt:
        raise MetricSpecError("question must be non-empty")
    spec = validate_metric_spec(metric_spec)
    if spec["operation"] == "derived_observable":
        if spec["observable_id"] != metric_id:
            raise MetricSpecError(
                "derived_observable.observable_id must equal the Tool metric"
            )
        required = [
            f"semantic_trace.{signal}"
            for signal in spec["required_signals"]
        ]
        required = list(dict.fromkeys([*required, "semantic_trace.physics_step"]))
        value_type = "number_or_null"
        evidence_kind = "caller_supplied_oracle_evidence_steps"
    elif spec["operation"] == "minimum_distance":
        required = [
            f"semantic_trace.{spec['left_signal']}",
            f"semantic_trace.{spec['right_signal']}",
            "semantic_trace.physics_step",
        ]
        value_type = "number_or_null"
        evidence_kind = "argmin_physics_step"
    elif spec["operation"] == "terminal_minimum_distance":
        required = [
            *[f"semantic_trace.{signal}" for signal in spec["left_signals"]],
            f"semantic_trace.{spec['right_signal']}",
            "semantic_trace.physics_step",
        ]
        value_type = "number_or_null"
        evidence_kind = "terminal_physics_step"
    elif spec["operation"] == "terminal_signal_difference":
        required = [
            f"semantic_trace.{spec['left_signal']}",
            f"semantic_trace.{spec['right_signal']}",
            "semantic_trace.physics_step",
        ]
        value_type = "number_or_null"
        evidence_kind = "terminal_physics_step"
    elif spec["operation"] == "terminal_signal_component":
        required = [
            f"semantic_trace.{spec['signal']}",
            "semantic_trace.physics_step",
        ]
        value_type = "number_or_null"
        evidence_kind = "terminal_physics_step"
    elif spec["operation"] == "event_count":
        required = [f"events.{spec['event']['event_type']}"]
        value_type = "integer"
        evidence_kind = "matching_event_physics_steps"
    else:
        required = [
            f"events.{spec['start_event']['event_type']}",
            f"events.{spec['end_event']['event_type']}",
        ]
        required = list(dict.fromkeys(required))
        value_type = "number_or_null"
        evidence_kind = "boundary_event_physics_steps"
    return {
        "schema_version": 1,
        "task_name": task,
        "metric": metric_id,
        "question": prompt,
        "route": "force_codegen",
        "reference_tool": None,
        "required_signals": required,
        "output_contract": {
            "source": f"typed_metric_spec_v{spec['schema_version']}",
            "metric_spec": spec,
            "value_type": value_type,
            "unit": spec["unit"],
            "passed": None,
            "evidence": evidence_kind,
        },
        "validation_requirements": {
            # The deterministic compiler is checked twice against its private
            # interpreter on every real episode.  A single safe rollout with a
            # legitimate zero event count must therefore remain valid.
            "min_episodes": 1,
            "distinct_reference_values": False,
            "required_reference_values": [],
        },
    }
