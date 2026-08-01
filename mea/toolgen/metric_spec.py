"""Typed semantic contracts and validation for proposal-derived Rule Tools.

Legacy MetricSpecs retain their independent interpreter.  A new derived
observable is provider-written Python admitted by a separate semantic review,
declared-signal AST checks, and deterministic read-only execution on real
telemetry.  It is trajectory measurement only, never success or reward
authority.  Callers may still supply a stronger independent numeric oracle.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from mea.providers.json_response import extract_json_response
from mea.toolkit.tools import TrajectoryView


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "MetricSpec.operation must be event_count, minimum_distance, "
            "terminal_signal_component, terminal_signal_difference, or "
            "time_between_events"
        )
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


def evaluate_metric_spec(
    metric_spec: Mapping[str, Any], trajectory: TrajectoryView
) -> dict[str, Any]:
    """Evaluate the private deterministic oracle for a typed metric."""

    spec = validate_metric_spec(metric_spec)
    if spec["operation"] == "derived_observable":
        raise MetricSpecError(
            "derived_observable has no built-in numeric interpreter; use "
            "ToolGen semantic-review/runtime validation or a caller-supplied "
            "independent oracle"
        )
    if spec["operation"] == "terminal_signal_difference":
        try:
            left = np.asarray(
                trajectory.trace[spec["left_signal"]],
                dtype=float,
            )
            right = np.asarray(
                trajectory.trace[spec["right_signal"]],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MetricSpecError(
                f"trajectory is missing a declared signal: {exc}"
            ) from exc
        component_index = _DIMENSION_INDEX[spec["component"]]
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[0] != right.shape[0]
        ):
            raise MetricSpecError(
                "declared signals must be aligned two-dimensional arrays"
            )
        if (
            not len(left)
            or component_index >= left.shape[1]
            or component_index >= right.shape[1]
        ):
            raise MetricSpecError(
                "declared signals do not contain the requested terminal "
                "component"
            )
        terminal_index = len(left) - 1
        left_value = float(left[terminal_index, component_index])
        right_value = float(right[terminal_index, component_index])
        finite = math.isfinite(left_value) and math.isfinite(right_value)
        signed_difference = left_value - right_value
        value = (
            abs(signed_difference)
            if finite and spec["absolute"]
            else signed_difference
        )
        physics = np.asarray(
            trajectory.trace.get(
                "physics_step", np.arange(len(left))
            ),
            dtype=int,
        )
        if physics.ndim != 1 or len(physics) != len(left):
            raise MetricSpecError(
                "physics_step must align with the declared signals"
            )
        return {
            "value": value if finite else None,
            "unit": spec["unit"],
            "passed": None,
            "evidence_steps": (
                [int(physics[terminal_index])] if finite else []
            ),
            "details": {
                "operation": spec["operation"],
                "left_signal": spec["left_signal"],
                "right_signal": spec["right_signal"],
                "component": spec["component"],
                "absolute": spec["absolute"],
                "left_terminal_value": left_value if finite else None,
                "right_terminal_value": right_value if finite else None,
                "terminal_index": terminal_index,
                "reason": "measured" if finite else "terminal_not_finite",
            },
        }
    if spec["operation"] == "terminal_signal_component":
        try:
            signal = np.asarray(trajectory.trace[spec["signal"]], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise MetricSpecError(
                f"trajectory is missing a declared signal: {exc}"
            ) from exc
        component_index = _DIMENSION_INDEX[spec["component"]]
        if signal.ndim != 2:
            raise MetricSpecError(
                "declared signal must be a two-dimensional array"
            )
        if not len(signal) or component_index >= signal.shape[1]:
            raise MetricSpecError(
                "declared signal does not contain the requested terminal component"
            )
        terminal_index = len(signal) - 1
        raw_value = float(signal[terminal_index, component_index])
        physics = np.asarray(
            trajectory.trace.get(
                "physics_step", np.arange(len(signal))
            ),
            dtype=int,
        )
        if physics.ndim != 1 or len(physics) != len(signal):
            raise MetricSpecError(
                "physics_step must align with the declared signal"
            )
        finite = math.isfinite(raw_value)
        value = abs(raw_value) if finite and spec["absolute"] else raw_value
        return {
            "value": value if finite else None,
            "unit": spec["unit"],
            "passed": None,
            "evidence_steps": [int(physics[terminal_index])] if finite else [],
            "details": {
                "operation": spec["operation"],
                "signal": spec["signal"],
                "component": spec["component"],
                "absolute": spec["absolute"],
                "terminal_index": terminal_index,
                "reason": "measured" if finite else "terminal_not_finite",
            },
        }
    if spec["operation"] == "event_count":
        selector = spec["event"]
        matches = [
            item for item in trajectory.events if _event_matches(item, selector)
        ]
        steps = sorted(
            {
                step
                for item in matches
                for step in [_event_step(item, selector)]
                if step is not None
            }
        )
        return {
            "value": len(matches),
            "unit": spec["unit"],
            "passed": None,
            "evidence_steps": steps,
            "details": {
                "operation": spec["operation"],
                "event": selector,
                "matching_count": len(matches),
                "reason": "measured",
            },
        }
    if spec["operation"] == "time_between_events":
        start = _first_event_boundary(trajectory.events, spec["start_event"])
        end = _first_event_boundary(trajectory.events, spec["end_event"])
        valid = bool(
            start is not None and end is not None and end[0] >= start[0]
        )
        steps = sorted(
            {item[1] for item in (start, end) if item is not None}
        )
        if start is None:
            reason = "start_event_missing"
        elif end is None:
            reason = "end_event_missing"
        elif end[0] < start[0]:
            reason = "end_event_precedes_start_event"
        else:
            reason = "measured"
        return {
            "value": float(end[0] - start[0]) if valid else None,
            "unit": spec["unit"],
            "passed": None,
            "evidence_steps": steps,
            "details": {
                "operation": spec["operation"],
                "start_event": spec["start_event"],
                "end_event": spec["end_event"],
                "start_simulation_time_seconds": start[0] if start else None,
                "end_simulation_time_seconds": end[0] if end else None,
                "start_physics_step": start[1] if start else None,
                "end_physics_step": end[1] if end else None,
                "reason": reason,
            },
        }
    try:
        left = np.asarray(trajectory.trace[spec["left_signal"]], dtype=float)
        right = np.asarray(trajectory.trace[spec["right_signal"]], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise MetricSpecError(f"trajectory is missing a declared signal: {exc}") from exc
    indices = [_DIMENSION_INDEX[item] for item in spec["dimensions"]]
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise MetricSpecError("declared signals must be aligned two-dimensional arrays")
    if not len(left) or max(indices) >= left.shape[1] or max(indices) >= right.shape[1]:
        raise MetricSpecError("declared signals do not contain the requested dimensions")
    left_view = left[:, indices]
    right_view = right[:, indices]
    valid = np.all(np.isfinite(left_view) & np.isfinite(right_view), axis=1)
    distances = np.linalg.norm(left_view - right_view, axis=1)
    masked = np.where(valid, distances, np.inf)
    index = int(np.argmin(masked))
    value = float(masked[index])
    if not math.isfinite(value):
        return {
            "value": None,
            "unit": spec["unit"],
            "passed": None,
            "evidence_steps": [],
            "details": {
                "operation": spec["operation"],
                "left_signal": spec["left_signal"],
                "right_signal": spec["right_signal"],
                "dimensions": list(spec["dimensions"]),
                "min_index": None,
                "reason": "no_finite_sample",
            },
        }
    physics = np.asarray(
        trajectory.trace.get("physics_step", np.arange(len(left))), dtype=int
    )
    step = int(physics[index])
    return {
        "value": value,
        "unit": spec["unit"],
        "passed": None,
        "evidence_steps": [step],
        "details": {
            "operation": spec["operation"],
            "left_signal": spec["left_signal"],
            "right_signal": spec["right_signal"],
            "dimensions": list(spec["dimensions"]),
            "min_index": index,
            "reason": "measured",
        },
    }


def _event_matches(event: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    if event.get("type") != selector["event_type"]:
        return False
    if selector["physical_only"] and event.get("physical_contact") is not True:
        return False
    actors = selector["actors"]
    if actors is None:
        return True
    # Current recorder artifacts expose stable TaskSchema actor ids alongside
    # legacy simulator names.  Query-derived selectors are expressed in those
    # advertised ids so same-name actor instances remain distinguishable.
    # Historical episodes have no actor_ids and retain their old name match.
    event_actors = event.get("actor_ids") or event.get("actors", [])
    return sorted(event_actors) == actors


def _event_fields(selector: Mapping[str, Any]) -> tuple[str, str]:
    if selector["event_type"] == "success_transition":
        return "simulation_time_seconds", "physics_step"
    if selector["physical_only"]:
        return (
            "first_physical_simulation_time_seconds",
            "first_physical_physics_step",
        )
    return "start_simulation_time_seconds", "start_physics_step"


def _event_step(event: Mapping[str, Any], selector: Mapping[str, Any]) -> int | None:
    _, step_field = _event_fields(selector)
    value = event.get(step_field)
    return (
        int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _first_event_boundary(
    events: Iterable[Mapping[str, Any]], selector: Mapping[str, Any]
) -> tuple[float, int] | None:
    time_field, step_field = _event_fields(selector)
    matches = []
    for event in events:
        if not _event_matches(event, selector):
            continue
        time_value = event.get(time_field)
        step_value = event.get(step_field)
        if (
            isinstance(time_value, (int, float))
            and not isinstance(time_value, bool)
            and math.isfinite(float(time_value))
            and isinstance(step_value, (int, float))
            and not isinstance(step_value, bool)
        ):
            matches.append((float(time_value), int(step_value)))
    return min(matches) if matches else None


def _compiled_event_filter(selector: Mapping[str, Any]) -> str:
    clauses = [f"item.get('type') == {selector['event_type']!r}"]
    if selector["physical_only"]:
        clauses.append("item.get('physical_contact') is True")
    if selector["actors"] is not None:
        clauses.append(
            "sorted(item.get('actor_ids') or item.get('actors', [])) == "
            f"{selector['actors']!r}"
        )
    return " and ".join(clauses)


def _compile_event_count_source(spec: Mapping[str, Any]) -> str:
    selector = spec["event"]
    _, step_field = _event_fields(selector)
    event_filter = _compiled_event_filter(selector)
    return f'''def generated_tool(trajectory):
    events = [
        item for item in trajectory.events
        if {event_filter}
    ]
    steps = sorted(list(set([
        int(item.get({step_field!r}))
        for item in events
        if item.get({step_field!r}) is not None
    ])))
    return {{
        "value": len(events),
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": steps,
        "details": {{
            "operation": {spec["operation"]!r},
            "event": {selector!r},
            "matching_count": len(events),
            "reason": "measured",
        }},
    }}
'''


def _compile_time_between_events_source(spec: Mapping[str, Any]) -> str:
    start_selector = spec["start_event"]
    end_selector = spec["end_event"]
    start_time, start_step = _event_fields(start_selector)
    end_time, end_step = _event_fields(end_selector)
    start_filter = _compiled_event_filter(start_selector)
    end_filter = _compiled_event_filter(end_selector)
    return f'''def generated_tool(trajectory):
    start_events = [
        item for item in trajectory.events
        if {start_filter}
    ]
    end_events = [
        item for item in trajectory.events
        if {end_filter}
    ]
    start_points = [
        (float(item.get({start_time!r})), int(item.get({start_step!r})))
        for item in start_events
        if item.get({start_time!r}) is not None
        and item.get({start_step!r}) is not None
    ]
    end_points = [
        (float(item.get({end_time!r})), int(item.get({end_step!r})))
        for item in end_events
        if item.get({end_time!r}) is not None
        and item.get({end_step!r}) is not None
    ]
    start = min(start_points) if start_points else None
    end = min(end_points) if end_points else None
    valid = bool(start is not None and end is not None and end[0] >= start[0])
    steps = sorted(list(set([
        item[1] for item in [start, end] if item is not None
    ])))
    if start is None:
        reason = "start_event_missing"
    elif end is None:
        reason = "end_event_missing"
    elif end[0] < start[0]:
        reason = "end_event_precedes_start_event"
    else:
        reason = "measured"
    return {{
        "value": float(end[0] - start[0]) if valid else None,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": steps,
        "details": {{
            "operation": {spec["operation"]!r},
            "start_event": {start_selector!r},
            "end_event": {end_selector!r},
            "start_simulation_time_seconds": start[0] if start else None,
            "end_simulation_time_seconds": end[0] if end else None,
            "start_physics_step": start[1] if start else None,
            "end_physics_step": end[1] if end else None,
            "reason": reason,
        }},
    }}
'''


def _compile_terminal_signal_component_source(spec: Mapping[str, Any]) -> str:
    component_index = _DIMENSION_INDEX[spec["component"]]
    value_expression = "abs(raw_value)" if spec["absolute"] else "raw_value"
    return f'''def generated_tool(trajectory):
    signal = np.asarray(trajectory.trace[{spec["signal"]!r}], dtype=float)
    terminal_index = len(signal) - 1
    raw_value = float(signal[terminal_index, {component_index}])
    finite = bool(np.isfinite(raw_value))
    value = {value_expression} if finite else None
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    return {{
        "value": value,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": [int(physics[terminal_index])] if finite else [],
        "details": {{
            "operation": {spec["operation"]!r},
            "signal": {spec["signal"]!r},
            "component": {spec["component"]!r},
            "absolute": {spec["absolute"]!r},
            "terminal_index": terminal_index,
            "reason": "measured" if finite else "terminal_not_finite",
        }},
    }}
'''


def _compile_terminal_signal_difference_source(spec: Mapping[str, Any]) -> str:
    component_index = _DIMENSION_INDEX[spec["component"]]
    value_expression = (
        "abs(signed_difference)"
        if spec["absolute"]
        else "signed_difference"
    )
    return f'''def generated_tool(trajectory):
    left = np.asarray(trajectory.trace[{spec["left_signal"]!r}], dtype=float)
    right = np.asarray(trajectory.trace[{spec["right_signal"]!r}], dtype=float)
    terminal_index = len(left) - 1
    left_value = float(left[terminal_index, {component_index}])
    right_value = float(right[terminal_index, {component_index}])
    finite = bool(np.isfinite(left_value) and np.isfinite(right_value))
    signed_difference = left_value - right_value
    value = {value_expression} if finite else None
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    return {{
        "value": value,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": [int(physics[terminal_index])] if finite else [],
        "details": {{
            "operation": {spec["operation"]!r},
            "left_signal": {spec["left_signal"]!r},
            "right_signal": {spec["right_signal"]!r},
            "component": {spec["component"]!r},
            "absolute": {spec["absolute"]!r},
            "left_terminal_value": left_value if finite else None,
            "right_terminal_value": right_value if finite else None,
            "terminal_index": terminal_index,
            "reason": "measured" if finite else "terminal_not_finite",
        }},
    }}
'''


def compile_metric_spec_source(metric_spec: Mapping[str, Any]) -> str:
    """Compile a MetricSpec to auditable Python accepted by ToolGen's AST gate."""

    spec = validate_metric_spec(metric_spec)
    if spec["operation"] == "derived_observable":
        raise MetricSpecError(
            "derived_observable has no compatibility compiler; provide a "
            "ToolGen provider or an exact validated registry match"
        )
    if spec["operation"] == "event_count":
        return _compile_event_count_source(spec)
    if spec["operation"] == "time_between_events":
        return _compile_time_between_events_source(spec)
    if spec["operation"] == "terminal_signal_component":
        return _compile_terminal_signal_component_source(spec)
    if spec["operation"] == "terminal_signal_difference":
        return _compile_terminal_signal_difference_source(spec)
    indices = [_DIMENSION_INDEX[item] for item in spec["dimensions"]]
    return f'''def generated_tool(trajectory):
    left = np.asarray(trajectory.trace[{spec["left_signal"]!r}], dtype=float)
    right = np.asarray(trajectory.trace[{spec["right_signal"]!r}], dtype=float)
    left_view = left[:, {indices!r}]
    right_view = right[:, {indices!r}]
    valid = np.all(np.isfinite(left_view) & np.isfinite(right_view), axis=1)
    distances = np.linalg.norm(left_view - right_view, axis=1)
    masked = np.where(valid, distances, np.inf)
    index = int(np.argmin(masked))
    value = float(masked[index])
    if not np.isfinite(value):
        return {{
            "value": None,
            "unit": {spec["unit"]!r},
            "passed": None,
            "evidence_steps": [],
            "details": {{
                "operation": {spec["operation"]!r},
                "left_signal": {spec["left_signal"]!r},
                "right_signal": {spec["right_signal"]!r},
                "dimensions": {spec["dimensions"]!r},
                "min_index": None,
                "reason": "no_finite_sample",
            }},
        }}
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    step = int(physics[index])
    return {{
        "value": value,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": [step],
        "details": {{
            "operation": {spec["operation"]!r},
            "left_signal": {spec["left_signal"]!r},
            "right_signal": {spec["right_signal"]!r},
            "dimensions": {spec["dimensions"]!r},
            "min_index": index,
            "reason": "measured",
        }},
    }}
'''


def build_task_code_context(
    child_run_dir: str | Path,
    *,
    task_proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact TaskGen-code context consumed by typed ToolGen."""

    child = Path(child_run_dir).expanduser().resolve()
    manifest_path = child / "manifest.json"
    if not manifest_path.is_file():
        raise MetricSpecError(f"TaskGen manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = child / "task.py"
    bundle_path = child / "generation/task_artifact_bundle.json"
    bundle = (
        json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle_path.is_file()
        else None
    )
    return {
        "schema_version": 1,
        "task_name": manifest.get("task_name"),
        "task_module": manifest.get("task_module"),
        "generation_kind": manifest.get("generation_kind"),
        "task_proposal": deepcopy(dict(task_proposal)) if task_proposal else None,
        "task_source": (
            {
                "path": "task.py",
                "sha256": _file_sha256(source),
                "excerpt": source.read_text(encoding="utf-8")[:4000],
            }
            if source.is_file()
            else None
        ),
        "task_artifact_bundle": bundle,
    }


def _provider_codegen_prompt(
    *,
    repo_root: Path,
    metric: str,
    question: str,
    metric_spec: Mapping[str, Any],
    trajectory: TrajectoryView,
    task_code_context: Mapping[str, Any] | None,
    previous_error: str | None,
) -> str:
    """Build the compact prompt for a real Python ToolGen attempt."""

    contract = (repo_root / "mea/toolgen/README.Agent.md").read_text(
        encoding="utf-8"
    )
    operation = str(metric_spec["operation"])
    normal_reason = "measured"
    null_reasons = {
        "derived_observable": ["no_finite_sample"],
        "minimum_distance": ["no_finite_sample"],
        "terminal_signal_component": ["terminal_not_finite"],
        "terminal_signal_difference": ["terminal_not_finite"],
        "event_count": [],
        "time_between_events": [
            "start_event_missing",
            "end_event_missing",
            "end_event_precedes_start_event",
        ],
    }[operation]
    result_contract = {
        "passed": None,
        "evidence_steps": "plain Python int physics steps",
        "details.operation": operation,
        "details.reason": normal_reason,
        "details.allowed_null_reasons": null_reasons,
        "json_native_scalars_only": True,
    }
    telemetry = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(trajectory.trace.items())
    }
    repair = (
        "\nPREVIOUS VALIDATION FAILURE:\n"
        + previous_error
        + "\nRepair only the reported failure and return the complete function.\n"
        if previous_error
        else ""
    )
    return f"""You are ToolGen in ManipEvalAgent.
Write the Python implementation of one Query-induced Rule Tool.  The typed
MetricSpec below is the semantic contract, not source code to copy.
For derived_observable, implement the provider-proposed description over only
its declared telemetry signals; it is not a pre-registered metric operator.
Implement the observable from the recorded trajectory.  For a new derived
observable, a separate semantic reviewer will inspect this complete source
without changing it.  The result is also restricted to declared signals,
checked by an AST allowlist, executed twice on real telemetry, validated for a
finite scalar/null value, the requested unit and trace-bound evidence steps,
and rejected if episode artifacts change.  This Tool is measurement evidence
only and must not define task success or reward.

METRIC ID:
{metric}

QUESTION:
{question}

SEMANTIC ORACLE CONTRACT:
{json.dumps(metric_spec, ensure_ascii=False, indent=2)}

REQUIRED RESULT SEMANTICS:
{json.dumps(result_contract, ensure_ascii=False, indent=2)}

REAL TELEMETRY SURFACE:
{json.dumps(telemetry, ensure_ascii=False, indent=2)}

TASKGEN CONTEXT:
{json.dumps(task_code_context, ensure_ascii=False, indent=2)}

TOOL CONTRACT:
{contract}
{repair}
Return exactly one Python fenced block containing the complete
def generated_tool(trajectory): function and nothing else.
"""


def _validate_metric_source(
    *,
    source_text: str,
    metric: str,
    spec: Mapping[str, Any],
    episodes: list[Path],
    trajectories: list[TrajectoryView],
    oracle_evaluator: Callable[[TrajectoryView], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Run static, determinism, semantic, and artifact-preservation gates."""

    from mea.toolgen.prototype import (
        ToolGenError,
        execute_generated_tool,
        validate_generated_tool,
    )

    try:
        validate_generated_tool(source_text)
        if spec["operation"] == "derived_observable":
            _validate_derived_signal_access(source_text, spec)
    except ToolGenError as exc:
        raise MetricSpecError(f"generated Python failed the static gate: {exc}") from exc
    except MetricSpecError:
        raise
    rows: list[dict[str, Any]] = []
    values: list[Any] = []
    for episode, trajectory in zip(episodes, trajectories):
        before = {
            name: _file_sha256(episode / name)
            for name in _CORE_ARTIFACTS
            if (episode / name).is_file()
        }
        try:
            first = execute_generated_tool(source_text, episode, tool_name=metric)
            second = execute_generated_tool(source_text, episode, tool_name=metric)
        except ToolGenError as exc:
            raise MetricSpecError(
                f"generated Python failed on real telemetry: {exc}"
            ) from exc
        generated = {
            key: first.get(key)
            for key in ("value", "unit", "passed", "evidence_steps", "details")
        }
        if spec["operation"] == "derived_observable":
            oracle = (
                _validate_external_oracle_result(
                    oracle_evaluator(trajectory),
                    spec=spec,
                    trajectory=trajectory,
                )
                if oracle_evaluator is not None
                else _validate_external_oracle_result(
                    generated,
                    spec=spec,
                    trajectory=trajectory,
                )
            )
        else:
            oracle = evaluate_metric_spec(spec, trajectory)
        deterministic = _canonical(first) == _canonical(second)
        semantic_differences = (
            _metric_semantic_differences(generated, oracle)
            if oracle_evaluator is not None
            or spec["operation"] != "derived_observable"
            else []
        )
        oracle_agreement = (
            not semantic_differences
            if oracle_evaluator is not None
            or spec["operation"] != "derived_observable"
            else None
        )
        semantic_contract_valid = not semantic_differences
        after = {
            name: _file_sha256(episode / name)
            for name in _CORE_ARTIFACTS
            if (episode / name).is_file()
        }
        if (
            not deterministic
            or not semantic_contract_valid
            or before != after
        ):
            raise MetricSpecError(
                "generated Python validation failed: "
                + _canonical(
                    {
                        "deterministic": deterministic,
                        "oracle_agreement": oracle_agreement,
                        "semantic_contract_valid": semantic_contract_valid,
                        "artifacts_unchanged": before == after,
                        "semantic_differences": semantic_differences,
                        "expected": _metric_semantic_projection(oracle),
                        "actual": _metric_semantic_projection(generated),
                    }
                )
            )
        values.append(oracle.get("value"))
        rows.append(
            {
                "episode_dir": str(episode),
                "policy_name": trajectory.metadata.get("policy_name"),
                "seed": trajectory.metadata.get("seed"),
                "generated_result": first,
                "oracle_projection": oracle,
                "deterministic": deterministic,
                "oracle_agreement": oracle_agreement,
                "semantic_contract_valid": semantic_contract_valid,
                "validation_authority": (
                    "caller_supplied_independent_numeric_oracle"
                    if oracle_evaluator is not None
                    else "toolgen_semantic_review_runtime"
                    if spec["operation"] == "derived_observable"
                    else "typed_metric_spec_interpreter"
                ),
                "artifacts_unchanged": before == after,
            }
        )
    return rows, values


_DERIVED_STANDARD_TRACE_KEYS = {
    "physics_step",
    "policy_step",
    "simulation_time_seconds",
}


def _validate_derived_signal_access(
    source_text: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Restrict a derived Tool to its declared trace surface."""

    tree = ast.parse(source_text)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "trajectory":
                if node.attr != "trace":
                    raise MetricSpecError(
                        "derived_observable may access only trajectory.trace"
                    )
                parent = parents.get(node)
                if not (
                    isinstance(parent, ast.Subscript)
                    and parent.value is node
                ):
                    raise MetricSpecError(
                        "derived_observable must access trajectory.trace "
                        "directly with a literal field name"
                    )
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "trajectory"
            and value.attr == "trace"
        ):
            continue
        key = node.slice
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise MetricSpecError(
                "derived_observable trace access must use a literal field name"
            )
        used.add(key.value)
    declared = set(spec["required_signals"])
    undeclared = sorted(used - declared - _DERIVED_STANDARD_TRACE_KEYS)
    missing = sorted(declared - used)
    if undeclared:
        raise MetricSpecError(
            "derived_observable source uses undeclared telemetry signals: "
            f"{undeclared}"
        )
    if missing:
        raise MetricSpecError(
            "derived_observable source does not use declared signals: "
            f"{missing}"
        )
    return {
        "required_signals": sorted(declared),
        "used_trace_keys": sorted(used),
    }


_SEMANTIC_REVIEW_CHECKS = {
    "implements_metric_description",
    "uses_only_declared_signals",
    "preserves_requested_unit",
    "returns_diagnostic_not_success",
}


def _validate_semantic_review(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricSpecError("ToolGen semantic review must be an object")
    review = deepcopy(dict(value))
    if set(review) != {"schema_version", "status", "checks", "reason"}:
        raise MetricSpecError(
            "ToolGen semantic review fields must be exactly "
            "schema_version/status/checks/reason"
        )
    checks = review.get("checks")
    if (
        review.get("schema_version") != 1
        or review.get("status") not in {"approved", "rejected"}
        or not isinstance(checks, Mapping)
        or set(checks) != _SEMANTIC_REVIEW_CHECKS
        or any(type(item) is not bool for item in checks.values())
        or not isinstance(review.get("reason"), str)
        or not review["reason"].strip()
    ):
        raise MetricSpecError("ToolGen semantic review contract is invalid")
    if review["status"] != "approved" or not all(checks.values()):
        raise MetricSpecError(
            "ToolGen semantic reviewer rejected the generated Tool: "
            + review["reason"].strip()
        )
    review["checks"] = dict(checks)
    review["reason"] = review["reason"].strip()
    return review


def _semantic_review_prompt(
    *,
    metric: str,
    metric_spec: Mapping[str, Any],
    source_text: str,
) -> str:
    return f"""You are ToolGen's separate semantic-review pass.
Review one generated trajectory-measurement Tool against its MetricSpec.
You are a development-agent proxy, not an independent human or model.
Approve only if the code directly implements the description, accesses only
required_signals plus physics/policy/time indices, preserves the requested
unit, and returns passed=None rather than defining success or reward.
Do not rewrite the code and do not infer task success.

METRIC:
{metric}

METRIC SPEC:
{json.dumps(metric_spec, ensure_ascii=False, indent=2)}

GENERATED SOURCE:
```python
{source_text.rstrip()}
```

Return strict JSON with exactly:
{{
  "schema_version": 1,
  "status": "approved" or "rejected",
  "checks": {{
    "implements_metric_description": true or false,
    "uses_only_declared_signals": true or false,
    "preserves_requested_unit": true or false,
    "returns_diagnostic_not_success": true or false
  }},
  "reason": "one concise sentence"
}}
"""


def _validate_external_oracle_result(
    value: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    trajectory: TrajectoryView,
) -> dict[str, Any]:
    """Normalize the caller-owned oracle without trusting generated Tool code."""

    if not isinstance(value, Mapping):
        raise MetricSpecError(
            "derived_observable oracle must return a result object"
        )
    result = deepcopy(dict(value))
    required = {"value", "unit", "passed", "evidence_steps", "details"}
    if set(result) != required:
        raise MetricSpecError(
            "derived_observable oracle result fields must be exactly "
            f"{sorted(required)}"
        )
    measured = result["value"]
    if (
        measured is not None
        and (
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or not math.isfinite(float(measured))
        )
    ):
        raise MetricSpecError(
            "derived_observable oracle value must be finite numeric or null"
        )
    if result["unit"] != spec["unit"] or result["passed"] is not None:
        raise MetricSpecError(
            "derived_observable oracle must preserve unit and passed=null"
        )
    steps = result["evidence_steps"]
    physics_steps = {
        int(item)
        for item in np.asarray(
            trajectory.trace["physics_step"],
            dtype=int,
        )
    }
    if (
        not isinstance(steps, list)
        or any(type(item) is not int or item not in physics_steps for item in steps)
        or steps != sorted(set(steps))
    ):
        raise MetricSpecError(
            "derived_observable oracle evidence_steps must be unique ordered "
            "physics steps from the fixture or live trajectory"
        )
    details = result["details"]
    if not isinstance(details, Mapping):
        raise MetricSpecError(
            "derived_observable oracle details must be an object"
        )
    expected_reason = "measured" if measured is not None else "no_finite_sample"
    if (
        details.get("operation") != "derived_observable"
        or details.get("reason") != expected_reason
    ):
        raise MetricSpecError(
            "derived_observable oracle details must declare its operation "
            "and measured/no_finite_sample reason"
        )
    result["value"] = float(measured) if measured is not None else None
    result["details"] = deepcopy(dict(details))
    return result


def _metric_semantic_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project only fields that define the metric's observable semantics."""

    details = payload.get("details")
    details = details if isinstance(details, Mapping) else {}
    return {
        "value": payload.get("value"),
        "unit": payload.get("unit"),
        "passed": payload.get("passed"),
        "evidence_steps": payload.get("evidence_steps"),
        "details": {
            "operation": details.get("operation"),
            "reason": details.get("reason"),
        },
    }


def _metric_values_equal(actual: Any, expected: Any) -> bool:
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
    return actual == expected


def _metric_semantic_differences(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[str]:
    actual_projection = _metric_semantic_projection(actual)
    expected_projection = _metric_semantic_projection(expected)
    differences = []
    for field in ("value", "unit", "passed", "evidence_steps"):
        if not _metric_values_equal(
            actual_projection[field],
            expected_projection[field],
        ):
            differences.append(field)
    for field in ("operation", "reason"):
        if (
            actual_projection["details"][field]
            != expected_projection["details"][field]
        ):
            differences.append(f"details.{field}")
    return differences


def execute_metric_spec(
    *,
    task_name: str,
    metric: str,
    question: str,
    metric_spec: Mapping[str, Any],
    episode_dirs: Iterable[str | Path],
    output_dir: str | Path,
    fixture_episode_dirs: Iterable[str | Path] = (),
    oracle_evaluator: Callable[[TrajectoryView], Mapping[str, Any]] | None = None,
    task_code_context: Mapping[str, Any] | None = None,
    registry_dir: str | Path | None = None,
    provider: Any | None = None,
    model: str | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Generate, validate, and optionally register one Query-induced Tool."""

    from mea.toolgen.prototype import (
        ToolGenError,
        extract_generated_tool,
        validate_generated_tool,
    )
    from mea.toolgen.registry import (
        find_run_local_registration,
        public_registration_summary,
        register_run_local_tool,
    )

    spec = validate_metric_spec(metric_spec)
    tool_spec = metric_spec_tool_spec(
        task_name=task_name,
        metric=metric,
        question=question,
        metric_spec=spec,
    )
    context = deepcopy(dict(task_code_context)) if task_code_context else None
    if context is not None and context.get("task_name") != task_name:
        raise MetricSpecError("TaskGen code context belongs to a different task")
    episodes = [Path(item).expanduser().resolve() for item in episode_dirs]
    if not episodes or len(set(episodes)) != len(episodes):
        raise MetricSpecError(
            "MetricSpec validation needs at least one unique telemetry episode"
        )
    fixtures = [
        Path(item).expanduser().resolve()
        for item in fixture_episode_dirs
    ]
    if len(set(fixtures)) != len(fixtures) or set(fixtures).intersection(episodes):
        raise MetricSpecError(
            "fixture and live telemetry episode paths must be unique"
        )
    if spec["operation"] == "derived_observable" and (
        bool(fixtures) != (oracle_evaluator is not None)
    ):
        raise MetricSpecError(
            "caller-supplied derived_observable fixtures and numeric oracle "
            "must be provided together"
        )
    if spec["operation"] != "derived_observable" and (
        fixtures or oracle_evaluator is not None
    ):
        raise MetricSpecError(
            "caller-supplied fixtures/oracle are only valid for "
            "derived_observable"
        )
    validation_episodes = [*fixtures, *episodes]
    validation_trajectories = [
        TrajectoryView(path) for path in validation_episodes
    ]
    trajectories = validation_trajectories[len(fixtures) :]
    for trajectory in validation_trajectories:
        if (
            trajectory.metadata.get("task_name") != task_name
            or trajectory.schema.get("task_name") != task_name
        ):
            raise MetricSpecError("MetricSpec episode task/schema does not match")
    if spec["operation"] in {
        "derived_observable",
        "minimum_distance",
        "terminal_signal_component",
        "terminal_signal_difference",
    }:
        required_signals = (
            set(spec["required_signals"])
            if spec["operation"] == "derived_observable"
            else {spec["left_signal"], spec["right_signal"]}
            if spec["operation"]
            in {"minimum_distance", "terminal_signal_difference"}
            else {spec["signal"]}
        )
        for trajectory in validation_trajectories:
            missing = sorted(required_signals - set(trajectory.trace))
            if missing:
                raise MetricSpecError(
                    f"MetricSpec signals are absent from TaskSchema telemetry: {missing}"
                )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise MetricSpecError(f"MetricSpec output already exists: {destination}")
    destination.mkdir(parents=True)
    _write_json(destination / "metric_spec.json", spec)
    _write_json(destination / "tool_spec.json", tool_spec)
    if context is not None:
        _write_json(destination / "task_code_context.json", context)

    def validate_source(
        source_text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
        validation_rows, values = _validate_metric_source(
            source_text=source_text,
            metric=metric,
            spec=spec,
            episodes=validation_episodes,
            trajectories=validation_trajectories,
            oracle_evaluator=oracle_evaluator,
        )
        return (
            validation_rows[len(fixtures) :],
            validation_rows[: len(fixtures)],
            values,
        )

    registry_match = None
    if registry_dir is not None:
        registry_match = find_run_local_registration(
            registry_dir, tool_spec=tool_spec, episode_dirs=episodes
        )
    semantic_review: dict[str, Any] | None = None
    automatic_derived_validation = (
        spec["operation"] == "derived_observable"
        and oracle_evaluator is None
    )
    if registry_match is not None and automatic_derived_validation:
        stored_review = (
            registry_match["registration"]
            .get("validation", {})
            .get("semantic_review")
        )
        try:
            semantic_review = _validate_semantic_review(stored_review)
        except MetricSpecError:
            # A legacy derived Tool without this review contract is not an
            # exact match for the new mainline validation authority.
            registry_match = None
    if registry_match is not None:
        source_path = registry_match["source_path"]
        route = "run_local_reuse"
        generation: dict[str, Any] | None = None
        rows, fixture_rows, values = validate_source(
            source_path.read_text(encoding="utf-8")
        )
    elif provider is not None:
        if not isinstance(model, str) or not model.strip():
            raise MetricSpecError("provider Python codegen requires model")
        attempts_dir = destination / "attempts"
        attempts_dir.mkdir()
        failures: list[dict[str, Any]] = []
        rows = []
        fixture_rows = []
        values = []
        source_text: str | None = None
        successful_attempt: int | None = None
        attempt_limit = max(1, min(int(max_attempts), 3))
        for attempt_index in range(attempt_limit):
            attempt_dir = attempts_dir / f"attempt_{attempt_index}"
            attempt_dir.mkdir()
            prompt = _provider_codegen_prompt(
                repo_root=Path(__file__).resolve().parents[2],
                metric=metric,
                question=question,
                metric_spec=spec,
                trajectory=trajectories[0],
                task_code_context=context,
                previous_error=(
                    failures[-1]["message"] if failures else None
                ),
            )
            (attempt_dir / "prompt.md").write_text(prompt, encoding="utf-8")
            try:
                response = provider.text(
                    prompt,
                    model=model.strip(),
                    system=(
                        "Return exactly one Python code fence containing the "
                        "complete generated_tool(trajectory) function."
                    ),
                    max_tokens=1800,
                    temperature=0.0,
                )
                (attempt_dir / "response.txt").write_text(
                    response + "\n", encoding="utf-8"
                )
                candidate = extract_generated_tool(response)
                (attempt_dir / "generated_tool.py").write_text(
                    candidate, encoding="utf-8"
                )
                candidate_review = None
                if automatic_derived_validation:
                    validate_generated_tool(candidate)
                    _validate_derived_signal_access(candidate, spec)
                    review_prompt = _semantic_review_prompt(
                        metric=metric,
                        metric_spec=spec,
                        source_text=candidate,
                    )
                    (attempt_dir / "review_prompt.md").write_text(
                        review_prompt,
                        encoding="utf-8",
                    )
                    review_response = provider.text(
                        review_prompt,
                        model=model.strip(),
                        system="Return only strict ToolGen semantic-review JSON.",
                        max_tokens=500,
                        temperature=0.0,
                    )
                    (attempt_dir / "review_response.txt").write_text(
                        review_response + "\n",
                        encoding="utf-8",
                    )
                    candidate_review = _validate_semantic_review(
                        extract_json_response(review_response)
                    )
                    _write_json(
                        attempt_dir / "semantic_review.json",
                        candidate_review,
                    )
                (
                    candidate_rows,
                    candidate_fixture_rows,
                    candidate_values,
                ) = validate_source(
                    candidate
                )
            except Exception as exc:
                failure = {
                    "attempt_index": attempt_index,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "provider": deepcopy(
                        dict(getattr(provider, "last_metadata", {}))
                    ),
                }
                failures.append(failure)
                _write_json(
                    attempt_dir / "validation.json",
                    {"valid": False, **failure},
                )
                continue
            source_text = candidate
            rows = candidate_rows
            fixture_rows = candidate_fixture_rows
            values = candidate_values
            successful_attempt = attempt_index
            semantic_review = candidate_review
            _write_json(
                attempt_dir / "validation.json",
                {
                    "valid": True,
                    "episode_count": len(rows),
                    "deterministic": True,
                    "oracle_agreement": (
                        True if oracle_evaluator is not None else None
                    ),
                    "semantic_contract_valid": True,
                    "semantic_review": semantic_review,
                    "artifacts_unchanged": True,
                },
            )
            break
        if source_text is None:
            raise MetricSpecError(
                "provider failed to generate a valid Python Tool: "
                + " | ".join(item["message"] for item in failures)
            )
        source_path = destination / "generated_tool.py"
        source_path.write_text(source_text, encoding="utf-8")
        route = "provider_python_codegen"
        generation = {
            "successful_attempt": successful_attempt,
            "attempt_count": len(failures) + 1,
            "failures": failures,
            "model_requested": model.strip(),
            "provider": deepcopy(dict(getattr(provider, "last_metadata", {}))),
            "semantic_review": semantic_review,
        }
    else:
        if spec["operation"] == "derived_observable":
            raise MetricSpecError(
                "derived_observable registry miss requires a provider"
            )
        source_path = destination / "generated_tool.py"
        source_path.write_text(compile_metric_spec_source(spec), encoding="utf-8")
        try:
            validate_generated_tool(source_path.read_text(encoding="utf-8"))
        except ToolGenError as exc:  # pragma: no cover - compiler invariant guard
            raise MetricSpecError(
                f"compiled MetricSpec failed the ToolGen static gate: {exc}"
            ) from exc
        route = "typed_metric_spec_compile"
        generation = None
        rows, fixture_rows, values = validate_source(
            source_path.read_text(encoding="utf-8")
        )
    finite_values = [float(item) for item in values if isinstance(item, (int, float))]
    if any(not math.isfinite(item) for item in finite_values):
        raise MetricSpecError("MetricSpec oracle produced a non-finite value")

    registration = None
    if registry_match is None and registry_dir is not None:
        registration = register_run_local_tool(
            registry_dir,
            tool_spec=tool_spec,
            episode_dirs=episodes,
            source_path=source_path,
            generation_registration={
                "tool": metric,
                "validated_episode_count": len(rows) + len(fixture_rows),
                "validated_property_scenario_count": len(fixture_rows),
                "oracle_kind": (
                    "toolgen_semantic_review_runtime_v1"
                    if automatic_derived_validation
                    else f"typed_metric_spec_v{spec['schema_version']}"
                ),
                "oracle_agreement_required": not automatic_derived_validation,
                "semantic_review_required": automatic_derived_validation,
            },
            generation_manifest={
                "successful_attempt": (
                    generation.get("successful_attempt")
                    if generation is not None
                    else None
                ),
                "model_requested": (
                    generation.get("model_requested")
                    if generation is not None
                    else None
                ),
                "generator_source_sha256": _file_sha256(source_path),
                "contract_sha256": hashlib.sha256(_canonical(spec).encode()).hexdigest(),
                "example_validation": [
                    _metric_semantic_projection(row["oracle_projection"])
                    for row in fixture_rows
                ],
                "semantic_review": semantic_review,
            },
            validation_episodes=[
                {
                    "episode_dir": str(episode),
                    "policy_name": row["policy_name"],
                    "seed": row["seed"],
                    "oracle_value": row["oracle_projection"].get("value"),
                }
                for episode, row in zip(episodes, rows)
            ],
        )
    elif registry_match is not None:
        registration = registry_match
    result = {
        "schema_version": 1,
        "status": "passed",
        "route": route,
        "provider_called": generation is not None,
        "generation": generation,
        "source_path": str(source_path),
        "tool_spec": tool_spec,
        "task_code_context_consumed": context is not None,
        "validation_authority": (
            "toolgen_semantic_review_runtime"
            if automatic_derived_validation
            else "caller_supplied_independent_numeric_oracle"
            if spec["operation"] == "derived_observable"
            else "typed_metric_spec_interpreter"
        ),
        "semantic_review": semantic_review,
        "fixtures": fixture_rows,
        "episodes": rows,
        "registration": (
            public_registration_summary(registration) if registration else None
        ),
        "limitations": [
            (
                "provider-defined derived observable over declared telemetry"
                if spec["operation"] == "derived_observable"
                else f"typed semantic oracle: {spec['operation']}"
            ),
            (
                "provider-generated Python"
                if generation is not None
                else "reused validated generated Python"
                if route == "run_local_reuse"
                else "provider-free compatibility compiler"
            ),
            (
                "semantic review plus declared-signal, deterministic, finite-"
                "result, evidence-step, and artifact-immutability gates; no "
                "success or reward authority"
                if automatic_derived_validation
                else "output is checked twice against the caller-supplied "
                "numeric oracle on fixtures and live episodes"
                if spec["operation"] == "derived_observable"
                else "output is checked twice against the trusted "
                "interpreter on each live episode; live values need not differ"
            ),
        ],
    }
    _write_json(destination / "execution.json", result)
    return result


__all__ = [
    "MetricSpecError",
    "build_task_code_context",
    "compile_metric_spec_source",
    "evaluate_metric_spec",
    "execute_metric_spec",
    "metric_spec_tool_spec",
    "validate_metric_spec",
]
