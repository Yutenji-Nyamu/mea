"""Trusted MetricSpec interpreter over recorded trajectories."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from mea.toolkit.tools import TrajectoryView

from .metric_schema import MetricSpecError, _DIMENSION_INDEX, validate_metric_spec

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
    if spec["operation"] == "terminal_minimum_distance":
        signal_names = [*spec["left_signals"], spec["right_signal"]]
        try:
            arrays = {
                name: np.asarray(trajectory.trace[name], dtype=float)
                for name in signal_names
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise MetricSpecError(
                f"trajectory is missing a declared signal: {exc}"
            ) from exc
        lengths = {array.shape[0] for array in arrays.values() if array.ndim == 2}
        if len(lengths) != 1 or len(arrays) != sum(
            array.ndim == 2 for array in arrays.values()
        ):
            raise MetricSpecError(
                "declared signals must be aligned two-dimensional arrays"
            )
        sample_count = next(iter(lengths))
        indices = [_DIMENSION_INDEX[item] for item in spec["dimensions"]]
        if not sample_count or any(
            max(indices) >= array.shape[1] for array in arrays.values()
        ):
            raise MetricSpecError(
                "declared signals do not contain the requested terminal dimensions"
            )
        terminal_index = sample_count - 1
        target = arrays[spec["right_signal"]][terminal_index, indices]
        distances = {
            signal: float(
                np.linalg.norm(arrays[signal][terminal_index, indices] - target)
            )
            for signal in spec["left_signals"]
        }
        finite = {
            signal: value
            for signal, value in distances.items()
            if math.isfinite(value)
        }
        winner = min(finite, key=finite.get) if finite else None
        physics = np.asarray(
            trajectory.trace.get("physics_step", np.arange(sample_count)),
            dtype=int,
        )
        if physics.ndim != 1 or len(physics) != sample_count:
            raise MetricSpecError(
                "physics_step must align with the declared signals"
            )
        return {
            "value": finite[winner] if winner is not None else None,
            "unit": spec["unit"],
            "passed": None,
            "evidence_steps": (
                [int(physics[terminal_index])] if winner is not None else []
            ),
            "details": {
                "operation": spec["operation"],
                "left_signals": list(spec["left_signals"]),
                "right_signal": spec["right_signal"],
                "dimensions": list(spec["dimensions"]),
                "selected_left_signal": winner,
                "terminal_index": terminal_index,
                "reason": "measured" if winner is not None else "terminal_not_finite",
            },
        }
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
