"""External-oracle validation and result comparison for MetricSpec Tools."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from mea.toolkit.tools import TrajectoryView

from .metric_schema import MetricSpecError


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
