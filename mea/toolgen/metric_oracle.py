"""Independent semantic review and oracle comparison for MetricSpec Tools."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from mea.toolkit.tools import TrajectoryView

from .metric_schema import MetricSpecError


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
