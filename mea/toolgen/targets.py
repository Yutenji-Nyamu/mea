"""Oracle contracts for exact and composition-based ToolGen targets."""

from __future__ import annotations

from typing import Any

from mea.toolkit.tools import TOOL_CATALOG, TrajectoryView


PICKUP_TO_CONTACT_METRIC = "pickup_to_first_contact_time"


COMPOSITE_TARGETS: dict[str, dict[str, Any]] = {
    PICKUP_TO_CONTACT_METRIC: {
        "description": (
            "Elapsed simulator time from the first hammer pickup threshold "
            "crossing to the first strict physical hammer-block contact."
        ),
        "oracle_kind": "composite_trusted_tools",
        "supporting_examples": [
            "first_hammer_pickup_step",
            "first_contact_step",
            "time_to_success",
        ],
        "unit": "s",
        "available_schema_keys": [
            "pickup_height_threshold_m",
            "physics_timestep_seconds",
        ],
        "null_semantics": (
            "value is null when pickup or strict physical contact is absent, "
            "or when contact precedes pickup"
        ),
        "reason_contract": {
            "pickup_missing": "pickup_not_observed",
            "contact_missing_after_pickup": "contact_not_observed_after_pickup",
            "contact_before_pickup": "contact_precedes_pickup",
            "valid_measurement": "measured",
        },
        "details_contract": {
            "pickup_detected": "boolean",
            "contact_detected": "boolean",
            "ordering_valid": "boolean",
            "pickup_physics_step": "integer_or_null",
            "contact_physics_step": "integer_or_null",
            "pickup_time_seconds": "number_or_null",
            "contact_time_seconds": "number_or_null",
            "duration_physics_steps": "integer_or_null",
            "pickup_height_threshold_m": "number",
            "reason": "measured_or_missing_reason",
        },
    }
}


def target_definition(
    target_metric: str,
    *,
    reference_tool: str | None = None,
) -> dict[str, Any]:
    """Return the generation/oracle contract for one supported target."""

    if target_metric in COMPOSITE_TARGETS:
        if reference_tool is not None:
            raise KeyError(
                f"composite target {target_metric} must not name a reference_tool"
            )
        return {"metric": target_metric, **COMPOSITE_TARGETS[target_metric]}
    reference = reference_tool or target_metric
    if reference not in TOOL_CATALOG or target_metric != reference:
        raise KeyError(f"unsupported ToolGen target: {target_metric}")
    return {
        "metric": target_metric,
        "description": TOOL_CATALOG[reference]["description"],
        "oracle_kind": "exact_trusted_tool",
        "supporting_examples": [reference],
        "reference_tool": reference,
    }


def _projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": result.get("value"),
        "unit": result.get("unit"),
        "passed": result.get("passed"),
        "evidence_steps": list(result.get("evidence_steps", [])),
        "details": dict(result.get("details", {})),
    }


def _pickup_to_first_contact_time(
    trajectory: TrajectoryView,
) -> dict[str, Any]:
    pickup_result = TOOL_CATALOG["first_hammer_pickup_step"]["function"](
        trajectory
    )
    contact_result = TOOL_CATALOG["first_contact_step"]["function"](
        trajectory
    )
    pickup_step = pickup_result.get("value")
    contact_step = contact_result.get("value")
    pickup_detected = pickup_step is not None
    contact_detected = contact_step is not None
    ordering_valid = bool(
        pickup_detected
        and contact_detected
        and int(contact_step) >= int(pickup_step)
    )
    timestep = float(trajectory.schema["physics_timestep_seconds"])
    pickup_time = pickup_result.get("details", {}).get(
        "simulation_time_seconds"
    )
    contact_time = contact_result.get("details", {}).get(
        "simulation_time_seconds"
    )
    if pickup_detected and pickup_time is None:
        pickup_time = int(pickup_step) * timestep
    if contact_detected and contact_time is None:
        contact_time = int(contact_step) * timestep
    duration_steps = (
        int(contact_step) - int(pickup_step) if ordering_valid else None
    )
    value = (
        float(contact_time) - float(pickup_time)
        if ordering_valid
        else None
    )
    evidence_steps = sorted(
        {
            int(step)
            for step in (pickup_step, contact_step)
            if step is not None
        }
    )
    if not pickup_detected:
        reason = "pickup_not_observed"
    elif not contact_detected:
        reason = "contact_not_observed_after_pickup"
    elif not ordering_valid:
        reason = "contact_precedes_pickup"
    else:
        reason = "measured"
    return {
        "value": value,
        "unit": "s",
        "passed": None,
        "evidence_steps": evidence_steps,
        "details": {
            "pickup_detected": pickup_detected,
            "contact_detected": contact_detected,
            "ordering_valid": ordering_valid,
            "pickup_physics_step": (
                int(pickup_step) if pickup_detected else None
            ),
            "contact_physics_step": (
                int(contact_step) if contact_detected else None
            ),
            "pickup_time_seconds": (
                float(pickup_time) if pickup_detected else None
            ),
            "contact_time_seconds": (
                float(contact_time) if contact_detected else None
            ),
            "duration_physics_steps": duration_steps,
            "pickup_height_threshold_m": float(
                trajectory.schema.get("pickup_height_threshold_m", 0.03)
            ),
            "reason": reason,
        },
    }


def evaluate_target_oracle(
    target_metric: str,
    trajectory: TrajectoryView,
    *,
    reference_tool: str | None = None,
) -> dict[str, Any]:
    """Evaluate an exact Trusted Tool or a deterministic composition oracle."""

    definition = target_definition(
        target_metric,
        reference_tool=reference_tool,
    )
    if definition["oracle_kind"] == "exact_trusted_tool":
        return _projection(
            TOOL_CATALOG[definition["reference_tool"]]["function"](trajectory)
        )
    if target_metric == PICKUP_TO_CONTACT_METRIC:
        return _pickup_to_first_contact_time(trajectory)
    raise KeyError(f"no oracle evaluator for target: {target_metric}")
