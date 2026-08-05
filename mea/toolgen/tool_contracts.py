"""Trusted Tool requests and executable ToolSpec contracts."""

from __future__ import annotations

from typing import Any

from mea.toolkit.tools import TOOL_CATALOG

from .targets import (
    BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
    COMPOSITE_TARGETS,
    PICKUP_TO_CONTACT_METRIC,
    target_definition,
)

class ToolOrchestrationError(RuntimeError):
    """Raised when a ToolSpec or its runtime inputs violate the contract."""


CONTACT_METRIC = "hammer_block_contact_ever"
CONTACT_QUESTION = "钃濊壊鏂瑰潡鍦烘櫙涓紝閿ゅ瓙鏄惁涓庢柟鍧楀彂鐢熻繃涓ユ牸鐗╃悊鎺ヨЕ锛?
CONTACT_REQUIRED_SIGNALS = [
    "hammer_block_contact_intervals",
    "physics_step_index",
]
CONTACT_OUTPUT_CONTRACT = {
    "value_type": "boolean",
    "unit": None,
    "passed_rule": "equals_value",
    "evidence_rule": "first_physical_contact_physics_step_or_empty",
}
CONTACT_VALIDATION_REQUIREMENTS = {
    "force_codegen": {
        "min_episodes": 2,
        "distinct_reference_values": True,
        "required_reference_values": [False, True],
    },
    "reuse": {
        "min_episodes": 1,
        "distinct_reference_values": False,
        "required_reference_values": [],
    },
}
PICKUP_TO_CONTACT_QUESTION = (
    "钃濊壊鏂瑰潡鍦烘櫙涓紝浠庨敜瀛愰娆℃姮鍗囪揪鍒?pickup 闃堝€煎埌棣栨涓ユ牸鐗╃悊鎺ヨЕ鏂瑰潡锛?
    "缁忚繃澶氬皯绉掞紵"
)
PICKUP_TO_CONTACT_REQUIRED_SIGNALS = [
    "semantic_trace.hammer_position",
    "semantic_trace.physics_step",
    "semantic_trace.simulation_time_seconds",
    "events.hammer_block_contact_intervals",
    "schema.pickup_height_threshold_m",
]
PICKUP_TO_CONTACT_OUTPUT_CONTRACT = {
    "value_type": "number_or_null",
    "unit": "s",
    "passed_rule": "always_null",
    "evidence_rule": "pickup_and_contact_physics_steps_or_available",
    "null_rule": "missing_pickup_or_contact_or_invalid_order",
}
PICKUP_TO_CONTACT_VALIDATION_REQUIREMENTS = {
    "min_episodes": 2,
    "distinct_reference_values": True,
    "required_reference_values": [],
}
BELL_ACTIVE_TCP_MIN_XY_ERROR_QUESTION = (
    "What was the minimum XY distance between the official active-arm TCP "
    "and the bell contact point during this rollout?"
)
TOOL_SPEC_KEYS = {
    "schema_version",
    "task_name",
    "metric",
    "question",
    "route",
    "reference_tool",
    "required_signals",
    "output_contract",
    "validation_requirements",
}


def contact_tool_request() -> dict[str, Any]:
    """Return a route-free request for strict hammer-block contact."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": CONTACT_METRIC,
        "question": CONTACT_QUESTION,
    }


def pickup_to_contact_tool_request() -> dict[str, Any]:
    """Return a route-free request for pickup-to-contact duration."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": PICKUP_TO_CONTACT_METRIC,
        "question": PICKUP_TO_CONTACT_QUESTION,
    }


def bell_active_tcp_min_xy_error_tool_request() -> dict[str, Any]:
    """Request the object-position diagnostic used for both bell sides."""

    return {
        "schema_version": 1,
        "task_name": "click_bell",
        "metric": BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
        "question": BELL_ACTIVE_TCP_MIN_XY_ERROR_QUESTION,
    }


def official_success_tool_request(task_name: str) -> dict[str, Any]:
    """Return a route-free request for one schema-backed official outcome."""

    if not isinstance(task_name, str) or not task_name.strip():
        raise ToolOrchestrationError("task_name must be a non-empty string")
    return {
        "schema_version": 1,
        "task_name": task_name.strip(),
        "metric": "official_check_success",
        "question": "Did the rollout satisfy the official RoboTwin success check?",
    }


def bbh_distractor_success_tool_request() -> dict[str, Any]:
    """Request the outcome from the validated provider-written BBH checker."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": "bbh_target_without_distractor_success",
        "question": (
            "Did the rollout hit the target block while avoiding every "
            "contact with the look-alike distractor?"
        ),
    }


def click_bell_distractor_success_tool_request() -> dict[str, Any]:
    """Request the validated provider-written ClickBell checker outcome."""

    return {
        "schema_version": 1,
        "task_name": "click_bell",
        "metric": "click_target_without_distractor_success",
        "question": (
            "Did the rollout press the intended bell with the correct arm "
            "without any latched contact with the look-alike bell?"
        ),
    }


def hammer_left_camera_contact_count_tool_request() -> dict[str, Any]:
    """Request the bounded BBH unintended-contact proxy from Trusted Tools."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": "hammer_left_camera_contact_count",
        "question": (
            "How many physical hammer-left_camera contact intervals occurred?"
        ),
    }


def time_to_success_tool_request(task_name: str) -> dict[str, Any]:
    """Request the trusted first-success timestamp for an official task.

    Aggregate Toolkit computes the cross-seed mean and dispersion.  Keeping
    this request route-free lets the Tool router prove that the implementation
    came from the audited Trusted Tool catalog rather than model-generated
    measurement code.
    """

    if not isinstance(task_name, str) or not task_name.strip():
        raise ToolOrchestrationError("task_name must be a non-empty string")
    return {
        "schema_version": 1,
        "task_name": task_name.strip(),
        "metric": "time_to_success",
        "question": "When did the rollout first satisfy the official success check?",
    }


def contact_tool_spec(route: str) -> dict[str, Any]:
    """Return the exact first-version contact ToolSpec for a route."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": CONTACT_METRIC,
        "question": CONTACT_QUESTION,
        "route": route,
        "reference_tool": CONTACT_METRIC,
        "required_signals": list(CONTACT_REQUIRED_SIGNALS),
        "output_contract": dict(CONTACT_OUTPUT_CONTRACT),
        "validation_requirements": {
            **CONTACT_VALIDATION_REQUIREMENTS[route],
            "required_reference_values": list(
                CONTACT_VALIDATION_REQUIREMENTS[route][
                    "required_reference_values"
                ]
            ),
        },
    }


def pickup_to_contact_tool_spec(route: str = "force_codegen") -> dict[str, Any]:
    """Return the first genuinely new, composition-validated ToolSpec."""

    if route != "force_codegen":
        raise ToolOrchestrationError(
            "pickup_to_first_contact_time 灏氭湭杩涘叆 Trusted catalog锛屽彧鍏佽 force_codegen"
        )
    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": PICKUP_TO_CONTACT_METRIC,
        "question": PICKUP_TO_CONTACT_QUESTION,
        "route": route,
        "reference_tool": None,
        "required_signals": list(PICKUP_TO_CONTACT_REQUIRED_SIGNALS),
        "output_contract": dict(PICKUP_TO_CONTACT_OUTPUT_CONTRACT),
        "validation_requirements": {
            **PICKUP_TO_CONTACT_VALIDATION_REQUIREMENTS,
            "required_reference_values": [],
        },
    }


def _composite_tool_spec(
    metric: str,
    question: str,
    task_name: str,
    *,
    route: str = "force_codegen",
) -> dict[str, Any]:
    if route != "force_codegen" or metric not in COMPOSITE_TARGETS:
        raise ToolOrchestrationError("composite targets require force_codegen")
    definition = target_definition(metric)
    supported = set(definition.get("supported_task_names", []))
    if task_name not in supported:
        raise ToolOrchestrationError(
            f"ToolSpec metric {metric!r} is incompatible with task {task_name!r}"
        )
    if metric == PICKUP_TO_CONTACT_METRIC:
        spec = pickup_to_contact_tool_spec(route)
        spec["question"] = question
        return spec
    return {
        "schema_version": 1,
        "task_name": task_name,
        "metric": metric,
        "question": question,
        "route": route,
        "reference_tool": None,
        "required_signals": list(definition.get("required_signals", [])),
        "output_contract": dict(definition.get("output_contract", {})),
        "validation_requirements": {
            **definition.get("validation_requirements", {}),
            "required_reference_values": list(
                definition.get("validation_requirements", {}).get(
                    "required_reference_values", []
                )
            ),
        },
    }


def _generic_trusted_tool_spec(
    metric: str,
    question: str,
    task_name: str,
) -> dict[str, Any]:
    """Build the internal routeful envelope for any exact catalog match."""

    if metric not in TOOL_CATALOG:
        raise ToolOrchestrationError(f"unknown Trusted Tool metric: {metric}")
    return {
        "schema_version": 1,
        "task_name": task_name,
        "metric": metric,
        "question": question,
        "route": "reuse",
        "reference_tool": metric,
        "required_signals": [],
        "output_contract": {"source": "trusted_tool_catalog"},
        "validation_requirements": {
            "min_episodes": 1,
            "distinct_reference_values": False,
            "required_reference_values": [],
        },
    }


def validate_tool_spec(
    value: Any,
    *,
    expected_route: str | None = None,
    expected_metric: str | None = None,
) -> dict[str, Any]:
    """Validate the intentionally narrow ToolSpec emitted by the Plan Agent."""

    if not isinstance(value, dict):
        raise ToolOrchestrationError("ToolSpec 蹇呴』鏄?JSON object")
    keys = set(value)
    if keys != TOOL_SPEC_KEYS:
        missing = sorted(TOOL_SPEC_KEYS - keys)
        extra = sorted(keys - TOOL_SPEC_KEYS)
        raise ToolOrchestrationError(
            f"ToolSpec fields 涓嶅尮閰嶏紝missing={missing}, extra={extra}"
        )
    route = value.get("route")
    if route not in {"reuse", "force_codegen"}:
        raise ToolOrchestrationError("ToolSpec.route 鍙厑璁?reuse 鎴?force_codegen")
    if expected_route is not None and route != expected_route:
        raise ToolOrchestrationError(
            f"ToolSpec.route 蹇呴』鏄湰杞害瀹氱殑 {expected_route}"
        )
    metric = value.get("metric")
    if expected_metric is not None and metric != expected_metric:
        raise ToolOrchestrationError(
            f"ToolSpec.metric 蹇呴』鏄湰杞害瀹氱殑 {expected_metric}"
        )
    if metric == CONTACT_METRIC:
        expected = contact_tool_spec(route)
    elif metric in COMPOSITE_TARGETS:
        question = value.get("question")
        task_name = value.get("task_name")
        if not isinstance(question, str) or not question.strip():
            raise ToolOrchestrationError("ToolSpec.question must be non-empty")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ToolOrchestrationError("ToolSpec.task_name must be non-empty")
        expected = _composite_tool_spec(
            metric, question.strip(), task_name.strip(), route=route
        )
    elif route == "reuse" and metric in TOOL_CATALOG:
        question = value.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolOrchestrationError("ToolSpec.question must be non-empty")
        task_name = value.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ToolOrchestrationError("ToolSpec.task_name must be non-empty")
        supported = set(
            TOOL_CATALOG[metric].get("supported_task_names", [])
        )
        if "*" not in supported and task_name not in supported:
            raise ToolOrchestrationError(
                f"ToolSpec metric {metric!r} is incompatible with task {task_name!r}"
            )
        expected = _generic_trusted_tool_spec(
            metric, question.strip(), task_name.strip()
        )
    else:
        raise ToolOrchestrationError(f"褰撳墠鏈敞鍐?ToolSpec metric: {metric}")
    question = value.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ToolOrchestrationError("ToolSpec.question must be non-empty")
    expected["question"] = question.strip()
    for field in TOOL_SPEC_KEYS - {"route", "question"}:
        if value.get(field) != expected[field]:
            raise ToolOrchestrationError(
                f"ToolSpec.{field} 蹇呴』绛変簬宸查獙璇佺殑 {metric} contract"
            )
    return expected
