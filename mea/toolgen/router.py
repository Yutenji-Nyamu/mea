"""Deterministic routing for route-free Tool requests.

The router never uses fuzzy text similarity to decide whether code is reused or
generated.  A metric identifier must exactly match either the Trusted Tool
catalog or a registered composite target.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from mea.toolkit.tools import TOOL_CATALOG, public_tool_catalog

from .targets import COMPOSITE_TARGETS
from .metric_spec import MetricSpecError, validate_metric_spec


class ToolRouterError(RuntimeError):
    """Raised when a route-free Tool request violates its contract."""


TOOL_REQUEST_KEYS = {
    "schema_version",
    "task_name",
    "metric",
    "question",
}
TOOL_REQUEST_V2_KEYS = TOOL_REQUEST_KEYS | {"metric_spec"}


def validate_tool_request(
    value: Any,
    *,
    expected_metric: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize the small semantic request emitted by planning."""

    if not isinstance(value, dict):
        raise ToolRouterError("tool_request must be a JSON object")
    schema_version = value.get("schema_version")
    expected_keys = (
        TOOL_REQUEST_KEYS
        if schema_version == 1
        else TOOL_REQUEST_V2_KEYS
        if schema_version == 2
        else None
    )
    if expected_keys is None:
        raise ToolRouterError("tool_request.schema_version must be 1 or 2")
    keys = set(value)
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        raise ToolRouterError(
            f"tool_request fields do not match: missing={missing}, extra={extra}"
        )
    task_name = value.get("task_name")
    if not isinstance(task_name, str) or not task_name.strip():
        raise ToolRouterError("tool_request.task_name must be a non-empty string")
    metric = value.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise ToolRouterError("tool_request.metric must be a non-empty string")
    if expected_metric is not None and metric.strip() != expected_metric:
        raise ToolRouterError(
            f"tool_request.metric must be the expected {expected_metric}"
        )
    question = value.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ToolRouterError("tool_request.question must be a non-empty string")
    result = {
        "schema_version": schema_version,
        "task_name": task_name.strip(),
        "metric": metric.strip(),
        "question": question.strip(),
    }
    if schema_version == 2:
        try:
            result["metric_spec"] = validate_metric_spec(value.get("metric_spec"))
        except MetricSpecError as exc:
            raise ToolRouterError(str(exc)) from exc
    return result


def _with_snapshot_hash(snapshot: dict[str, Any]) -> dict[str, Any]:
    unhashed = dict(snapshot)
    unhashed.pop("snapshot_sha256", None)
    encoded = json.dumps(
        unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    snapshot["snapshot_sha256"] = hashlib.sha256(encoded).hexdigest()
    return snapshot


def catalog_snapshot() -> dict[str, Any]:
    """Return the compact executable registries used for one route decision."""

    composite_targets = [
        {
            "metric": metric,
            "description": target.get("description"),
            "oracle_kind": target.get("oracle_kind"),
            "supporting_examples": list(target.get("supporting_examples", [])),
        }
        for metric, target in sorted(COMPOSITE_TARGETS.items())
    ]
    snapshot = {
        "schema_version": 1,
        "matching_policy": "strict_exact_metric_id",
        "trusted_tools": sorted(
            public_tool_catalog(), key=lambda item: item["name"]
        ),
        "composite_targets": composite_targets,
        "typed_metric_spec": {
            "schema_version": 1,
            "operations": [
                "event_count",
                "minimum_distance",
                "time_between_events",
            ],
            "execution": "compile_validate_register",
        },
    }
    return _with_snapshot_hash(snapshot)


def route_tool_request(
    value: Any,
    *,
    run_local_registration: dict[str, Any] | None = None,
    reviewed_registration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an exact Tool request to reuse, force-codegen, or unsupported."""

    request = validate_tool_request(value)
    snapshot = catalog_snapshot()
    metric = request["metric"]
    task_name = request["task_name"]
    trusted_entry = TOOL_CATALOG.get(metric)
    supported_task_names = (
        set(trusted_entry.get("supported_task_names", []))
        if trusted_entry is not None
        else set()
    )
    trusted_task_supported = trusted_entry is not None and (
        "*" in supported_task_names or task_name in supported_task_names
    )
    composite_entry = COMPOSITE_TARGETS.get(metric)
    composite_task_supported = bool(
        composite_entry
        and task_name in set(composite_entry.get("supported_task_names", []))
    )
    metric_spec = request.get("metric_spec")
    if metric_spec is not None and (
        metric in TOOL_CATALOG or metric in COMPOSITE_TARGETS
    ):
        raise ToolRouterError(
            "typed MetricSpec metric ids cannot override a registered metric"
        )

    if run_local_registration is not None:
        if (
            run_local_registration.get("scope") != "run_local"
            or run_local_registration.get("status") != "validated"
            or run_local_registration.get("target_metric") != metric
        ):
            raise ToolRouterError("invalid run-local registration match")
        snapshot["run_local_match"] = deepcopy(run_local_registration)
        _with_snapshot_hash(snapshot)

    if reviewed_registration is not None:
        if (
            reviewed_registration.get("scope") != "reviewed_persistent"
            or reviewed_registration.get("status") != "approved"
            or reviewed_registration.get("target_metric") != metric
            or reviewed_registration.get("task_name") != task_name
        ):
            raise ToolRouterError("invalid reviewed persistent registration match")
        snapshot["reviewed_match"] = deepcopy(reviewed_registration)
        _with_snapshot_hash(snapshot)

    if metric_spec is not None:
        status = "resolved"
        route = "typed_metric_spec_compile"
        registry = "typed_metric_spec_v1"
        reference_tool = None
        reason = "validated typed MetricSpec can be compiled and differentially gated"
    elif trusted_task_supported:
        status = "resolved"
        route = "reuse"
        registry = "trusted_tool_catalog"
        reference_tool = metric
        reason = "exact metric identifier matched a Trusted Tool"
    elif (
        composite_task_supported
        and metric in COMPOSITE_TARGETS
        and run_local_registration is not None
    ):
        status = "resolved"
        route = "run_local_reuse"
        registry = "evaluation_local_tool_registry"
        reference_tool = run_local_registration["tool_id"]
        reason = (
            "exact ToolSpec, code, and telemetry schema hashes matched a "
            "validated evaluation-local Tool"
        )
    elif (
        composite_task_supported
        and metric in COMPOSITE_TARGETS
        and reviewed_registration is not None
    ):
        status = "resolved"
        route = "reviewed_persistent_reuse"
        registry = "reviewed_tool_registry"
        reference_tool = reviewed_registration["tool_id"]
        reason = (
            "explicit approval plus exact ToolSpec, code, and telemetry schema "
            "hashes matched a reviewed persistent Tool"
        )
    elif composite_task_supported and metric in COMPOSITE_TARGETS:
        status = "resolved"
        route = "force_codegen"
        registry = "composite_target_registry"
        reference_tool = None
        reason = "exact metric identifier matched a registered composite target"
    else:
        status = "unsupported"
        route = None
        registry = None
        reference_tool = None
        reason = (
            "metric is not compatible with the requested task"
            if metric in TOOL_CATALOG or metric in COMPOSITE_TARGETS
            else "metric did not exactly match an executable registry entry"
        )

    decision = {
        "schema_version": 1,
        "status": status,
        "matching_policy": "strict_exact_metric_id",
        "requested_route": "auto",
        "resolved_route": route,
        "task_name": request["task_name"],
        "metric": metric,
        "exact_match": status == "resolved",
        "matched_registry": registry,
        "reference_tool": reference_tool,
        "provider_required": route == "force_codegen",
        "reason": reason,
        "catalog_snapshot_sha256": snapshot["snapshot_sha256"],
    }
    if run_local_registration is not None and route == "run_local_reuse":
        decision["run_local_registration"] = deepcopy(
            run_local_registration
        )
    if reviewed_registration is not None and route == "reviewed_persistent_reuse":
        decision["reviewed_registration"] = deepcopy(reviewed_registration)
    return {
        "tool_request": request,
        "catalog_snapshot": deepcopy(snapshot),
        "route_decision": decision,
    }
