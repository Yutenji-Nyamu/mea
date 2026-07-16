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


class ToolRouterError(RuntimeError):
    """Raised when a route-free Tool request violates its contract."""


TOOL_REQUEST_KEYS = {
    "schema_version",
    "task_name",
    "metric",
    "question",
}


def validate_tool_request(
    value: Any,
    *,
    expected_metric: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize the small semantic request emitted by planning."""

    if not isinstance(value, dict):
        raise ToolRouterError("tool_request must be a JSON object")
    keys = set(value)
    if keys != TOOL_REQUEST_KEYS:
        missing = sorted(TOOL_REQUEST_KEYS - keys)
        extra = sorted(keys - TOOL_REQUEST_KEYS)
        raise ToolRouterError(
            f"tool_request fields do not match: missing={missing}, extra={extra}"
        )
    if value.get("schema_version") != 1:
        raise ToolRouterError("tool_request.schema_version must be 1")
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
    return {
        "schema_version": 1,
        "task_name": task_name.strip(),
        "metric": metric.strip(),
        "question": question.strip(),
    }


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
    }
    return _with_snapshot_hash(snapshot)


def route_tool_request(
    value: Any,
    *,
    run_local_registration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an exact Tool request to reuse, force-codegen, or unsupported."""

    request = validate_tool_request(value)
    snapshot = catalog_snapshot()
    metric = request["metric"]
    task_supported = request["task_name"] == "beat_block_hammer"

    if run_local_registration is not None:
        if (
            run_local_registration.get("scope") != "run_local"
            or run_local_registration.get("status") != "validated"
            or run_local_registration.get("target_metric") != metric
        ):
            raise ToolRouterError("invalid run-local registration match")
        snapshot["run_local_match"] = deepcopy(run_local_registration)
        _with_snapshot_hash(snapshot)

    if task_supported and metric in TOOL_CATALOG:
        status = "resolved"
        route = "reuse"
        registry = "trusted_tool_catalog"
        reference_tool = metric
        reason = "exact metric identifier matched a Trusted Tool"
    elif (
        task_supported
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
    elif task_supported and metric in COMPOSITE_TARGETS:
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
            "task is unsupported"
            if not task_supported
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
    return {
        "tool_request": request,
        "catalog_snapshot": deepcopy(snapshot),
        "route_decision": decision,
    }
