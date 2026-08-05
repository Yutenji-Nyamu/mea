"""Independent-oracle availability binding for open Tool requests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .open_request_contract import OpenToolRequestError, _text

_UNSUPPORTED_RESPONSE_KEYS = {
    "schema_version",
    "status",
    "reason_code",
    "reason",
}


def _unsupported_tool_response(
    raw_response: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    requested_need: str,
    provider: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Turn an unavailable derived oracle into an explicit method result."""

    oracle = context["artifact_context"]["oracle_broker"][
        "derived_observable"
    ]
    explicitly_unsupported = raw_response.get("status") == "unsupported"
    metric_spec = raw_response.get("metric_spec")
    requests_derived = (
        isinstance(metric_spec, Mapping)
        and metric_spec.get("operation") == "derived_observable"
    )
    if not explicitly_unsupported and not requests_derived:
        return None
    if oracle["status"] == "available":
        if explicitly_unsupported:
            raise OpenToolRequestError(
                "provider reported a missing derived oracle although the "
                "ToolArtifactContext supplies one"
            )
        return None
    if explicitly_unsupported:
        if set(raw_response) != _UNSUPPORTED_RESPONSE_KEYS:
            raise OpenToolRequestError(
                "unsupported Tool response fields must be exactly "
                f"{sorted(_UNSUPPORTED_RESPONSE_KEYS)}"
            )
        if raw_response.get("schema_version") != 1:
            raise OpenToolRequestError(
                "unsupported Tool response schema_version must be 1"
            )
        if (
            raw_response.get("reason_code")
            != "independent_oracle_broker_unavailable"
        ):
            raise OpenToolRequestError(
                "unsupported Tool response reason_code is invalid"
            )
        _text(raw_response.get("reason"), "unsupported reason")
    return {
        "schema_version": 1,
        "status": "unsupported",
        "artifact_kind": "rule_tool",
        "source": "provider_query_induced_tool_request",
        "reason_code": oracle["reason_code"],
        "reason": oracle["reason"],
        "requested_need": _text(requested_need, "requested_need"),
        "tool_request": None,
        "context": deepcopy(dict(context)),
        "provider": deepcopy(dict(provider)),
    }
