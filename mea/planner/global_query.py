"""Bounded global routing from one open query to an existing MEA planner."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping

from .catalog import catalog_task, validate_act_catalog
from .click_bell import CLICK_BELL_ADAPTIVE_ASPECTS
from .prototype import EXPECTED_POLICY, MAX_ROUNDS, SUB_ASPECT_CATALOG


class GlobalRouteError(ValueError):
    """Raised when a global route proposal exceeds the trusted ACT catalog."""


_ROUTE_KEYS = {
    "schema_version",
    "decision",
    "task_name",
    "task_profile",
    "evaluation_goal",
    "requested_aspect_ids",
    "first_aspect_id",
    "unsupported_aspect_ids",
}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GlobalRouteError(f"{field} must be a non-empty string")
    return value.strip()


def _unique_text_list(value: Any, field: str, *, allow_empty: bool) -> list[str]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise GlobalRouteError(f"{field} must be a {qualifier} string list")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise GlobalRouteError(f"{field} must not contain duplicates")
    return normalized


def validate_route_selection(
    value: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a model proposal without accepting executable parameters."""

    trusted_catalog = validate_act_catalog(catalog)
    if not isinstance(value, Mapping) or set(value) != _ROUTE_KEYS:
        raise GlobalRouteError(
            f"GlobalRouteSelection fields must be exactly {sorted(_ROUTE_KEYS)}"
        )
    proposal = deepcopy(dict(value))
    if proposal.get("schema_version") != 1:
        raise GlobalRouteError("GlobalRouteSelection schema_version must be 1")
    decision = proposal.get("decision")
    if decision not in {"route", "unsupported"}:
        raise GlobalRouteError("decision must be route or unsupported")
    proposal["evaluation_goal"] = _require_text(
        proposal.get("evaluation_goal"), "evaluation_goal"
    )
    requested = _unique_text_list(
        proposal.get("requested_aspect_ids"),
        "requested_aspect_ids",
        allow_empty=decision == "unsupported",
    )
    unsupported = _unique_text_list(
        proposal.get("unsupported_aspect_ids"),
        "unsupported_aspect_ids",
        allow_empty=decision == "route",
    )
    proposal["requested_aspect_ids"] = requested
    proposal["unsupported_aspect_ids"] = unsupported

    supported_aspects = {
        aspect["aspect_id"]
        for task in trusted_catalog["tasks"]
        for aspect in task["aspects"]
    }
    if decision == "unsupported":
        if any(
            proposal.get(field) is not None
            for field in ("task_name", "task_profile", "first_aspect_id")
        ):
            raise GlobalRouteError(
                "unsupported decision requires null task, profile, and first aspect"
            )
        if requested or not unsupported:
            raise GlobalRouteError(
                "unsupported decision requires no requested aspects and at least one gap"
            )
        false_gaps = sorted(set(unsupported) & supported_aspects)
        if false_gaps:
            raise GlobalRouteError(
                f"supported aspects cannot be declared unsupported: {false_gaps}"
            )
        return proposal

    if unsupported:
        raise GlobalRouteError("routed selection cannot contain unsupported aspects")
    task_name = _require_text(proposal.get("task_name"), "task_name")
    try:
        task = catalog_task(trusted_catalog, task_name)
    except ValueError as exc:
        raise GlobalRouteError(str(exc)) from exc
    profile = _require_text(proposal.get("task_profile"), "task_profile")
    if profile != task["task_profile"]:
        raise GlobalRouteError(
            f"task_profile {profile!r} is not trusted for {task_name!r}"
        )
    available_aspects = {aspect["aspect_id"] for aspect in task["aspects"]}
    unknown = sorted(set(requested) - available_aspects)
    if unknown:
        raise GlobalRouteError(f"unsupported routed aspects for {task_name}: {unknown}")
    if len(requested) > int(task["max_rounds"]):
        raise GlobalRouteError("requested aspects exceed the trusted round budget")
    first_aspect = _require_text(proposal.get("first_aspect_id"), "first_aspect_id")
    if first_aspect not in requested:
        raise GlobalRouteError("first_aspect_id must be one of requested_aspect_ids")
    proposal.update(
        {
            "task_name": task_name,
            "task_profile": profile,
            "first_aspect_id": first_aspect,
        }
    )
    return proposal


def _extract_json_response(response: str) -> dict[str, Any]:
    source = str(response).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", source, re.DOTALL)
    if fenced:
        source = fenced.group(1).strip()
    start = source.find("{")
    end = source.rfind("}")
    if start < 0 or end < start:
        raise GlobalRouteError("global route response contains no JSON object")
    try:
        value = json.loads(source[start : end + 1])
    except json.JSONDecodeError as exc:
        raise GlobalRouteError("global route response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise GlobalRouteError("global route response must be a JSON object")
    return value


def _compact_history(history_context: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in (history_context or [])[:3]:
        if not isinstance(item, dict):
            continue
        planning = item.get("planning") if isinstance(item.get("planning"), dict) else {}
        compatibility = (
            item.get("compatibility")
            if isinstance(item.get("compatibility"), dict)
            else {}
        )
        compact.append(
            {
                "evaluation_id": item.get("evaluation_id"),
                "similarity": item.get("similarity"),
                "user_request": item.get("user_request"),
                "task_name": item.get("task_name"),
                "requested_template_ids": planning.get("requested_template_ids", []),
                "first_template_id": planning.get("first_template_id"),
                "planning_state": planning.get("planning_state"),
                "same_policy": compatibility.get("same_policy"),
                "same_checkpoint": compatibility.get("same_checkpoint"),
            }
        )
    return compact


def build_global_route_prompt(
    user_request: str,
    catalog: Mapping[str, Any],
    history_context: list[dict[str, Any]] | None = None,
) -> str:
    """Build a catalog-only prompt with compact completed-plan provenance."""

    request = _require_text(user_request, "user_request")
    trusted_catalog = validate_act_catalog(catalog)
    if trusted_catalog["tasks"]:
        example_task = trusted_catalog["tasks"][0]
        example_aspect = example_task["aspects"][0]["aspect_id"]
        example = {
            "schema_version": 1,
            "decision": "route",
            "task_name": example_task["task_name"],
            "task_profile": example_task["task_profile"],
            "evaluation_goal": "evaluate the requested supported capability",
            "requested_aspect_ids": [example_aspect],
            "first_aspect_id": example_aspect,
            "unsupported_aspect_ids": [],
        }
    else:
        example = {
            "schema_version": 1,
            "decision": "unsupported",
            "task_name": None,
            "task_profile": None,
            "evaluation_goal": "report that no ACT-ready route is available",
            "requested_aspect_ids": [],
            "first_aspect_id": None,
            "unsupported_aspect_ids": ["capability.no_act_ready_task"],
        }
    return f"""You are the bounded global Plan Agent for an ACT-only MEA reproduction.
Select one ACT-ready task, its single trusted profile, the query-relevant
aspects, and the first aspect.  Use only the catalog below.  Never output paths,
Python, modules, checkpoints, seeds, gates, tools, variants, or execution fields.
If the query requires any capability outside the catalog, return
decision=\"unsupported\", null task/profile/first_aspect, no requested aspects,
and list the unsupported semantic aspect ids.  Historical plans are planning
priors only and never current-run execution evidence.

USER QUERY:
{request}

TRUSTED ACT EVALUATION CATALOG:
{json.dumps(trusted_catalog, ensure_ascii=False, indent=2)}

SIMILAR COMPLETED PLAN HISTORY:
{json.dumps(_compact_history(history_context), ensure_ascii=False, indent=2)}

Return strict JSON with exactly this shape:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


class GlobalQueryRouter:
    """Ask a model for one semantic route and enforce the trusted catalog."""

    def __init__(self, provider: Any, *, model: str, catalog: Mapping[str, Any]):
        self.provider = provider
        self.model = _require_text(model, "model")
        self.catalog = validate_act_catalog(catalog)
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_trace: dict[str, Any] | None = None

    def route(
        self,
        user_request: str,
        *,
        history_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prompt = build_global_route_prompt(
            user_request, self.catalog, history_context=history_context
        )
        self.last_prompt = prompt
        self.last_responses = []
        errors: list[str] = []
        selection = None
        attempt_count = 0
        for _attempt in range(2):
            attempt_count += 1
            attempt_prompt = prompt
            if errors:
                attempt_prompt += (
                    "\nPREVIOUS VALIDATION ERROR:\n"
                    + errors[-1]
                    + "\nReturn a complete corrected JSON object.\n"
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict GlobalRouteSelection JSON.",
                    max_tokens=700,
                    temperature=0.0,
                )
                self.last_responses.append(str(response))
                selection = validate_route_selection(
                    _extract_json_response(str(response)), self.catalog
                )
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if selection is None:
            self.last_trace = {
                "schema_version": 1,
                "catalog_sha256": self.catalog["catalog_sha256"],
                "provider_called": True,
                "attempt_count": attempt_count,
                "validation_errors": errors,
                "provider_metadata": dict(
                    getattr(self.provider, "last_metadata", {})
                ),
            }
            raise GlobalRouteError(f"global route failed twice: {errors}")
        resolved = None
        if selection["decision"] == "route":
            task = catalog_task(self.catalog, selection["task_name"])
            selected_aspects = set(selection["requested_aspect_ids"])
            resolved = {
                "task_name": task["task_name"],
                "task_family": task["task_family"],
                "task_profile": task["task_profile"],
                "planner_kind": task["planner_kind"],
                "checkpoint": deepcopy(task["checkpoint"]),
                "aspects": [
                    deepcopy(aspect)
                    for aspect in task["aspects"]
                    if aspect["aspect_id"] in selected_aspects
                ],
            }
        result = {
            "schema_version": 1,
            "selection": selection,
            "resolved": resolved,
            "catalog_sha256": self.catalog["catalog_sha256"],
            "provider_called": True,
            "attempt_count": attempt_count,
            "validation_errors": errors,
            "provider_metadata": dict(getattr(self.provider, "last_metadata", {})),
        }
        self.last_trace = deepcopy(result)
        return result


def _validated_routed_task(
    selection: Mapping[str, Any], catalog: Mapping[str, Any], task_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = validate_route_selection(selection, catalog)
    if normalized["decision"] != "route" or normalized["task_name"] != task_name:
        raise GlobalRouteError(f"selection is not a routed {task_name} evaluation")
    return normalized, catalog_task(catalog, task_name)


def route_to_click_proposal(
    selection: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Translate a validated route to ClickBellEvaluationProposal fields."""

    route, _task = _validated_routed_task(selection, catalog, "click_bell")
    unknown = sorted(
        set(route["requested_aspect_ids"]) - set(CLICK_BELL_ADAPTIVE_ASPECTS)
    )
    if unknown:
        raise GlobalRouteError(f"click_bell proposal has unknown aspects: {unknown}")
    return {
        "schema_version": 1,
        "task_name": "click_bell",
        "evaluation_goal": route["evaluation_goal"],
        "requested_aspect_ids": list(route["requested_aspect_ids"]),
        "first_aspect_id": route["first_aspect_id"],
    }


def route_to_bbh_proposal(
    selection: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Translate a validated route to the existing BBH EvaluationProposal."""

    route, task = _validated_routed_task(selection, catalog, "beat_block_hammer")
    aspect_map = {
        aspect["aspect_id"]: list(aspect["template_ids"])
        for aspect in task["aspects"]
    }
    requested_templates = [
        template_id
        for aspect_id in route["requested_aspect_ids"]
        for template_id in aspect_map[aspect_id]
    ]
    first_template = aspect_map[route["first_aspect_id"]][0]
    if any(template_id not in SUB_ASPECT_CATALOG for template_id in requested_templates):
        raise GlobalRouteError("BBH catalog route no longer matches SUB_ASPECT_CATALOG")
    return {
        "schema_version": 5,
        "task_name": "beat_block_hammer",
        "policy": deepcopy(EXPECTED_POLICY),
        "evaluation_goal": route["evaluation_goal"],
        "requested_template_ids": requested_templates,
        "first_template_id": first_template,
        "max_rounds": MAX_ROUNDS,
    }


def route_to_planner_proposal(
    selection: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Dispatch a validated route to one existing planner proposal schema."""

    route = validate_route_selection(selection, catalog)
    if route["decision"] != "route":
        raise GlobalRouteError("unsupported selection has no executable planner proposal")
    task = catalog_task(catalog, route["task_name"])
    proposal = (
        route_to_click_proposal(route, catalog)
        if route["task_name"] == "click_bell"
        else route_to_bbh_proposal(route, catalog)
    )
    return {
        "schema_version": 1,
        "task_name": route["task_name"],
        "task_profile": route["task_profile"],
        "planner_kind": task["planner_kind"],
        "proposal": proposal,
    }


__all__ = [
    "GlobalQueryRouter",
    "GlobalRouteError",
    "build_global_route_prompt",
    "route_to_bbh_proposal",
    "route_to_click_proposal",
    "route_to_planner_proposal",
    "validate_route_selection",
]
