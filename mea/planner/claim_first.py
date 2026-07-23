"""Claim-first open-Query planning without a predeclared aspect itinerary.

The existing adaptive planner deliberately selects from trusted executable
templates.  That is useful for production routing, but it cannot demonstrate
the paper's stronger claim that the Plan Agent discovers a small set of
sub-aspects while evaluation is in progress.  This module owns that earlier,
semantic planning step.

Only three kinds of information enter the model prompt:

* the original Query;
* a projection of policy/simulator generation capabilities that removes
  aspect ids, template ids, and navigation order;
* evidence from rounds that have already completed.

The resulting semantic proposal is intentionally not executable.  TaskGen and
ToolGen must subsequently resolve its requested perturbation and tool need.
There is no deterministic aspect fallback: provider failure is surfaced rather
than silently restoring a scripted route.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.taskgen import extract_json_response


class ClaimFirstPlanError(ValueError):
    """Raised when open-Query inputs or a semantic proposal are invalid."""


_CAPABILITY_KEYS = {
    "schema_version",
    "policy_card",
    "simulator_card",
    "generation_card",
}
_GENERATION_KEYS = {"taskgen_operations", "toolgen"}
_TASKGEN_OPERATION_KEYS = {
    "operation",
    "controlled_axis",
    "generation_mode",
    "allowed_change_roots",
}
_TOOLGEN_KEYS = {
    "retrieve_first",
    "can_generate_rule_metric",
    "can_generate_vqa_question",
}
_EVIDENCE_KEYS = {
    "schema_version",
    "round_id",
    "tested_sub_aspect",
    "tested_hypothesis",
    "tested_perturbation",
    "outcome",
    "evidence_summary",
    "limitations",
}
_PROPOSAL_KEYS = {
    "schema_version",
    "action",
    "sub_aspect",
    "hypothesis",
    "requested_perturbation",
    "task_need",
    "tool_need",
    "rationale",
}
_PERTURBATION_KEYS = {"description", "controlled_changes", "preserve"}
_NEED_KEYS = {"required", "description"}
_TOOL_NEED_KEYS = _NEED_KEYS | {"reuse_first"}
_OUTCOMES = {"success", "failure", "ambiguous"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_CAPABILITY_KEYS = {
    "aspect_id",
    "available_aspect_ids",
    "template_id",
    "template_ids",
    "fallback_step",
    "navigation_options",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimFirstPlanError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _text_list(value: Any, field: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise ClaimFirstPlanError(f"{field} must be a list")
    result = [_text(item, f"{field}[]") for item in value]
    if not allow_empty and not result:
        raise ClaimFirstPlanError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise ClaimFirstPlanError(f"{field} must not contain duplicates")
    return result


def _assert_no_navigation_keys(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(set(value) & _FORBIDDEN_CAPABILITY_KEYS)
        if forbidden:
            raise ClaimFirstPlanError(
                f"{field} contains predeclared navigation fields: {forbidden}"
            )
        for key, nested in value.items():
            _assert_no_navigation_keys(nested, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_navigation_keys(nested, field=f"{field}[{index}]")


def validate_open_query_capabilities(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a semantic capability projection with no aspect itinerary."""

    if not isinstance(value, Mapping) or set(value) != _CAPABILITY_KEYS:
        raise ClaimFirstPlanError(
            f"OpenQueryCapabilities fields must be exactly "
            f"{sorted(_CAPABILITY_KEYS)}"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != 1:
        raise ClaimFirstPlanError("OpenQueryCapabilities.schema_version must be 1")
    _assert_no_navigation_keys(result, field="OpenQueryCapabilities")

    policy = result.get("policy_card")
    simulator = result.get("simulator_card")
    generation = result.get("generation_card")
    if not isinstance(policy, Mapping) or not policy:
        raise ClaimFirstPlanError("policy_card must be a non-empty object")
    if not isinstance(simulator, Mapping) or not simulator:
        raise ClaimFirstPlanError("simulator_card must be a non-empty object")
    if not isinstance(generation, Mapping) or set(generation) != _GENERATION_KEYS:
        raise ClaimFirstPlanError(
            f"generation_card fields must be exactly {sorted(_GENERATION_KEYS)}"
        )
    operations = generation.get("taskgen_operations")
    if not isinstance(operations, list):
        raise ClaimFirstPlanError("taskgen_operations must be a list")
    normalized_operations: list[dict[str, Any]] = []
    for index, raw in enumerate(operations):
        if not isinstance(raw, Mapping) or set(raw) != _TASKGEN_OPERATION_KEYS:
            raise ClaimFirstPlanError(
                "each taskgen operation must have exactly "
                f"{sorted(_TASKGEN_OPERATION_KEYS)}"
            )
        operation = {
            "operation": _text(raw.get("operation"), f"taskgen_operations[{index}].operation"),
            "controlled_axis": _optional_text(
                raw.get("controlled_axis"),
                f"taskgen_operations[{index}].controlled_axis",
            ),
            "generation_mode": _optional_text(
                raw.get("generation_mode"),
                f"taskgen_operations[{index}].generation_mode",
            ),
            "allowed_change_roots": _text_list(
                raw.get("allowed_change_roots"),
                f"taskgen_operations[{index}].allowed_change_roots",
            ),
        }
        if operation not in normalized_operations:
            normalized_operations.append(operation)
    toolgen = generation.get("toolgen")
    if not isinstance(toolgen, Mapping) or set(toolgen) != _TOOLGEN_KEYS:
        raise ClaimFirstPlanError(
            f"toolgen fields must be exactly {sorted(_TOOLGEN_KEYS)}"
        )
    if any(not isinstance(toolgen.get(key), bool) for key in _TOOLGEN_KEYS):
        raise ClaimFirstPlanError("all toolgen capability flags must be bool")
    result["policy_card"] = deepcopy(dict(policy))
    result["simulator_card"] = deepcopy(dict(simulator))
    result["generation_card"] = {
        "taskgen_operations": normalized_operations,
        "toolgen": deepcopy(dict(toolgen)),
    }
    return result


def project_open_query_capabilities(
    planning_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove catalog navigation from a trusted runtime PlanningContext.

    The caller is expected to validate/build ``planning_context`` through the
    repository-owned context adapter first.  This projection deliberately
    omits adapter aspect/template ids as well as the simulator's redundant
    ``available_aspect_ids`` field.
    """

    if not isinstance(planning_context, Mapping):
        raise ClaimFirstPlanError("PlanningContext must be an object")
    policy = planning_context.get("policy_card")
    simulator = planning_context.get("simulator_card")
    adapter = planning_context.get("adapter_view")
    if not isinstance(policy, Mapping):
        raise ClaimFirstPlanError("PlanningContext.policy_card must be an object")
    if not isinstance(simulator, Mapping):
        raise ClaimFirstPlanError("PlanningContext.simulator_card must be an object")
    if not isinstance(adapter, Mapping):
        raise ClaimFirstPlanError("PlanningContext.adapter_view must be an object")
    templates = adapter.get("templates")
    if not isinstance(templates, list):
        raise ClaimFirstPlanError("PlanningContext adapter templates must be a list")

    operations: list[dict[str, Any]] = []
    for index, template in enumerate(templates):
        if not isinstance(template, Mapping):
            raise ClaimFirstPlanError(
                f"PlanningContext adapter template {index} must be an object"
            )
        item = {
            "operation": template.get("taskgen_operation"),
            "controlled_axis": template.get("controlled_axis"),
            "generation_mode": template.get("generation_mode"),
            "allowed_change_roots": deepcopy(
                template.get("allowed_change_roots")
            ),
        }
        if item not in operations:
            operations.append(item)

    projected_simulator = {
        key: deepcopy(nested)
        for key, nested in simulator.items()
        if key != "available_aspect_ids"
    }
    return validate_open_query_capabilities(
        {
            "schema_version": 1,
            "policy_card": deepcopy(dict(policy)),
            "simulator_card": projected_simulator,
            "generation_card": {
                "taskgen_operations": operations,
                "toolgen": {
                    "retrieve_first": True,
                    "can_generate_rule_metric": True,
                    "can_generate_vqa_question": True,
                },
            },
        }
    )


def validate_open_query_evidence(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the compact evidence-only history shown to the Plan Agent."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ClaimFirstPlanError("evidence_history must be a sequence")
    result: list[dict[str, Any]] = []
    seen_round_ids: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _EVIDENCE_KEYS:
            raise ClaimFirstPlanError(
                f"evidence_history[{index}] fields must be exactly "
                f"{sorted(_EVIDENCE_KEYS)}"
            )
        item = deepcopy(dict(raw))
        if item.get("schema_version") != 1:
            raise ClaimFirstPlanError(
                f"evidence_history[{index}].schema_version must be 1"
            )
        round_id = _text(item.get("round_id"), f"evidence_history[{index}].round_id")
        if round_id in seen_round_ids:
            raise ClaimFirstPlanError(f"duplicate evidence round_id: {round_id!r}")
        seen_round_ids.add(round_id)
        outcome = item.get("outcome")
        if outcome not in _OUTCOMES:
            raise ClaimFirstPlanError(
                f"evidence_history[{index}].outcome must be one of "
                f"{sorted(_OUTCOMES)}"
            )
        item.update(
            {
                "round_id": round_id,
                "tested_sub_aspect": _text(
                    item.get("tested_sub_aspect"),
                    f"evidence_history[{index}].tested_sub_aspect",
                ),
                "tested_hypothesis": _text(
                    item.get("tested_hypothesis"),
                    f"evidence_history[{index}].tested_hypothesis",
                ),
                "tested_perturbation": _text(
                    item.get("tested_perturbation"),
                    f"evidence_history[{index}].tested_perturbation",
                ),
                "outcome": outcome,
                "evidence_summary": _text(
                    item.get("evidence_summary"),
                    f"evidence_history[{index}].evidence_summary",
                ),
                "limitations": _text_list(
                    item.get("limitations"),
                    f"evidence_history[{index}].limitations",
                ),
            }
        )
        result.append(item)
    return result


def _validate_need(
    value: Any,
    *,
    field: str,
    tool: bool,
) -> dict[str, Any]:
    keys = _TOOL_NEED_KEYS if tool else _NEED_KEYS
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ClaimFirstPlanError(f"{field} fields must be exactly {sorted(keys)}")
    required = value.get("required")
    if not isinstance(required, bool):
        raise ClaimFirstPlanError(f"{field}.required must be bool")
    description = _optional_text(value.get("description"), f"{field}.description")
    if required != (description is not None):
        raise ClaimFirstPlanError(
            f"{field}.description must be present exactly when required=true"
        )
    result = {"required": required, "description": description}
    if tool:
        if value.get("reuse_first") is not True:
            raise ClaimFirstPlanError(f"{field}.reuse_first must be true")
        result["reuse_first"] = True
    return result


def validate_open_query_plan_proposal(
    value: Mapping[str, Any],
    *,
    has_evidence: bool,
) -> dict[str, Any]:
    """Validate one semantic next-step decision without catalog enumeration."""

    if not isinstance(value, Mapping) or set(value) != _PROPOSAL_KEYS:
        raise ClaimFirstPlanError(
            f"OpenQueryPlanProposal fields must be exactly "
            f"{sorted(_PROPOSAL_KEYS)}"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != 1:
        raise ClaimFirstPlanError("OpenQueryPlanProposal.schema_version must be 1")
    action = result.get("action")
    if action not in {"continue", "stop"}:
        raise ClaimFirstPlanError("OpenQueryPlanProposal.action must be continue or stop")
    result["hypothesis"] = _text(result.get("hypothesis"), "hypothesis")
    result["rationale"] = _text(result.get("rationale"), "rationale")
    result["task_need"] = _validate_need(
        result.get("task_need"), field="task_need", tool=False
    )
    result["tool_need"] = _validate_need(
        result.get("tool_need"), field="tool_need", tool=True
    )

    if action == "stop":
        if not has_evidence:
            raise ClaimFirstPlanError("stop requires at least one completed evidence round")
        if result.get("sub_aspect") is not None:
            raise ClaimFirstPlanError("stop must set sub_aspect to null")
        if result.get("requested_perturbation") is not None:
            raise ClaimFirstPlanError("stop must set requested_perturbation to null")
        if result["task_need"]["required"] or result["tool_need"]["required"]:
            raise ClaimFirstPlanError("stop cannot request TaskGen or ToolGen work")
        return result

    sub_aspect = _text(result.get("sub_aspect"), "sub_aspect")
    if not _IDENTIFIER.fullmatch(sub_aspect):
        raise ClaimFirstPlanError(
            "sub_aspect must be a semantic identifier using letters, digits, . _ or -"
        )
    perturbation = result.get("requested_perturbation")
    if not isinstance(perturbation, Mapping) or set(perturbation) != _PERTURBATION_KEYS:
        raise ClaimFirstPlanError(
            "requested_perturbation fields must be exactly "
            f"{sorted(_PERTURBATION_KEYS)}"
        )
    result["sub_aspect"] = sub_aspect
    result["requested_perturbation"] = {
        "description": _text(
            perturbation.get("description"), "requested_perturbation.description"
        ),
        "controlled_changes": _text_list(
            perturbation.get("controlled_changes"),
            "requested_perturbation.controlled_changes",
            allow_empty=False,
        ),
        "preserve": _text_list(
            perturbation.get("preserve"), "requested_perturbation.preserve"
        ),
    }
    return result


def open_query_input_digest(
    user_query: str,
    capabilities: Mapping[str, Any],
    evidence_history: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the exact semantic inputs used for one provider decision."""

    payload = {
        "user_query": _text(user_query, "user_query"),
        "capabilities": validate_open_query_capabilities(capabilities),
        "evidence_history": validate_open_query_evidence(evidence_history),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ClaimFirstOpenQueryAgent:
    """Ask a provider to discover the next sub-aspect from evidence."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = _text(model, "model")
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _prompt(
        user_query: str,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
    ) -> str:
        example = {
            "schema_version": 1,
            "action": "continue",
            "sub_aspect": "semantic.sub_aspect_discovered_now",
            "hypothesis": "A falsifiable statement this one round will test.",
            "requested_perturbation": {
                "description": "One bounded, diagnostic perturbation.",
                "controlled_changes": ["the single factor intentionally changed"],
                "preserve": ["task identity", "policy checkpoint"],
            },
            "task_need": {
                "required": True,
                "description": "Scene or success-check work TaskGen must provide.",
            },
            "tool_need": {
                "required": True,
                "description": "Observable or metric ToolGen must retrieve or generate.",
                "reuse_first": True,
            },
            "rationale": "Why this is the most informative next test for the Query.",
        }
        return f"""You are the claim-first Plan Agent in ManipEvalAgent.
Discover a small set of evaluation sub-aspects online.  There is no candidate
aspect list, template itinerary, success-then-switch script, or fallback route.
Choose only the single most informative next experiment for the original
Query, using the policy/simulator capabilities and completed evidence below.

For action=continue, invent a precise semantic sub_aspect identifier and one
falsifiable hypothesis.  Request a bounded perturbation supported by the
capability cards.  State whether TaskGen must create/alter the task and whether
ToolGen must retrieve or generate an observable.  A new tool need may be named
even when it is not in an existing metric list.  Avoid repeating a tested
perturbation unless ambiguous evidence requires a more observable version.

Use success to probe the most consequential remaining uncertainty; use failure
to discriminate a causal failure hypothesis; use ambiguous evidence to improve
observability or isolate the confound.  Stop only when the completed evidence
already answers the original Query.  For action=stop set sub_aspect and
requested_perturbation to null, both needs to required=false/description=null,
and express the evidence-supported conclusion in hypothesis.

ORIGINAL QUERY:
{user_query}

POLICY AND SIMULATOR CAPABILITIES:
{json.dumps(capabilities, ensure_ascii=False, indent=2)}

COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):
{json.dumps(evidence_history, ensure_ascii=False, indent=2)}

Return strict JSON with exactly these fields:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""

    def propose(
        self,
        user_query: str,
        *,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        query = _text(user_query, "user_query")
        trusted_capabilities = validate_open_query_capabilities(capabilities)
        trusted_evidence = validate_open_query_evidence(evidence_history)
        prompt = self._prompt(query, trusted_capabilities, trusted_evidence)
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []

        proposal: dict[str, Any] | None = None
        for _attempt in range(2):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + "\nReturn one complete corrected JSON object.\n"
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict OpenQueryPlanProposal JSON.",
                    max_tokens=900,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                proposal = validate_open_query_plan_proposal(
                    extract_json_response(response),
                    has_evidence=bool(trusted_evidence),
                )
                break
            except Exception as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        if proposal is None:
            raise ClaimFirstPlanError(
                "provider failed two open-Query proposal attempts: "
                + " | ".join(self.last_errors)
            )
        return {
            "schema_version": 1,
            "source": "provider_claim_first_open_query",
            "input_digest": open_query_input_digest(
                query, trusted_capabilities, trusted_evidence
            ),
            "proposal": proposal,
            "provider": {
                "model_requested": self.model,
                "called": True,
                "attempt_count": len(self.last_responses),
                "errors": list(self.last_errors),
                "last_metadata": deepcopy(
                    dict(getattr(self.provider, "last_metadata", {}))
                ),
            },
        }


__all__ = [
    "ClaimFirstOpenQueryAgent",
    "ClaimFirstPlanError",
    "open_query_input_digest",
    "project_open_query_capabilities",
    "validate_open_query_capabilities",
    "validate_open_query_evidence",
    "validate_open_query_plan_proposal",
]
