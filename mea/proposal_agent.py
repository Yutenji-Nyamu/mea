"""Bounded model-facing Proposal Agent above TaskGen and ToolGen.

This is deliberately small: it demonstrates that an open query can produce a
new semantic TaskProposal and an aspect-specific ToolProposal without adding a
new hard-coded template.  The evaluation target is supplied by PlanSession and
therefore cannot be changed by the model.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from mea.capability_adapter import registered_capability_contracts
from mea.execution_vqa import QUESTION_CATALOG, validate_run_local_question_spec
from mea.proposals import (
    ProposalError,
    tool_request_from_proposal,
    validate_task_proposal,
    validate_tool_proposal,
)
from mea.taskgen import extract_json_response
from mea.taskgen.click_bell import validate_click_bell_variant_hint
from mea.toolgen import route_tool_request


class ProposalAgentError(RuntimeError):
    """Raised when a provider cannot produce a valid bounded proposal."""


_BUNDLE_KEYS = {"schema_version", "task_proposal", "tool_proposal"}

_TYPED_IDENTIFIERS_BY_TASK: dict[str, dict[str, list[Any]]] = {
    "beat_block_hammer": {
        "trace_signals": [
            "hammer_position",
            "block_position",
            "hammer_functional_position",
            "block_functional_position",
            "left_tcp_position",
            "right_tcp_position",
        ],
        # This is intentionally not every raw simulator contact.  The first
        # safety slice admits only one expected task pair and one observed,
        # precisely named unintended-contact proxy.
        "contact_actor_pairs": [
            ["020_hammer", "box"],
            ["020_hammer", "left_camera"],
        ],
    },
    "click_bell": {
        "trace_signals": [
            "bell_position",
            "bell_contact_position",
            "left_tcp_position",
            "right_tcp_position",
        ],
        "contact_actor_pairs": [],
    },
}


def _target_aspect(target: Mapping[str, Any], aspect_id: str) -> dict[str, Any]:
    for aspect in target.get("aspects") or []:
        if isinstance(aspect, Mapping) and aspect.get("aspect_id") == aspect_id:
            return deepcopy(dict(aspect))
    raise ProposalAgentError(
        f"aspect {aspect_id!r} is unavailable for bound task {target.get('task_name')!r}"
    )


def proposal_capability_mode(task_name: str, aspect_id: str) -> str:
    """Return the honest proposal mode for one currently supported axis."""

    if task_name == "click_bell" and aspect_id == "object_position":
        return "novel_bounded"
    return "registered_reuse"


def _proposal_card(
    target: Mapping[str, Any],
    aspect_id: str,
    *,
    base_template_id: str | None = None,
    capability_mode: str | None = None,
) -> dict[str, Any]:
    task_name = str(target.get("task_name") or "")
    aspect = _target_aspect(target, aspect_id)
    contracts = [
        item
        for item in registered_capability_contracts(task_name)
        if item["aspect"]["aspect_id"] == aspect_id
    ]
    if not contracts:
        raise ProposalAgentError("bound aspect has no materializable TaskGen contract")
    first = next(
        (
            item
            for item in contracts
            if base_template_id is not None
            and item["template_id"] == base_template_id
        ),
        contracts[0],
    )
    mode = capability_mode or proposal_capability_mode(task_name, aspect_id)
    if mode not in {"novel_bounded", "registered_reuse"}:
        raise ProposalAgentError(f"unsupported proposal capability mode: {mode!r}")
    if mode == "novel_bounded" and not (
        task_name == "click_bell" and aspect_id == "object_position"
    ):
        raise ProposalAgentError(
            "novel bounded changes are not implemented for this task/aspect"
        )
    if mode == "novel_bounded":
        change_contract = {
            "bell": {
                "position_mode": "fixed",
                "xy": (
                    "two finite numbers; x in [-0.25,-0.05] or [0.05,0.25], "
                    "y in [-0.20,0.0]"
                ),
            }
        }
        example_changes = {
            "bell": {"position_mode": "fixed", "xy": [-0.14, -0.12]}
        }
    else:
        change_contract = {
            "mode": "registered_reuse_only",
            "allowed_change_roots": first["taskgen"]["allowed_change_roots"],
            "required_changes": first["taskgen"]["changes"],
        }
        example_changes = deepcopy(first["taskgen"]["changes"])
    vqa_candidates: list[str] = []
    for contract in contracts:
        for phenomenon_id in contract["vqa"]["phenomenon_ids"]:
            if phenomenon_id not in vqa_candidates:
                vqa_candidates.append(phenomenon_id)
    reference_question = deepcopy(QUESTION_CATALOG[vqa_candidates[0]])
    reference_question.update(
        {
            "id": (
                f"run_local.{task_name}.{aspect_id.replace('.', '_')}."
                "query_observation"
            ),
            "question": (
                "Does the rollout visibly show the robot making task-relevant "
                "progress under the query-generated variation?"
            ),
        }
    )
    reference_question = validate_run_local_question_spec(reference_question)
    typed_identifiers = deepcopy(
        _TYPED_IDENTIFIERS_BY_TASK.get(
            task_name, {"trace_signals": [], "contact_actor_pairs": []}
        )
    )
    registered_example = {
        "schema_version": 1,
        "task_proposal": {
            "schema_version": 1,
            "proposal_id": f"{aspect_id}.query_generated_1",
            "task_name": task_name,
            "aspect_id": aspect_id,
            "intent": "evaluate a query-relevant bounded variation",
            "capability_id": first["taskgen"]["capability_id"],
            "reuse_first": True,
            "changes": example_changes,
            "preserve_success_semantics": True,
        },
        "tool_proposal": {
            "schema_version": 2,
            "proposal_id": f"{aspect_id}.query_generated_1.tool",
            "task_name": task_name,
            "aspect_id": aspect_id,
            "evaluation_goal": "measure task outcome and visible behavior",
            "metric": first["tool"]["metric"],
            "question": "What simulator measurement best diagnoses this aspect?",
            "vqa_phenomenon_ids": [
                *vqa_candidates,
                reference_question["id"],
            ],
            "vqa_question_specs": [reference_question],
            "reuse_first": True,
        },
    }
    typed_example = deepcopy(registered_example)
    typed_example["tool_proposal"].update(
        {
            "schema_version": 3,
            "proposal_id": f"{aspect_id}.query_metric_1.tool",
            "metric": "query_right_tcp_target_xy_distance",
            "question": "What was the minimum XY TCP-target distance?",
            "metric_spec": {
                "schema_version": 1,
                "operation": "minimum_distance",
                "left_signal": "right_tcp_position",
                "right_signal": (
                    "bell_contact_position"
                    if task_name == "click_bell"
                    else "block_position"
                ),
                "dimensions": ["x", "y"],
                "unit": "m",
                "null_semantics": "null_if_no_finite_sample",
            },
        }
    )
    if (
        task_name == "beat_block_hammer"
        and aspect_id == "safety.hammer_left_camera_contact"
    ):
        typed_example["tool_proposal"].update(
            {
                "metric": "query_hammer_left_camera_contact_count",
                "question": (
                    "How many physical hammer-left_camera contact intervals occurred?"
                ),
                "metric_spec": {
                    "schema_version": 1,
                    "operation": "event_count",
                    "event": {
                        "event_type": "contact_interval",
                        "actors": ["020_hammer", "left_camera"],
                        "physical_only": True,
                    },
                    "unit": "count",
                    "null_semantics": "zero_if_absent",
                },
            }
        )
    return {
        "task_name": task_name,
        "aspect": aspect,
        "proposal_capability_mode": mode,
        "base_template_id": first["template_id"],
        "taskgen": {
            "capability_id": first["taskgen"]["capability_id"],
            "reuse_first": True,
            "preserve_success_semantics": True,
            "change_contract": change_contract,
            "registered_changes_to_avoid": [
                contract["taskgen"]["changes"] for contract in contracts
            ],
        },
        "toolgen": {
            "metric_candidates": sorted(
                {contract["tool"]["metric"] for contract in contracts}
            ),
            "typed_metric_spec_v1": {
                "optional": True,
                "proposal_schema_version": 3,
                "metric_id": (
                    "new lower_snake_case id that does not collide with a "
                    "registered metric"
                ),
                "operations": [
                    "event_count",
                    "minimum_distance",
                    "time_between_events",
                ],
                "allowed_identifiers": typed_identifiers,
                "exact_variants": {
                    "minimum_distance": {
                        "fields": [
                            "schema_version",
                            "operation",
                            "left_signal",
                            "right_signal",
                            "dimensions",
                            "unit",
                            "null_semantics",
                        ],
                        "dimensions": [["x", "y"], ["x", "y", "z"]],
                        "unit": "m",
                        "null_semantics": "null_if_no_finite_sample",
                    },
                    "event_count": {
                        "fields": [
                            "schema_version",
                            "operation",
                            "event",
                            "unit",
                            "null_semantics",
                        ],
                        "event_fields": [
                            "event_type",
                            "actors",
                            "physical_only",
                        ],
                        "unit": "count",
                        "null_semantics": "zero_if_absent",
                    },
                    "time_between_events": {
                        "fields": [
                            "schema_version",
                            "operation",
                            "start_event",
                            "end_event",
                            "unit",
                            "null_semantics",
                        ],
                        "event_fields": [
                            "event_type",
                            "actors",
                            "physical_only",
                        ],
                        "unit": "s",
                        "null_semantics": "null_if_missing_or_reversed",
                    },
                },
                "authority": (
                    "bounded telemetry DSL only; never arbitrary Python"
                ),
            },
            "vqa_phenomenon_candidates": vqa_candidates,
            "reuse_first": True,
        },
        "example": registered_example,
        "typed_metric_example": typed_example,
    }


def build_proposal_prompt(
    user_query: str,
    target: Mapping[str, Any],
    aspect_id: str,
    *,
    base_template_id: str | None = None,
    capability_mode: str | None = None,
    planning_context: Mapping[str, Any] | None = None,
) -> str:
    query = str(user_query).strip()
    if not query:
        raise ProposalAgentError("user_query must be non-empty")
    card = _proposal_card(
        target,
        aspect_id,
        base_template_id=base_template_id,
        capability_mode=capability_mode,
    )
    context_text = (
        json.dumps(planning_context, ensure_ascii=False, indent=2)
        if planning_context is not None
        else "not supplied; use only the bound target and capability card"
    )
    variation_instruction = (
        "Propose one new bounded task variation that is not exactly equal to a "
        "registered change."
        if card["proposal_capability_mode"] == "novel_bounded"
        else "Reuse exactly the registered task changes; only author the intent and "
        "smallest useful Tool/VQA assignment."
    )
    return f"""You are the bounded TaskGen/ToolGen Proposal Agent for MEA.
The policy evaluation is already bound to one task and checkpoint.  Do not
change task, policy, checkpoint, aspect, success semantics, or executable
fields.  {variation_instruction} Then assign either one listed Rule metric or,
when the query needs a new measurement, ToolProposal v3 with one bounded
typed_metric_spec_v1 and a new metric id.  Select the smallest useful subset
of the listed VQA phenomena.  TaskGen and ToolGen independently
retrieve or generate implementations after validating this semantic proposal.

USER QUERY:
{query}

BOUND EVALUATION TARGET:
{json.dumps(target, ensure_ascii=False, indent=2)}

TRUSTED POLICY/SIMULATOR/ADAPTER CONTEXT:
{context_text}

PROPOSAL CAPABILITY CARD:
{json.dumps(card, ensure_ascii=False, indent=2)}

Return one strict ProposalBundle JSON object.  Use this exact registered Tool
shape when an existing metric is enough:
{json.dumps(card['example'], ensure_ascii=False, indent=2)}

Or use this exact ToolProposal v3 shape when the query genuinely needs a new
typed metric; copy only identifiers admitted by allowed_identifiers:
{json.dumps(card['typed_metric_example'], ensure_ascii=False, indent=2)}
"""


def _validate_typed_metric_identifiers(
    tool: Mapping[str, Any], card: Mapping[str, Any]
) -> None:
    if tool.get("schema_version") != 3:
        return
    spec = tool.get("metric_spec")
    if not isinstance(spec, Mapping):
        raise ProposalError("ToolProposal v3 must contain metric_spec")
    identifiers = card["toolgen"]["typed_metric_spec_v1"][
        "allowed_identifiers"
    ]
    allowed_signals = set(identifiers["trace_signals"])
    allowed_pairs = {
        tuple(sorted(pair)) for pair in identifiers["contact_actor_pairs"]
    }
    if spec.get("operation") == "minimum_distance":
        requested = {spec.get("left_signal"), spec.get("right_signal")}
        if not requested <= allowed_signals:
            raise ProposalError(
                "MetricSpec trace signals are outside the bound TaskSchema"
            )
        return
    selector_fields = (
        ("event",)
        if spec.get("operation") == "event_count"
        else ("start_event", "end_event")
    )
    for field in selector_fields:
        selector = spec.get(field)
        if not isinstance(selector, Mapping):
            raise ProposalError(f"MetricSpec.{field} must be an event selector")
        actors = selector.get("actors")
        if actors is not None and tuple(sorted(actors)) not in allowed_pairs:
            raise ProposalError(
                f"MetricSpec.{field}.actors are outside the bound actor pairs"
            )


def validate_proposal_bundle(
    value: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    aspect_id: str,
    require_novel_changes: bool = True,
    base_template_id: str | None = None,
    capability_mode: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BUNDLE_KEYS:
        raise ProposalError(
            f"ProposalBundle fields must be exactly {sorted(_BUNDLE_KEYS)}"
        )
    if value.get("schema_version") != 1:
        raise ProposalError("ProposalBundle.schema_version must be 1")
    task_name = str(target.get("task_name") or "")
    _target_aspect(target, aspect_id)
    task = validate_task_proposal(
        value.get("task_proposal"), expected_task_name=task_name
    )
    if task["aspect_id"] != aspect_id:
        raise ProposalError("TaskProposal changed the selected aspect")
    tool = validate_tool_proposal(
        value.get("tool_proposal"),
        expected_task_name=task_name,
        expected_aspect_id=aspect_id,
    )
    card = _proposal_card(
        target,
        aspect_id,
        base_template_id=base_template_id,
        capability_mode=capability_mode,
    )
    _validate_typed_metric_identifiers(tool, card)
    if task["capability_id"] != card["taskgen"]["capability_id"]:
        raise ProposalError("TaskProposal changed the selected capability")
    proposed_tool_request = tool_request_from_proposal(tool)
    routed = route_tool_request(proposed_tool_request)
    typed_metric = (
        tool["schema_version"] == 3
        and routed["route_decision"]["resolved_route"]
        == "typed_metric_spec_compile"
    )
    if (
        tool["metric"] not in card["toolgen"]["metric_candidates"]
        and not typed_metric
    ):
        raise ProposalError("ToolProposal selected a metric outside the capability card")
    run_local_ids = {
        item["id"] for item in tool.get("vqa_question_specs", [])
    }
    selected_catalog_ids = set(tool["vqa_phenomenon_ids"]) - run_local_ids
    unknown_vqa = sorted(
        selected_catalog_ids - set(card["toolgen"]["vqa_phenomenon_candidates"])
    )
    if unknown_vqa:
        raise ProposalError(f"ToolProposal selected unavailable VQA phenomena: {unknown_vqa}")
    if any(item not in QUESTION_CATALOG for item in selected_catalog_ids):
        raise ProposalError("ToolProposal selected an unregistered VQA phenomenon")
    registered_changes = card["taskgen"]["registered_changes_to_avoid"]
    if require_novel_changes and task["changes"] in registered_changes:
        raise ProposalError("TaskProposal repeated an exact registered template")
    if (
        card["proposal_capability_mode"] == "registered_reuse"
        and task["changes"] != card["taskgen"]["change_contract"]["required_changes"]
    ):
        raise ProposalError(
            "this task/aspect currently supports registered changes only"
        )
    if task_name == "click_bell" and aspect_id == "object_position":
        try:
            task["changes"] = validate_click_bell_variant_hint(task["changes"])
        except RuntimeError as exc:
            raise ProposalError(f"invalid click_bell position proposal: {exc}") from exc
    if routed["route_decision"]["status"] != "resolved":
        raise ProposalError("ToolProposal cannot be resolved by ToolGen")
    return {
        "schema_version": 1,
        "task_proposal": task,
        "tool_proposal": tool,
        "tool_route_preview": routed["route_decision"],
    }


class BoundedProposalAgent:
    """Ask one model for a proposal while runtime owns all executable details."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = str(model)
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    def propose(
        self,
        user_query: str,
        *,
        target: Mapping[str, Any],
        aspect_id: str,
        require_novel_changes: bool = True,
        base_template_id: str | None = None,
        capability_mode: str | None = None,
        planning_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = capability_mode or proposal_capability_mode(
            str(target.get("task_name") or ""), aspect_id
        )
        if mode == "registered_reuse":
            require_novel_changes = False
        prompt = build_proposal_prompt(
            user_query,
            target,
            aspect_id,
            base_template_id=base_template_id,
            capability_mode=mode,
            planning_context=planning_context,
        )
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []
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
                    system="Return only strict ProposalBundle JSON.",
                    max_tokens=1000,
                    temperature=0.0,
                )
                self.last_responses.append(str(response))
                return validate_proposal_bundle(
                    extract_json_response(str(response)),
                    target=target,
                    aspect_id=aspect_id,
                    require_novel_changes=require_novel_changes,
                    base_template_id=base_template_id,
                    capability_mode=mode,
                )
            except Exception as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        raise ProposalAgentError(f"proposal failed twice: {self.last_errors}")


__all__ = [
    "BoundedProposalAgent",
    "ProposalAgentError",
    "build_proposal_prompt",
    "validate_proposal_bundle",
    "proposal_capability_mode",
]
