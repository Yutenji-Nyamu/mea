"""Bounded model-facing Proposal Agent above TaskGen and ToolGen.

This is deliberately small: it demonstrates that an open query can produce a
new semantic TaskProposal and an aspect-specific ToolProposal without adding a
new hard-coded template.  The evaluation target is supplied by PlanSession and
therefore cannot be changed by the model.
"""

from __future__ import annotations

import json
import math
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
from mea.taskgen import (
    default_bbh_success_spec_v2,
    experimental_bbh_success_spec_v2,
    extract_json_response,
    success_spec_validation_report,
)
from mea.taskgen.click_bell import validate_click_bell_variant_hint
from mea.toolgen import route_tool_request


class ProposalAgentError(RuntimeError):
    """Raised when a provider cannot produce a valid bounded proposal."""


_BUNDLE_KEYS = {"schema_version", "task_proposal", "tool_proposal"}
_EXPERIMENTAL_SUCCESS_MODE = "experimental_success_bounded"

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
    if mode not in {
        "novel_bounded",
        "registered_reuse",
        _EXPERIMENTAL_SUCCESS_MODE,
    }:
        raise ProposalAgentError(f"unsupported proposal capability mode: {mode!r}")
    if mode == "novel_bounded" and not (
        task_name == "click_bell" and aspect_id == "object_position"
    ):
        raise ProposalAgentError(
            "novel bounded changes are not implemented for this task/aspect"
        )
    if mode == _EXPERIMENTAL_SUCCESS_MODE:
        if not (
            task_name == "beat_block_hammer"
            and aspect_id == "object_appearance.color"
            and first["taskgen"]["capability_id"] == "object_appearance.color"
            and first["taskgen"]["generation_mode"] == "force_codegen"
        ):
            raise ProposalAgentError(
                "experimental SuccessSpec proposals are capability-gated to "
                "beat_block_hammer/object_appearance.color force_codegen"
            )
        change_contract = {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": (
                    "exactly three finite numbers in [0,1]; the complete changes "
                    "object must not equal a registered template"
                ),
            }
        }
        example_changes = {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.25, 0.25, 0.75],
            }
        }
    elif mode == "novel_bounded":
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
    if mode == _EXPERIMENTAL_SUCCESS_MODE:
        # The scene color is query-generated, so the registered blue-specific
        # visual predicate would be semantically wrong for most proposals.
        vqa_candidates = [
            item for item in vqa_candidates if item != "block_color_blue"
        ]
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
    experimental_success = experimental_bbh_success_spec_v2(
        thresholds_m=(0.025, 0.025)
    )
    registered_example = {
        "schema_version": 1,
        "task_proposal": {
            "schema_version": (
                2 if mode == _EXPERIMENTAL_SUCCESS_MODE else 1
            ),
            "proposal_id": f"{aspect_id}.query_generated_1",
            "task_name": task_name,
            "aspect_id": aspect_id,
            "intent": "evaluate a query-relevant bounded variation",
            "capability_id": first["taskgen"]["capability_id"],
            "reuse_first": True,
            "changes": example_changes,
            "preserve_success_semantics": (
                mode != _EXPERIMENTAL_SUCCESS_MODE
            ),
            **(
                {"success_spec": experimental_success}
                if mode == _EXPERIMENTAL_SUCCESS_MODE
                else {}
            ),
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
            "preserve_success_semantics": (
                mode != _EXPERIMENTAL_SUCCESS_MODE
            ),
            "change_contract": change_contract,
            "registered_changes_to_avoid": [
                contract["taskgen"]["changes"] for contract in contracts
            ],
            "success_semantics": (
                {
                    "proposal_schema_version": 2,
                    "selected_track": "experimental",
                    "execution_authority": (
                        "compiled_success_spec_experimental_bounded"
                    ),
                    "official_reference": {
                        "track": "official",
                        "execution_authority": "official_check_success",
                        "success_spec": default_bbh_success_spec_v2(),
                    },
                    "experimental_contract": {
                        "track": "experimental",
                        "official_equivalent": False,
                        "compile_probe_acceptance_eligible": True,
                        "act_runtime_eligible": True,
                        "outcome_label": "generated_check_success",
                        "allowed_example": experimental_success,
                        "degrees_of_freedom": (
                            "only the two planar thresholds inside the trusted "
                            "experimental envelope"
                        ),
                    },
                    "reporting_contract": (
                        "official and experimental outcomes are separate labeled "
                        "channels; an experimental outcome must never be reported "
                        "as official policy success"
                    ),
                }
                if mode == _EXPERIMENTAL_SUCCESS_MODE
                else {
                    "proposal_schema_version": 1,
                    "selected_track": "official",
                    "execution_authority": "official_check_success",
                    "preserve_success_semantics": True,
                }
            ),
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
    if card["proposal_capability_mode"] == _EXPERIMENTAL_SUCCESS_MODE:
        variation_instruction = (
            "Propose one new bounded BBH appearance variation and TaskProposal v2. "
            "Copy the trusted experimental SuccessSpec structure exactly; only its "
            "two planar thresholds may vary inside the stated envelope. Keep the "
            "official result and experimental result as separately labeled tracks, "
            "and never describe the experimental predicate as official success."
        )
        success_instruction = (
            "The runtime capability gate explicitly permits the bounded experimental "
            "SuccessSpec shown in the card; no other success-semantics change is "
            "permitted."
        )
    elif card["proposal_capability_mode"] == "novel_bounded":
        variation_instruction = (
            "Propose one new bounded task variation that is not exactly equal to a "
            "registered change."
        )
        success_instruction = "Do not change success semantics."
    else:
        variation_instruction = (
            "Reuse exactly the registered task changes; only author the intent and "
            "smallest useful Tool/VQA assignment."
        )
        success_instruction = "Do not change success semantics."
    return f"""You are the bounded TaskGen/ToolGen Proposal Agent for MEA.
The policy evaluation is already bound to one task and checkpoint.  Do not
change task, policy, checkpoint, aspect, or executable fields.
{success_instruction}  {variation_instruction} Then assign either one listed Rule metric or,
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

For ToolProposal v2/v3, vqa_question_specs must be a non-empty list and the
set of run_local.* ids in vqa_phenomenon_ids must exactly equal the ids in
vqa_question_specs.  Prefer the registered ToolProposal v2 example verbatim
when no genuinely new visual question is needed.
"""


def _repair_vqa_question_binding(
    value: Mapping[str, Any], card: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one bounded structural repair to a malformed v2/v3 VQA binding."""

    repaired = deepcopy(dict(value))
    tool = repaired.get("tool_proposal")
    if not isinstance(tool, Mapping) or tool.get("schema_version") not in {2, 3}:
        raise ProposalError("bounded VQA repair requires ToolProposal v2 or v3")
    normalized_tool = deepcopy(dict(tool))
    reference = deepcopy(
        card["example"]["tool_proposal"]["vqa_question_specs"][0]
    )
    allowed_catalog = list(card["toolgen"]["vqa_phenomenon_candidates"])
    raw_phenomena = normalized_tool.get("vqa_phenomenon_ids")
    retained_catalog = (
        [
            item
            for item in raw_phenomena
            if isinstance(item, str) and item in allowed_catalog
        ]
        if isinstance(raw_phenomena, list)
        else []
    )
    if not retained_catalog:
        retained_catalog = [allowed_catalog[0]]
    normalized_tool["vqa_phenomenon_ids"] = [
        *dict.fromkeys(retained_catalog),
        reference["id"],
    ]
    normalized_tool["vqa_question_specs"] = [reference]
    repaired["tool_proposal"] = normalized_tool
    return repaired, {
        "schema_version": 1,
        "action": "bind_card_reference_vqa_question",
        "reference_question_id": reference["id"],
        "retained_catalog_phenomenon_ids": list(dict.fromkeys(retained_catalog)),
        "semantic_fields_changed": [
            "tool_proposal.vqa_phenomenon_ids",
            "tool_proposal.vqa_question_specs",
        ],
        "executable_fields_changed": [],
    }


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


def _validate_experimental_bbh_scene_changes(
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the sole scene envelope paired with public SuccessSpec v2."""

    if set(changes) != {"block"} or not isinstance(changes.get("block"), Mapping):
        raise ProposalError(
            "experimental BBH scene changes must contain exactly changes.block"
        )
    block = dict(changes["block"])
    expected = {"position_mode", "yaw_mode", "scale", "color"}
    if set(block) != expected:
        raise ProposalError(
            f"experimental BBH changes.block fields must be exactly {sorted(expected)}"
        )
    if (
        block["position_mode"] != "official_random"
        or block["yaw_mode"] != "official_random"
    ):
        raise ProposalError(
            "experimental BBH appearance preserves official position/yaw sampling"
        )
    scale = block["scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or abs(float(scale) - 1.0) > 1e-12
    ):
        raise ProposalError(
            "experimental BBH appearance preserves official block scale 1.0"
        )
    color = block["color"]
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ProposalError(
            "experimental BBH block.color must contain exactly three channels"
        )
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, (int, float))
        or not math.isfinite(float(channel))
        or not 0.0 <= float(channel) <= 1.0
        for channel in color
    ):
        raise ProposalError(
            "experimental BBH block.color channels must be finite numbers in [0,1]"
        )
    return {
        "block": {
            "position_mode": "official_random",
            "yaw_mode": "official_random",
            "scale": 1.0,
            "color": [float(channel) for channel in color],
        }
    }


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
    experimental_mode = (
        card["proposal_capability_mode"] == _EXPERIMENTAL_SUCCESS_MODE
    )
    if experimental_mode:
        if (
            task["schema_version"] != 2
            or task["preserve_success_semantics"] is not False
        ):
            raise ProposalError(
                "experimental SuccessSpec mode requires TaskProposal v2 with "
                "preserve_success_semantics=false"
            )
        report = success_spec_validation_report(task["success_spec"])
        if (
            report["official_equivalent"]
            or not report["act_eligible"]
            or not report["experimental_bounded"]
        ):
            raise ProposalError(
                "experimental SuccessSpec mode requires the non-official bounded "
                "ACT envelope"
            )
        task["changes"] = _validate_experimental_bbh_scene_changes(
            task["changes"]
        )
    elif (
        task["schema_version"] != 1
        or task["preserve_success_semantics"] is not True
    ):
        raise ProposalError(
            "TaskProposal v2 is disabled without the explicit experimental "
            "SuccessSpec capability mode"
        )
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
    result = {
        "schema_version": 1,
        "task_proposal": task,
        "tool_proposal": tool,
        "tool_route_preview": routed["route_decision"],
    }
    if experimental_mode:
        result["success_semantics_comparison"] = {
            "schema_version": 1,
            "selected_track": "experimental",
            "official": {
                "execution_authority": "official_check_success",
                "success_spec": default_bbh_success_spec_v2(),
                "result": None,
                "result_status": "not_measured_by_proposal",
            },
            "experimental": {
                "execution_authority": (
                    "compiled_success_spec_experimental_bounded"
                ),
                "success_spec": deepcopy(task["success_spec"]),
                "result": None,
                "result_status": "pending_materialization_or_rollout",
                "act_runtime_eligible": True,
                "outcome_label": "generated_check_success",
            },
            "reporting_contract": (
                "Keep official and experimental outcomes as separate channels. "
                "Null means unmeasured. A later rollout may populate the "
                "generated_check_success channel, which is never relabeled as "
                "official policy success."
            ),
        }
    return result


class BoundedProposalAgent:
    """Ask one model for a proposal while runtime owns all executable details."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = str(model)
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []
        self.last_repairs: list[dict[str, Any]] = []

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
        elif mode == _EXPERIMENTAL_SUCCESS_MODE:
            # The public agent may reach the non-official success path only
            # through this explicit runtime-owned mode.  It must also produce a
            # new scene variation rather than relabel a registered official run.
            require_novel_changes = True
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
        self.last_repairs = []
        card = _proposal_card(
            target,
            aspect_id,
            base_template_id=base_template_id,
            capability_mode=mode,
        )
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
                raw = extract_json_response(str(response))
                try:
                    return validate_proposal_bundle(
                        raw,
                        target=target,
                        aspect_id=aspect_id,
                        require_novel_changes=require_novel_changes,
                        base_template_id=base_template_id,
                        capability_mode=mode,
                    )
                except ProposalError as exc:
                    tool = raw.get("tool_proposal")
                    raw_specs = (
                        tool.get("vqa_question_specs")
                        if isinstance(tool, Mapping)
                        else None
                    )
                    vqa_binding_error = (
                        isinstance(tool, Mapping)
                        and tool.get("schema_version") in {2, 3}
                        and (
                            not isinstance(raw_specs, list)
                            or not raw_specs
                            or "vqa_question_specs" in str(exc)
                            or "run-local phenomenon ids" in str(exc)
                            or "run-local VQA question" in str(exc)
                        )
                    )
                    if not vqa_binding_error:
                        raise
                    repaired, trace = _repair_vqa_question_binding(raw, card)
                    validated = validate_proposal_bundle(
                        repaired,
                        target=target,
                        aspect_id=aspect_id,
                        require_novel_changes=require_novel_changes,
                        base_template_id=base_template_id,
                        capability_mode=mode,
                    )
                    trace.update(
                        {
                            "attempt_index": _attempt + 1,
                            "trigger": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    self.last_repairs.append(trace)
                    return validated
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
