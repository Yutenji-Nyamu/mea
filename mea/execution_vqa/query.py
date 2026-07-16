"""Bounded query contracts for execution-time visual observations.

The Plan Agent and ToolGen outputs are not allowed to inject arbitrary Vision
prompts.  This module maps their audited identifiers to a small catalog of
visual questions.  Unknown identifiers fall back to the legacy three-question
profile so existing callers keep the previous behaviour.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


class ExecutionVQAQueryError(ValueError):
    """Raised when a dynamic Execution VQA query violates its contract."""


QUESTION_CATALOG: dict[str, dict[str, Any]] = {
    "block_color_blue": {
        "question_type": "categorical_attribute_match",
        "target_role": "target_object",
        "question": "Does the target block appear blue in the rollout?",
        "visual_scope": "appearance_only",
        "numeric_authority": "no_numeric_oracle",
    },
    "hammer_visibly_lifted": {
        "question_type": "visible_state_change",
        "target_role": "manipulated_tool",
        "question": (
            "Is the hammer visibly lifted above its initial resting height?"
        ),
        "visual_scope": "rollout_change",
        "numeric_authority": "simulator_pickup_threshold_is_authoritative",
    },
    "block_visibly_displaced": {
        "question_type": "visible_state_change",
        "target_role": "target_object",
        "question": (
            "Is the target block visibly displaced from its initial pose?"
        ),
        "visual_scope": "rollout_change",
        "numeric_authority": "simulator_pose_is_authoritative_when_available",
    },
    "bell_visibly_pressed": {
        "question_type": "visible_state_change",
        "target_role": "task_target",
        "question": "Does the robot visibly press or actuate the target bell?",
        "visual_scope": "rollout_change",
        "numeric_authority": "official_check_success_is_authoritative",
    },
    "bottle_visibly_repositioned": {
        "question_type": "visible_state_change",
        "target_role": "manipulated_object",
        "question": (
            "Is the target bottle visibly moved from its initial resting pose "
            "to the elevated side placement?"
        ),
        "visual_scope": "rollout_change",
        "numeric_authority": "official_check_success_is_authoritative",
    },
    "roller_visibly_lifted": {
        "question_type": "visible_state_change",
        "target_role": "manipulated_object",
        "question": "Is the target roller visibly lifted by both robot arms?",
        "visual_scope": "rollout_change",
        "numeric_authority": "official_check_success_is_authoritative",
    },
}

# Keep the implicit legacy profile frozen even as task-specific catalog entries
# are added.  Existing callers must never receive a click_bell question.
LEGACY_PHENOMENON_IDS = (
    "block_color_blue",
    "hammer_visibly_lifted",
    "block_visibly_displaced",
)
ALL_PHENOMENON_IDS = tuple(QUESTION_CATALOG)

# Exact, trusted identifiers only.  Neither a model-produced task instruction
# nor ToolSpec.question is copied into the Vision prompt.
TEMPLATE_QUESTION_RULES: dict[str, tuple[str, ...]] = {
    "object_appearance.color_blue": ("block_color_blue",),
}
TASK_TEMPLATE_QUESTION_RULES: dict[tuple[str, str], tuple[str, ...]] = {
    ("click_bell", "task_execution.official_baseline"): (
        "bell_visibly_pressed",
    ),
    ("adjust_bottle", "task_execution.official_baseline"): (
        "bottle_visibly_repositioned",
    ),
    ("grab_roller", "task_execution.official_baseline"): (
        "roller_visibly_lifted",
    ),
}
SUB_ASPECT_QUESTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("object_appearance.color", ("block_color_blue",)),
    (
        "performance.pickup_to_contact_timing",
        ("hammer_visibly_lifted", "block_visibly_displaced"),
    ),
)
METRIC_QUESTION_RULES: dict[str, tuple[str, ...]] = {
    "hammer_block_contact_ever": (
        "hammer_visibly_lifted",
        "block_visibly_displaced",
    ),
    "first_contact_step": (
        "hammer_visibly_lifted",
        "block_visibly_displaced",
    ),
    "first_hammer_pickup_step": ("hammer_visibly_lifted",),
    "hammer_pickup_height": ("hammer_visibly_lifted",),
    "pickup_to_first_contact_time": (
        "hammer_visibly_lifted",
        "block_visibly_displaced",
    ),
}
TASK_METRIC_QUESTION_RULES: dict[tuple[str, str], tuple[str, ...]] = {
    ("click_bell", "official_check_success"): ("bell_visibly_pressed",),
    ("adjust_bottle", "official_check_success"): (
        "bottle_visibly_repositioned",
    ),
    ("grab_roller", "official_check_success"): (
        "roller_visibly_lifted",
    ),
}

QUERY_KEYS = {
    "schema_version",
    "profile",
    "task_name",
    "template_id",
    "sub_aspect",
    "tool_metric",
    "phenomenon_ids",
    "questions",
    "selection_reasons",
    "answer_contract",
}
QUESTION_KEYS = {
    "id",
    "question_type",
    "target_role",
    "question",
    "visual_scope",
    "numeric_authority",
}
ANSWER_CONTRACT = {
    "required_response_keys": [
        "phenomena",
        "confidence",
        "frame_ids",
        "numeric_consistency",
        "conflicts",
    ],
    "phenomenon_item_keys": [
        "id",
        "observed",
        "description",
        "confidence",
        "frame_ids",
    ],
    "observed_type": "boolean_or_null",
    "numeric_consistency_values": ["consistent", "conflict", "uncertain"],
}


def _optional_identifier(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ExecutionVQAQueryError(f"{field} must be null or a non-empty string")
    return value.strip()


def _tool_metric(tool_contract: Mapping[str, Any] | None) -> str | None:
    if tool_contract is None:
        return None
    if not isinstance(tool_contract, Mapping):
        raise ExecutionVQAQueryError("tool_contract must be a mapping or null")
    return _optional_identifier(tool_contract.get("metric"), field="tool metric")


def _append_unique(destination: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in destination:
            destination.append(value)


def build_execution_vqa_query(
    *,
    task_name: str | None = None,
    template_id: str | None = None,
    sub_aspect: str | None = None,
    tool_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, allowlisted visual query contract.

    Context fields only select catalog entries.  Free-form Plan/Tool text is
    intentionally ignored.  Calling this function without context returns the
    original three phenomena in their original order.
    """

    task = _optional_identifier(task_name, field="task_name")
    template = _optional_identifier(template_id, field="template_id")
    aspect = _optional_identifier(sub_aspect, field="sub_aspect")
    metric = _tool_metric(tool_contract)
    context_supplied = any((task, template, aspect, metric))
    selected: list[str] = []
    reasons: list[str] = []

    task_template_key = (task, template)
    if task_template_key in TASK_TEMPLATE_QUESTION_RULES:
        _append_unique(selected, TASK_TEMPLATE_QUESTION_RULES[task_template_key])
        reasons.append(f"task_template:{task}:{template}")
    if template in TEMPLATE_QUESTION_RULES:
        _append_unique(selected, TEMPLATE_QUESTION_RULES[template])
        reasons.append(f"template:{template}")
    if aspect:
        for prefix, question_ids in SUB_ASPECT_QUESTION_RULES:
            if aspect == prefix or aspect.startswith(prefix + "."):
                _append_unique(selected, question_ids)
                reasons.append(f"sub_aspect:{prefix}")
                break
    task_metric_key = (task, metric)
    if task_metric_key in TASK_METRIC_QUESTION_RULES:
        _append_unique(selected, TASK_METRIC_QUESTION_RULES[task_metric_key])
        reasons.append(f"task_metric:{task}:{metric}")
    elif metric in METRIC_QUESTION_RULES:
        _append_unique(selected, METRIC_QUESTION_RULES[metric])
        reasons.append(f"tool_metric:{metric}")

    if not selected:
        selected = list(LEGACY_PHENOMENON_IDS)
        reasons.append(
            "legacy_default:no_context"
            if not context_supplied
            else "legacy_fallback:no_allowlisted_rule"
        )

    questions = []
    for phenomenon_id in ALL_PHENOMENON_IDS:
        if phenomenon_id not in selected:
            continue
        questions.append(
            {
                "id": phenomenon_id,
                **deepcopy(QUESTION_CATALOG[phenomenon_id]),
            }
        )

    query = {
        "schema_version": 1,
        "profile": "dynamic_v1" if context_supplied else "legacy_v1",
        "task_name": task,
        "template_id": template,
        "sub_aspect": aspect,
        "tool_metric": metric,
        "phenomenon_ids": [item["id"] for item in questions],
        "questions": questions,
        "selection_reasons": reasons,
        "answer_contract": deepcopy(ANSWER_CONTRACT),
    }
    return validate_execution_vqa_query(query)


def validate_execution_vqa_query(value: Any) -> dict[str, Any]:
    """Validate a complete query before it is interpolated into a prompt."""

    if not isinstance(value, dict) or set(value) != QUERY_KEYS:
        raise ExecutionVQAQueryError(
            f"query fields must be exactly {sorted(QUERY_KEYS)}"
        )
    if value.get("schema_version") != 1:
        raise ExecutionVQAQueryError("query.schema_version must be 1")
    if value.get("profile") not in {"legacy_v1", "dynamic_v1"}:
        raise ExecutionVQAQueryError("query.profile is not allowlisted")
    for field in ("task_name", "template_id", "sub_aspect", "tool_metric"):
        _optional_identifier(value.get(field), field=field)

    ids = value.get("phenomenon_ids")
    if not isinstance(ids, list) or not ids:
        raise ExecutionVQAQueryError("query.phenomenon_ids must be non-empty")
    if len(ids) != len(set(ids)):
        raise ExecutionVQAQueryError("query.phenomenon_ids must be unique")
    if any(item not in QUESTION_CATALOG for item in ids):
        raise ExecutionVQAQueryError("query contains a non-allowlisted phenomenon")

    questions = value.get("questions")
    if not isinstance(questions, list) or len(questions) != len(ids):
        raise ExecutionVQAQueryError("query.questions must match phenomenon_ids")
    normalized_questions = []
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or set(question) != QUESTION_KEYS:
            raise ExecutionVQAQueryError(f"query.questions[{index}] has invalid fields")
        phenomenon_id = question.get("id")
        if phenomenon_id != ids[index]:
            raise ExecutionVQAQueryError("question order must match phenomenon_ids")
        expected = {"id": phenomenon_id, **QUESTION_CATALOG[phenomenon_id]}
        if question != expected:
            raise ExecutionVQAQueryError(
                f"query.questions[{index}] must equal the trusted catalog entry"
            )
        normalized_questions.append(deepcopy(expected))

    reasons = value.get("selection_reasons")
    if not isinstance(reasons, list) or not reasons:
        raise ExecutionVQAQueryError("query.selection_reasons must be non-empty")
    if any(not isinstance(item, str) or not item for item in reasons):
        raise ExecutionVQAQueryError("query.selection_reasons contains invalid values")
    if value.get("answer_contract") != ANSWER_CONTRACT:
        raise ExecutionVQAQueryError("query.answer_contract must be the fixed contract")

    result = deepcopy(value)
    result["questions"] = normalized_questions
    result["answer_contract"] = deepcopy(ANSWER_CONTRACT)
    return result
