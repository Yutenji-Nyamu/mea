"""Bounded query contracts for execution-time visual observations.

The Plan Agent and ToolGen outputs are not allowed to inject arbitrary Vision
prompts.  This module maps audited identifiers to a small committed catalog and
admits only tightly bounded, self-contained ``run_local.*`` question specs.
Unknown context still falls back to the legacy three-question profile so
existing callers keep the previous behaviour.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.capability_adapter import (
    CapabilityAdapterError,
    resolve_capability_contract,
)


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
        "question": ("Is the hammer visibly lifted above its initial resting height?"),
        "visual_scope": "rollout_change",
        "numeric_authority": "simulator_pickup_threshold_is_authoritative",
    },
    "block_visibly_displaced": {
        "question_type": "visible_state_change",
        "target_role": "target_object",
        "question": ("Is the target block visibly displaced from its initial pose?"),
        "visual_scope": "rollout_change",
        "numeric_authority": "simulator_pose_is_authoritative_when_available",
    },
    "hammer_avoids_unintended_collision": {
        "question_type": "visible_unintended_contact",
        "target_role": "manipulated_tool",
        "question": (
            "Does the hammer avoid visibly colliding with the left camera "
            "actor while the robot executes the task?"
        ),
        "visual_scope": "rollout_change",
        "numeric_authority": (
            "hammer_left_camera_contact_count_is_authoritative_for_this_proxy"
        ),
    },
    "target_block_visible": {
        "question_type": "visible_target_under_scene_shift",
        "target_role": "target_object",
        "question": (
            "Is the intended target block clearly visible in the generated "
            "distractor scene?"
        ),
        "visual_scope": "scene_appearance",
        "numeric_authority": "simulator_actor_identity_is_authoritative",
    },
    "lookalike_distractor_visible": {
        "question_type": "visible_target_under_scene_shift",
        "target_role": "distractor",
        "question": (
            "Is the physically similar distractor clearly visible as a "
            "separate object near the intended target?"
        ),
        "visual_scope": "scene_appearance",
        "numeric_authority": "simulator_actor_identity_is_authoritative",
    },
    "distractor_not_struck": {
        "question_type": "visible_unintended_contact",
        "target_role": "distractor",
        "question": (
            "Does the rollout avoid visibly striking the lookalike distractor "
            "while acting on the intended target?"
        ),
        "visual_scope": "rollout_change",
        "numeric_authority": (
            "generated_checker_and_simulator_contacts_are_authoritative"
        ),
    },
    "bell_visibly_pressed": {
        "question_type": "visible_state_change",
        "target_role": "task_target",
        "question": "Does the robot visibly press or actuate the target bell?",
        "visual_scope": "rollout_change",
        "numeric_authority": "official_check_success_is_authoritative",
    },
    "bell_target_selected_among_clutter": {
        "question_type": "visible_target_selection",
        "target_role": "task_target",
        "question": (
            "Among the tabletop clutter, does the robot visibly act on the "
            "target bell rather than a distractor?"
        ),
        "visual_scope": "rollout_change",
        "numeric_authority": "official_check_success_is_authoritative",
    },
    "bell_visible_with_unseen_background_texture": {
        "question_type": "visible_target_under_scene_shift",
        "target_role": "task_target",
        "question": (
            "With the unseen wall and table textures, does the target bell "
            "remain clearly visible and distinguishable?"
        ),
        "visual_scope": "scene_appearance",
        "numeric_authority": "simulator_texture_info_is_authoritative",
    },
    "bell_visible_under_random_lighting": {
        "question_type": "visible_target_under_scene_shift",
        "target_role": "task_target",
        "question": (
            "Under the randomized static lighting, does the target bell remain "
            "clearly visible without unusable under- or over-exposure?"
        ),
        "visual_scope": "scene_appearance",
        "numeric_authority": "simulator_light_configuration_is_authoritative",
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
    ("beat_block_hammer", "safety.hammer_left_camera_contact.official"): (
        "hammer_avoids_unintended_collision",
    ),
    ("beat_block_hammer", "robustness.distractor_avoidance.lookalike"): (
        "target_block_visible",
        "lookalike_distractor_visible",
        "distractor_not_struck",
    ),
    ("click_bell", "task_execution.official_baseline"): ("bell_visibly_pressed",),
    ("click_bell", "object_position.left_fixed"): ("bell_visibly_pressed",),
    ("click_bell", "object_position.right_fixed"): ("bell_visibly_pressed",),
    ("click_bell", "robustness.scene_clutter.official_table"): (
        "bell_visibly_pressed",
        "bell_target_selected_among_clutter",
    ),
    ("click_bell", "scene_background_texture.unseen"): (
        "bell_visibly_pressed",
        "bell_visible_with_unseen_background_texture",
    ),
    ("click_bell", "scene_lighting.static_random"): (
        "bell_visibly_pressed",
        "bell_visible_under_random_lighting",
    ),
    ("click_bell", "performance.completion_time_stability.official"): (
        "bell_visibly_pressed",
    ),
    ("adjust_bottle", "task_execution.official_baseline"): (
        "bottle_visibly_repositioned",
    ),
    ("grab_roller", "task_execution.official_baseline"): ("roller_visibly_lifted",),
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
    ("click_bell", "time_to_success"): ("bell_visibly_pressed",),
    ("adjust_bottle", "official_check_success"): ("bottle_visibly_repositioned",),
    ("grab_roller", "official_check_success"): ("roller_visibly_lifted",),
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
RUN_LOCAL_QUESTION_MAX_CHARS = 240
_RUN_LOCAL_PHENOMENON_ID = re.compile(
    r"^run_local\.[a-z0-9](?:[a-z0-9_.-]{0,94}[a-z0-9])?$"
)
RUN_LOCAL_QUESTION_TYPES = frozenset(
    item["question_type"] for item in QUESTION_CATALOG.values()
)
RUN_LOCAL_TARGET_ROLES = frozenset(
    item["target_role"] for item in QUESTION_CATALOG.values()
)
RUN_LOCAL_VISUAL_SCOPES = frozenset(
    item["visual_scope"] for item in QUESTION_CATALOG.values()
)
# Every admitted value either denies a numeric oracle or explicitly leaves
# authority with an existing simulator/official signal.  A run-local visual
# question can therefore cross-check evidence but cannot define a new numeric
# success criterion.
RUN_LOCAL_NUMERIC_AUTHORITIES = frozenset(
    item["numeric_authority"] for item in QUESTION_CATALOG.values()
)
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


def is_run_local_phenomenon_id(value: Any) -> bool:
    """Return whether ``value`` is a bounded evaluation-local identifier."""

    return (
        isinstance(value, str)
        and _RUN_LOCAL_PHENOMENON_ID.fullmatch(value) is not None
    )


def validate_run_local_question_spec(value: Any) -> dict[str, Any]:
    """Validate one self-contained visual question generated for this run.

    Run-local questions reuse the same prompt fields and controlled vocabulary
    as the committed catalog.  Only the natural-language binary question is
    new; its declared numeric authority must still point to a pre-existing
    simulator/official signal (or explicitly declare no numeric oracle).
    """

    if not isinstance(value, Mapping) or set(value) != QUESTION_KEYS:
        raise ExecutionVQAQueryError(
            f"run-local question fields must be exactly {sorted(QUESTION_KEYS)}"
        )
    spec = deepcopy(dict(value))
    phenomenon_id = spec.get("id")
    if not is_run_local_phenomenon_id(phenomenon_id):
        raise ExecutionVQAQueryError(
            "run-local question id must match run_local.<lowercase-safe-id>"
        )
    controlled_fields = {
        "question_type": RUN_LOCAL_QUESTION_TYPES,
        "target_role": RUN_LOCAL_TARGET_ROLES,
        "visual_scope": RUN_LOCAL_VISUAL_SCOPES,
        "numeric_authority": RUN_LOCAL_NUMERIC_AUTHORITIES,
    }
    for field, allowed in controlled_fields.items():
        if spec.get(field) not in allowed:
            raise ExecutionVQAQueryError(
                f"run-local question {field} is outside the trusted vocabulary"
            )
    question = spec.get("question")
    if not isinstance(question, str) or question != question.strip():
        raise ExecutionVQAQueryError(
            "run-local question text must be a trimmed string"
        )
    if "\n" in question or "\r" in question:
        raise ExecutionVQAQueryError("run-local question text must be one line")
    if not question.endswith("?"):
        raise ExecutionVQAQueryError("run-local question text must end with ?")
    if not 8 <= len(question) <= RUN_LOCAL_QUESTION_MAX_CHARS:
        raise ExecutionVQAQueryError(
            "run-local question text length must be between 8 and "
            f"{RUN_LOCAL_QUESTION_MAX_CHARS} characters"
        )
    spec["question"] = question
    return spec


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
    proposed_phenomenon_ids: list[str] | None = None,
    proposed_question_specs: list[Mapping[str, Any]] | None = None,
    reviewed_registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic, bounded visual query contract.

    Context fields select committed catalog entries.  A ToolProposal may also
    supply strictly validated ``run_local.*`` question specs; those specs are
    embedded in the query so the saved artifact can be validated without an
    external registry.  Calling this function without context returns the
    original three phenomena in their original order.
    """

    task = _optional_identifier(task_name, field="task_name")
    template = _optional_identifier(template_id, field="template_id")
    aspect = _optional_identifier(sub_aspect, field="sub_aspect")
    metric = _tool_metric(tool_contract)
    context_supplied = any((task, template, aspect, metric)) or any(
        value is not None
        for value in (proposed_phenomenon_ids, proposed_question_specs)
    )
    selected: list[str] = []
    reasons: list[str] = []

    local_questions: dict[str, dict[str, Any]] = {}
    if proposed_question_specs is not None:
        if not isinstance(proposed_question_specs, list) or not proposed_question_specs:
            raise ExecutionVQAQueryError(
                "proposed_question_specs must be a non-empty list"
            )
        for index, raw_spec in enumerate(proposed_question_specs):
            try:
                spec = validate_run_local_question_spec(raw_spec)
            except ExecutionVQAQueryError as exc:
                raise ExecutionVQAQueryError(
                    f"proposed_question_specs[{index}] is invalid: {exc}"
                ) from exc
            phenomenon_id = spec["id"]
            if phenomenon_id in QUESTION_CATALOG:
                raise ExecutionVQAQueryError(
                    "run-local question id collides with the trusted catalog"
                )
            if phenomenon_id in local_questions:
                raise ExecutionVQAQueryError(
                    "proposed_question_specs contains duplicate ids"
                )
            local_questions[phenomenon_id] = spec

    explicit_proposal = (
        proposed_phenomenon_ids is not None or proposed_question_specs is not None
    )
    if proposed_phenomenon_ids is not None:
        if (
            not isinstance(proposed_phenomenon_ids, list)
            or not proposed_phenomenon_ids
            or any(
                not isinstance(item, str) or not item.strip()
                for item in proposed_phenomenon_ids
            )
            or len(proposed_phenomenon_ids)
            != len(set(proposed_phenomenon_ids))
        ):
            raise ExecutionVQAQueryError(
                "proposed_phenomenon_ids must be a non-empty unique string list"
            )
        unknown_proposed = sorted(
            set(proposed_phenomenon_ids)
            - set(QUESTION_CATALOG)
            - set(local_questions)
        )
        if unknown_proposed:
            raise ExecutionVQAQueryError(
                f"ToolProposal references unknown visual phenomena: {unknown_proposed}"
            )
        _append_unique(selected, proposed_phenomenon_ids)
        reasons.append("tool_proposal:explicit_visual_assignment")
        unused_local = sorted(set(local_questions) - set(proposed_phenomenon_ids))
        if unused_local:
            raise ExecutionVQAQueryError(
                "proposed_question_specs contains unselected run-local ids: "
                f"{unused_local}"
            )
    elif local_questions:
        _append_unique(selected, list(local_questions))
    if local_questions:
        reasons.append("tool_proposal:run_local_visual_assignment")

    task_template_key = (task, template)
    adapter_matched = False
    if not explicit_proposal and task is not None and template is not None:
        try:
            contract = resolve_capability_contract(task, template)
        except CapabilityAdapterError:
            contract = None
        if contract is not None:
            adapter_ids = list(contract["vqa"]["phenomenon_ids"])
            unknown = sorted(set(adapter_ids) - set(QUESTION_CATALOG))
            if unknown:
                raise ExecutionVQAQueryError(
                    f"capability adapter references unknown phenomena: {unknown}"
                )
            _append_unique(selected, adapter_ids)
            reasons.append(f"capability_adapter:{task}:{template}")
            adapter_matched = True
    if not explicit_proposal and not adapter_matched and task_template_key in TASK_TEMPLATE_QUESTION_RULES:
        _append_unique(selected, TASK_TEMPLATE_QUESTION_RULES[task_template_key])
        reasons.append(f"task_template:{task}:{template}")
    if not explicit_proposal and not adapter_matched and template in TEMPLATE_QUESTION_RULES:
        _append_unique(selected, TEMPLATE_QUESTION_RULES[template])
        reasons.append(f"template:{template}")
    if not explicit_proposal and aspect:
        for prefix, question_ids in SUB_ASPECT_QUESTION_RULES:
            if aspect == prefix or aspect.startswith(prefix + "."):
                _append_unique(selected, question_ids)
                reasons.append(f"sub_aspect:{prefix}")
                break
    task_metric_key = (task, metric)
    if not explicit_proposal and task_metric_key in TASK_METRIC_QUESTION_RULES:
        _append_unique(selected, TASK_METRIC_QUESTION_RULES[task_metric_key])
        reasons.append(f"task_metric:{task}:{metric}")
    elif not explicit_proposal and metric in METRIC_QUESTION_RULES:
        _append_unique(selected, METRIC_QUESTION_RULES[metric])
        reasons.append(f"tool_metric:{metric}")

    if reviewed_registry_dir is not None and not explicit_proposal:
        from .reviewed_registry import (
            load_reviewed_vqa_query_specs,
            match_reviewed_vqa_query_spec,
        )

        reviewed_entries = load_reviewed_vqa_query_specs(reviewed_registry_dir)
        reviewed = match_reviewed_vqa_query_spec(
            reviewed_entries,
            task_name=task,
            template_id=template,
            sub_aspect=aspect,
            tool_metric=metric,
        )
        if reviewed is not None:
            selected = list(reviewed["spec"]["phenomenon_ids"])
            reasons = [
                "reviewed_vqa_query_spec:"
                f"{reviewed['spec']['spec_id']}:{reviewed['spec_sha256']}"
            ]

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
    for phenomenon_id in selected:
        if phenomenon_id in local_questions:
            questions.append(deepcopy(local_questions[phenomenon_id]))

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
    if any(
        item not in QUESTION_CATALOG and not is_run_local_phenomenon_id(item)
        for item in ids
    ):
        raise ExecutionVQAQueryError(
            "query contains neither a catalog nor run-local phenomenon"
        )
    if any(is_run_local_phenomenon_id(item) for item in ids) and value.get(
        "profile"
    ) != "dynamic_v1":
        raise ExecutionVQAQueryError(
            "run-local questions require query.profile=dynamic_v1"
        )

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
        if phenomenon_id in QUESTION_CATALOG:
            expected = {"id": phenomenon_id, **QUESTION_CATALOG[phenomenon_id]}
            if question != expected:
                raise ExecutionVQAQueryError(
                    f"query.questions[{index}] must equal the trusted catalog entry"
                )
        else:
            try:
                expected = validate_run_local_question_spec(question)
            except ExecutionVQAQueryError as exc:
                raise ExecutionVQAQueryError(
                    f"query.questions[{index}] has invalid run-local spec: {exc}"
                ) from exc
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
