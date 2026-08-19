"""Runtime experiment candidates discovered from an open Query.

An ``ExperimentCandidate`` is the semantic hand-off between planning and the
TaskGen, ToolGen, and VQA stages.  It deliberately carries no catalog template
identifier.  Each materialization need is independently optional and typed, so a
trajectory-only Query can request a Rule Tool without inventing a new scene,
checker, or VQA Tool, while an official-task-only Query explicitly requests
reuse of the simulator's success predicate.  Legacy ``tool_need``
candidates remain accepted and are normalized to the independent Rule/VQA
representation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Any, Mapping

from .semantic_coverage import (
    SemanticCoverageError,
    build_candidate_intent_alignment,
    validate_evaluation_intent,
    validate_intent_alignment,
)

class ExperimentCandidateError(ValueError):
    """Raised when a runtime experiment candidate is malformed."""


_CANDIDATE_BASE_KEYS = {
    "schema_version",
    "candidate_id",
    "source_query",
    "base_task",
    "semantic_concern",
}
_LEGACY_CANDIDATE_KEYS = _CANDIDATE_BASE_KEYS | {
    "scene_need",
    "checker_need",
    "tool_need",
}
_TYPED_CANDIDATE_KEYS = _CANDIDATE_BASE_KEYS | {
    "scene_need",
    "checker_need",
    "rule_tool_need",
    "vqa_tool_need",
}
_CANONICAL_CANDIDATE_KEYS = _TYPED_CANDIDATE_KEYS | {"tool_need"}
_SEMANTIC_CANDIDATE_KEYS = {"evaluation_intent", "intent_alignment"}
_NEED_KEYS = {"kind", "description", "reuse_first"}
_SCENE_NEED_KEYS = _NEED_KEYS | {"controlled_changes"}
_SCENE_DELTA_KEYS = {
    "actor",
    "property",
    "axis",
    "signed_delta",
    "unit",
    "reference",
}
_NEED_KINDS = frozenset({"reuse", "adapt", "generate", "measure", "vqa"})
_FIELD_KINDS = {
    "scene_need": frozenset({"reuse", "adapt", "generate"}),
    "checker_need": frozenset({"reuse", "adapt", "generate"}),
    "rule_tool_need": frozenset({"reuse", "adapt", "generate", "measure"}),
    "vqa_tool_need": frozenset({"reuse", "adapt", "generate", "vqa"}),
    # Accepted only on the compatibility input boundary.
    "tool_need": frozenset({"reuse", "adapt", "generate", "measure", "vqa"}),
}
_LEGACY_NEED_KIND = {
    "scene_need": "adapt",
    "checker_need": "generate",
    "rule_tool_need": "measure",
    "vqa_tool_need": "vqa",
    "tool_need": "measure",
}
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentCandidateError(f"{field} must be a non-empty string")
    return value.strip()


def _candidate_id(value: Any) -> str:
    candidate_id = _text(value, "ExperimentCandidate.candidate_id")
    if not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ExperimentCandidateError(
            "ExperimentCandidate.candidate_id must contain only letters, "
            "digits, dot, underscore, colon, or hyphen"
        )
    return candidate_id


def validate_scene_delta(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one executable same-seed position delta from the Plan Agent."""

    if not isinstance(value, Mapping) or set(value) != _SCENE_DELTA_KEYS:
        raise ExperimentCandidateError(
            "controlled scene delta fields must be exactly "
            f"{sorted(_SCENE_DELTA_KEYS)}"
        )
    actor = _text(value.get("actor"), "controlled scene delta.actor")
    if value.get("property") != "position":
        raise ExperimentCandidateError(
            "controlled scene delta.property must be position"
        )
    axis = value.get("axis")
    if axis not in {"x", "y", "z"}:
        raise ExperimentCandidateError(
            "controlled scene delta.axis must be x, y, or z"
        )
    signed_delta = value.get("signed_delta")
    if (
        isinstance(signed_delta, bool)
        or not isinstance(signed_delta, (int, float))
        or not math.isfinite(float(signed_delta))
        or float(signed_delta) == 0.0
    ):
        raise ExperimentCandidateError(
            "controlled scene delta.signed_delta must be a finite non-zero "
            "number"
        )
    if value.get("unit") != "m":
        raise ExperimentCandidateError(
            "controlled scene delta.unit must be m"
        )
    if value.get("reference") != "same_seed_official_reset":
        raise ExperimentCandidateError(
            "controlled scene delta.reference must be "
            "same_seed_official_reset"
        )
    return {
        "actor": actor,
        "property": "position",
        "axis": axis,
        "signed_delta": float(signed_delta),
        "unit": "m",
        "reference": "same_seed_official_reset",
    }


def validate_scene_deltas(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ExperimentCandidateError(
            "ExperimentCandidate.scene_need.controlled_changes must be a "
            "non-empty list"
        )
    result = [validate_scene_delta(item) for item in value]
    identities = [
        (item["actor"], item["property"], item["axis"])
        for item in result
    ]
    if len(identities) != len(set(identities)):
        raise ExperimentCandidateError(
            "controlled scene deltas must target unique actor/property/axis "
            "tuples"
        )
    return result


def _slug(value: str, *, field: str) -> str:
    slug = _SLUG_SEPARATOR.sub(".", _text(value, field).casefold()).strip(".")
    if not slug:
        raise ExperimentCandidateError(f"{field} cannot produce a candidate id")
    return slug


def _need(
    value: Any,
    *,
    field: str,
    legacy_string: bool,
) -> dict[str, Any] | None:
    if value is None:
        if legacy_string:
            raise ExperimentCandidateError(
                f"ExperimentCandidate v1 {field} must be a non-empty string"
            )
        return None
    if isinstance(value, str):
        description = _text(value, f"ExperimentCandidate.{field}")
        return {
            "kind": _LEGACY_NEED_KIND[field],
            "description": description,
            "reuse_first": field
            in {"tool_need", "rule_tool_need", "vqa_tool_need"},
        }
    allowed_shapes = (
        (_NEED_KEYS, _SCENE_NEED_KEYS)
        if field == "scene_need"
        else (_NEED_KEYS,)
    )
    if not isinstance(value, Mapping) or set(value) not in allowed_shapes:
        raise ExperimentCandidateError(
            f"ExperimentCandidate.{field} must be null, a legacy string, or "
            f"an object with exactly one of "
            f"{[sorted(shape) for shape in allowed_shapes]}"
        )
    need = deepcopy(dict(value))
    kind = need.get("kind")
    if kind not in _NEED_KINDS or kind not in _FIELD_KINDS[field]:
        raise ExperimentCandidateError(
            f"ExperimentCandidate.{field}.kind must be one of "
            f"{sorted(_FIELD_KINDS[field])}"
        )
    need["description"] = _text(
        need.get("description"),
        f"ExperimentCandidate.{field}.description",
    )
    if not isinstance(need.get("reuse_first"), bool):
        raise ExperimentCandidateError(
            f"ExperimentCandidate.{field}.reuse_first must be bool"
        )
    if field == "scene_need" and "controlled_changes" in need:
        need["controlled_changes"] = validate_scene_deltas(
            need["controlled_changes"]
        )
    return need


def experiment_candidate_need_kinds(
    value: Mapping[str, Any],
) -> frozenset[str]:
    """Return the stages explicitly requested by one normalized candidate."""

    candidate = validate_experiment_candidate(value)
    return frozenset(
        field.removesuffix("_need")
        for field in (
            "scene_need",
            "checker_need",
            "rule_tool_need",
            "vqa_tool_need",
        )
        if candidate[field] is not None
    )


def validate_experiment_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the planner-to-generation hand-off.

    Version 1 required three strings and version 2 originally exposed one
    combined ``tool_need``.  Both shapes remain accepted.  The normalized
    version-2 object carries independent Rule/VQA needs plus a deprecated
    single-tool alias for older read-only callers.
    """

    supplied_keys = set(value) if isinstance(value, Mapping) else set()
    semantic_keys = supplied_keys & _SEMANTIC_CANDIDATE_KEYS
    shape_keys = supplied_keys - _SEMANTIC_CANDIDATE_KEYS
    if (
        not isinstance(value, Mapping)
        or semantic_keys not in (set(), _SEMANTIC_CANDIDATE_KEYS)
        or frozenset(shape_keys)
        not in {
            frozenset(_LEGACY_CANDIDATE_KEYS),
            frozenset(_TYPED_CANDIDATE_KEYS),
            frozenset(_CANONICAL_CANDIDATE_KEYS),
        }
    ):
        raise ExperimentCandidateError(
            "ExperimentCandidate fields must be exactly one of the accepted "
            "shapes: legacy "
            f"{sorted(_LEGACY_CANDIDATE_KEYS)} or typed "
            f"{sorted(_TYPED_CANDIDATE_KEYS)} shape"
        )
    candidate = deepcopy(dict(value))
    schema_version = candidate.get("schema_version")
    if schema_version not in {1, 2}:
        raise ExperimentCandidateError(
            "ExperimentCandidate.schema_version must be 1 or 2"
        )
    candidate["candidate_id"] = _candidate_id(candidate.get("candidate_id"))
    for field in ("source_query", "base_task", "semantic_concern"):
        candidate[field] = _text(
            candidate.get(field), f"ExperimentCandidate.{field}"
        )
    for field in ("scene_need", "checker_need"):
        candidate[field] = _need(
            candidate.get(field),
            field=field,
            legacy_string=schema_version == 1,
        )

    if "rule_tool_need" in candidate or "vqa_tool_need" in candidate:
        rule_tool_need = _need(
            candidate.get("rule_tool_need"),
            field="rule_tool_need",
            legacy_string=False,
        )
        vqa_tool_need = _need(
            candidate.get("vqa_tool_need"),
            field="vqa_tool_need",
            legacy_string=False,
        )
    else:
        legacy_tool_need = _need(
            candidate.get("tool_need"),
            field="tool_need",
            legacy_string=schema_version == 1,
        )
        if (
            legacy_tool_need is not None
            and legacy_tool_need["kind"] == "vqa"
        ):
            rule_tool_need = None
            vqa_tool_need = {**legacy_tool_need, "kind": "vqa"}
        else:
            rule_tool_need = legacy_tool_need
            vqa_tool_need = None
    candidate["rule_tool_need"] = rule_tool_need
    candidate["vqa_tool_need"] = vqa_tool_need
    if not any(
        need is not None
        for need in (
            candidate["scene_need"],
            candidate["checker_need"],
            rule_tool_need,
            vqa_tool_need,
        )
    ):
        raise ExperimentCandidateError(
            "ExperimentCandidate must request at least one typed need"
        )

    # This alias is intentionally lossy when both Tool types are requested.
    # Production runtime code must consume the two canonical fields.
    candidate["tool_need"] = deepcopy(
        rule_tool_need if rule_tool_need is not None else vqa_tool_need
    )
    candidate["schema_version"] = 2
    normalized = {
        key: candidate[key]
        for key in (
            "schema_version",
            "candidate_id",
            "source_query",
            "base_task",
            "semantic_concern",
            "scene_need",
            "checker_need",
            "rule_tool_need",
            "vqa_tool_need",
            "tool_need",
        )
    }
    if semantic_keys:
        try:
            intent = validate_evaluation_intent(
                candidate["evaluation_intent"]
            )
            supplied_alignment = validate_intent_alignment(
                candidate["intent_alignment"]
            )
            alignment = build_candidate_intent_alignment(
                intent,
                semantic_concern=candidate["semantic_concern"],
                scene_need=candidate["scene_need"],
                checker_need=candidate["checker_need"],
                rule_tool_need=rule_tool_need,
                vqa_tool_need=vqa_tool_need,
                declared_relationship=supplied_alignment["relationship"],
                declared_rationale=supplied_alignment["rationale"],
            )
        except SemanticCoverageError as exc:
            raise ExperimentCandidateError(
                f"invalid semantic coverage contract: {exc}"
            ) from exc
        if intent["source_query"] != candidate["source_query"]:
            raise ExperimentCandidateError(
                "EvaluationIntent.source_query differs from candidate source_query"
            )
        normalized["evaluation_intent"] = intent
        normalized["intent_alignment"] = alignment
    return normalized


def build_experiment_candidate(
    *,
    source_query: str,
    base_task: str,
    semantic_concern: str,
    scene_need: str | Mapping[str, Any] | None = None,
    checker_need: str | Mapping[str, Any] | None = None,
    tool_need: str | Mapping[str, Any] | None = None,
    rule_tool_need: str | Mapping[str, Any] | None = None,
    vqa_tool_need: str | Mapping[str, Any] | None = None,
    candidate_id: str | None = None,
    evaluation_intent: Mapping[str, Any] | None = None,
    intent_relationship: str | None = None,
    intent_relationship_rationale: str | None = None,
) -> dict[str, Any]:
    """Build one catalog-independent runtime experiment candidate."""

    task = _text(base_task, "base_task")
    concern = _text(semantic_concern, "semantic_concern")
    scene = _need(scene_need, field="scene_need", legacy_string=False)
    checker = _need(checker_need, field="checker_need", legacy_string=False)
    if tool_need is not None and (
        rule_tool_need is not None or vqa_tool_need is not None
    ):
        raise ExperimentCandidateError(
            "legacy tool_need cannot be combined with rule_tool_need or "
            "vqa_tool_need"
        )
    legacy_tool = _need(
        tool_need, field="tool_need", legacy_string=False
    )
    rule_tool = _need(
        rule_tool_need, field="rule_tool_need", legacy_string=False
    )
    vqa_tool = _need(
        vqa_tool_need, field="vqa_tool_need", legacy_string=False
    )
    if legacy_tool is not None:
        if legacy_tool["kind"] == "vqa":
            vqa_tool = {**legacy_tool, "kind": "vqa"}
        else:
            rule_tool = legacy_tool
    normalized_intent: dict[str, Any] | None = None
    if evaluation_intent is not None:
        try:
            normalized_intent = validate_evaluation_intent(evaluation_intent)
        except SemanticCoverageError as exc:
            raise ExperimentCandidateError(
                f"invalid semantic coverage contract: {exc}"
            ) from exc
    experiment_digest = hashlib.sha256(
        json.dumps(
            {
                "base_task": task,
                "semantic_concern": concern,
                "scene_need": scene,
                "checker_need": checker,
                "rule_tool_need": rule_tool,
                "vqa_tool_need": vqa_tool,
                "evaluation_intent": normalized_intent,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    resolved_id = (
        _candidate_id(candidate_id)
        if candidate_id is not None
        else "dynamic."
        + _slug(task, field="base_task")
        + "."
        + _slug(concern, field="semantic_concern")
        + "."
        + experiment_digest
    )
    value: dict[str, Any] = {
        "schema_version": 2,
        "candidate_id": resolved_id,
        "source_query": source_query,
        "base_task": task,
        "semantic_concern": concern,
        "scene_need": scene,
        "checker_need": checker,
        "rule_tool_need": rule_tool,
        "vqa_tool_need": vqa_tool,
    }
    if normalized_intent is not None:
        try:
            intent = normalized_intent
            alignment = build_candidate_intent_alignment(
                intent,
                semantic_concern=concern,
                scene_need=scene,
                checker_need=checker,
                rule_tool_need=rule_tool,
                vqa_tool_need=vqa_tool,
                declared_relationship=intent_relationship,
                declared_rationale=intent_relationship_rationale,
            )
        except SemanticCoverageError as exc:
            raise ExperimentCandidateError(
                f"invalid semantic coverage contract: {exc}"
            ) from exc
        value["evaluation_intent"] = intent
        value["intent_alignment"] = alignment
    elif (
        intent_relationship is not None
        or intent_relationship_rationale is not None
    ):
        raise ExperimentCandidateError(
            "intent relationship requires evaluation_intent"
        )
    return validate_experiment_candidate(value)


__all__ = [
    "ExperimentCandidateError",
    "build_experiment_candidate",
    "experiment_candidate_need_kinds",
    "validate_scene_delta",
    "validate_scene_deltas",
    "validate_experiment_candidate",
]
