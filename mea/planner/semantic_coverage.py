"""Semantic coverage contracts from an open Query to runtime artifacts.

The planner is allowed to refine an experiment after observing evidence, but
it must not silently replace the first Query-derived concern with an easier
nearby diagnostic before that concern is tested. ``EvaluationIntent`` freezes
that first candidate's semantics before runtime task binding.
``ImplementationTrace`` records whether one generated candidate directly
implements that intent, is only a diagnostic proxy, or is unsupported. Query
answer sufficiency remains owned by ``QueryContract``.

The contract is deliberately small.  It does not attempt to prove natural
language equivalence; it combines explicit planner declarations with
TaskGen/ToolGen validation facts and fails toward an explicit proxy label when
the candidate wording does not preserve the requested change and observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence


class SemanticCoverageError(ValueError):
    """Raised when an intent, alignment, or implementation trace is malformed."""


_INTENT_KEYS = {
    "schema_version",
    "intent_id",
    "source_query",
    "original_concern",
    "hypothesis",
    "requested_change",
    "preserved_conditions",
    "required_observation",
}
_ALIGNMENT_KEYS = {
    "schema_version",
    "relationship",
    "rationale",
    "matched_intent_fields",
    "unmatched_intent_fields",
}
_TRACE_KEYS = {
    "schema_version",
    "intent_id",
    "candidate_id",
    "stage",
    "relationship",
    "rationale",
    "covered_intent_fields",
    "uncovered_intent_fields",
    "pending_intent_fields",
    "coverage_status",
    "repair_required",
    "validation_evidence",
}
_INTENT_REQUIREMENT_FIELDS = (
    "requested_change",
    "preserved_conditions",
    "hypothesis",
    "required_observation",
)
_RELATIONSHIPS = {"direct", "diagnostic_proxy", "unsupported"}
_STAGES = {"candidate", "taskgen", "execution"}
_COVERAGE_STATUSES = {"complete", "partial", "not_covered"}
_TOKEN = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_NO_SCENE_CHANGE = re.compile(
    r"\b(?:keep|reuse|use)\b.{0,40}\b(?:official|unchanged)\b"
    r"|\b(?:official|scene|appearance)\b.{0,40}\bunchanged\b"
    r"|保持.{0,24}(?:官方场景|场景|外观).{0,12}不变"
    r"|复用官方场景",
    re.IGNORECASE,
)
_EXPLICIT_SCENE_CHANGE = re.compile(
    r"\b(?:add|enlarge|increase|move|recolor|reduce|remove|replace|resize|"
    r"rotate|scale|shrink|swap)\w*\b"
    r"|(?:增加|增大|放大|移动|换色|重着色|减小|缩小|移除|替换|旋转|缩放)",
    re.IGNORECASE,
)
_STOPWORDS = {
    "about",
    "after",
    "against",
    "between",
    "check",
    "does",
    "from",
    "into",
    "keep",
    "measure",
    "policy",
    "require",
    "task",
    "that",
    "the",
    "their",
    "then",
    "this",
    "while",
    "with",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticCoverageError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SemanticCoverageError(f"{field} must be a string list")
    result = [
        _text(item, f"{field}[]")
        for item in value
    ]
    if len(result) != len(set(result)):
        raise SemanticCoverageError(f"{field} must not contain duplicates")
    if not allow_empty and not result:
        raise SemanticCoverageError(f"{field} must not be empty")
    return result


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]


def build_evaluation_intent(
    *,
    source_query: str,
    original_concern: str,
    hypothesis: str,
    requested_change: str,
    required_observation: str,
    preserved_conditions: Sequence[str] = (),
) -> dict[str, Any]:
    """Freeze one Query-derived candidate before task/checkpoint binding."""

    payload = {
        "source_query": _text(source_query, "source_query"),
        "original_concern": _text(original_concern, "original_concern"),
        "hypothesis": _text(hypothesis, "hypothesis"),
        "requested_change": _text(requested_change, "requested_change"),
        "preserved_conditions": _string_list(
            preserved_conditions, "preserved_conditions"
        ),
        "required_observation": _text(
            required_observation, "required_observation"
        ),
    }
    return validate_evaluation_intent(
        {
            "schema_version": 1,
            "intent_id": f"intent.{_digest(payload)}",
            **payload,
        }
    )


def evaluation_intent_from_free_concern(
    concern: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an intent from the provider-authored catalog-free concern."""

    if not isinstance(concern, Mapping):
        raise SemanticCoverageError("free concern must be an object")
    source_query = _text(
        concern.get("source_query"),
        "free_concern.source_query",
    )
    requested_variation = _text(
        concern.get("requested_variation"),
        "free_concern.requested_variation",
    )
    preserved_conditions = list(
        dict.fromkeys(
            [
                *_extract_preserved_conditions(source_query),
                *_extract_preserved_conditions(requested_variation),
            ]
        )
    )
    return build_evaluation_intent(
        source_query=source_query,
        original_concern=concern.get("sub_aspect"),
        hypothesis=concern.get("hypothesis"),
        requested_change=requested_variation,
        required_observation=concern.get("measurement_need"),
        preserved_conditions=preserved_conditions,
    )


def _extract_preserved_conditions(requested_change: str) -> list[str]:
    """Preserve explicit keep/fixed clauses instead of discarding them."""

    clauses: list[str] = []
    matched_spans: list[tuple[int, int]] = []
    patterns = (
        re.compile(
            r"\bwhile\s+(?:keeping|holding|leaving|maintaining|preserving)"
            r"\s+(.+?)"
            r"(?:\s+(?:fixed|unchanged|constant))?(?=[.!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:keep|maintain|preserve)\s+(.+?)\s+"
            r"(?:fixed|unchanged|constant)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:^|(?<=[;:.!?]))\s*\b(?:please\s+)?"
            r"(?:keep|maintain|preserve)"
            r"\s+(.+?)(?=[.!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhile\s+(.+?)\s+"
            r"(?:remain(?:s)?|stay(?:s)?|is|are)\s+"
            r"(?:fixed|unchanged|constant)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwith\s+(.+?)\s+(?:fixed|unchanged|constant)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwithout\s+(?:changing|altering|moving|modifying)\s+"
            r"(.+?)(?=[.!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwithout\s+(?:any\s+)?changes?\s+to\s+"
            r"(.+?)(?=[.!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:leave|hold)\s+(.+?)\s+"
            r"(?:fixed|unchanged|constant)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:do\s+not|don't|must\s+not)\s+"
            r"(?:change|alter|move|modify)\s+(.+?)(?=[.!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(r"(?:保持|保留)(.+?)(?:不变|固定)"),
        re.compile(r"(?:不要|不得|不能)(?:改变|移动|修改)(.+?)(?:[。！？]|$)"),
    )
    for pattern in patterns:
        for match in pattern.finditer(requested_change):
            matched_spans.append(match.span())
            clause = re.sub(
                r"\s+(?:fixed|unchanged|constant)\s*$",
                "",
                match.group(1).strip(" ,.;"),
                flags=re.IGNORECASE,
            )
            if clause:
                clauses.append(clause)
    unmatched = list(requested_change)
    for start, end in matched_spans:
        unmatched[start:end] = " " * (end - start)
    preservation_cue = re.search(
        r"(?:^|[;:.!?]\s*)\b(?:please\s+)?"
        r"(?:keep|maintain|preserve)\s+"
        r"|\bwhile\s+(?:keeping|holding|leaving|maintaining|preserving)\s+"
        r"|\b(?:keep|maintain|preserve)\b.{0,80}\b"
        r"(?:fixed|unchanged|constant)\b"
        r"|\b(?:remain|stay)s?\s+(?:fixed|unchanged|constant)\b"
        r"|\bwithout\s+(?:(?:any\s+)?changes?\s+to|"
        r"(?:changing|altering|moving|modifying))\b"
        r"|\b(?:leave|hold)\b.{0,80}\b"
        r"(?:fixed|unchanged|constant)\b"
        r"|\b(?:do\s+not|don't|must\s+not)\s+"
        r"(?:change|alter|move|modify)\b"
        r"|(?:保持|保留).{0,80}(?:不变|固定)"
        r"|(?:不要|不得|不能)(?:改变|移动|修改)",
        "".join(unmatched),
        re.IGNORECASE,
    )
    if preservation_cue:
        raise SemanticCoverageError(
            "explicit preservation clause could not be normalized: "
            f"{requested_change!r}"
        )
    if not clauses:
        return []
    conditions: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        parts = re.split(
            r"\s*,\s*|\s+\b(?:and|or)\b\s+|、|，|和|或",
            clause,
        )
        for part in parts:
            condition = re.sub(
                r"^(?:and|or|及|和|或)\s+",
                "",
                part.strip(),
                flags=re.IGNORECASE,
            ).removeprefix("its ").strip()
            if not condition:
                continue
            semantic_key = re.sub(
                r"^(?:the|its|a|an)\s+",
                "",
                condition.casefold(),
            )
            if semantic_key in seen:
                continue
            seen.add(semantic_key)
            conditions.append(condition)
    return conditions


def validate_evaluation_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INTENT_KEYS:
        raise SemanticCoverageError(
            "EvaluationIntent fields must be exactly "
            f"{sorted(_INTENT_KEYS)}"
        )
    intent = deepcopy(dict(value))
    if intent.get("schema_version") != 1:
        raise SemanticCoverageError("EvaluationIntent schema_version must be 1")
    for field in (
        "intent_id",
        "source_query",
        "original_concern",
        "hypothesis",
        "requested_change",
        "required_observation",
    ):
        intent[field] = _text(intent.get(field), f"EvaluationIntent.{field}")
    intent["preserved_conditions"] = _string_list(
        intent.get("preserved_conditions"),
        "EvaluationIntent.preserved_conditions",
    )
    expected_id = "intent." + _digest(
        {
            key: intent[key]
            for key in (
                "source_query",
                "original_concern",
                "hypothesis",
                "requested_change",
                "preserved_conditions",
                "required_observation",
            )
        }
    )
    if intent["intent_id"] != expected_id:
        raise SemanticCoverageError(
            "EvaluationIntent.intent_id does not match its semantic content"
        )
    return intent


def _semantic_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    result: set[str] = set()
    for token in _TOKEN.findall(normalized):
        if token.isascii():
            if len(token) >= 3 and token not in _STOPWORDS:
                result.add(token)
            continue
        # Character bigrams preserve useful Chinese semantics without a
        # language-specific tokenizer. Single-character runs remain usable.
        if len(token) == 1:
            result.add(token)
        else:
            result.update(
                token[index : index + 2]
                for index in range(len(token) - 1)
            )
    return result


def _semantic_match(requirement: str, implementation_text: str) -> bool:
    requirement_normalized = unicodedata.normalize(
        "NFKC", requirement
    ).casefold().strip()
    implementation_normalized = unicodedata.normalize(
        "NFKC", implementation_text
    ).casefold()
    if requirement_normalized in implementation_normalized:
        return True
    required = _semantic_tokens(requirement)
    implemented = _semantic_tokens(implementation_text)
    if not required:
        return False
    overlap = len(required & implemented)
    minimum = 1 if len(required) <= 2 else max(
        2, math.ceil(len(required) * 0.4)
    )
    return overlap >= minimum


def _requests_no_scene_change(requested_change: str) -> bool:
    """Distinguish an unchanged scene from a changed scene with invariants."""

    return bool(
        _NO_SCENE_CHANGE.search(requested_change)
        and not _EXPLICIT_SCENE_CHANGE.search(requested_change)
    )


def validate_intent_alignment(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ALIGNMENT_KEYS:
        raise SemanticCoverageError(
            "IntentAlignment fields must be exactly "
            f"{sorted(_ALIGNMENT_KEYS)}"
        )
    alignment = deepcopy(dict(value))
    if alignment.get("schema_version") != 1:
        raise SemanticCoverageError("IntentAlignment schema_version must be 1")
    if alignment.get("relationship") not in _RELATIONSHIPS:
        raise SemanticCoverageError(
            f"IntentAlignment.relationship must be one of "
            f"{sorted(_RELATIONSHIPS)}"
        )
    alignment["rationale"] = _text(
        alignment.get("rationale"), "IntentAlignment.rationale"
    )
    for field in ("matched_intent_fields", "unmatched_intent_fields"):
        alignment[field] = _string_list(
            alignment.get(field), f"IntentAlignment.{field}"
        )
        unknown = set(alignment[field]) - set(_INTENT_REQUIREMENT_FIELDS)
        if unknown:
            raise SemanticCoverageError(
                f"IntentAlignment.{field} contains unknown fields: "
                f"{sorted(unknown)}"
            )
    if set(alignment["matched_intent_fields"]) & set(
        alignment["unmatched_intent_fields"]
    ):
        raise SemanticCoverageError(
            "IntentAlignment matched/unmatched fields must be disjoint"
        )
    if set(alignment["matched_intent_fields"]) | set(
        alignment["unmatched_intent_fields"]
    ) != set(_INTENT_REQUIREMENT_FIELDS):
        raise SemanticCoverageError(
            "IntentAlignment must classify every candidate-contract field"
        )
    if (
        alignment["relationship"] == "direct"
        and alignment["unmatched_intent_fields"]
    ):
        raise SemanticCoverageError(
            "direct IntentAlignment cannot leave intent fields unmatched"
        )
    return alignment


def build_candidate_intent_alignment(
    intent: Mapping[str, Any],
    *,
    semantic_concern: str,
    scene_need: Mapping[str, Any] | None,
    checker_need: Mapping[str, Any] | None,
    rule_tool_need: Mapping[str, Any] | None = None,
    vqa_tool_need: Mapping[str, Any] | None = None,
    tool_need: Mapping[str, Any] | None = None,
    declared_relationship: str | None = None,
    declared_rationale: str | None = None,
) -> dict[str, Any]:
    """Classify a candidate conservatively against its frozen intent."""

    normalized = validate_evaluation_intent(intent)
    semantic_text = _text(semantic_concern, "semantic_concern")
    scene_text = (
        str(scene_need.get("description") or "")
        if isinstance(scene_need, Mapping)
        else ""
    )
    checker_text = (
        str(checker_need.get("description") or "")
        if isinstance(checker_need, Mapping)
        else ""
    )
    tool_text = "\n".join(
        str(need.get("description") or "")
        for need in (rule_tool_need, vqa_tool_need, tool_need)
        if isinstance(need, Mapping)
    )
    implementations = {
        "requested_change": "\n".join((semantic_text, scene_text)),
        "preserved_conditions": "\n".join(
            (semantic_text, scene_text, checker_text)
        ),
        "hypothesis": "\n".join((semantic_text, checker_text, tool_text)),
        "required_observation": "\n".join(
            (semantic_text, checker_text, tool_text)
        ),
    }
    matched = []
    for field in _INTENT_REQUIREMENT_FIELDS:
        if field == "requested_change":
            no_scene_change = (
                _requests_no_scene_change(normalized["requested_change"])
            )
            if scene_need is None:
                matched_field = no_scene_change
            elif no_scene_change:
                matched_field = False
            else:
                matched_field = _semantic_match(
                    normalized[field], implementations[field]
                )
        elif field == "preserved_conditions":
            conditions = normalized["preserved_conditions"]
            matched_field = (
                not conditions
                or (
                    scene_need is None
                    and _requests_no_scene_change(
                        normalized["requested_change"]
                    )
                )
                or all(
                    _semantic_match(condition, implementations[field])
                    for condition in conditions
                )
            )
        else:
            matched_field = _semantic_match(
                normalized[field], implementations[field]
            )
        if matched_field:
            matched.append(field)
    unmatched = [
        field for field in _INTENT_REQUIREMENT_FIELDS if field not in matched
    ]
    inferred = "direct" if not unmatched else "diagnostic_proxy"
    relationship = declared_relationship or inferred
    if relationship not in _RELATIONSHIPS:
        raise SemanticCoverageError(
            f"declared relationship must be one of {sorted(_RELATIONSHIPS)}"
        )
    if relationship == "direct" and inferred != "direct":
        raise SemanticCoverageError(
            "candidate cannot declare direct coverage while contract "
            f"fields are unmatched: {unmatched}"
        )
    rationale = (
        _text(declared_rationale, "declared_rationale")
        if declared_rationale is not None
        else (
            "Candidate preserves the requested change, hypothesis, and "
            "observation semantics."
            if relationship == "direct"
            else
            "Candidate is a nearby diagnostic, not a direct implementation "
            f"of candidate-contract fields: {unmatched}."
        )
    )
    return validate_intent_alignment(
        {
            "schema_version": 1,
            "relationship": relationship,
            "rationale": rationale,
            "matched_intent_fields": matched,
            "unmatched_intent_fields": unmatched,
        }
    )


def _taskgen_facts(
    validation: Mapping[str, Any] | None,
) -> dict[str, bool | None]:
    if validation is None:
        return {
            "scene_change_passed": None,
            "checker_fixtures_passed": None,
            "visual_diagnosis_passed": None,
            "preserved_conditions_verified": None,
        }
    preflight = validation.get("preflight")
    if not isinstance(preflight, Mapping):
        preflight = validation
    fixtures = (
        validation.get("checker_fixtures")
        if isinstance(validation.get("checker_fixtures"), list)
        else preflight.get("checker_fixtures")
    )
    fixture_passed = (
        bool(fixtures)
        and all(
            isinstance(item, Mapping) and item.get("passed") is True
            for item in fixtures
        )
        if isinstance(fixtures, list)
        else None
    )
    visual = preflight.get("vision_validation")
    return {
        "scene_change_passed": (
            preflight.get("scene_change_passed")
            if isinstance(preflight.get("scene_change_passed"), bool)
            else None
        ),
        "checker_fixtures_passed": fixture_passed,
        "visual_diagnosis_passed": (
            visual.get("passed")
            if isinstance(visual, Mapping)
            and isinstance(visual.get("passed"), bool)
            else None
        ),
        "preserved_conditions_verified": (
            preflight.get("preserved_conditions_verified")
            if isinstance(
                preflight.get("preserved_conditions_verified"), bool
            )
            else None
        ),
    }


def build_implementation_trace(
    candidate: Mapping[str, Any],
    *,
    taskgen_validation: Mapping[str, Any] | None = None,
    tool_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project semantic and runtime coverage for one candidate.

    Candidates predating ``EvaluationIntent`` return ``None`` unchanged.
    """

    intent_value = candidate.get("evaluation_intent")
    alignment_value = candidate.get("intent_alignment")
    if intent_value is None and alignment_value is None:
        return None
    intent = validate_evaluation_intent(intent_value)
    alignment = validate_intent_alignment(alignment_value)
    candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
    relationship = alignment["relationship"]
    facts = _taskgen_facts(taskgen_validation)
    tool_passed = (
        tool_evaluation.get("status") == "passed"
        if isinstance(tool_evaluation, Mapping)
        else None
    )
    stage = (
        "execution"
        if tool_evaluation is not None
        else "taskgen"
        if taskgen_validation is not None
        else "candidate"
    )
    covered: list[str] = []
    uncovered = list(alignment["unmatched_intent_fields"])
    pending: list[str] = []
    if relationship != "direct":
        uncovered = list(_INTENT_REQUIREMENT_FIELDS)
    else:
        scene_requested = candidate.get("scene_need") is not None
        checker_requested = candidate.get("checker_need") is not None
        taskgen_requested = scene_requested or checker_requested
        observation_requested = any(
            candidate.get(field) is not None
            for field in (
                "scene_need",
                "checker_need",
                "rule_tool_need",
                "vqa_tool_need",
                "tool_need",
            )
        )
        preserve_required = bool(intent["preserved_conditions"])
        if not preserve_required or not taskgen_requested:
            covered.append("preserved_conditions")
        if taskgen_validation is None:
            (
                pending if scene_requested else covered
            ).append("requested_change")
            (
                pending if checker_requested else covered
            ).append("hypothesis")
            if (
                preserve_required
                and taskgen_requested
                and "preserved_conditions"
                in alignment["matched_intent_fields"]
            ):
                pending.append("preserved_conditions")
        else:
            scene_valid = (
                not scene_requested
                or (
                    facts["scene_change_passed"] is True
                    and facts["visual_diagnosis_passed"] is not False
                )
            )
            checker_valid = (
                not checker_requested
                or facts["checker_fixtures_passed"] is True
            )
            (covered if scene_valid else uncovered).append(
                "requested_change"
            )
            if preserve_required and taskgen_requested:
                preserve_verified = facts["preserved_conditions_verified"]
                (
                    covered
                    if preserve_verified is True
                    else uncovered
                    if preserve_verified is False
                    else pending
                ).append("preserved_conditions")
            (covered if checker_valid else uncovered).append("hypothesis")
        if tool_evaluation is None:
            (
                pending if observation_requested else uncovered
            ).append("required_observation")
        else:
            (
                covered
                if observation_requested and tool_passed
                else uncovered
            ).append("required_observation")
    covered = list(dict.fromkeys(covered))
    uncovered = list(dict.fromkeys(uncovered))
    pending = [
        field
        for field in dict.fromkeys(pending)
        if field not in covered and field not in uncovered
    ]
    status = (
        "complete"
        if len(covered) == len(_INTENT_REQUIREMENT_FIELDS)
        and not uncovered
        and not pending
        else "partial"
        if covered
        else "not_covered"
    )
    taskgen_owned_uncovered = set(uncovered) & {
        "requested_change",
        "preserved_conditions",
        "hypothesis",
    }
    trace = {
        "schema_version": 1,
        "intent_id": intent["intent_id"],
        "candidate_id": candidate_id,
        "stage": stage,
        "relationship": relationship,
        "rationale": alignment["rationale"],
        "covered_intent_fields": covered,
        "uncovered_intent_fields": uncovered,
        "pending_intent_fields": pending,
        "coverage_status": status,
        "repair_required": bool(
            relationship == "direct"
            and taskgen_validation is not None
            and taskgen_owned_uncovered
        ),
        "validation_evidence": {
            **facts,
            "tool_evaluation_passed": tool_passed,
        },
    }
    return validate_implementation_trace(trace)


def validate_implementation_trace(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TRACE_KEYS:
        raise SemanticCoverageError(
            "ImplementationTrace fields must be exactly "
            f"{sorted(_TRACE_KEYS)}"
        )
    trace = deepcopy(dict(value))
    if trace.get("schema_version") != 1:
        raise SemanticCoverageError(
            "ImplementationTrace schema_version must be 1"
        )
    for field in ("intent_id", "candidate_id", "rationale"):
        trace[field] = _text(
            trace.get(field), f"ImplementationTrace.{field}"
        )
    if trace.get("stage") not in _STAGES:
        raise SemanticCoverageError(
            f"ImplementationTrace.stage must be one of {sorted(_STAGES)}"
        )
    if trace.get("relationship") not in _RELATIONSHIPS:
        raise SemanticCoverageError(
            "ImplementationTrace.relationship is invalid"
        )
    classified: set[str] = set()
    for field in (
        "covered_intent_fields",
        "uncovered_intent_fields",
        "pending_intent_fields",
    ):
        trace[field] = _string_list(
            trace.get(field), f"ImplementationTrace.{field}"
        )
        unknown = set(trace[field]) - set(_INTENT_REQUIREMENT_FIELDS)
        if unknown:
            raise SemanticCoverageError(
                f"ImplementationTrace.{field} contains unknown fields: "
                f"{sorted(unknown)}"
            )
        if classified & set(trace[field]):
            raise SemanticCoverageError(
                "ImplementationTrace coverage fields must be disjoint"
            )
        classified.update(trace[field])
    if classified != set(_INTENT_REQUIREMENT_FIELDS):
        raise SemanticCoverageError(
            "ImplementationTrace must classify every candidate-contract field"
        )
    if trace.get("coverage_status") not in _COVERAGE_STATUSES:
        raise SemanticCoverageError(
            "ImplementationTrace.coverage_status is invalid"
        )
    if not isinstance(trace.get("repair_required"), bool):
        raise SemanticCoverageError(
            "ImplementationTrace.repair_required must be bool"
        )
    evidence = trace.get("validation_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "scene_change_passed",
        "checker_fixtures_passed",
        "visual_diagnosis_passed",
        "preserved_conditions_verified",
        "tool_evaluation_passed",
    }:
        raise SemanticCoverageError(
            "ImplementationTrace.validation_evidence has invalid fields"
        )
    if any(
        item is not None and not isinstance(item, bool)
        for item in evidence.values()
    ):
        raise SemanticCoverageError(
            "ImplementationTrace validation evidence must be bool or null"
        )
    trace["validation_evidence"] = deepcopy(dict(evidence))
    return trace


def advance_implementation_trace_with_tool(
    trace: Mapping[str, Any],
    tool_evaluation: Mapping[str, Any] | None,
    *,
    rule_required: bool = True,
    vqa_evaluation: Mapping[str, Any] | None = None,
    vqa_required: bool = False,
) -> dict[str, Any]:
    """Advance a trace using only the independently requested evidence types."""

    advanced = validate_implementation_trace(trace)
    rule_passed = (
        isinstance(tool_evaluation, Mapping)
        and tool_evaluation.get("status") == "passed"
    )
    vqa_passed = (
        isinstance(vqa_evaluation, Mapping)
        and vqa_evaluation.get("status") == "passed"
    )
    passed = bool(
        (not rule_required or rule_passed)
        and (not vqa_required or vqa_passed)
        and (rule_required or vqa_required)
    )
    advanced["stage"] = "execution"
    advanced["validation_evidence"]["tool_evaluation_passed"] = passed
    field = "required_observation"
    for key in (
        "covered_intent_fields",
        "uncovered_intent_fields",
        "pending_intent_fields",
    ):
        advanced[key] = [
            item for item in advanced[key] if item != field
        ]
    if advanced["relationship"] == "direct" and passed:
        advanced["covered_intent_fields"].append(field)
    else:
        advanced["uncovered_intent_fields"].append(field)
    advanced["coverage_status"] = (
        "complete"
        if len(advanced["covered_intent_fields"])
        == len(_INTENT_REQUIREMENT_FIELDS)
        else "partial"
        if advanced["covered_intent_fields"]
        else "not_covered"
    )
    return validate_implementation_trace(advanced)


__all__ = [
    "SemanticCoverageError",
    "advance_implementation_trace_with_tool",
    "build_candidate_intent_alignment",
    "build_evaluation_intent",
    "build_implementation_trace",
    "evaluation_intent_from_free_concern",
    "validate_evaluation_intent",
    "validate_implementation_trace",
    "validate_intent_alignment",
]
