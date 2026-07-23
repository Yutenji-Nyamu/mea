"""Deterministic answer scope and fail-closed limitation projection."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping


class AnswerScopeError(ValueError):
    """Raised when final feedback omits an evidence-required limitation."""


_SCOPE_KEYS = {
    "schema_version",
    "sample_count",
    "seeds",
    "tested_candidate_ids",
    "untested_candidate_ids",
    "unsupported_capabilities",
    "evidence_conflict",
    "termination",
    "claim_verdict",
    "required_limitations",
}
_LIMITATION_KEYS = {"code", "text"}
_TERMINATIONS = {
    "evidence_sufficient",
    "budget_exhausted",
    "continue",
    "pipeline_invalid",
    "unknown",
}


def _dedupe(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _collect_seeds(evidence: Mapping[str, Any]) -> list[int]:
    raw: list[Any] = []
    if "seed" in evidence:
        raw.append(evidence.get("seed"))
    if isinstance(evidence.get("seeds"), list):
        raw.extend(evidence["seeds"])
    for round_evidence in evidence.get("rounds", []):
        if not isinstance(round_evidence, Mapping):
            continue
        if isinstance(round_evidence.get("seeds"), list):
            raw.extend(round_evidence["seeds"])
        for episode in round_evidence.get("episodes", []):
            if isinstance(episode, Mapping):
                raw.append(episode.get("seed"))
    seeds: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value not in seeds:
            seeds.append(value)
    return seeds


def _sample_count(evidence: Mapping[str, Any], seeds: list[int]) -> int | None:
    aggregate = None
    observations = evidence.get("observations")
    if isinstance(observations, Mapping):
        aggregate = observations.get("aggregate")
    if not isinstance(aggregate, Mapping):
        aggregate = evidence.get("aggregate")
    # Prefer execution metadata because Aggregate may include expert and policy
    # cohorts in the same unique-episode count.
    for value in (evidence.get("total_episodes"),):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    round_counts = [
        item.get("num_episodes")
        for item in evidence.get("rounds", [])
        if isinstance(item, Mapping)
    ]
    if round_counts and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in round_counts
    ):
        return sum(round_counts)
    value = evidence.get("num_episodes")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    value = (aggregate or {}).get("unique_episode_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return len(seeds) if seeds else None


def _execution_conflict(evidence: Mapping[str, Any]) -> bool:
    observations = evidence.get("observations")
    if isinstance(observations, Mapping) and observations.get(
        "execution_vqa_conflict"
    ) is True:
        return True
    direct = evidence.get("execution_vqa")
    if isinstance(direct, Mapping) and direct.get("evidence_conflict") is True:
        return True
    assessment = _query_assessment(evidence)
    if assessment is not None:
        candidates = assessment.get("conflict_candidate_ids")
        if isinstance(candidates, list) and bool(candidates):
            return True
    return any(
        isinstance(item, Mapping)
        and isinstance(item.get("execution_vqa"), Mapping)
        and item["execution_vqa"].get("evidence_conflict") is True
        for item in evidence.get("rounds", [])
    )


def _unsupported_capabilities(evidence: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    limitations = evidence.get("limitations")
    if isinstance(limitations, Mapping):
        values.extend(
            limitations.get("global_route_unsupported_capabilities") or []
        )
    global_route = evidence.get("global_query_route")
    if isinstance(global_route, Mapping):
        selection = global_route.get("selection")
        if isinstance(selection, Mapping):
            values.extend(selection.get("unsupported_capabilities") or [])
    normalized: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            text = value.strip()
        elif isinstance(value, Mapping):
            task_name = value.get("task_name")
            aspect_id = value.get("aspect_id")
            if not isinstance(task_name, str) or not isinstance(aspect_id, str):
                continue
            text = f"{task_name}:{aspect_id}"
        else:
            continue
        if text not in normalized:
            normalized.append(text)
    return normalized


def _query_assessment(evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("query_sufficiency", "query_sufficiency_assessment"):
        value = evidence.get(key)
        if isinstance(value, Mapping):
            return value
    plan = evidence.get("plan")
    if isinstance(plan, Mapping):
        value = plan.get("query_sufficiency")
        if isinstance(value, Mapping):
            return value
    return None


def _tested_candidates(evidence: Mapping[str, Any]) -> list[str]:
    assessment = _query_assessment(evidence)
    if assessment is not None and isinstance(
        assessment.get("observed_candidate_ids"), list
    ):
        return _dedupe(
            [
                str(item)
                for item in assessment["observed_candidate_ids"]
                if isinstance(item, str) and item
            ]
        )
    plan = evidence.get("plan")
    if isinstance(plan, Mapping) and isinstance(
        plan.get("completed_template_ids"), list
    ):
        return _dedupe(
            [
                str(item)
                for item in plan["completed_template_ids"]
                if isinstance(item, str) and item
            ]
        )
    result = []
    for item in evidence.get("rounds", []):
        if not isinstance(item, Mapping):
            continue
        round_plan = item.get("round_plan")
        sources = [item, round_plan] if isinstance(round_plan, Mapping) else [item]
        candidate = next(
            (
                source.get("template_id")
                for source in sources
                if isinstance(source.get("template_id"), str)
                and source.get("template_id")
            ),
            None,
        )
        if candidate is not None and candidate not in result:
            result.append(candidate)
    return result


def _untested_candidates(evidence: Mapping[str, Any]) -> list[str]:
    assessment = _query_assessment(evidence)
    if assessment is not None and isinstance(
        assessment.get("untested_candidate_ids"), list
    ):
        return _dedupe(
            [
                str(item)
                for item in assessment["untested_candidate_ids"]
                if isinstance(item, str) and item
            ]
        )
    plan = evidence.get("plan")
    if isinstance(plan, Mapping) and isinstance(
        plan.get("remaining_template_ids"), list
    ):
        return _dedupe(
            [
                str(item)
                for item in plan["remaining_template_ids"]
                if isinstance(item, str) and item
            ]
        )
    return []


def _termination(evidence: Mapping[str, Any]) -> tuple[str, str | None]:
    observations = evidence.get("observations")
    if isinstance(observations, Mapping) and observations.get(
        "pipeline_passed"
    ) is False:
        return "pipeline_invalid", None
    assessment = _query_assessment(evidence)
    if assessment is not None:
        reason = assessment.get("stop_reason")
        verdict = assessment.get("claim_verdict")
        sufficient = assessment.get("evidence_sufficient")
        should_stop = assessment.get("should_stop")
        valid_query_stop = (
            reason == "evidence_sufficient"
            and sufficient is True
            and should_stop is True
        ) or (
            reason == "budget_exhausted"
            and sufficient is False
            and should_stop is True
        ) or (
            reason == "continue"
            and sufficient is False
            and should_stop is False
        )
        if valid_query_stop:
            return str(reason), str(verdict) if verdict is not None else None
        raise AnswerScopeError(
            "query sufficiency assessment has an inconsistent stop verdict"
        )
    plan = evidence.get("plan")
    if isinstance(plan, Mapping):
        budget_remaining = plan.get("round_budget_remaining")
        planning_state = plan.get("planning_state")
        if (
            budget_remaining == 0
            and isinstance(planning_state, str)
            and planning_state.startswith("stopped_after_round_")
        ):
            return "budget_exhausted", None
    return "unknown", None


def _canonical_limitations(
    *,
    sample_count: int | None,
    seeds: list[int],
    untested: list[str],
    unsupported: list[str],
    conflict: bool,
    termination: str,
) -> list[dict[str, str]]:
    limitations = [
        {
            "code": "sample_count_and_seeds",
            "text": (
                f"Evidence contains N={sample_count} policy episodes at seeds "
                f"{seeds}."
                if sample_count is not None and seeds
                else f"Evidence contains N={sample_count} policy episodes; seeds are unavailable."
                if sample_count is not None
                else f"Policy episode count is unavailable; observed seeds are {seeds}."
                if seeds
                else "Policy episode count and seeds are unavailable."
            ),
        }
    ]
    if untested:
        limitations.append(
            {
                "code": "untested_candidates",
                "text": f"Untested candidates remain: {untested}.",
            }
        )
    if unsupported:
        limitations.append(
            {
                "code": "unsupported_capabilities",
                "text": f"Unsupported requested capabilities remain: {unsupported}.",
            }
        )
    if conflict:
        limitations.append(
            {
                "code": "evidence_conflict",
                "text": "Execution VQA conflicts with another evidence source; the conflict remains unresolved.",
            }
        )
    termination_text = {
        "evidence_sufficient": (
            "The run stopped because the finite query-sufficiency contract was "
            "satisfied; this is not a statistical generalization guarantee."
        ),
        "budget_exhausted": (
            "The run stopped because its round budget was exhausted before the "
            "query-sufficiency contract was satisfied."
        ),
        "continue": (
            "The query-sufficiency contract requires more evidence; the current "
            "answer is interim."
        ),
        "pipeline_invalid": (
            "The evaluation pipeline is invalid, so it cannot support a policy "
            "performance conclusion."
        ),
        "unknown": (
            "No validated query-sufficiency stop verdict is present in the evidence."
        ),
    }[termination]
    limitations.append(
        {
            "code": f"termination_{termination}",
            "text": termination_text,
        }
    )
    return limitations


def build_answer_scope(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Project facts that every final answer must expose as limitations."""

    if not isinstance(evidence, Mapping):
        raise AnswerScopeError("evidence must be an object")
    seeds = _collect_seeds(evidence)
    sample_count = _sample_count(evidence, seeds)
    tested = _tested_candidates(evidence)
    untested = _untested_candidates(evidence)
    unsupported = _unsupported_capabilities(evidence)
    conflict = _execution_conflict(evidence)
    termination, claim_verdict = _termination(evidence)
    limitations = _canonical_limitations(
        sample_count=sample_count,
        seeds=seeds,
        untested=untested,
        unsupported=unsupported,
        conflict=conflict,
        termination=termination,
    )
    return validate_answer_scope(
        {
            "schema_version": 1,
            "sample_count": sample_count,
            "seeds": seeds,
            "tested_candidate_ids": tested,
            "untested_candidate_ids": untested,
            "unsupported_capabilities": unsupported,
            "evidence_conflict": conflict,
            "termination": termination,
            "claim_verdict": claim_verdict,
            "required_limitations": limitations,
        }
    )


def validate_answer_scope(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SCOPE_KEYS:
        raise AnswerScopeError(
            f"AnswerScope fields must be exactly {sorted(_SCOPE_KEYS)}"
        )
    scope = deepcopy(dict(value))
    if scope.get("schema_version") != 1:
        raise AnswerScopeError("AnswerScope schema_version must be 1")
    count = scope.get("sample_count")
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count < 0
    ):
        raise AnswerScopeError("sample_count must be a non-negative integer or null")
    seeds = scope.get("seeds")
    if (
        not isinstance(seeds, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise AnswerScopeError("seeds must be a unique integer list")
    for field in (
        "tested_candidate_ids",
        "untested_candidate_ids",
        "unsupported_capabilities",
    ):
        values = scope.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or len(values) != len(set(values))
        ):
            raise AnswerScopeError(f"{field} must be a unique string list")
    if set(scope["tested_candidate_ids"]) & set(scope["untested_candidate_ids"]):
        raise AnswerScopeError("tested and untested candidates must be disjoint")
    if not isinstance(scope.get("evidence_conflict"), bool):
        raise AnswerScopeError("evidence_conflict must be boolean")
    if scope.get("termination") not in _TERMINATIONS:
        raise AnswerScopeError(
            f"termination must be one of {sorted(_TERMINATIONS)}"
        )
    verdict = scope.get("claim_verdict")
    if verdict is not None and (not isinstance(verdict, str) or not verdict):
        raise AnswerScopeError("claim_verdict must be a non-empty string or null")
    limitations = scope.get("required_limitations")
    if not isinstance(limitations, list) or not limitations:
        raise AnswerScopeError("required_limitations must be a non-empty list")
    codes = []
    for index, item in enumerate(limitations):
        if not isinstance(item, Mapping) or set(item) != _LIMITATION_KEYS:
            raise AnswerScopeError(
                f"required_limitations[{index}] must contain code and text"
            )
        code = item.get("code")
        text = item.get("text")
        if not isinstance(code, str) or not code:
            raise AnswerScopeError(f"required_limitations[{index}].code is invalid")
        if not isinstance(text, str) or not text:
            raise AnswerScopeError(f"required_limitations[{index}].text is invalid")
        codes.append(code)
    if len(codes) != len(set(codes)):
        raise AnswerScopeError("required limitation codes must be unique")
    expected_limitations = _canonical_limitations(
        sample_count=count,
        seeds=seeds,
        untested=scope["untested_candidate_ids"],
        unsupported=scope["unsupported_capabilities"],
        conflict=scope["evidence_conflict"],
        termination=scope["termination"],
    )
    if limitations != expected_limitations:
        raise AnswerScopeError(
            "required_limitations do not match the structured answer scope"
        )
    return scope


def project_answer_scope(
    feedback: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Append canonical limitations and bind their codes to the final answer."""

    if not isinstance(feedback, Mapping):
        raise AnswerScopeError("feedback must be an object")
    normalized_scope = validate_answer_scope(scope)
    projected = deepcopy(dict(feedback))
    limitations = projected.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) or not item for item in limitations
    ):
        raise AnswerScopeError("feedback.limitations must be a string list")
    canonical = [
        item["text"] for item in normalized_scope["required_limitations"]
    ]
    projected["limitations"] = _dedupe([*limitations, *canonical])
    projected["answer_scope"] = normalized_scope
    projected["limitation_codes"] = [
        item["code"] for item in normalized_scope["required_limitations"]
    ]
    return validate_answer_scope_projection(projected, normalized_scope)


def validate_answer_scope_projection(
    feedback: Mapping[str, Any],
    expected_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed if structured scope or any canonical limitation is absent."""

    if not isinstance(feedback, Mapping):
        raise AnswerScopeError("feedback must be an object")
    embedded = feedback.get("answer_scope")
    if not isinstance(embedded, Mapping):
        raise AnswerScopeError("feedback is missing structured answer_scope")
    scope = validate_answer_scope(embedded)
    if expected_scope is not None and scope != validate_answer_scope(expected_scope):
        raise AnswerScopeError("feedback answer_scope differs from evidence")
    required_codes = [item["code"] for item in scope["required_limitations"]]
    codes = feedback.get("limitation_codes")
    if codes != required_codes:
        raise AnswerScopeError(
            "feedback limitation_codes do not exactly match answer_scope"
        )
    limitations = feedback.get("limitations")
    if not isinstance(limitations, list):
        raise AnswerScopeError("feedback.limitations must be a list")
    missing = [
        item["code"]
        for item in scope["required_limitations"]
        if item["text"] not in limitations
    ]
    if missing:
        raise AnswerScopeError(
            f"feedback omitted evidence-required limitations: {missing}"
        )
    conclusion_parts = [
        feedback.get("answer"),
        feedback.get("evaluation_scope"),
        *(feedback.get("findings") or []),
    ]
    conclusion = "\n".join(
        item for item in conclusion_parts if isinstance(item, str)
    )
    contradictions: list[str] = []
    if scope["termination"] != "evidence_sufficient" and re.search(
        r"\b(?:the\s+)?evidence\s+(?:is|was)\s+sufficient\b"
        r"|\bsufficient\s+evidence\b"
        r"|证据(?:已经|已)?(?:充分|足够)"
        r"|足以(?:证明|建立).{0,12}泛化",
        conclusion,
        re.IGNORECASE,
    ):
        contradictions.append("claims evidence sufficiency without that stop verdict")
    if scope["termination"] == "evidence_sufficient" and re.search(
        r"\bbudget\s+(?:was\s+)?exhausted\b|\bhard\s+cap\b"
        r"|预算(?:已经|已)?耗尽|达到.{0,8}(?:轮次|预算)上限",
        conclusion,
        re.IGNORECASE,
    ):
        contradictions.append("claims budget exhaustion despite evidence sufficiency")
    if scope["untested_candidate_ids"] and re.search(
        r"\b(?:all|every)\s+(?:candidate|variant|condition)s?\s+"
        r"(?:was|were|has\s+been|have\s+been)?\s*tested\b"
        r"|所有.{0,12}(?:候选|变体|条件).{0,8}(?:已测试|测试完|已覆盖)"
        r"|全部.{0,12}(?:候选|变体|条件).{0,8}(?:已测试|测试完|已覆盖)",
        conclusion,
        re.IGNORECASE,
    ):
        contradictions.append("claims complete testing while candidates remain")
    if scope["unsupported_capabilities"] and re.search(
        r"\bno\s+unsupported\s+(?:capabilit(?:y|ies)|request)"
        r"|\ball\s+requested\s+capabilities\s+(?:are|were)\s+supported\b"
        r"|没有不支持.{0,8}(?:能力|请求)|所有.{0,12}能力.{0,8}(?:均)?支持",
        conclusion,
        re.IGNORECASE,
    ):
        contradictions.append("denies recorded unsupported capabilities")
    if scope["evidence_conflict"] and re.search(
        r"\bno\s+evidence\s+conflicts?\b"
        r"|\bevidence\s+sources?\s+(?:agree|are\s+consistent)\b"
        r"|证据(?:来源)?(?:没有|无)冲突|证据(?:来源)?(?:完全)?一致",
        conclusion,
        re.IGNORECASE,
    ):
        contradictions.append("denies the recorded evidence conflict")
    if contradictions:
        raise AnswerScopeError(
            "feedback contradicts structured answer_scope: "
            + "; ".join(contradictions)
        )
    return deepcopy(dict(feedback))


__all__ = [
    "AnswerScopeError",
    "build_answer_scope",
    "project_answer_scope",
    "validate_answer_scope",
    "validate_answer_scope_projection",
]
