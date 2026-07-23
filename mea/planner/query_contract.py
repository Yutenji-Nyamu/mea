"""Deterministic query semantics and evidence-sufficiency contracts.

The adaptive planner decides *what may be tested next*.  This module owns the
separate question of whether the evidence collected so far is logically
sufficient for the user's query.  Keeping those concerns separate prevents a
route's initial aspect list from silently becoming the stopping rule.

The contract deliberately supports only claims whose finite-domain semantics
can be checked without another model call:

* ``universal``: every required candidate passes;
* ``existential``: at least one required candidate passes;
* ``comparative``: two explicitly named groups have enough scored evidence;
* ``diagnostic``: a failure has an evidence-backed diagnosis, or the entire
  required finite domain has been checked without observing a failure.

This is an MEA reliability extension, not a contract defined by the paper.
The paper describes a small dynamically discovered aspect set and stopping
after sufficient evidence, but does not formalize quantified truth conditions.
This module makes that otherwise implicit decision auditable.  It is a bounded
protocol, not a statistical generalization guarantee.

The prototype trusts upstream evidence normalization.  In particular,
comparative scores are assumed to share a preregistered metric, unit, and
direction, while diagnostic text is assumed to have already passed an
independent evidence/causal review.  This module does not establish either
assumption by itself.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Iterable, Mapping


class QuerySufficiencyError(ValueError):
    """Raised when a query contract or its evidence is malformed."""


CLAIM_TYPES = frozenset(
    {"universal", "existential", "comparative", "diagnostic"}
)
OUTCOMES = frozenset({"pass", "fail", "unknown", "conflict"})

_CONTRACT_KEYS = {
    "schema_version",
    "claim_type",
    "candidate_universe",
    "required_coverage",
    "round_budget",
    "comparison_groups",
}
_COVERAGE_KEYS = {
    "candidate_ids",
    "minimum_evaluated",
    "minimum_per_group",
}
_EVIDENCE_KEYS = {"candidate_id", "outcome", "score", "diagnosis"}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuerySufficiencyError(f"{field} must be a non-empty string")
    return value.strip()


def _unique_text_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise QuerySufficiencyError(f"{field} must be a {qualifier} string list")
    result = [_text(item, f"{field}[]") for item in value]
    if len(result) != len(set(result)):
        raise QuerySufficiencyError(f"{field} must not contain duplicates")
    return result


def infer_claim_type(user_query: str) -> str:
    """Conservatively infer a finite-domain claim type from an open query.

    Ambiguous questions default to ``diagnostic``.  This avoids treating a
    generic "generalization" question as a universal theorem.
    """

    query = _text(user_query, "user_query").casefold()
    patterns = (
        (
            "comparative",
            r"\b(compare|comparison|versus|vs\.?|better|worse|difference)\b"
            r"|比较|对比|优于|劣于|差异",
        ),
        (
            "universal",
            r"\b(all|every|each|across\s+all|for\s+any)\b"
            r"|所有|全部|每个|任意一个|任何一个",
        ),
        (
            "existential",
            r"\b(any\s+one|at\s+least\s+one|exists?|some)\b"
            r"|至少(?:有)?一个|是否有一个|存在一个",
        ),
    )
    for claim_type, pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return claim_type
    return "diagnostic"


def build_query_sufficiency_contract(
    user_query: str,
    *,
    candidate_universe: Iterable[str],
    required_candidate_ids: Iterable[str] | None = None,
    round_budget: int,
    claim_type: str | None = None,
    minimum_evaluated: int | None = None,
    comparison_groups: Mapping[str, Iterable[str]] | None = None,
    minimum_per_group: int | None = None,
) -> dict[str, Any]:
    """Build and validate an explicit query-sufficiency contract."""

    universe = [str(item) for item in candidate_universe]
    required = (
        list(universe)
        if required_candidate_ids is None
        else [str(item) for item in required_candidate_ids]
    )
    resolved_type = str(claim_type or infer_claim_type(user_query))
    resolved_minimum = (
        len(required)
        if minimum_evaluated is None
        and resolved_type in {"universal", "existential"}
        else 1
        if minimum_evaluated is None
        else minimum_evaluated
    )
    resolved_group_minimum = (
        1
        if resolved_type == "comparative" and minimum_per_group is None
        else minimum_per_group
    )
    groups = (
        {
            str(name): [str(item) for item in candidate_ids]
            for name, candidate_ids in comparison_groups.items()
        }
        if comparison_groups is not None
        else None
    )
    return validate_query_sufficiency_contract(
        {
            "schema_version": 1,
            "claim_type": resolved_type,
            "candidate_universe": universe,
            "required_coverage": {
                "candidate_ids": required,
                "minimum_evaluated": resolved_minimum,
                "minimum_per_group": resolved_group_minimum,
            },
            "round_budget": round_budget,
            "comparison_groups": groups,
        }
    )


def validate_query_sufficiency_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a finite-domain sufficiency contract exactly."""

    if not isinstance(value, Mapping) or set(value) != _CONTRACT_KEYS:
        raise QuerySufficiencyError(
            f"QuerySufficiencyContract fields must be exactly {sorted(_CONTRACT_KEYS)}"
        )
    contract = deepcopy(dict(value))
    if contract.get("schema_version") != 1:
        raise QuerySufficiencyError(
            "QuerySufficiencyContract schema_version must be 1"
        )
    claim_type = contract.get("claim_type")
    if claim_type not in CLAIM_TYPES:
        raise QuerySufficiencyError(
            f"claim_type must be one of {sorted(CLAIM_TYPES)}"
        )
    universe = _unique_text_list(
        contract.get("candidate_universe"), "candidate_universe"
    )
    raw_coverage = contract.get("required_coverage")
    if not isinstance(raw_coverage, Mapping) or set(raw_coverage) != _COVERAGE_KEYS:
        raise QuerySufficiencyError(
            f"required_coverage fields must be exactly {sorted(_COVERAGE_KEYS)}"
        )
    required = _unique_text_list(
        raw_coverage.get("candidate_ids"),
        "required_coverage.candidate_ids",
    )
    outside = sorted(set(required) - set(universe))
    if outside:
        raise QuerySufficiencyError(
            f"required coverage leaves the candidate universe: {outside}"
        )
    minimum = raw_coverage.get("minimum_evaluated")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 1
        or minimum > len(required)
    ):
        raise QuerySufficiencyError(
            "required_coverage.minimum_evaluated must be in "
            f"[1, {len(required)}]"
        )
    budget = contract.get("round_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise QuerySufficiencyError("round_budget must be a positive integer")

    raw_groups = contract.get("comparison_groups")
    group_minimum = raw_coverage.get("minimum_per_group")
    groups: dict[str, list[str]] | None = None
    if claim_type == "comparative":
        if not isinstance(raw_groups, Mapping) or len(raw_groups) != 2:
            raise QuerySufficiencyError(
                "comparative claims require exactly two comparison_groups"
            )
        groups = {
            _text(name, "comparison_groups key"): _unique_text_list(
                list(candidate_ids),
                f"comparison_groups.{name}",
            )
            for name, candidate_ids in raw_groups.items()
        }
        flat = [item for candidate_ids in groups.values() for item in candidate_ids]
        if len(flat) != len(set(flat)):
            raise QuerySufficiencyError("comparison_groups must be disjoint")
        if set(flat) != set(required):
            raise QuerySufficiencyError(
                "comparison_groups must partition required_coverage.candidate_ids"
            )
        smallest_group = min(len(items) for items in groups.values())
        if (
            isinstance(group_minimum, bool)
            or not isinstance(group_minimum, int)
            or group_minimum < 1
            or group_minimum > smallest_group
        ):
            raise QuerySufficiencyError(
                "minimum_per_group must be a positive integer no larger than "
                "the smallest comparison group"
            )
    else:
        if raw_groups is not None or group_minimum is not None:
            raise QuerySufficiencyError(
                "comparison_groups and minimum_per_group are only valid for "
                "comparative claims"
            )

    contract["candidate_universe"] = universe
    contract["required_coverage"] = {
        "candidate_ids": required,
        "minimum_evaluated": minimum,
        "minimum_per_group": group_minimum,
    }
    contract["comparison_groups"] = groups
    return contract


def _validate_candidate_evidence(
    value: Mapping[str, Any],
    *,
    universe: set[str],
    index: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_KEYS:
        raise QuerySufficiencyError(
            f"candidate_evidence[{index}] fields must be exactly "
            f"{sorted(_EVIDENCE_KEYS)}"
        )
    item = deepcopy(dict(value))
    candidate_id = _text(item.get("candidate_id"), f"candidate_evidence[{index}].candidate_id")
    if candidate_id not in universe:
        raise QuerySufficiencyError(
            f"candidate evidence leaves the candidate universe: {candidate_id!r}"
        )
    outcome = item.get("outcome")
    if outcome not in OUTCOMES:
        raise QuerySufficiencyError(
            f"candidate_evidence[{index}].outcome must be one of {sorted(OUTCOMES)}"
        )
    score = item.get("score")
    if score is not None and (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise QuerySufficiencyError(
            f"candidate_evidence[{index}].score must be finite or null"
        )
    diagnosis = item.get("diagnosis")
    if diagnosis is not None:
        diagnosis = _text(
            diagnosis, f"candidate_evidence[{index}].diagnosis"
        )
    item["candidate_id"] = candidate_id
    item["score"] = None if score is None else float(score)
    item["diagnosis"] = diagnosis
    return item


def _candidate_states(
    evidence: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        by_candidate.setdefault(item["candidate_id"], []).append(item)
    states: dict[str, dict[str, Any]] = {}
    for candidate_id, records in by_candidate.items():
        decisive = {item["outcome"] for item in records} & {"pass", "fail"}
        explicit_conflict = any(item["outcome"] == "conflict" for item in records)
        if explicit_conflict or len(decisive) > 1:
            outcome = "conflict"
        elif decisive:
            outcome = next(iter(decisive))
        else:
            outcome = "unknown"
        scored = [item["score"] for item in records if item["score"] is not None]
        score = (
            sum(scored) / len(scored)
            if scored
            else 1.0
            if outcome == "pass"
            else 0.0
            if outcome == "fail"
            else None
        )
        diagnoses = [
            item["diagnosis"] for item in records if item["diagnosis"] is not None
        ]
        states[candidate_id] = {
            "outcome": outcome,
            "score": score,
            "diagnoses": diagnoses,
            "observation_count": len(records),
        }
    return states


def assess_query_sufficiency(
    contract: Mapping[str, Any],
    candidate_evidence: Iterable[Mapping[str, Any]],
    *,
    completed_rounds: int | None = None,
) -> dict[str, Any]:
    """Apply asymmetric finite-domain stopping semantics to cached evidence."""

    normalized = validate_query_sufficiency_contract(contract)
    raw_evidence = list(candidate_evidence)
    evidence = [
        _validate_candidate_evidence(
            item,
            universe=set(normalized["candidate_universe"]),
            index=index,
        )
        for index, item in enumerate(raw_evidence)
    ]
    rounds = len(evidence) if completed_rounds is None else completed_rounds
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise QuerySufficiencyError("completed_rounds must be a non-negative integer")
    if rounds < len(evidence):
        raise QuerySufficiencyError(
            "completed_rounds cannot be smaller than candidate evidence count"
        )
    if rounds > normalized["round_budget"]:
        raise QuerySufficiencyError(
            "completed_rounds exceeds the query sufficiency round budget"
        )

    states = _candidate_states(evidence)
    required = list(normalized["required_coverage"]["candidate_ids"])
    decisive = [
        candidate_id
        for candidate_id in required
        if states.get(candidate_id, {}).get("outcome") in {"pass", "fail"}
    ]
    passed = [
        candidate_id
        for candidate_id in required
        if states.get(candidate_id, {}).get("outcome") == "pass"
    ]
    failed = [
        candidate_id
        for candidate_id in required
        if states.get(candidate_id, {}).get("outcome") == "fail"
    ]
    conflicts = [
        candidate_id
        for candidate_id in required
        if states.get(candidate_id, {}).get("outcome") == "conflict"
    ]
    unknown = [
        candidate_id
        for candidate_id in required
        if states.get(candidate_id, {}).get("outcome") == "unknown"
    ]
    untested_required = [
        candidate_id for candidate_id in required if candidate_id not in states
    ]

    claim_type = normalized["claim_type"]
    sufficient = False
    verdict = "inconclusive"
    statistics: dict[str, Any] = {}
    rationale = "The query contract still has unresolved required evidence."

    if claim_type == "universal":
        if failed:
            sufficient = True
            verdict = "refuted"
            rationale = (
                "A definitive failing candidate falsifies the universal claim."
            )
        elif len(passed) == len(required):
            sufficient = True
            verdict = "supported"
            rationale = (
                "Every candidate in the finite required coverage passed."
            )
    elif claim_type == "existential":
        if passed:
            sufficient = True
            verdict = "supported"
            rationale = (
                "A definitive passing candidate witnesses the existential claim."
            )
        elif len(failed) == len(required):
            sufficient = True
            verdict = "refuted"
            rationale = (
                "Every candidate in the finite required coverage failed."
            )
    elif claim_type == "comparative":
        group_statistics: dict[str, Any] = {}
        enough_groups = True
        minimum_per_group = normalized["required_coverage"]["minimum_per_group"]
        for name, group_candidates in normalized["comparison_groups"].items():
            group_scores = [
                states[candidate_id]["score"]
                for candidate_id in group_candidates
                if states.get(candidate_id, {}).get("outcome")
                in {"pass", "fail"}
                and states[candidate_id]["score"] is not None
            ]
            group_statistics[name] = {
                "evaluated": len(group_scores),
                "mean_score": (
                    sum(group_scores) / len(group_scores) if group_scores else None
                ),
            }
            if len(group_scores) < minimum_per_group:
                enough_groups = False
        statistics["comparison_groups"] = group_statistics
        if (
            enough_groups
            and len(decisive)
            >= normalized["required_coverage"]["minimum_evaluated"]
        ):
            names = list(normalized["comparison_groups"])
            first_mean = group_statistics[names[0]]["mean_score"]
            second_mean = group_statistics[names[1]]["mean_score"]
            sufficient = True
            if math.isclose(first_mean, second_mean, rel_tol=0.0, abs_tol=1e-12):
                verdict = "tie_observed"
            elif first_mean > second_mean:
                verdict = f"{names[0]}_higher_observed"
            else:
                verdict = f"{names[1]}_higher_observed"
            rationale = (
                "Both comparison groups meet the preregistered finite evidence "
                "minimum; the verdict describes only their observed scores."
            )
    else:
        diagnosed_failures = [
            candidate_id
            for candidate_id in failed
            if states[candidate_id]["diagnoses"]
        ]
        statistics["diagnosed_failure_candidate_ids"] = diagnosed_failures
        if (
            diagnosed_failures
            and len(decisive)
            >= normalized["required_coverage"]["minimum_evaluated"]
        ):
            sufficient = True
            verdict = "diagnosed"
            rationale = (
                "A measured failure has an evidence-backed diagnosis and the "
                "minimum diagnostic coverage is met."
            )
        elif len(passed) == len(required):
            sufficient = True
            verdict = "no_failure_observed"
            rationale = (
                "The entire finite required domain was checked and no failure "
                "was observed; this does not prove failures are impossible."
            )

    budget_remaining = max(normalized["round_budget"] - rounds, 0)
    if sufficient:
        should_stop = True
        stop_reason = "evidence_sufficient"
    elif budget_remaining <= 0:
        should_stop = True
        stop_reason = "budget_exhausted"
        rationale = (
            "The bounded rollout budget ended before the query sufficiency "
            "contract was satisfied."
        )
    else:
        should_stop = False
        stop_reason = "continue"

    diagnostic_repeats = (
        [
            candidate_id
            for candidate_id in failed
            if not states[candidate_id]["diagnoses"]
        ]
        if claim_type == "diagnostic"
        else []
    )
    recommended = [
        *conflicts,
        *unknown,
        *untested_required,
        *diagnostic_repeats,
    ]
    recommended = list(dict.fromkeys(recommended))
    if claim_type == "comparative" and not sufficient:
        group_stats = statistics.get("comparison_groups", {})
        minimum_per_group = normalized["required_coverage"]["minimum_per_group"]
        for name, candidates in normalized["comparison_groups"].items():
            if group_stats.get(name, {}).get("evaluated", 0) >= minimum_per_group:
                continue
            for candidate_id in candidates:
                if candidate_id not in decisive and candidate_id not in recommended:
                    recommended.append(candidate_id)
    if not recommended and not should_stop:
        recommended = [
            candidate_id
            for candidate_id in normalized["candidate_universe"]
            if candidate_id not in states
        ]

    observed = [
        candidate_id
        for candidate_id in normalized["candidate_universe"]
        if candidate_id in states
    ]
    untested = [
        candidate_id
        for candidate_id in normalized["candidate_universe"]
        if candidate_id not in states
    ]
    limitations = [
        (
            "This is a finite-domain stopping prototype, not a statistical "
            "generalization guarantee."
        )
    ]
    if claim_type == "comparative":
        limitations.append(
            "Comparative scores are trusted upstream inputs; their metric, "
            "unit, direction, and cross-group comparability must be "
            "preregistered and are not independently validated here."
        )
    if claim_type == "diagnostic":
        limitations.append(
            "Diagnosis strings are trusted upstream evidence labels; this "
            "contract does not independently infer or validate causality."
        )
    return {
        "schema_version": 1,
        "contract": normalized,
        "should_stop": should_stop,
        "stop_reason": stop_reason,
        "claim_verdict": verdict,
        "evidence_sufficient": sufficient,
        "completed_rounds": rounds,
        "round_budget": normalized["round_budget"],
        "budget_remaining": budget_remaining,
        "observed_candidate_ids": observed,
        "decisive_candidate_ids": decisive,
        "conflict_candidate_ids": conflicts,
        "unknown_candidate_ids": unknown,
        "untested_required_candidate_ids": untested_required,
        "untested_candidate_ids": untested,
        "recommended_candidate_ids": recommended,
        "rationale": rationale,
        "statistics": statistics,
        "limitations": limitations,
    }


__all__ = [
    "CLAIM_TYPES",
    "OUTCOMES",
    "QuerySufficiencyError",
    "assess_query_sufficiency",
    "build_query_sufficiency_contract",
    "infer_claim_type",
    "validate_query_sufficiency_contract",
]
