"""Deterministic query semantics and evidence-sufficiency contracts.

The adaptive planner decides *what may be tested next*.  This module owns the
separate question of whether the evidence collected so far is logically
sufficient for the user's query.  Keeping those concerns separate prevents a
route's initial aspect list from silently becoming the stopping rule.

The v1 contract supports claims whose finite-domain semantics can be checked
without another model call:

* ``universal``: every required candidate passes;
* ``existential``: at least one required candidate passes;
* ``failure_enumeration``: every required candidate has a decisive outcome,
  so the complete finite-domain failure set can be reported;
* ``comparative``: two explicitly named groups have enough scored evidence;
* ``diagnostic``: a failure has an evidence-backed diagnosis, or the entire
  required finite domain has been checked without observing a failure.

The backward-compatible v2 extension makes the candidate-domain boundary
explicit.  An open candidate universe may grow as the planner discovers new
concerns.  Existential searches may name either ``pass`` or ``fail`` as their
witness outcome, while exhaustive and worst-case conclusions remain
inconclusive until the candidate universe is explicitly closed.

Version 3 also owns whether an official control is logically required.  This
keeps the control decision with the Query truth contract instead of forcing
every trajectory, observability, or Tool-only question through an unrelated
policy-success control.  Version-1 and version-2 inputs remain accepted and
normalize to version 3 with ``control_requirement="required"``.

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
    {
        "universal",
        "existential",
        "failure_enumeration",
        "comparative",
        "diagnostic",
        "worst_case",
    }
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
_OPEN_CONTRACT_KEYS = _CONTRACT_KEYS | {
    "candidate_universe_closed",
    "existential_witness_outcome",
}
_CONTROL_CONTRACT_KEYS = _OPEN_CONTRACT_KEYS | {"control_requirement"}
_COVERAGE_KEYS = {
    "candidate_ids",
    "minimum_evaluated",
    "minimum_per_group",
}
_EVIDENCE_KEYS = {"candidate_id", "outcome", "score", "diagnosis"}
_EXISTENTIAL_WITNESS_OUTCOMES = frozenset({"pass", "fail"})
_CONTROL_REQUIREMENTS = frozenset({"required", "not_required"})
_CONTROL_REQUIRED_QUERY = re.compile(
    r"\b(?:generaliz|robust|attribute|appearance|pose|position|instance|"
    r"variant|perturb|compare|comparison|versus|worst[- ]?case)\w*\b"
    r"|泛化|鲁棒|属性|外观|姿态|位置|实例|变体|扰动|比较|对比|最差",
    re.IGNORECASE,
)
_OFFICIAL_ONLY_QUERY = re.compile(
    r"\b(?:official (?:scene|task) only|"
    r"only (?:the )?official (?:scene|task)|baseline only)\b"
    r"|\bonly\s+(?:the\s+)?official"
    r"(?:\s+[a-z0-9_.-]+){1,3}\s+task\b"
    r"|只(?:验证|测试)官方|仅(?:验证|测试)官方|"
    r"只用官方(?:场景|任务)|仅用官方(?:场景|任务)",
    re.IGNORECASE,
)
_CONTROL_FREE_QUERY = re.compile(
    r"\b(?:trajectory|telemetry|motion|jerk|oscillat|wobbl|smooth|"
    r"velocity|acceleration|pre[- ]?contact|before contact)\w*\b"
    r"|轨迹|遥测|运动|抖动|急动|平滑|速度|加速度|接触前"
    r"|官方(?:场景|任务)(?:能否|是否)",
    re.IGNORECASE,
)


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
    # These are truth-condition markers, not task/aspect keywords.  Keeping
    # the Chinese forms explicit avoids constraining concern discovery while
    # giving a Chinese Query the same stopping semantics as its English form.
    if (
        any(marker in query for marker in ("列出", "枚举", "汇总", "哪些"))
        and any(marker in query for marker in ("失败", "失效"))
    ):
        return "failure_enumeration"
    chinese_markers = (
        ("worst_case", ("最差", "最弱表现", "表现最低")),
        (
            "failure_enumeration",
            ("列出所有失败", "枚举失败", "全部失败模式", "哪些条件会失败"),
        ),
        ("comparative", ("比较", "对比", "优于", "劣于", "差异")),
        ("universal", ("所有", "全部", "每个", "任意一个", "任何一个")),
        (
            "existential",
            (
                "至少一个",
                "是否有一个",
                "是否存在",
                "存不存在",
                "有没有一种",
                "存在一个",
                "哪一种",
                "哪一个",
                "暴露弱点",
                "找到反例",
            ),
        ),
    )
    for claim_type, markers in chinese_markers:
        if any(marker in query for marker in markers):
            return claim_type
    patterns = (
        (
            "worst_case",
            r"\b(?:worst[- ]?case|worst[- ]performing|lowest[- ]performing)\b"
            r"|最差|最弱表现",
        ),
        (
            "failure_enumeration",
            r"\b(?:enumerate|list|catalog(?:ue)?)\b.{0,48}"
            r"\b(?:failures?|failing|failure\s+modes?)\b"
            r"|\b(?:all|complete)\s+(?:observed\s+)?"
            r"(?:failures?|failing\s+candidates?|failure\s+modes?)\b"
            r"|(?:列出|枚举|汇总).{0,24}(?:失败|失效)"
            r"|(?:哪些|全部|所有).{0,12}(?:候选|条件|模式).{0,12}"
            r"(?:失败|失效)",
        ),
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
            r"|至少(?:有)?一个|是否有一个|是否存在|存不存在"
            r"|有没有一种|存在一个",
        ),
    )
    for claim_type, pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return claim_type
    return "diagnostic"


def infer_existential_witness_outcome(user_query: str) -> str:
    """Return the outcome that satisfies one existential Query.

    Existence-of-success Queries keep the conventional ``pass`` witness.
    Queries asking for a weakness, failure, or counterexample instead require
    a measured ``fail`` witness.
    """

    query = _text(user_query, "user_query").casefold()
    if (
        re.search(
            r"\b(?:fail(?:s|ed|ing|ure)?|weakness|counterexample|breaks?)\b",
            query,
        )
        or any(
            marker in query
            for marker in ("失败", "失效", "弱点", "反例", "崩溃")
        )
    ):
        return "fail"
    return "pass"


def infer_control_requirement(
    user_query: str,
    *,
    semantic_context: Mapping[str, Any] | None = None,
) -> str:
    """Decide whether the Query needs a separate official control rollout.

    Trajectory and telemetry questions can measure the official scene directly.
    Generalization, perturbation, and comparison claims still need a control
    anchor. Ambiguous cases default to ``required``. This decision never rejects
    a Query and an explicit QueryContract remains authoritative.
    """

    query_text = _text(user_query, "user_query")
    if _OFFICIAL_ONLY_QUERY.search(query_text):
        return "not_required"
    if _CONTROL_REQUIRED_QUERY.search(query_text):
        return "required"
    if _CONTROL_FREE_QUERY.search(query_text):
        return "not_required"
    parts = [query_text]
    if semantic_context is not None:
        if not isinstance(semantic_context, Mapping):
            raise QuerySufficiencyError("semantic_context must be an object")
        for field in (
            "sub_aspect",
            "hypothesis",
            "requested_variation",
            "measurement_need",
        ):
            value = semantic_context.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    text = "\n".join(parts)
    if _CONTROL_REQUIRED_QUERY.search(text):
        return "required"
    if _CONTROL_FREE_QUERY.search(text):
        return "not_required"
    return "required"


def query_is_official_only(user_query: str) -> bool:
    """Return whether the Query explicitly limits evaluation to the official task."""

    return _OFFICIAL_ONLY_QUERY.search(_text(user_query, "user_query")) is not None


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
    candidate_universe_closed: bool | None = None,
    existential_witness_outcome: str | None = None,
    control_requirement: str = "required",
) -> dict[str, Any]:
    """Build and validate an explicit query-sufficiency contract.

    The builder emits the canonical version-3 schema.  Legacy version-1 and
    version-2 serialized contracts are accepted by the validator.
    """

    universe = [str(item) for item in candidate_universe]
    required = (
        list(universe)
        if required_candidate_ids is None
        else [str(item) for item in required_candidate_ids]
    )
    resolved_type = str(claim_type or infer_claim_type(user_query))
    if (
        existential_witness_outcome is not None
        and resolved_type != "existential"
    ):
        raise QuerySufficiencyError(
            "existential_witness_outcome is only valid for existential claims"
        )
    use_open_semantics = (
        candidate_universe_closed is not None
        or existential_witness_outcome is not None
        or resolved_type == "worst_case"
    )
    resolved_minimum = (
        0
        if minimum_evaluated is None
        and not required
        and use_open_semantics
        and candidate_universe_closed is False
        else
        len(required)
        if minimum_evaluated is None
        and resolved_type
        in {"universal", "existential", "failure_enumeration", "worst_case"}
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
    contract = {
        "schema_version": 3,
        "claim_type": resolved_type,
        "candidate_universe": universe,
        "required_coverage": {
            "candidate_ids": required,
            "minimum_evaluated": resolved_minimum,
            "minimum_per_group": resolved_group_minimum,
        },
        "round_budget": round_budget,
        "comparison_groups": groups,
        "candidate_universe_closed": (
            True
            if candidate_universe_closed is None
            else candidate_universe_closed
        ),
        "existential_witness_outcome": (
            str(
                existential_witness_outcome
                or infer_existential_witness_outcome(user_query)
            )
            if resolved_type == "existential"
            else None
        ),
        "control_requirement": control_requirement,
    }
    return validate_query_sufficiency_contract(contract)


def validate_query_sufficiency_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a finite-domain sufficiency contract exactly."""

    if not isinstance(value, Mapping):
        raise QuerySufficiencyError(
            "QuerySufficiencyContract must be an object"
        )
    schema_version = value.get("schema_version")
    expected_keys = (
        _CONTRACT_KEYS
        if schema_version == 1
        else _OPEN_CONTRACT_KEYS
        if schema_version == 2
        else _CONTROL_CONTRACT_KEYS
        if schema_version == 3
        else None
    )
    if expected_keys is None:
        raise QuerySufficiencyError(
            "QuerySufficiencyContract schema_version must be 1, 2, or 3"
        )
    if set(value) != expected_keys:
        raise QuerySufficiencyError(
            "QuerySufficiencyContract fields must be exactly "
            f"{sorted(expected_keys)} for schema_version {schema_version}"
        )
    contract = deepcopy(dict(value))
    if schema_version == 1:
        contract["candidate_universe_closed"] = True
        contract["existential_witness_outcome"] = (
            "pass" if contract.get("claim_type") == "existential" else None
        )
    if schema_version in {1, 2}:
        contract["control_requirement"] = "required"
    contract["schema_version"] = 3
    claim_type = contract.get("claim_type")
    if claim_type not in CLAIM_TYPES:
        raise QuerySufficiencyError(
            f"claim_type must be one of {sorted(CLAIM_TYPES)}"
        )
    allow_empty_open_universe = bool(
        contract.get("candidate_universe_closed") is False
    )
    universe = _unique_text_list(
        contract.get("candidate_universe"),
        "candidate_universe",
        allow_empty=allow_empty_open_universe,
    )
    raw_coverage = contract.get("required_coverage")
    if not isinstance(raw_coverage, Mapping) or set(raw_coverage) != _COVERAGE_KEYS:
        raise QuerySufficiencyError(
            f"required_coverage fields must be exactly {sorted(_COVERAGE_KEYS)}"
        )
    required = _unique_text_list(
        raw_coverage.get("candidate_ids"),
        "required_coverage.candidate_ids",
        allow_empty=allow_empty_open_universe,
    )
    outside = sorted(set(required) - set(universe))
    if outside:
        raise QuerySufficiencyError(
            f"required coverage leaves the candidate universe: {outside}"
        )
    minimum = raw_coverage.get("minimum_evaluated")
    minimum_floor = 0 if not required and allow_empty_open_universe else 1
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < minimum_floor
        or minimum > len(required)
    ):
        raise QuerySufficiencyError(
            "required_coverage.minimum_evaluated must be in "
            f"[{minimum_floor}, {len(required)}]"
        )
    if claim_type == "failure_enumeration" and minimum != len(required):
        raise QuerySufficiencyError(
            "failure_enumeration requires every required candidate to be "
            "decisively evaluated"
        )
    if claim_type == "worst_case":
        if schema_version == 1:
            raise QuerySufficiencyError(
                "worst_case requires QuerySufficiencyContract schema_version "
                "2 or 3"
            )
        if minimum != len(required):
            raise QuerySufficiencyError(
                "worst_case requires every required candidate to be "
                "decisively evaluated"
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

    universe_closed = contract.get("candidate_universe_closed")
    if not isinstance(universe_closed, bool):
        raise QuerySufficiencyError(
            "candidate_universe_closed must be bool"
        )
    witness = contract.get("existential_witness_outcome")
    if claim_type == "existential":
        if witness not in _EXISTENTIAL_WITNESS_OUTCOMES:
            raise QuerySufficiencyError(
                "existential_witness_outcome must be pass or fail"
            )
    elif witness is not None:
        raise QuerySufficiencyError(
            "existential_witness_outcome is only valid for existential claims"
        )
    control_requirement = contract.get("control_requirement")
    if control_requirement not in _CONTROL_REQUIREMENTS:
        raise QuerySufficiencyError(
            "control_requirement must be required or not_required"
        )

    contract["candidate_universe"] = universe
    contract["required_coverage"] = {
        "candidate_ids": required,
        "minimum_evaluated": minimum,
        "minimum_per_group": group_minimum,
    }
    contract["comparison_groups"] = groups
    return contract


def extend_query_candidate_universe(
    contract: Mapping[str, Any],
    candidate_ids: Iterable[str],
    *,
    candidate_universe_closed: bool | None = None,
) -> dict[str, Any]:
    """Append dynamically discovered candidates to an open-world contract.

    A legacy finite contract is promoted to v3 and reopened.  Comparative
    contracts are intentionally excluded because a new candidate also needs a
    preregistered group assignment, which this small API cannot infer.
    """

    normalized = validate_query_sufficiency_contract(contract)
    if isinstance(candidate_ids, (str, bytes, bytearray)):
        raise QuerySufficiencyError(
            "candidate_ids must be an iterable of candidate-id strings"
        )
    additions = _unique_text_list(
        [str(item) for item in candidate_ids],
        "candidate_ids",
        allow_empty=True,
    )
    if normalized["claim_type"] == "comparative" and additions:
        raise QuerySufficiencyError(
            "comparative candidate discovery requires an explicit group assignment"
        )
    universe = list(normalized["candidate_universe"])
    new_ids = [item for item in additions if item not in universe]
    universe.extend(new_ids)
    required = list(normalized["required_coverage"]["candidate_ids"])
    previous_required_count = len(required)
    required.extend(item for item in new_ids if item not in required)
    minimum = normalized["required_coverage"]["minimum_evaluated"]
    if (
        (
            normalized["claim_type"]
            in {"universal", "existential", "failure_enumeration", "worst_case"}
            or previous_required_count == 0
        )
        and minimum == previous_required_count
    ):
        minimum = len(required)
    promoted = {
        **normalized,
        "schema_version": 3,
        "candidate_universe": universe,
        "candidate_universe_closed": (
            False
            if candidate_universe_closed is None
            else candidate_universe_closed
        ),
        "existential_witness_outcome": (
            normalized.get("existential_witness_outcome", "pass")
            if normalized["claim_type"] == "existential"
            else None
        ),
        "control_requirement": normalized["control_requirement"],
        "required_coverage": {
            **normalized["required_coverage"],
            "candidate_ids": required,
            "minimum_evaluated": minimum,
        },
    }
    return validate_query_sufficiency_contract(promoted)


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
    universe_closed = normalized.get("candidate_universe_closed", True)
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
        elif not universe_closed:
            rationale = (
                "The universal candidate universe remains open, so the "
                "exhaustive Query verdict must remain inconclusive."
            )
        elif len(passed) == len(required):
            sufficient = True
            verdict = "supported"
            rationale = (
                "Every candidate in the finite required coverage passed."
            )
    elif claim_type == "existential":
        witness_outcome = normalized.get(
            "existential_witness_outcome", "pass"
        )
        witnesses = passed if witness_outcome == "pass" else failed
        non_witnesses = failed if witness_outcome == "pass" else passed
        statistics["existential_witness_outcome"] = witness_outcome
        statistics["witness_candidate_ids"] = list(witnesses)
        if witnesses:
            sufficient = True
            verdict = (
                "counterexample_found"
                if witness_outcome == "fail"
                else "supported"
            )
            rationale = (
                f"A definitive {witness_outcome} candidate witnesses the "
                "existential claim."
            )
        elif universe_closed and len(non_witnesses) == len(required):
            sufficient = True
            verdict = "refuted"
            rationale = (
                "The closed required candidate universe contains no requested "
                "existential witness."
            )
    elif claim_type == "failure_enumeration":
        statistics["failure_candidate_ids"] = list(failed)
        statistics["passed_candidate_ids"] = list(passed)
        if universe_closed and len(decisive) == len(required):
            sufficient = True
            verdict = "failure_set_enumerated"
            rationale = (
                "Every candidate in the finite required coverage has a "
                "decisive outcome, so the complete finite-domain failure set "
                "is known."
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
    elif claim_type == "worst_case":
        if universe_closed and len(decisive) == len(required):
            scores = {
                candidate_id: states[candidate_id]["score"]
                for candidate_id in required
                if states[candidate_id]["score"] is not None
            }
            if len(scores) == len(required):
                worst_score = min(scores.values())
                worst_ids = [
                    candidate_id
                    for candidate_id in required
                    if math.isclose(
                        scores[candidate_id],
                        worst_score,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ]
                sufficient = True
                verdict = "worst_case_observed"
                statistics["worst_score"] = worst_score
                statistics["worst_candidate_ids"] = worst_ids
                rationale = (
                    "Every candidate in the closed required universe has a "
                    "comparable score, so its observed worst case is known."
                )
        elif not universe_closed:
            rationale = (
                "The worst-case candidate universe remains open; an unseen "
                "candidate may have a lower score."
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
        elif universe_closed and len(passed) == len(required):
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
    if should_stop:
        # Keep uncovered candidates in the explicit untested fields below,
        # but never encode them as a next action after either semantic
        # sufficiency or a hard budget stop.
        recommended = []

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
    if not universe_closed:
        limitations.append(
            "The candidate universe is open; exhaustive, no-counterexample, "
            "and worst-case conclusions are not licensed."
        )
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
    if claim_type == "failure_enumeration":
        limitations.append(
            "The enumerated failures are complete only for the explicitly "
            "required finite candidate domain."
        )
    if claim_type == "worst_case":
        limitations.append(
            "Worst-case scores are trusted upstream inputs and are comparable "
            "only under a shared metric, unit, and direction."
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
        "candidate_universe_closed": universe_closed,
        "candidate_discovery_required": bool(
            not universe_closed and not sufficient and budget_remaining > 0
        ),
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
    "extend_query_candidate_universe",
    "infer_claim_type",
    "infer_control_requirement",
    "query_is_official_only",
    "validate_query_sufficiency_contract",
]
