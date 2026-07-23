"""Pre-registered, artifact-only fixed/adaptive efficiency comparisons.

The protocol deliberately separates three things that are easy to conflate:

* a matched design (the two arms freeze the same scientific identity);
* resource accounting (starts and retries, not successful completions); and
* an efficiency result (which is meaningful only when the original Query
  conclusion agrees).

Nothing in this module starts a provider, simulator, expert, probe, or ACT
rollout.  Synthetic fixtures exercise the contract without becoming empirical
policy evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.planner.query_contract import (
    QuerySufficiencyError,
    build_query_sufficiency_contract,
    validate_query_sufficiency_contract,
)


PROTOCOL = "matched_fixed_adaptive_efficiency_v1"
FIXED_STRATEGY = "fixed_predeclared_v1"
ADAPTIVE_STRATEGY = "adaptive_query_sufficiency_v1"
MATCHED_FIELDS = (
    "query",
    "checkpoint",
    "candidate_suite",
    "seeds",
    "max_budget",
    "sufficiency_contract",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_KINDS = {"synthetic_fixture", "cached_artifact", "live_rollout"}
_VERDICTS = {
    "supported_in_tested_scope",
    "not_supported_in_tested_scope",
    "inconclusive",
}


class MatchedEfficiencyError(ValueError):
    """Raised when a fixed/adaptive pair violates its frozen contract."""


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MatchedEfficiencyError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(text):
        raise MatchedEfficiencyError(f"{field} must be a non-empty identifier")
    return text


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MatchedEfficiencyError(f"{field} must be non-empty text")
    return value.strip()


def _count(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MatchedEfficiencyError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise MatchedEfficiencyError(
            f"{field} must be a finite number >= {minimum}"
        )
    return float(value)


def _checkpoint(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedEfficiencyError(f"{field} must be an object")
    checkpoint_id = _text(value.get("checkpoint_id"), field=f"{field}.checkpoint_id")
    artifact_sha256 = value.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or not _SHA256.fullmatch(
        artifact_sha256
    ):
        raise MatchedEfficiencyError(
            f"{field}.artifact_sha256 must be 64 lowercase hex characters"
        )
    return {
        "checkpoint_id": checkpoint_id,
        "artifact_sha256": artifact_sha256,
    }


def _candidate_suite(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise MatchedEfficiencyError(f"{field} must be a non-empty list")
    result = [
        _identifier(candidate, field=f"{field}[{index}]")
        for index, candidate in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise MatchedEfficiencyError(f"{field} contains duplicate candidates")
    return result


def _seeds(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise MatchedEfficiencyError(f"{field} must be a non-empty list")
    result = [
        _count(seed, field=f"{field}[{index}]")
        for index, seed in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise MatchedEfficiencyError(f"{field} contains duplicate seeds")
    return result


def _max_budget(value: Any, *, field: str) -> dict[str, Any]:
    expected = {
        "act_episode_starts",
        "expert_starts",
        "probe_starts",
        "provider_retries",
        "wall_seconds",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MatchedEfficiencyError(
            f"{field} fields must be exactly {sorted(expected)}"
        )
    return {
        "act_episode_starts": _count(
            value["act_episode_starts"],
            field=f"{field}.act_episode_starts",
            minimum=1,
        ),
        "expert_starts": _count(
            value["expert_starts"], field=f"{field}.expert_starts"
        ),
        "probe_starts": _count(
            value["probe_starts"], field=f"{field}.probe_starts"
        ),
        "provider_retries": _count(
            value["provider_retries"], field=f"{field}.provider_retries"
        ),
        "wall_seconds": _finite(
            value["wall_seconds"], field=f"{field}.wall_seconds", minimum=0.001
        ),
    }


def _sufficiency_contract(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedEfficiencyError(f"{field} must be an object")
    try:
        result = validate_query_sufficiency_contract(value)
    except QuerySufficiencyError as exc:
        raise MatchedEfficiencyError(
            f"{field} is not a valid QuerySufficiencyContract: {exc}"
        ) from exc
    # The canonical contract itself is frozen; no parallel contract_id exists.
    canonical_sha256(result)
    return result


def _arm(value: Any, *, arm_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedEfficiencyError(f"arms.{arm_name} must be an object")
    expected_strategy = (
        FIXED_STRATEGY if arm_name == "fixed" else ADAPTIVE_STRATEGY
    )
    if value.get("strategy") != expected_strategy:
        raise MatchedEfficiencyError(
            f"arms.{arm_name}.strategy must be {expected_strategy}"
        )
    query = _text(value.get("query"), field=f"arms.{arm_name}.query")
    candidate_suite = _candidate_suite(
        value.get("candidate_suite"),
        field=f"arms.{arm_name}.candidate_suite",
    )
    sufficiency_contract = _sufficiency_contract(
        value.get("sufficiency_contract"),
        field=f"arms.{arm_name}.sufficiency_contract",
    )
    if sufficiency_contract["candidate_universe"] != candidate_suite:
        raise MatchedEfficiencyError(
            f"arms.{arm_name}.sufficiency_contract candidate_universe must "
            "exactly match candidate_suite"
        )
    return {
        "strategy": expected_strategy,
        "query": query,
        "query_sha256": canonical_sha256(query),
        "checkpoint": _checkpoint(
            value.get("checkpoint"), field=f"arms.{arm_name}.checkpoint"
        ),
        "candidate_suite": candidate_suite,
        "seeds": _seeds(value.get("seeds"), field=f"arms.{arm_name}.seeds"),
        "max_budget": _max_budget(
            value.get("max_budget"), field=f"arms.{arm_name}.max_budget"
        ),
        "sufficiency_contract": sufficiency_contract,
        "sufficiency_contract_sha256": canonical_sha256(
            sufficiency_contract
        ),
    }


def validate_matched_preregistration(value: Any) -> dict[str, Any]:
    """Normalize a two-arm preregistration and fail closed on any mismatch."""

    if not isinstance(value, Mapping):
        raise MatchedEfficiencyError("preregistration must be an object")
    if value.get("schema_version") != 1 or value.get("protocol") != PROTOCOL:
        raise MatchedEfficiencyError(
            f"preregistration must use schema_version=1 and protocol={PROTOCOL}"
        )
    study_id = _identifier(value.get("study_id"), field="study_id")
    raw_arms = value.get("arms")
    if not isinstance(raw_arms, Mapping) or set(raw_arms) != {"fixed", "adaptive"}:
        raise MatchedEfficiencyError("arms must contain exactly fixed and adaptive")
    arms = {
        name: _arm(raw_arms[name], arm_name=name)
        for name in ("fixed", "adaptive")
    }
    mismatch = [
        field
        for field in MATCHED_FIELDS
        if arms["fixed"][field] != arms["adaptive"][field]
    ]
    if mismatch:
        raise MatchedEfficiencyError(
            "matched arms differ in frozen fields: " + ", ".join(mismatch)
        )
    expected_samples = len(arms["fixed"]["candidate_suite"]) * len(
        arms["fixed"]["seeds"]
    )
    if arms["fixed"]["max_budget"]["act_episode_starts"] < expected_samples:
        raise MatchedEfficiencyError(
            "shared ACT budget cannot cover the fixed candidate-by-seed suite"
        )
    normalized = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "study_id": study_id,
        "matched_fields": list(MATCHED_FIELDS),
        "arms": arms,
    }
    normalized["preregistration_sha256"] = canonical_sha256(normalized)
    return normalized


def build_matched_preregistration(
    *,
    study_id: str,
    query: str,
    checkpoint: Mapping[str, Any],
    candidate_suite: Sequence[str],
    seeds: Sequence[int],
    max_budget: Mapping[str, Any],
    sufficiency_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build two arms from one shared scientific identity."""

    shared = {
        "query": query,
        "checkpoint": dict(checkpoint),
        "candidate_suite": list(candidate_suite),
        "seeds": list(seeds),
        "max_budget": dict(max_budget),
        "sufficiency_contract": deepcopy(dict(sufficiency_contract)),
    }
    return validate_matched_preregistration(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "study_id": study_id,
            "arms": {
                "fixed": {"strategy": FIXED_STRATEGY, **deepcopy(shared)},
                "adaptive": {
                    "strategy": ADAPTIVE_STRATEGY,
                    **deepcopy(shared),
                },
            },
        }
    )


def _resource_usage(value: Any, *, budget: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "act_episode_starts",
        "completed_policy_trials",
        "policy_steps",
        "expert_starts",
        "probe_starts",
        "provider_logical_calls",
        "provider_transport_attempts",
        "wall_seconds",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MatchedEfficiencyError(
            f"resource_usage fields must be exactly {sorted(expected)}"
        )
    result = {
        "act_episode_starts": _count(
            value["act_episode_starts"],
            field="resource_usage.act_episode_starts",
        ),
        "completed_policy_trials": _count(
            value["completed_policy_trials"],
            field="resource_usage.completed_policy_trials",
        ),
        "policy_steps": _count(
            value["policy_steps"], field="resource_usage.policy_steps"
        ),
        "expert_starts": _count(
            value["expert_starts"], field="resource_usage.expert_starts"
        ),
        "probe_starts": _count(
            value["probe_starts"], field="resource_usage.probe_starts"
        ),
        "provider_logical_calls": _count(
            value["provider_logical_calls"],
            field="resource_usage.provider_logical_calls",
        ),
        "provider_transport_attempts": _count(
            value["provider_transport_attempts"],
            field="resource_usage.provider_transport_attempts",
        ),
        "wall_seconds": _finite(
            value["wall_seconds"], field="resource_usage.wall_seconds"
        ),
    }
    if result["provider_transport_attempts"] < result["provider_logical_calls"]:
        raise MatchedEfficiencyError(
            "provider transport attempts cannot be smaller than logical calls"
        )
    result["provider_retries"] = (
        result["provider_transport_attempts"]
        - result["provider_logical_calls"]
    )
    for resource in (
        "act_episode_starts",
        "expert_starts",
        "probe_starts",
        "provider_retries",
    ):
        if result[resource] > budget[resource]:
            raise MatchedEfficiencyError(
                f"{resource} exceeds the preregistered shared budget"
            )
    if result["wall_seconds"] > budget["wall_seconds"]:
        raise MatchedEfficiencyError(
            "wall_seconds exceeds the preregistered shared budget"
        )
    return result


def _observations(
    value: Any,
    *,
    arm_name: str,
    candidate_suite: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise MatchedEfficiencyError("observations must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    allowed = {(candidate, seed) for candidate in candidate_suite for seed in seeds}
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise MatchedEfficiencyError(f"observations[{index}] must be an object")
        candidate = _identifier(
            raw.get("candidate_id"), field=f"observations[{index}].candidate_id"
        )
        seed = _count(raw.get("seed"), field=f"observations[{index}].seed")
        identity = (candidate, seed)
        if identity not in allowed:
            raise MatchedEfficiencyError(
                f"observation {identity} is outside the frozen suite"
            )
        if identity in seen:
            raise MatchedEfficiencyError(f"duplicate observation {identity}")
        seen.add(identity)
        result.append(
            {
                "candidate_id": candidate,
                "seed": seed,
                "evidence": deepcopy(raw.get("evidence")),
            }
        )
    if arm_name == "fixed" and seen != allowed:
        missing = sorted(allowed - seen)
        raise MatchedEfficiencyError(
            f"fixed arm must cover the complete frozen suite; missing {missing}"
        )
    return result


def _conclusion(value: Any, *, query_sha256: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedEfficiencyError("conclusion must be an object")
    if value.get("query_sha256") != query_sha256:
        raise MatchedEfficiencyError(
            "conclusion does not bind to the original frozen Query"
        )
    verdict = value.get("verdict")
    if verdict not in _VERDICTS:
        raise MatchedEfficiencyError(
            f"conclusion.verdict must be one of {sorted(_VERDICTS)}"
        )
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise MatchedEfficiencyError(
            "conclusion.limitations must explicitly bound the answer"
        )
    return {
        "query_sha256": query_sha256,
        "conclusion_key": _identifier(
            value.get("conclusion_key"), field="conclusion.conclusion_key"
        ),
        "verdict": verdict,
        "answer": _text(value.get("answer"), field="conclusion.answer"),
        "limitations": [
            _text(item, field=f"conclusion.limitations[{index}]")
            for index, item in enumerate(limitations)
        ],
    }


def _result(
    value: Any,
    *,
    arm_name: str,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedEfficiencyError(f"{arm_name} result must be an object")
    arm = preregistration["arms"][arm_name]
    if value.get("strategy") != arm["strategy"]:
        raise MatchedEfficiencyError(f"{arm_name} result strategy mismatch")
    if value.get("preregistration_sha256") != preregistration[
        "preregistration_sha256"
    ]:
        raise MatchedEfficiencyError(
            f"{arm_name} result preregistration hash mismatch"
        )
    source = value.get("evidence_source")
    if source not in _SOURCE_KINDS:
        raise MatchedEfficiencyError(
            f"evidence_source must be one of {sorted(_SOURCE_KINDS)}"
        )
    if value.get("status") != "completed":
        raise MatchedEfficiencyError(
            "only completed arms can enter a matched comparison"
        )
    observations = _observations(
        value.get("observations"),
        arm_name=arm_name,
        candidate_suite=arm["candidate_suite"],
        seeds=arm["seeds"],
    )
    usage = _resource_usage(
        value.get("resource_usage"), budget=arm["max_budget"]
    )
    if usage["act_episode_starts"] < len(observations):
        raise MatchedEfficiencyError(
            "ACT episode starts cannot be smaller than completed observations"
        )
    if usage["completed_policy_trials"] < len(observations):
        raise MatchedEfficiencyError(
            "completed policy trials cannot be smaller than completed observations"
        )
    if usage["completed_policy_trials"] > usage["act_episode_starts"]:
        raise MatchedEfficiencyError(
            "completed policy trials cannot exceed ACT episode starts"
        )
    stopping = value.get("stopping")
    if not isinstance(stopping, Mapping):
        raise MatchedEfficiencyError("stopping must be an object")
    reason = stopping.get("reason")
    allowed_reasons = (
        {"fixed_suite_complete"}
        if arm_name == "fixed"
        else {"sufficiency_reached", "budget_exhausted"}
    )
    if reason not in allowed_reasons:
        raise MatchedEfficiencyError(
            f"{arm_name} stopping.reason must be one of {sorted(allowed_reasons)}"
        )
    sufficiency_met = stopping.get("sufficiency_met")
    if not isinstance(sufficiency_met, bool):
        raise MatchedEfficiencyError("stopping.sufficiency_met must be boolean")
    if reason == "sufficiency_reached" and not sufficiency_met:
        raise MatchedEfficiencyError(
            "sufficiency_reached requires sufficiency_met=true"
        )
    return {
        "strategy": arm["strategy"],
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "status": "completed",
        "evidence_source": source,
        "observations": observations,
        "resource_usage": usage,
        "stopping": {
            "reason": reason,
            "sufficiency_met": sufficiency_met,
        },
        "conclusion": _conclusion(
            value.get("conclusion"), query_sha256=arm["query_sha256"]
        ),
    }


def compare_matched_results(
    preregistration: Mapping[str, Any],
    fixed_result: Mapping[str, Any],
    adaptive_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit one pair without executing either strategy."""

    prereg = validate_matched_preregistration(preregistration)
    fixed = _result(fixed_result, arm_name="fixed", preregistration=prereg)
    adaptive = _result(
        adaptive_result, arm_name="adaptive", preregistration=prereg
    )
    if fixed["evidence_source"] != adaptive["evidence_source"]:
        raise MatchedEfficiencyError(
            "matched arms must use the same evidence_source"
        )
    conclusion_fields = ("conclusion_key", "verdict")
    conclusion_agreement = all(
        fixed["conclusion"][field] == adaptive["conclusion"][field]
        for field in conclusion_fields
    )
    resources: dict[str, dict[str, Any]] = {}
    for name in (
        "act_episode_starts",
        "completed_policy_trials",
        "policy_steps",
        "expert_starts",
        "probe_starts",
        "provider_logical_calls",
        "provider_transport_attempts",
        "provider_retries",
        "wall_seconds",
    ):
        fixed_value = fixed["resource_usage"][name]
        adaptive_value = adaptive["resource_usage"][name]
        resources[name] = {
            "fixed": fixed_value,
            "adaptive": adaptive_value,
            "fixed_minus_adaptive": fixed_value - adaptive_value,
        }
    act_savings = resources["act_episode_starts"]["fixed_minus_adaptive"]
    synthetic = fixed["evidence_source"] == "synthetic_fixture"
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "study_id": prereg["study_id"],
        "mode": "artifact_only_no_execution",
        "calls_started_by_comparison": {
            "provider": 0,
            "simulator": 0,
            "expert": 0,
            "probe": 0,
            "act": 0,
        },
        "preregistration_sha256": prereg["preregistration_sha256"],
        "matched_identity": {
            field: deepcopy(prereg["arms"]["fixed"][field])
            for field in MATCHED_FIELDS
        },
        "sufficiency_contract_sha256": prereg["arms"]["fixed"][
            "sufficiency_contract_sha256"
        ],
        "evidence_source": fixed["evidence_source"],
        "arms": {"fixed": fixed, "adaptive": adaptive},
        "resource_comparison": resources,
        "original_query_conclusion": {
            "comparison_fields": list(conclusion_fields),
            "agrees": conclusion_agreement,
            "fixed": {
                field: fixed["conclusion"][field] for field in conclusion_fields
            },
            "adaptive": {
                field: adaptive["conclusion"][field]
                for field in conclusion_fields
            },
        },
        "act_start_savings": act_savings,
        "adaptive_used_fewer_act_episode_starts": act_savings > 0,
        "zero_act_savings": act_savings == 0,
        "efficiency_pattern_passed": act_savings > 0 and conclusion_agreement,
        "empirical_policy_claim_eligible": False,
        "empirical_policy_claim_ineligible_reason": (
            "synthetic_fixture"
            if synthetic
            else "result_manifest_provenance_not_independently_audited_here"
        ),
        "paper_table_eligible": False,
        "paper_reference_configuration": {
            "constructed_task_trials_per_task": 5,
            "agent_runs": 10,
            "source": "paper_appendix_A.1.1_and_tables_2_5",
        },
        "paper_reference_configuration_met": False,
        "resource_semantics": {
            "act_episode_starts": (
                "episode start attempts, including attempts that may not complete"
            ),
            "completed_policy_trials": "completed policy episodes",
            "evaluation_samples": (
                "unique frozen candidate-by-seed observations; derived separately"
            ),
            "policy_steps": "recorded ACT inference steps",
            "paper_reported_sample_count": (
                "not inferred from episode starts, completed trials, or policy steps"
            ),
        },
        "evaluation_samples": {
            "fixed": len(fixed["observations"]),
            "adaptive": len(adaptive["observations"]),
        },
        "paper_reported_sample_count": None,
        "limitations": [
            (
                "Synthetic fixtures validate protocol logic only; they are not "
                "policy-performance or sampling-efficiency evidence."
                if synthetic
                else "Artifact declarations require independent provenance audit before an empirical claim."
            ),
            "A resource saving is not conclusion-preserving unless the original Query conclusion agrees.",
            "Starts and provider retries are reported separately from successful completions.",
            "This pair does not satisfy the paper target of five trials per constructed task and ten agent runs.",
            "Paper sample count is not silently equated with ACT episode starts or policy steps.",
            "This minimal pair is not a paper-table reproduction.",
        ],
    }


def build_synthetic_demonstrations() -> dict[str, Any]:
    """Return both required functional fixtures: savings and zero savings."""

    sufficiency_contract = build_query_sufficiency_contract(
        "Does at least one frozen candidate work?",
        candidate_universe=["position_left", "instance_base0"],
        required_candidate_ids=["position_left", "instance_base0"],
        round_budget=2,
        claim_type="existential",
    )
    prereg = build_matched_preregistration(
        study_id="synthetic_matched_protocol_smoke",
        query="Does the policy generalize across the frozen candidate suite?",
        checkpoint={
            "checkpoint_id": "synthetic_act_checkpoint",
            "artifact_sha256": "a" * 64,
        },
        candidate_suite=["position_left", "instance_base0"],
        seeds=[100502],
        max_budget={
            "act_episode_starts": 2,
            "expert_starts": 2,
            "probe_starts": 2,
            "provider_retries": 4,
            "wall_seconds": 120.0,
        },
        sufficiency_contract=sufficiency_contract,
    )
    query_hash = prereg["arms"]["fixed"]["query_sha256"]

    def result(
        *,
        strategy: str,
        candidate_ids: Sequence[str],
        act_starts: int,
        wall_seconds: float,
        stopping_reason: str,
    ) -> dict[str, Any]:
        return {
            "strategy": strategy,
            "preregistration_sha256": prereg["preregistration_sha256"],
            "status": "completed",
            "evidence_source": "synthetic_fixture",
            "observations": [
                {
                    "candidate_id": candidate,
                    "seed": 100502,
                    "evidence": {"synthetic_success": True},
                }
                for candidate in candidate_ids
            ],
            "resource_usage": {
                "act_episode_starts": act_starts,
                "completed_policy_trials": act_starts,
                "policy_steps": 10 * act_starts,
                "expert_starts": act_starts,
                "probe_starts": act_starts,
                "provider_logical_calls": 3 * act_starts,
                "provider_transport_attempts": 3 * act_starts + 1,
                "wall_seconds": wall_seconds,
            },
            "stopping": {
                "reason": stopping_reason,
                "sufficiency_met": True,
            },
            "conclusion": {
                "query_sha256": query_hash,
                "conclusion_key": "supported_frozen_suite_smoke",
                "verdict": "supported_in_tested_scope",
                "answer": "Synthetic evidence supports only the fixture conclusion.",
                "limitations": [
                    "Synthetic fixture; no policy rollout was executed."
                ],
            },
        }

    fixed = result(
        strategy=FIXED_STRATEGY,
        candidate_ids=["position_left", "instance_base0"],
        act_starts=2,
        wall_seconds=20.0,
        stopping_reason="fixed_suite_complete",
    )
    early_adaptive = result(
        strategy=ADAPTIVE_STRATEGY,
        candidate_ids=["position_left"],
        act_starts=1,
        wall_seconds=11.0,
        stopping_reason="sufficiency_reached",
    )
    full_adaptive = result(
        strategy=ADAPTIVE_STRATEGY,
        candidate_ids=["position_left", "instance_base0"],
        act_starts=2,
        wall_seconds=21.0,
        stopping_reason="sufficiency_reached",
    )
    return {
        "schema_version": 1,
        "mode": "synthetic_functional_demonstration",
        "preregistration": prereg,
        "scenarios": {
            "adaptive_one_vs_fixed_two": compare_matched_results(
                prereg, fixed, early_adaptive
            ),
            "adaptive_two_vs_fixed_two_zero_savings": compare_matched_results(
                prereg, fixed, full_adaptive
            ),
        },
        "empirical_policy_claim_eligible": False,
    }


__all__ = [
    "ADAPTIVE_STRATEGY",
    "FIXED_STRATEGY",
    "MATCHED_FIELDS",
    "MatchedEfficiencyError",
    "PROTOCOL",
    "build_matched_preregistration",
    "build_synthetic_demonstrations",
    "canonical_sha256",
    "compare_matched_results",
    "validate_matched_preregistration",
]
