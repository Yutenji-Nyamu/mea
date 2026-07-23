"""Artifact-only aggregators for small, real ManipEvalAgent claim demos.

The functions in this module do not start policies, providers, simulators, or
annotation agents.  They import explicitly labelled live manifests or
development-agent annotations and compute deterministic summaries.  There is
no synthetic/default-result mode: a caller must supply every observation.

These protocols are intentionally small.  In particular, a two-policy ranking
is labelled as a toy pilot and development-agent labels are never described as
human validity evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence


EFFICIENCY_PROTOCOL = "paper_claim_small_efficiency_v1"
RANKING_PROTOCOL = "paper_claim_policy_ranking_v1"
PROXY_VALIDITY_PROTOCOL = "paper_claim_proxy_validity_v1"
CODEGEN_ABLATION_PROTOCOL = "paper_claim_codegen_ablation_v1"
PROPOSAL_PROMPT_ABLATION_PROTOCOL = (
    "paper_claim_proposal_prompt_ablation_v1"
)
ERROR_DISTRIBUTION_PROTOCOL = "paper_claim_error_distribution_v1"

PAPER_VQA_CONDITIONS = (
    "clean",
    "scene_clutter",
    "background_texture",
    "lighting",
)
PLAN_PAPER_CATEGORIES = (
    "generalization_object",
    "generalization_scene",
    "performance",
    "safety",
    "language_or_multitask",
)
ERROR_STAGES = ("plan", "taskgen", "toolgen", "simulator", "other")
TASKGEN_ABLATION_CONDITIONS = (
    "complete",
    "base",
    "no_rag",
    "no_visual_self_check",
    "no_readme_agent",
)
TOOLGEN_ABLATION_CONDITIONS = ("complete", "no_rag")

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_REAL_EVIDENCE_SOURCES = {
    EFFICIENCY_PROTOCOL: "live_act_rollout",
    RANKING_PROTOCOL: "live_policy_rollout",
    PROXY_VALIDITY_PROTOCOL: "development_agent_proxy",
    CODEGEN_ABLATION_PROTOCOL: "live_provider_codegen",
    PROPOSAL_PROMPT_ABLATION_PROTOCOL: "live_provider_structured_proposal",
    ERROR_DISTRIBUTION_PROTOCOL: "live_runtime_operation_log",
}


class PaperClaimDemoError(ValueError):
    """Raised when an imported claim-demo artifact is incomplete or ambiguous."""


def canonical_sha256(value: Any) -> str:
    """Return a stable hash for an imported JSON value."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PaperClaimDemoError(f"input is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperClaimDemoError(f"{field} must be an object")
    return value


def _items(value: Any, *, field: str, minimum: int = 1) -> list[Any]:
    if not isinstance(value, list) or len(value) < minimum:
        raise PaperClaimDemoError(
            f"{field} must be a list with at least {minimum} item(s)"
        )
    return value


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(text):
        raise PaperClaimDemoError(f"{field} must be a non-empty identifier")
    return text


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PaperClaimDemoError(f"{field} must be non-empty text")
    return value.strip()


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise PaperClaimDemoError(f"{field} must be boolean")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PaperClaimDemoError(f"{field} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PaperClaimDemoError(f"{field} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise PaperClaimDemoError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise PaperClaimDemoError(f"{field} must be <= {maximum}")
    return result


def _unique_strings(
    value: Any, *, field: str, minimum: int = 1
) -> list[str]:
    result = [
        _identifier(item, field=f"{field}[{index}]")
        for index, item in enumerate(
            _items(value, field=field, minimum=minimum)
        )
    ]
    if len(result) != len(set(result)):
        raise PaperClaimDemoError(f"{field} contains duplicates")
    return result


def _header(value: Any, *, protocol: str) -> Mapping[str, Any]:
    root = _object(value, field="manifest")
    if root.get("schema_version") != 1 or root.get("protocol") != protocol:
        raise PaperClaimDemoError(
            f"manifest must use schema_version=1 and protocol={protocol}"
        )
    expected_source = _REAL_EVIDENCE_SOURCES[protocol]
    if root.get("evidence_source") != expected_source:
        raise PaperClaimDemoError(
            f"{protocol} requires evidence_source={expected_source}; "
            "synthetic or inferred sources are not accepted"
        )
    _identifier(root.get("study_id"), field="study_id")
    return root


def _result_envelope(
    *,
    protocol: str,
    study_id: str,
    source: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": f"{protocol}_result",
        "study_id": study_id,
        "input_manifest_sha256": canonical_sha256(source),
        **dict(result),
    }


def _rollout_trials(
    value: Any,
    *,
    field: str,
    candidates: Sequence[str],
    seeds: Sequence[int],
    require_complete_grid: bool,
    require_score: bool = False,
    require_outcome_binding: bool = False,
    allowed_pairs: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    rows = _items(value, field=field)
    allowed = allowed_pairs or {
        (candidate, seed) for candidate in candidates for seed in seeds
    }
    seen: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _object(raw, field=f"{field}[{index}]")
        candidate = _identifier(
            row.get("candidate_id"), field=f"{field}[{index}].candidate_id"
        )
        seed = _integer(row.get("seed"), field=f"{field}[{index}].seed")
        identity = (candidate, seed)
        if identity not in allowed:
            raise PaperClaimDemoError(
                f"{field}[{index}] candidate/seed is outside the frozen suite"
            )
        if identity in seen:
            raise PaperClaimDemoError(
                f"{field} contains duplicate candidate/seed {identity}"
            )
        seen.add(identity)
        normalized_row = {
            "trial_id": _identifier(
                row.get("trial_id"), field=f"{field}[{index}].trial_id"
            ),
            "candidate_id": candidate,
            "seed": seed,
            "rollout_ref": _text(
                row.get("rollout_ref"), field=f"{field}[{index}].rollout_ref"
            ),
        }
        episode_status = str(row.get("episode_status"))
        if episode_status not in {"completed", "runtime_error"}:
            raise PaperClaimDemoError(
                f"{field}[{index}].episode_status must be completed or "
                "runtime_error"
            )
        normalized_row["episode_status"] = episode_status
        if require_score:
            if episode_status != "completed":
                raise PaperClaimDemoError(
                    f"{field}[{index}] ranking score requires a completed episode"
                )
            normalized_row["score"] = _number(
                row.get("score"), field=f"{field}[{index}].score"
            )
        else:
            normalized_row["success"] = _boolean(
                row.get("success"), field=f"{field}[{index}].success"
            )
            if episode_status == "runtime_error" and normalized_row["success"]:
                raise PaperClaimDemoError(
                    f"{field}[{index}] runtime_error cannot be policy success"
                )
        if require_outcome_binding:
            outcome_tool_sha256 = _text(
                row.get("outcome_tool_sha256"),
                field=f"{field}[{index}].outcome_tool_sha256",
            )
            if not re.fullmatch(r"[0-9a-f]{64}", outcome_tool_sha256):
                raise PaperClaimDemoError(
                    f"{field}[{index}].outcome_tool_sha256 must be lowercase "
                    "SHA-256"
                )
            normalized_row.update(
                {
                    "outcome_metric": _identifier(
                        row.get("outcome_metric"),
                        field=f"{field}[{index}].outcome_metric",
                    ),
                    "outcome_authority": _identifier(
                        row.get("outcome_authority"),
                        field=f"{field}[{index}].outcome_authority",
                    ),
                    "outcome_tool_sha256": outcome_tool_sha256,
                }
            )
        normalized.append(normalized_row)
    if require_complete_grid and seen != allowed:
        raise PaperClaimDemoError(
            f"{field} must contain the complete candidate-by-seed grid; "
            f"missing={sorted(allowed - seen)}"
        )
    return normalized


def evaluate_small_efficiency(value: Any) -> dict[str, Any]:
    """Compare a dense fixed arm with a smaller adaptive ACT arm."""

    root = _header(value, protocol=EFFICIENCY_PROTOCOL)
    study_id = _identifier(root.get("study_id"), field="study_id")
    candidates = _unique_strings(
        root.get("candidate_universe"), field="candidate_universe"
    )
    seeds = [
        _integer(seed, field=f"seeds[{index}]")
        for index, seed in enumerate(_items(root.get("seeds"), field="seeds"))
    ]
    if len(seeds) != len(set(seeds)):
        raise PaperClaimDemoError("seeds contains duplicates")
    claim_type = str(root.get("claim_type"))
    if claim_type not in {
        "universal_all_candidates",
        "thresholded_success_rate",
    }:
        raise PaperClaimDemoError(
            "claim_type must be universal_all_candidates or "
            "thresholded_success_rate"
        )
    comparison_design = str(root.get("comparison_design"))
    if comparison_design not in {
        "independent_live_arms",
        "cached_prefix_counterfactual",
    }:
        raise PaperClaimDemoError(
            "comparison_design must be independent_live_arms or "
            "cached_prefix_counterfactual"
        )
    comparison_timing = str(root.get("comparison_timing"))
    expected_timing = (
        "post_hoc_after_observed_outcomes"
        if comparison_design == "cached_prefix_counterfactual"
        else "pre_registered_before_outcomes"
    )
    if comparison_timing != expected_timing:
        raise PaperClaimDemoError(
            f"{comparison_design} requires comparison_timing="
            f"{expected_timing}"
        )
    cost_scope = str(root.get("cost_scope"))
    if cost_scope not in {"policy_episode_wall_only", "full_agent_wall"}:
        raise PaperClaimDemoError(
            "cost_scope must be policy_episode_wall_only or full_agent_wall"
        )
    adaptive_stop_assessment_ref = _text(
        root.get("adaptive_stop_assessment_ref"),
        field="adaptive_stop_assessment_ref",
    )
    declared_pairs: set[tuple[str, int]] | None = None
    if root.get("candidate_seed_pairs") is not None:
        raw_pairs = _items(
            root.get("candidate_seed_pairs"), field="candidate_seed_pairs"
        )
        declared_pairs = set()
        for index, raw_pair in enumerate(raw_pairs):
            pair = _object(
                raw_pair, field=f"candidate_seed_pairs[{index}]"
            )
            if set(pair) != {"candidate_id", "seed"}:
                raise PaperClaimDemoError(
                    "candidate_seed_pairs entries require candidate_id and seed"
                )
            candidate = _identifier(
                pair.get("candidate_id"),
                field=f"candidate_seed_pairs[{index}].candidate_id",
            )
            seed = _integer(
                pair.get("seed"),
                field=f"candidate_seed_pairs[{index}].seed",
            )
            identity = (candidate, seed)
            if candidate not in candidates or seed not in seeds:
                raise PaperClaimDemoError(
                    "candidate_seed_pairs entry is outside candidate_universe/seeds"
                )
            if identity in declared_pairs:
                raise PaperClaimDemoError(
                    f"duplicate candidate_seed_pairs entry: {identity}"
                )
            declared_pairs.add(identity)
        if {candidate for candidate, _ in declared_pairs} != set(candidates):
            raise PaperClaimDemoError(
                "candidate_seed_pairs must cover every candidate"
            )
    query = _text(root.get("query"), field="query")
    rule = _object(root.get("conclusion_rule"), field="conclusion_rule")
    if rule.get("metric") != "success_rate" or rule.get("operator") != ">=":
        raise PaperClaimDemoError(
            "conclusion_rule must be success_rate >= threshold"
        )
    threshold = _number(
        rule.get("threshold"),
        field="conclusion_rule.threshold",
        minimum=0.0,
        maximum=1.0,
    )
    if claim_type == "universal_all_candidates" and threshold != 1.0:
        raise PaperClaimDemoError(
            "universal_all_candidates requires success_rate >= 1.0"
        )
    arms: dict[str, dict[str, Any]] = {}
    normalized_trials: dict[str, list[dict[str, Any]]] = {}
    for arm_name, require_grid in (("fixed", True), ("adaptive", False)):
        raw_arm = _object(root.get(arm_name), field=arm_name)
        if raw_arm.get("arm") != arm_name:
            raise PaperClaimDemoError(f"{arm_name}.arm must be {arm_name}")
        trials = _rollout_trials(
            raw_arm.get("trials"),
            field=f"{arm_name}.trials",
            candidates=candidates,
            seeds=seeds,
            require_complete_grid=require_grid,
            require_outcome_binding=True,
            allowed_pairs=declared_pairs,
        )
        normalized_trials[arm_name] = trials
        successes = sum(row["success"] for row in trials)
        success_rate = successes / len(trials)
        tested_pairs = {
            (row["candidate_id"], row["seed"]) for row in trials
        }
        complete_pairs = declared_pairs or {
            (candidate, seed) for candidate in candidates for seed in seeds
        }
        if claim_type == "universal_all_candidates":
            conclusion = (
                "refuted_in_sample"
                if successes < len(trials)
                else "supported_in_complete_finite_suite"
                if tested_pairs == complete_pairs
                else "inconclusive_in_sample"
            )
        else:
            conclusion = (
                "supported_in_sample"
                if success_rate >= threshold
                else "not_supported_in_sample"
            )
        outcome_bindings = {
            (
                row["outcome_metric"],
                row["outcome_authority"],
                row["outcome_tool_sha256"],
            )
            for row in trials
        }
        if len(outcome_bindings) != 1:
            raise PaperClaimDemoError(
                f"{arm_name}.trials must use one frozen outcome binding"
            )
        arms[arm_name] = {
            "run_id": _identifier(
                raw_arm.get("run_id"), field=f"{arm_name}.run_id"
            ),
            "policy_id": _identifier(
                raw_arm.get("policy_id"), field=f"{arm_name}.policy_id"
            ),
            "checkpoint_id": _text(
                raw_arm.get("checkpoint_id"),
                field=f"{arm_name}.checkpoint_id",
            ),
            "wall_seconds": _number(
                raw_arm.get("wall_seconds"),
                field=f"{arm_name}.wall_seconds",
                minimum=0.0,
            ),
            "arm_trace_ref": _text(
                raw_arm.get("arm_trace_ref"),
                field=f"{arm_name}.arm_trace_ref",
            ),
            "act_rollouts": len(trials),
            "completed_policy_trials": sum(
                row["episode_status"] == "completed" for row in trials
            ),
            "runtime_error_trials": sum(
                row["episode_status"] == "runtime_error" for row in trials
            ),
            "successes": successes,
            "success_rate": success_rate,
            "conclusion": conclusion,
            "outcome_binding": {
                "metric": next(iter(outcome_bindings))[0],
                "authority": next(iter(outcome_bindings))[1],
                "tool_sha256": next(iter(outcome_bindings))[2],
            },
            "tested_candidate_seed_pairs": [
                [row["candidate_id"], row["seed"]] for row in trials
            ],
        }
    for field in ("policy_id", "checkpoint_id"):
        if arms["fixed"][field] != arms["adaptive"][field]:
            raise PaperClaimDemoError(
                f"fixed/adaptive {field} must identify the same policy"
            )
    if arms["fixed"]["outcome_binding"] != arms["adaptive"]["outcome_binding"]:
        raise PaperClaimDemoError(
            "fixed/adaptive trials must use the same frozen outcome binding"
        )
    fixed_by_pair = {
        (row["candidate_id"], row["seed"]): row
        for row in normalized_trials["fixed"]
    }
    adaptive_refs = {row["rollout_ref"] for row in normalized_trials["adaptive"]}
    fixed_refs = {row["rollout_ref"] for row in normalized_trials["fixed"]}
    if comparison_design == "cached_prefix_counterfactual":
        for row in normalized_trials["adaptive"]:
            fixed_row = fixed_by_pair.get((row["candidate_id"], row["seed"]))
            if fixed_row is None or fixed_row["rollout_ref"] != row["rollout_ref"]:
                raise PaperClaimDemoError(
                    "cached_prefix_counterfactual adaptive trials must reuse "
                    "the exact fixed-prefix rollout"
                )
    elif adaptive_refs & fixed_refs:
        raise PaperClaimDemoError(
            "independent_live_arms must not reuse rollout references"
        )
    fixed_count = arms["fixed"]["act_rollouts"]
    adaptive_count = arms["adaptive"]["act_rollouts"]
    fixed_wall = arms["fixed"]["wall_seconds"]
    adaptive_wall = arms["adaptive"]["wall_seconds"]
    agreement = arms["fixed"]["conclusion"] == arms["adaptive"]["conclusion"]
    act_saving = fixed_count - adaptive_count
    wall_saving = fixed_wall - adaptive_wall
    if comparison_design == "cached_prefix_counterfactual":
        claim_status = (
            "post_hoc_cached_counterfactual_protocol_demo"
            if agreement
            else "post_hoc_counterfactual_conclusion_disagrees"
        )
    elif not agreement:
        claim_status = "conclusion_disagrees"
    elif act_saving > 0 and wall_saving > 0:
        claim_status = "toy_efficiency_claim_supported"
    else:
        claim_status = "agreement_without_joint_savings"
    return _result_envelope(
        protocol=EFFICIENCY_PROTOCOL,
        study_id=study_id,
        source=root,
        result={
            "claim_scope": (
                "post_hoc_cached_prefix_counterfactual_not_observed_saving"
                if comparison_design == "cached_prefix_counterfactual"
                else "small_real_rollout_demo_not_full_benchmark"
            ),
            "query": query,
            "claim_type": claim_type,
            "comparison_design": comparison_design,
            "comparison_timing": comparison_timing,
            "cost_scope": cost_scope,
            "adaptive_stop_assessment_ref": adaptive_stop_assessment_ref,
            "candidate_universe": candidates,
            "seeds": seeds,
            "candidate_seed_pairs": (
                [
                    {"candidate_id": candidate, "seed": seed}
                    for candidate, seed in sorted(declared_pairs)
                ]
                if declared_pairs is not None
                else None
            ),
            "conclusion_rule": {
                "metric": "success_rate",
                "operator": ">=",
                "threshold": threshold,
            },
            "arms": arms,
            "conclusion_agreement": agreement,
            "act_rollout_saving": (
                act_saving
                if comparison_design == "independent_live_arms"
                else None
            ),
            "act_rollout_saving_fraction": (
                act_saving / fixed_count
                if comparison_design == "independent_live_arms"
                else None
            ),
            "counterfactual_avoidable_rollout_count": (
                act_saving
                if comparison_design == "cached_prefix_counterfactual"
                else None
            ),
            "counterfactual_avoidable_rollout_fraction": (
                act_saving / fixed_count
                if comparison_design == "cached_prefix_counterfactual"
                else None
            ),
            "wall_second_saving": (
                wall_saving
                if comparison_design == "independent_live_arms"
                else None
            ),
            "wall_second_saving_fraction": (
                wall_saving / fixed_wall if fixed_wall > 0 else None
            )
            if comparison_design == "independent_live_arms"
            else None,
            "avoided_rollout_wall_seconds_estimate": (
                wall_saving
                if comparison_design == "cached_prefix_counterfactual"
                else None
            ),
            "measured_independent_arm_wall_speedup": (
                comparison_design == "independent_live_arms"
            ),
            "claim_status": claim_status,
        },
    )


def _average_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for key, _ in ordered[index:end]:
            ranks[key] = average_rank
        index = end
    return ranks


def _spearman(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float | None:
    keys = sorted(left)
    if keys != sorted(right) or len(keys) < 2:
        raise PaperClaimDemoError(
            "Spearman inputs must contain the same two or more policies"
        )
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks.values()) / len(keys)
    right_mean = sum(right_ranks.values()) / len(keys)
    numerator = sum(
        (left_ranks[key] - left_mean) * (right_ranks[key] - right_mean)
        for key in keys
    )
    left_ss = sum((left_ranks[key] - left_mean) ** 2 for key in keys)
    right_ss = sum((right_ranks[key] - right_mean) ** 2 for key in keys)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def evaluate_policy_ranking(value: Any) -> dict[str, Any]:
    """Rank live policy rollouts and compare them with a frozen reference."""

    root = _header(value, protocol=RANKING_PROTOCOL)
    study_id = _identifier(root.get("study_id"), field="study_id")
    candidates = _unique_strings(
        root.get("candidate_universe"), field="candidate_universe"
    )
    seeds = [
        _integer(seed, field=f"seeds[{index}]")
        for index, seed in enumerate(_items(root.get("seeds"), field="seeds"))
    ]
    if len(seeds) != len(set(seeds)):
        raise PaperClaimDemoError("seeds contains duplicates")
    reference = _object(root.get("reference_scores"), field="reference_scores")
    reference_scores = {
        _identifier(policy_id, field="reference_scores key"): _number(
            score, field=f"reference_scores.{policy_id}"
        )
        for policy_id, score in reference.items()
    }
    policies = _items(root.get("policies"), field="policies", minimum=2)
    observed_scores: dict[str, float] = {}
    policy_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(policies):
        policy = _object(raw, field=f"policies[{index}]")
        policy_id = _identifier(
            policy.get("policy_id"), field=f"policies[{index}].policy_id"
        )
        if policy_id in observed_scores:
            raise PaperClaimDemoError(f"duplicate policy_id: {policy_id}")
        trials = _rollout_trials(
            policy.get("trials"),
            field=f"policies[{index}].trials",
            candidates=candidates,
            seeds=seeds,
            require_complete_grid=True,
            require_score=True,
        )
        mean_score = sum(row["score"] for row in trials) / len(trials)
        observed_scores[policy_id] = mean_score
        policy_rows.append(
            {
                "policy_id": policy_id,
                "checkpoint_id": _text(
                    policy.get("checkpoint_id"),
                    field=f"policies[{index}].checkpoint_id",
                ),
                "run_id": _identifier(
                    policy.get("run_id"), field=f"policies[{index}].run_id"
                ),
                "act_rollouts": len(trials),
                "mean_score": mean_score,
            }
        )
    if set(reference_scores) != set(observed_scores):
        raise PaperClaimDemoError(
            "reference_scores must contain exactly the evaluated policy ids"
        )
    observed_ranks = _average_ranks(observed_scores)
    reference_ranks = _average_ranks(reference_scores)
    rho = _spearman(observed_scores, reference_scores)
    policy_count = len(policy_rows)
    toy = policy_count == 2
    ranking_inconclusive_tie = rho is None
    exact_order_agreement = all(
        observed_ranks[key] == reference_ranks[key] for key in observed_ranks
    )
    return _result_envelope(
        protocol=RANKING_PROTOCOL,
        study_id=study_id,
        source=root,
        result={
            "claim_scope": (
                "two_policy_toy_ranking_not_table9"
                if toy
                else "limited_policy_ranking_not_full_table9"
            ),
            "reference_source_ref": _text(
                root.get("reference_source_ref"), field="reference_source_ref"
            ),
            "policy_count": policy_count,
            "two_policy_toy": toy,
            "policies": policy_rows,
            "observed_mean_scores": observed_scores,
            "reference_scores": reference_scores,
            "observed_ranks": observed_ranks,
            "reference_ranks": reference_ranks,
            "spearman_rho": rho,
            "exact_order_agreement": exact_order_agreement,
            "claim_status": (
                "toy_order_inconclusive_tie"
                if toy and ranking_inconclusive_tie
                else "limited_order_inconclusive_tie"
                if ranking_inconclusive_tie
                else "toy_order_agrees"
                if toy and exact_order_agreement
                else "toy_order_disagrees"
                if toy
                else "limited_order_agrees"
                if exact_order_agreement
                else "limited_order_disagrees"
            ),
        },
    )


def _binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        raise PaperClaimDemoError(
            "AUROC requires at least one positive and one negative item"
        )
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def evaluate_proxy_validity(value: Any) -> dict[str, Any]:
    """Compute Plan set metrics and four-condition VQA proxy metrics."""

    root = _header(value, protocol=PROXY_VALIDITY_PROTOCOL)
    study_id = _identifier(root.get("study_id"), field="study_id")
    plan = _object(root.get("plan"), field="plan")
    plan_reference_session = _identifier(
        plan.get("reference_session_id"), field="plan.reference_session_id"
    )
    plan_prediction_session = _identifier(
        plan.get("prediction_session_id"), field="plan.prediction_session_id"
    )
    if plan_reference_session == plan_prediction_session:
        raise PaperClaimDemoError(
            "Plan reference and prediction must come from separate sessions"
        )
    plan_items = _items(plan.get("items"), field="plan.items")
    true_positive = false_positive = false_negative = exact = 0
    category_counts = {category: 0 for category in PLAN_PAPER_CATEGORIES}
    category_confusion = {
        category: {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
        for category in PLAN_PAPER_CATEGORIES
    }
    normalized_plan: list[dict[str, Any]] = []
    for index, raw in enumerate(plan_items):
        item = _object(raw, field=f"plan.items[{index}]")
        category = str(item.get("paper_category"))
        if category not in PLAN_PAPER_CATEGORIES:
            raise PaperClaimDemoError(
                f"plan.items[{index}].paper_category must be one of "
                f"{list(PLAN_PAPER_CATEGORIES)}"
            )
        reference = set(
            _unique_strings(
                item.get("reference_aspects"),
                field=f"plan.items[{index}].reference_aspects",
            )
        )
        predicted = set(
            _unique_strings(
                item.get("predicted_aspects"),
                field=f"plan.items[{index}].predicted_aspects",
                minimum=0,
            )
        )
        true_positive += len(reference & predicted)
        false_positive += len(predicted - reference)
        false_negative += len(reference - predicted)
        exact += reference == predicted
        category_counts[category] += 1
        category_confusion[category]["tp"] += len(reference & predicted)
        category_confusion[category]["fp"] += len(predicted - reference)
        category_confusion[category]["fn"] += len(reference - predicted)
        category_confusion[category]["exact"] += reference == predicted
        normalized_plan.append(
            {
                "item_id": _identifier(
                    item.get("item_id"),
                    field=f"plan.items[{index}].item_id",
                ),
                "paper_category": category,
                "reference_aspects": sorted(reference),
                "predicted_aspects": sorted(predicted),
            }
        )
    missing_plan_categories = [
        category
        for category in PLAN_PAPER_CATEGORIES
        if not category_counts[category]
    ]
    if missing_plan_categories:
        raise PaperClaimDemoError(
            "Plan annotations are missing paper categories: "
            f"{missing_plan_categories}"
        )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    micro_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    category_metrics: dict[str, dict[str, Any]] = {}
    for category in PLAN_PAPER_CATEGORIES:
        counts = category_confusion[category]
        category_precision = (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if counts["tp"] + counts["fp"]
            else 0.0
        )
        category_recall = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if counts["tp"] + counts["fn"]
            else 0.0
        )
        category_f1 = (
            2
            * category_precision
            * category_recall
            / (category_precision + category_recall)
            if category_precision + category_recall
            else 0.0
        )
        category_metrics[category] = {
            "n": category_counts[category],
            "micro_precision": category_precision,
            "micro_recall": category_recall,
            "micro_f1": category_f1,
            "exact_set_match_rate": counts["exact"] / category_counts[category],
        }

    vqa = _object(root.get("vqa"), field="vqa")
    vqa_reference_session = _identifier(
        vqa.get("reference_session_id"), field="vqa.reference_session_id"
    )
    vqa_prediction_session = _identifier(
        vqa.get("prediction_session_id"), field="vqa.prediction_session_id"
    )
    if vqa_reference_session == vqa_prediction_session:
        raise PaperClaimDemoError(
            "VQA reference and prediction must come from separate sessions"
        )
    threshold = _number(
        vqa.get("threshold"),
        field="vqa.threshold",
        minimum=0.0,
        maximum=1.0,
    )
    by_condition: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    seen_vqa_ids: set[str] = set()
    for index, raw in enumerate(_items(vqa.get("items"), field="vqa.items")):
        item = _object(raw, field=f"vqa.items[{index}]")
        item_id = _identifier(
            item.get("item_id"), field=f"vqa.items[{index}].item_id"
        )
        if item_id in seen_vqa_ids:
            raise PaperClaimDemoError(f"duplicate VQA item_id: {item_id}")
        seen_vqa_ids.add(item_id)
        condition = str(item.get("condition"))
        if condition not in PAPER_VQA_CONDITIONS:
            raise PaperClaimDemoError(
                f"vqa.items[{index}].condition must be one of "
                f"{list(PAPER_VQA_CONDITIONS)}"
            )
        label = _boolean(
            item.get("reference_observed"),
            field=f"vqa.items[{index}].reference_observed",
        )
        score = _number(
            item.get("positive_score"),
            field=f"vqa.items[{index}].positive_score",
            minimum=0.0,
            maximum=1.0,
        )
        _text(
            item.get("evidence_ref"), field=f"vqa.items[{index}].evidence_ref"
        )
        by_condition[condition].append((label, score))
    missing_conditions = [
        condition for condition in PAPER_VQA_CONDITIONS if condition not in by_condition
    ]
    if missing_conditions:
        raise PaperClaimDemoError(
            f"VQA annotations are missing paper conditions: {missing_conditions}"
        )
    condition_results: dict[str, dict[str, Any]] = {}
    all_labels: list[bool] = []
    all_scores: list[float] = []
    for condition in PAPER_VQA_CONDITIONS:
        rows = by_condition[condition]
        labels = [row[0] for row in rows]
        scores = [row[1] for row in rows]
        correct = sum((score >= threshold) == label for label, score in rows)
        condition_results[condition] = {
            "n": len(rows),
            "accuracy": correct / len(rows),
            "auroc": _binary_auc(labels, scores),
        }
        all_labels.extend(labels)
        all_scores.extend(scores)
    overall_correct = sum(
        (score >= threshold) == label
        for label, score in zip(all_labels, all_scores)
    )
    return _result_envelope(
        protocol=PROXY_VALIDITY_PROTOCOL,
        study_id=study_id,
        source=root,
        result={
            "claim_scope": (
                "development_agent_proxy_not_human_or_independent_validity"
            ),
            "plan": {
                "n": len(normalized_plan),
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": micro_f1,
                "exact_set_match_rate": exact / len(normalized_plan),
                "categories": category_metrics,
                "reference_session_id": plan_reference_session,
                "prediction_session_id": plan_prediction_session,
            },
            "vqa": {
                "n": len(all_labels),
                "threshold": threshold,
                "accuracy": overall_correct / len(all_labels),
                "auroc": _binary_auc(all_labels, all_scores),
                "conditions": condition_results,
                "reference_session_id": vqa_reference_session,
                "prediction_session_id": vqa_prediction_session,
            },
            "claim_status": "development_proxy_metrics_only",
        },
    )


def _evaluate_ablation_attempts(
    value: Any,
    *,
    protocol: str,
    proposal_only: bool,
) -> dict[str, Any]:
    """Aggregate matched provider attempts without changing evidence type."""

    root = _header(value, protocol=protocol)
    study_id = _identifier(root.get("study_id"), field="study_id")
    attempts = _items(root.get("attempts"), field="attempts")
    seen_attempts: set[str] = set()
    seen_provider_refs: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    input_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    failure_stages: dict[str, int] = defaultdict(int)
    for index, raw in enumerate(attempts):
        attempt = _object(raw, field=f"attempts[{index}]")
        attempt_id = _identifier(
            attempt.get("attempt_id"), field=f"attempts[{index}].attempt_id"
        )
        if attempt_id in seen_attempts:
            raise PaperClaimDemoError(f"duplicate attempt_id: {attempt_id}")
        seen_attempts.add(attempt_id)
        provider_ref = _text(
            attempt.get("provider_attempt_ref"),
            field=f"attempts[{index}].provider_attempt_ref",
        )
        if provider_ref in seen_provider_refs:
            raise PaperClaimDemoError(
                f"duplicate provider_attempt_ref: {provider_ref}"
            )
        seen_provider_refs.add(provider_ref)
        component = str(attempt.get("component"))
        if component not in {"taskgen", "toolgen"}:
            raise PaperClaimDemoError(
                f"attempts[{index}].component must be taskgen or toolgen"
            )
        condition = str(attempt.get("condition"))
        allowed_conditions = (
            TASKGEN_ABLATION_CONDITIONS
            if component == "taskgen"
            else TOOLGEN_ABLATION_CONDITIONS
        )
        if condition not in allowed_conditions:
            raise PaperClaimDemoError(
                f"invalid {component} ablation condition: {condition}"
            )
        syntax_valid = _boolean(
            attempt.get("syntax_valid"),
            field=f"attempts[{index}].syntax_valid",
        )
        downstream_valid = _boolean(
            attempt.get("downstream_valid"),
            field=f"attempts[{index}].downstream_valid",
        )
        accepted = _boolean(
            attempt.get("accepted"), field=f"attempts[{index}].accepted"
        )
        if accepted and not (syntax_valid and downstream_valid):
            raise PaperClaimDemoError(
                f"accepted attempt {attempt_id} must pass both validation gates"
            )
        input_id = _identifier(
            attempt.get("input_id"), field=f"attempts[{index}].input_id"
        )
        _text(
            attempt.get("artifact_ref"), field=f"attempts[{index}].artifact_ref"
        )
        if proposal_only and attempt.get("artifact_kind") != (
            "structured_proposal_json"
        ):
            raise PaperClaimDemoError(
                f"attempts[{index}].artifact_kind must be "
                "structured_proposal_json"
            )
        failure_stage = attempt.get("failure_stage")
        if accepted:
            if failure_stage is not None:
                raise PaperClaimDemoError(
                    f"accepted attempt {attempt_id} cannot have failure_stage"
                )
        else:
            failure_stage = _identifier(
                failure_stage,
                field=f"attempts[{index}].failure_stage",
            )
            failure_stages[failure_stage] += 1
        row = {
            "syntax_valid": syntax_valid,
            "downstream_valid": downstream_valid,
            "accepted": accepted,
        }
        grouped[(component, condition)].append(row)
        input_sets[(component, condition)].add(input_id)
    results: dict[str, dict[str, Any]] = {"taskgen": {}, "toolgen": {}}
    coverage: dict[str, dict[str, Any]] = {}
    for component, required in (
        ("taskgen", TASKGEN_ABLATION_CONDITIONS),
        ("toolgen", TOOLGEN_ABLATION_CONDITIONS),
    ):
        present = [
            condition for condition in required if (component, condition) in grouped
        ]
        missing = [condition for condition in required if condition not in present]
        component_sets = [
            input_sets[(component, condition)] for condition in present
        ]
        paired_inputs = bool(component_sets) and all(
            values == component_sets[0] for values in component_sets[1:]
        )
        coverage[component] = {
            "present_conditions": present,
            "missing_conditions": missing,
            "paired_input_sets_across_present_conditions": paired_inputs,
            "paper_condition_complete": not missing and paired_inputs,
        }
        for condition in present:
            rows = grouped[(component, condition)]
            results[component][condition] = {
                "attempts": len(rows),
                "syntax_pass_rate": sum(row["syntax_valid"] for row in rows)
                / len(rows),
                "downstream_pass_rate": sum(
                    row["downstream_valid"] for row in rows
                )
                / len(rows),
                "accepted": sum(row["accepted"] for row in rows),
                "acceptance_rate": sum(row["accepted"] for row in rows)
                / len(rows),
                "input_ids": sorted(input_sets[(component, condition)]),
            }
    table3_ready = all(
        coverage[component]["paper_condition_complete"]
        for component in ("taskgen", "toolgen")
    )
    if proposal_only:
        for component_rows in results.values():
            for row in component_rows.values():
                row["json_parse_pass_rate"] = row.pop("syntax_pass_rate")
                row["proposal_proxy_pass_rate"] = row.pop(
                    "downstream_pass_rate"
                )
        for row in coverage.values():
            row["prompt_condition_complete"] = row.pop(
                "paper_condition_complete"
            )
    return _result_envelope(
        protocol=protocol,
        study_id=study_id,
        source=root,
        result={
            "claim_scope": (
                "observed_structured_proposal_attempts_not_codegen_or_table3"
                if proposal_only
                else "observed_codegen_attempts_not_paper_scale_table3"
            ),
            "attempt_count": len(attempts),
            "conditions": results,
            "coverage": coverage,
            "failure_stage_counts": dict(sorted(failure_stages.items())),
            (
                "prompt_condition_matrix_complete"
                if proposal_only
                else "table3_minimum_condition_coverage"
            ): table3_ready,
            "task_or_tool_code_generated_and_executed": (
                False if proposal_only else None
            ),
            "claim_status": (
                (
                    "proposal_prompt_matrix_complete_no_codegen_evidence"
                    if table3_ready
                    else "partial_proposal_prompt_matrix_no_codegen_evidence"
                )
                if proposal_only
                else (
                    "minimum_real_attempt_matrix_complete"
                    if table3_ready
                    else "partial_real_attempt_matrix"
                )
            ),
        },
    )


def evaluate_codegen_ablation(value: Any) -> dict[str, Any]:
    """Aggregate actual provider code-generation attempts by ablation arm."""

    return _evaluate_ablation_attempts(
        value,
        protocol=CODEGEN_ABLATION_PROTOCOL,
        proposal_only=False,
    )


def evaluate_proposal_prompt_ablation(value: Any) -> dict[str, Any]:
    """Aggregate structured proposal responses without calling them codegen."""

    return _evaluate_ablation_attempts(
        value,
        protocol=PROPOSAL_PROMPT_ABLATION_PROTOCOL,
        proposal_only=True,
    )


def evaluate_error_distribution(value: Any) -> dict[str, Any]:
    """Compute error rate and stage distribution from all runtime operations."""

    root = _header(value, protocol=ERROR_DISTRIBUTION_PROTOCOL)
    study_id = _identifier(root.get("study_id"), field="study_id")
    operations = _items(root.get("operations"), field="operations")
    seen: set[str] = set()
    totals = {stage: 0 for stage in ERROR_STAGES}
    errors = {stage: 0 for stage in ERROR_STAGES}
    error_codes: dict[str, int] = defaultdict(int)
    run_ids: set[str] = set()
    for index, raw in enumerate(operations):
        operation = _object(raw, field=f"operations[{index}]")
        operation_id = _identifier(
            operation.get("operation_id"),
            field=f"operations[{index}].operation_id",
        )
        if operation_id in seen:
            raise PaperClaimDemoError(f"duplicate operation_id: {operation_id}")
        seen.add(operation_id)
        run_ids.add(
            _identifier(
                operation.get("run_id"), field=f"operations[{index}].run_id"
            )
        )
        stage = str(operation.get("stage"))
        if stage not in ERROR_STAGES:
            raise PaperClaimDemoError(
                f"operations[{index}].stage must be one of {list(ERROR_STAGES)}"
            )
        _text(
            operation.get("operation_ref"),
            field=f"operations[{index}].operation_ref",
        )
        status = str(operation.get("status"))
        if status not in {"success", "error"}:
            raise PaperClaimDemoError(
                f"operations[{index}].status must be success or error"
            )
        totals[stage] += 1
        if status == "error":
            error_code = _identifier(
                operation.get("error_code"),
                field=f"operations[{index}].error_code",
            )
            errors[stage] += 1
            error_codes[error_code] += 1
        elif operation.get("error_code") is not None:
            raise PaperClaimDemoError(
                f"successful operation {operation_id} cannot have error_code"
            )
    operation_count = len(operations)
    error_count = sum(errors.values())
    stage_rows: dict[str, dict[str, Any]] = {}
    for stage in ERROR_STAGES:
        stage_rows[stage] = {
            "operations": totals[stage],
            "errors": errors[stage],
            "error_rate": errors[stage] / totals[stage] if totals[stage] else None,
            "share_of_all_errors": (
                errors[stage] / error_count if error_count else 0.0
            ),
        }
    error_rate = error_count / operation_count
    return _result_envelope(
        protocol=ERROR_DISTRIBUTION_PROTOCOL,
        study_id=study_id,
        source=root,
        result={
            "claim_scope": "observed_operations_not_paper_scale_error_audit",
            "run_count": len(run_ids),
            "operation_count": operation_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "paper_reference_error_rate": 0.05,
            "absolute_distance_from_paper_reference": abs(error_rate - 0.05),
            "stage_coverage": {
                "present": [stage for stage in ERROR_STAGES if totals[stage]],
                "missing": [stage for stage in ERROR_STAGES if not totals[stage]],
            },
            "stages": stage_rows,
            "error_code_counts": dict(sorted(error_codes.items())),
            "claim_status": "observed_error_distribution_only",
        },
    )


_EVALUATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    EFFICIENCY_PROTOCOL: evaluate_small_efficiency,
    RANKING_PROTOCOL: evaluate_policy_ranking,
    PROXY_VALIDITY_PROTOCOL: evaluate_proxy_validity,
    CODEGEN_ABLATION_PROTOCOL: evaluate_codegen_ablation,
    PROPOSAL_PROMPT_ABLATION_PROTOCOL: evaluate_proposal_prompt_ablation,
    ERROR_DISTRIBUTION_PROTOCOL: evaluate_error_distribution,
}


def evaluate_paper_claim_manifest(value: Any) -> dict[str, Any]:
    """Dispatch a real claim-demo manifest to its deterministic evaluator."""

    root = _object(value, field="manifest")
    protocol = root.get("protocol")
    evaluator = _EVALUATORS.get(str(protocol))
    if evaluator is None:
        raise PaperClaimDemoError(
            f"unsupported protocol {protocol!r}; expected one of "
            f"{sorted(_EVALUATORS)}"
        )
    return evaluator(root)
