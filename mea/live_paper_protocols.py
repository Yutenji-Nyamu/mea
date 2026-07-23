"""Fail-closed, no-execution protocols for the next paper-evidence runs.

The builders in this module preregister experiments.  The evaluators only
consume receipts from runs that happened elsewhere; they never start a
provider, simulator, expert, probe, or policy rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.paper_claim_demo import evaluate_policy_ranking


class LivePaperProtocolError(ValueError):
    """Raised when a paper-evidence manifest violates its frozen contract."""


EFFICIENCY_PROTOCOL = "click_bell_independent_live_efficiency_v1"
RANKING_PROTOCOL = "act_dp3_exact_seed_pair_v1"
TABLE3_PROTOCOL = "table3_real_codegen_ablation_v1"
PROXY_PROTOCOL = "plan_vqa_development_proxy_manifest_v1"

CLICK_BELL_CANDIDATES = (
    {
        "candidate_id": "left_base0",
        "task_name": "click_bell",
        "position": "left",
        "instance": 0,
    },
    {
        "candidate_id": "right_base0",
        "task_name": "click_bell",
        "position": "right",
        "instance": 0,
    },
    {
        "candidate_id": "left_base1",
        "task_name": "click_bell",
        "position": "left",
        "instance": 1,
    },
    {
        "candidate_id": "right_base1",
        "task_name": "click_bell",
        "position": "right",
        "instance": 1,
    },
)
_CANDIDATE_IDS = tuple(row["candidate_id"] for row in CLICK_BELL_CANDIDATES)
_EFFICIENCY_MODES = {
    "smoke_3act": {
        "fixed_candidates": ("left_base0", "right_base0"),
        "adaptive_min": 1,
        "adaptive_max": 1,
        "total_min": 3,
        "total_max": 3,
        "claim_scope": "three_act_mechanism_smoke_not_dense_reference",
    },
    "toy_5to7act": {
        "fixed_candidates": _CANDIDATE_IDS,
        "adaptive_min": 1,
        "adaptive_max": 3,
        "total_min": 5,
        "total_max": 7,
        "claim_scope": "independent_live_toy_not_paper_tables_1_2",
    },
}

TABLE3_CONDITIONS = (
    "complete",
    "base",
    "minus_rag",
    "minus_visual_self_check",
    "minus_readme_agent",
)
TABLE3_PROPOSALS = (
    {
        "proposal_id": "u01_click_bell_physical_decoy",
        "task_name": "click_bell",
        "prompt": "Add a visually similar physical decoy bell that must not be pressed.",
    },
    {
        "proposal_id": "u02_click_bell_partial_occlusion",
        "task_name": "click_bell",
        "prompt": "Partially occlude the bell contact point without moving the target.",
    },
    {
        "proposal_id": "u03_bbh_target_distractor",
        "task_name": "beat_block_hammer",
        "prompt": "Add a physical distractor block; hit only the designated target block.",
    },
    {
        "proposal_id": "u04_bbh_clearance",
        "task_name": "beat_block_hammer",
        "prompt": "Constrain hammer clearance around a nearby obstacle.",
    },
    {
        "proposal_id": "u05_bbh_precontact_motion",
        "task_name": "beat_block_hammer",
        "prompt": "Evaluate pre-contact smoothness before the first target contact.",
    },
)
PAPER_VQA_CONDITIONS = (
    "clean",
    "scene_clutter",
    "background_texture",
    "lighting",
)


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LivePaperProtocolError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LivePaperProtocolError(f"{field} must be an object")
    return dict(value)


def _items(value: Any, *, field: str, minimum: int = 0) -> list[Any]:
    if not isinstance(value, list) or len(value) < minimum:
        raise LivePaperProtocolError(f"{field} must be a list with >= {minimum} items")
    return list(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise LivePaperProtocolError(f"{field} must be non-empty text")
    return value.strip()


def _identifier(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if not all(ch.isalnum() or ch in "._-" for ch in text):
        raise LivePaperProtocolError(f"{field} must be an identifier")
    return text


def _sha256(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise LivePaperProtocolError(f"{field} must be 64 lowercase hex characters")
    return text


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LivePaperProtocolError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise LivePaperProtocolError(f"{field} must be finite and >= {minimum}")
    return float(value)


def _utc(value: Any, *, field: str) -> datetime:
    text = _text(value, field=field)
    if not text.endswith("Z"):
        raise LivePaperProtocolError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise LivePaperProtocolError(f"{field} is not a timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise LivePaperProtocolError(f"{field} must use UTC")
    return parsed


def _checkpoint(value: Any, *, field: str) -> dict[str, str]:
    row = _object(value, field=field)
    return {
        "checkpoint_id": _text(row.get("checkpoint_id"), field=f"{field}.checkpoint_id"),
        "artifact_sha256": _sha256(
            row.get("artifact_sha256"), field=f"{field}.artifact_sha256"
        ),
    }


def _seal(value: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[hash_field] = canonical_sha256(result)
    return result


def _verify_seal(value: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    supplied = _sha256(result.pop(hash_field, None), field=hash_field)
    expected = canonical_sha256(result)
    if supplied != expected:
        raise LivePaperProtocolError(f"{hash_field} mismatch")
    result[hash_field] = supplied
    return result


def build_click_bell_efficiency_preregistration(
    *,
    study_id: str,
    mode: str,
    checkpoint: Mapping[str, Any],
    seed: int,
    created_at_utc: str,
) -> dict[str, Any]:
    if mode not in _EFFICIENCY_MODES:
        raise LivePaperProtocolError(f"unknown efficiency mode: {mode}")
    spec = _EFFICIENCY_MODES[mode]
    _utc(created_at_utc, field="created_at_utc")
    body = {
        "schema_version": 1,
        "protocol": EFFICIENCY_PROTOCOL,
        "study_id": _identifier(study_id, field="study_id"),
        "created_at_utc": created_at_utc,
        "evidence_requirement": "independent_live_rollout_only",
        "mode": mode,
        "claim_scope": spec["claim_scope"],
        "query": (
            "Does at least one of the four frozen click_bell candidates fail, "
            "and which paired position/instance axis is directly contrasted?"
        ),
        "checkpoint": _checkpoint(checkpoint, field="checkpoint"),
        "seed": _integer(seed, field="seed"),
        "candidate_universe": deepcopy(list(CLICK_BELL_CANDIDATES)),
        "fixed_contract": {
            "candidate_ids": list(spec["fixed_candidates"]),
            "stop_reason": "fixed_suite_complete",
        },
        "adaptive_contract": {
            "candidate_ids": list(_CANDIDATE_IDS),
            "min_episode_starts": spec["adaptive_min"],
            "max_episode_starts": spec["adaptive_max"],
            "query_sufficient_rule": "at_least_one_completed_failure",
            "allowed_stop_reasons": ["query_sufficient", "budget_exhausted"],
        },
        "total_episode_start_contract": {
            "minimum": spec["total_min"],
            "maximum": spec["total_max"],
        },
        "conclusion_contract": {
            "score_semantics": "official_success_boolean",
            "overall_verdicts": [
                "weakness_observed",
                "frozen_suite_all_succeeded",
                "inconclusive",
            ],
            "axis_rule": "paired_binary_score_difference",
            "comparison_fields": ["overall_verdict", "weakness_axes"],
        },
        "provenance_contract": {
            "forbidden_designs": [
                "cached_prefix_counterfactual",
                "posthoc_arm_split",
                "shared_rollout_receipt",
            ],
            "required_attempt_fields": [
                "attempt_id",
                "candidate_id",
                "seed",
                "evidence_source",
                "rollout_ref",
                "started_at_utc",
                "ended_at_utc",
                "wall_seconds",
                "status",
                "success",
            ],
        },
        "execution_entrypoint": "policy/ACT/eval_mea.sh",
        "calls_started_by_preregistration": {
            "provider": 0,
            "simulator": 0,
            "expert": 0,
            "probe": 0,
            "act": 0,
        },
    }
    return _seal(body, hash_field="preregistration_sha256")


def validate_click_bell_efficiency_preregistration(value: Any) -> dict[str, Any]:
    row = _verify_seal(_object(value, field="preregistration"), hash_field="preregistration_sha256")
    if row.get("schema_version") != 1 or row.get("protocol") != EFFICIENCY_PROTOCOL:
        raise LivePaperProtocolError("unsupported efficiency preregistration")
    mode = row.get("mode")
    if mode not in _EFFICIENCY_MODES:
        raise LivePaperProtocolError("efficiency mode is not frozen")
    if row.get("candidate_universe") != list(CLICK_BELL_CANDIDATES):
        raise LivePaperProtocolError("candidate universe differs from the frozen four")
    rebuilt = build_click_bell_efficiency_preregistration(
        study_id=row.get("study_id"),
        mode=mode,
        checkpoint=row.get("checkpoint"),
        seed=row.get("seed"),
        created_at_utc=row.get("created_at_utc"),
    )
    if rebuilt != row:
        raise LivePaperProtocolError("preregistration contract was modified")
    return row


def _live_attempt(
    value: Any,
    *,
    field: str,
    seed: int,
    allowed_candidates: set[str],
    preregistered_at: datetime,
) -> dict[str, Any]:
    row = _object(value, field=field)
    expected = {
        "attempt_id",
        "candidate_id",
        "seed",
        "evidence_source",
        "rollout_ref",
        "started_at_utc",
        "ended_at_utc",
        "wall_seconds",
        "status",
        "success",
    }
    if set(row) != expected:
        raise LivePaperProtocolError(f"{field} fields must be exactly {sorted(expected)}")
    candidate = _identifier(row["candidate_id"], field=f"{field}.candidate_id")
    if candidate not in allowed_candidates:
        raise LivePaperProtocolError(f"{field} candidate is outside frozen arm")
    if row["seed"] != seed:
        raise LivePaperProtocolError(f"{field}.seed differs from exact seed")
    if row["evidence_source"] != "live_policy_rollout":
        raise LivePaperProtocolError(f"{field} must be a live rollout, never cached")
    start = _utc(row["started_at_utc"], field=f"{field}.started_at_utc")
    end = _utc(row["ended_at_utc"], field=f"{field}.ended_at_utc")
    if start < preregistered_at or end < start:
        raise LivePaperProtocolError(f"{field} timestamps violate preregistration order")
    wall = _number(row["wall_seconds"], field=f"{field}.wall_seconds")
    if wall > (end - start).total_seconds() + 1.0:
        raise LivePaperProtocolError(f"{field}.wall_seconds exceeds elapsed time")
    status = row["status"]
    if status not in {"completed", "runtime_error"}:
        raise LivePaperProtocolError(f"{field}.status is invalid")
    success = row["success"]
    if status == "completed" and not isinstance(success, bool):
        raise LivePaperProtocolError(f"{field}.success must be boolean when completed")
    if status == "runtime_error" and success is not None:
        raise LivePaperProtocolError(f"{field}.success must be null on runtime_error")
    return {
        "attempt_id": _identifier(row["attempt_id"], field=f"{field}.attempt_id"),
        "candidate_id": candidate,
        "seed": seed,
        "evidence_source": "live_policy_rollout",
        "rollout_ref": _text(row["rollout_ref"], field=f"{field}.rollout_ref"),
        "started_at_utc": row["started_at_utc"],
        "ended_at_utc": row["ended_at_utc"],
        "wall_seconds": wall,
        "status": status,
        "success": success,
    }


def _efficiency_arm(
    value: Any,
    *,
    arm: str,
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(value, field=f"{arm}_result")
    expected = {
        "schema_version",
        "protocol",
        "arm",
        "arm_run_id",
        "preregistration_sha256",
        "started_at_utc",
        "ended_at_utc",
        "wall_seconds",
        "stop_reason",
        "attempts",
    }
    if set(row) != expected:
        raise LivePaperProtocolError(f"{arm} result fields must be exactly {sorted(expected)}")
    if row["schema_version"] != 1 or row["protocol"] != f"{EFFICIENCY_PROTOCOL}_arm":
        raise LivePaperProtocolError(f"unsupported {arm} result")
    if row["arm"] != arm or row["preregistration_sha256"] != prereg["preregistration_sha256"]:
        raise LivePaperProtocolError(f"{arm} result is not bound to preregistration")
    preregistered_at = _utc(prereg["created_at_utc"], field="created_at_utc")
    start = _utc(row["started_at_utc"], field=f"{arm}.started_at_utc")
    end = _utc(row["ended_at_utc"], field=f"{arm}.ended_at_utc")
    if start < preregistered_at or end < start:
        raise LivePaperProtocolError(f"{arm} timestamps violate preregistration order")
    wall = _number(row["wall_seconds"], field=f"{arm}.wall_seconds")
    if wall > (end - start).total_seconds() + 1.0:
        raise LivePaperProtocolError(f"{arm}.wall_seconds exceeds elapsed time")
    allowed = (
        set(prereg["fixed_contract"]["candidate_ids"])
        if arm == "fixed"
        else set(prereg["adaptive_contract"]["candidate_ids"])
    )
    attempts = [
        _live_attempt(
            item,
            field=f"{arm}.attempts[{index}]",
            seed=prereg["seed"],
            allowed_candidates=allowed,
            preregistered_at=preregistered_at,
        )
        for index, item in enumerate(_items(row["attempts"], field=f"{arm}.attempts", minimum=1))
    ]
    identities = [item["attempt_id"] for item in attempts]
    refs = [item["rollout_ref"] for item in attempts]
    candidates = [item["candidate_id"] for item in attempts]
    if len(identities) != len(set(identities)) or len(refs) != len(set(refs)):
        raise LivePaperProtocolError(f"{arm} attempt ids and rollout refs must be unique")
    if len(candidates) != len(set(candidates)):
        raise LivePaperProtocolError(f"{arm} cannot retry or reuse a candidate in this bounded pilot")
    completed = {item["candidate_id"] for item in attempts if item["status"] == "completed"}
    if arm == "fixed":
        required = set(prereg["fixed_contract"]["candidate_ids"])
        if completed != required or len(attempts) != len(required):
            raise LivePaperProtocolError("fixed arm must complete its exact preregistered suite")
        if row["stop_reason"] != "fixed_suite_complete":
            raise LivePaperProtocolError("fixed arm must stop with fixed_suite_complete")
    else:
        contract = prereg["adaptive_contract"]
        if not contract["min_episode_starts"] <= len(attempts) <= contract["max_episode_starts"]:
            raise LivePaperProtocolError("adaptive arm start count violates frozen budget")
        has_failure = any(
            item["status"] == "completed" and item["success"] is False
            for item in attempts
        )
        if row["stop_reason"] == "query_sufficient":
            if not has_failure:
                raise LivePaperProtocolError("query_sufficient requires a completed failure")
        elif row["stop_reason"] == "budget_exhausted":
            if len(attempts) != contract["max_episode_starts"] or has_failure:
                raise LivePaperProtocolError("budget_exhausted does not match adaptive evidence")
        else:
            raise LivePaperProtocolError("invalid adaptive stop reason")
    return {
        "arm": arm,
        "arm_run_id": _identifier(row["arm_run_id"], field=f"{arm}.arm_run_id"),
        "started_at_utc": row["started_at_utc"],
        "ended_at_utc": row["ended_at_utc"],
        "wall_seconds": wall,
        "stop_reason": row["stop_reason"],
        "attempts": attempts,
    }


def _efficiency_conclusion(arm: Mapping[str, Any]) -> dict[str, Any]:
    scores = {
        item["candidate_id"]: item["success"]
        for item in arm["attempts"]
        if item["status"] == "completed"
    }
    failures = sorted(key for key, value in scores.items() if value is False)
    if failures:
        verdict = "weakness_observed"
    elif set(scores) == set(_CANDIDATE_IDS):
        verdict = "frozen_suite_all_succeeded"
    else:
        verdict = "inconclusive"
    axes: list[str] = []
    pairs = {
        "position": (
            ("left_base0", "right_base0"),
            ("left_base1", "right_base1"),
        ),
        "instance": (
            ("left_base0", "left_base1"),
            ("right_base0", "right_base1"),
        ),
    }
    for axis, comparisons in pairs.items():
        if any(
            left in scores and right in scores and scores[left] != scores[right]
            for left, right in comparisons
        ):
            axes.append(axis)
    return {
        "overall_verdict": verdict,
        "weakness_axes": axes,
        "observed_failure_candidates": failures,
        "tested_candidates": sorted(scores),
    }


def evaluate_click_bell_efficiency(
    preregistration: Any,
    fixed_result: Any,
    adaptive_result: Any,
) -> dict[str, Any]:
    prereg = validate_click_bell_efficiency_preregistration(preregistration)
    fixed = _efficiency_arm(fixed_result, arm="fixed", prereg=prereg)
    adaptive = _efficiency_arm(adaptive_result, arm="adaptive", prereg=prereg)
    if fixed["arm_run_id"] == adaptive["arm_run_id"]:
        raise LivePaperProtocolError("arms must have independent run ids")
    fixed_ids = {item["attempt_id"] for item in fixed["attempts"]}
    adaptive_ids = {item["attempt_id"] for item in adaptive["attempts"]}
    fixed_refs = {item["rollout_ref"] for item in fixed["attempts"]}
    adaptive_refs = {item["rollout_ref"] for item in adaptive["attempts"]}
    if fixed_ids & adaptive_ids or fixed_refs & adaptive_refs:
        raise LivePaperProtocolError("arms cannot share rollout receipts")
    total_starts = len(fixed["attempts"]) + len(adaptive["attempts"])
    budget = prereg["total_episode_start_contract"]
    if not budget["minimum"] <= total_starts <= budget["maximum"]:
        raise LivePaperProtocolError("pair violates frozen total ACT-start budget")
    conclusions = {
        "fixed": _efficiency_conclusion(fixed),
        "adaptive": _efficiency_conclusion(adaptive),
    }
    fields = prereg["conclusion_contract"]["comparison_fields"]
    agrees = all(conclusions["fixed"][field] == conclusions["adaptive"][field] for field in fields)
    act_saving = len(fixed["attempts"]) - len(adaptive["attempts"])
    wall_saving = fixed["wall_seconds"] - adaptive["wall_seconds"]
    technical_errors = sum(
        item["status"] != "completed"
        for arm in (fixed, adaptive)
        for item in arm["attempts"]
    )
    eligible_toy = (
        prereg["mode"] == "toy_5to7act"
        and technical_errors == 0
        and agrees
        and act_saving > 0
        and wall_saving > 0
    )
    return {
        "schema_version": 1,
        "protocol": f"{EFFICIENCY_PROTOCOL}_result",
        "study_id": prereg["study_id"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "comparison_design": "independent_live_arms",
        "cached_prefix_used": False,
        "mode": prereg["mode"],
        "claim_scope": prereg["claim_scope"],
        "arms": {"fixed": fixed, "adaptive": adaptive},
        "conclusions": conclusions,
        "conclusion_comparison_fields": fields,
        "original_query_conclusion_agrees": agrees,
        "resource_measurement": {
            "fixed_act_episode_starts": len(fixed["attempts"]),
            "adaptive_act_episode_starts": len(adaptive["attempts"]),
            "act_episode_start_saving": act_saving,
            "fixed_wall_seconds": fixed["wall_seconds"],
            "adaptive_wall_seconds": adaptive["wall_seconds"],
            "measured_wall_second_saving": wall_saving,
            "technical_runtime_errors": technical_errors,
        },
        "toy_efficiency_evidence_passed": eligible_toy,
        "paper_tables_1_2_eligible": False,
        "limitations": [
            "The three-ACT mode is a mechanism smoke, not a dense reference.",
            "The five-to-seven-ACT mode is one task, one checkpoint, and one seed.",
            "This protocol does not reproduce the paper trial or agent-run counts.",
        ],
    }


def build_ranking_preregistration(
    *,
    study_id: str,
    act_checkpoint: Mapping[str, Any],
    dp3_checkpoint: Mapping[str, Any],
    seeds: Sequence[int],
    created_at_utc: str,
    reference_source_ref: str,
    reference_scores: Mapping[str, float],
) -> dict[str, Any]:
    _utc(created_at_utc, field="created_at_utc")
    normalized_seeds = [_integer(seed, field="seeds[]") for seed in seeds]
    if len(normalized_seeds) != 3 or len(set(normalized_seeds)) != 3:
        raise LivePaperProtocolError("ranking pilot requires exactly three unique seeds")
    if set(reference_scores) != {"act", "dp3"}:
        raise LivePaperProtocolError("reference_scores must contain exactly act and dp3")
    body = {
        "schema_version": 1,
        "protocol": RANKING_PROTOCOL,
        "study_id": _identifier(study_id, field="study_id"),
        "created_at_utc": created_at_utc,
        "candidate_id": "bbh_official_demo_clean",
        "seeds": normalized_seeds,
        "policies": {
            "act": _checkpoint(act_checkpoint, field="act_checkpoint"),
            "dp3": _checkpoint(dp3_checkpoint, field="dp3_checkpoint"),
        },
        "reference_source_ref": _text(reference_source_ref, field="reference_source_ref"),
        "reference_scores": {
            key: _number(reference_scores[key], field=f"reference_scores.{key}")
            for key in ("act", "dp3")
        },
        "execution_entrypoints": {
            "act": "policy/ACT/eval_mea.sh",
            "dp3": "policy/DP3/eval.sh",
        },
        "rollout_contract": {
            "exact_trials_per_policy": 3,
            "exact_total_policy_rollouts": 6,
            "evidence_source": "live_policy_rollout",
            "tie_rule": "spearman_null_and_inconclusive",
        },
        "claim_scope": "two_policy_three_seed_pair_order_pilot_not_table9",
        "calls_started_by_preregistration": {
            "provider": 0,
            "simulator": 0,
            "expert": 0,
            "probe": 0,
            "act": 0,
        },
    }
    return _seal(body, hash_field="preregistration_sha256")


def validate_ranking_preregistration(value: Any) -> dict[str, Any]:
    row = _verify_seal(_object(value, field="ranking preregistration"), hash_field="preregistration_sha256")
    if row.get("schema_version") != 1 or row.get("protocol") != RANKING_PROTOCOL:
        raise LivePaperProtocolError("unsupported ranking preregistration")
    rebuilt = build_ranking_preregistration(
        study_id=row.get("study_id"),
        act_checkpoint=_object(row.get("policies"), field="policies").get("act"),
        dp3_checkpoint=_object(row.get("policies"), field="policies").get("dp3"),
        seeds=row.get("seeds"),
        created_at_utc=row.get("created_at_utc"),
        reference_source_ref=row.get("reference_source_ref"),
        reference_scores=row.get("reference_scores"),
    )
    if rebuilt != row:
        raise LivePaperProtocolError("ranking preregistration contract was modified")
    return row


def evaluate_exact_seed_ranking(preregistration: Any, result_manifest: Any) -> dict[str, Any]:
    prereg = validate_ranking_preregistration(preregistration)
    result = _object(result_manifest, field="ranking result")
    if result.get("schema_version") != 1 or result.get("protocol") != f"{RANKING_PROTOCOL}_runs":
        raise LivePaperProtocolError("unsupported ranking result manifest")
    if result.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise LivePaperProtocolError("ranking result is not bound to preregistration")
    policy_rows = _items(result.get("policies"), field="policies", minimum=2)
    if {row.get("policy_id") for row in policy_rows if isinstance(row, Mapping)} != {"act", "dp3"}:
        raise LivePaperProtocolError("ranking result must contain exactly ACT and DP3")
    converted: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    rollout_refs: set[str] = set()
    total_wall = 0.0
    preregistered_at = _utc(prereg["created_at_utc"], field="created_at_utc")
    for raw_policy in policy_rows:
        policy = _object(raw_policy, field="policy")
        policy_id = policy.get("policy_id")
        checkpoint = _checkpoint(policy.get("checkpoint"), field=f"{policy_id}.checkpoint")
        if checkpoint != prereg["policies"][policy_id]:
            raise LivePaperProtocolError(f"{policy_id} checkpoint differs from preregistration")
        run_id = _identifier(policy.get("run_id"), field=f"{policy_id}.run_id")
        if run_id in run_ids:
            raise LivePaperProtocolError("policy run ids must be independent")
        run_ids.add(run_id)
        trials = _items(policy.get("trials"), field=f"{policy_id}.trials", minimum=3)
        if len(trials) != 3:
            raise LivePaperProtocolError("each policy requires exactly three trials")
        seen_seeds: set[int] = set()
        converted_trials: list[dict[str, Any]] = []
        for index, raw_trial in enumerate(trials):
            trial = _object(raw_trial, field=f"{policy_id}.trials[{index}]")
            seed = _integer(trial.get("seed"), field="trial.seed")
            if seed not in prereg["seeds"] or seed in seen_seeds:
                raise LivePaperProtocolError("policy trials must cover exact unique seeds")
            seen_seeds.add(seed)
            if trial.get("evidence_source") != "live_policy_rollout":
                raise LivePaperProtocolError("ranking trials must be live, never cached")
            if trial.get("status") != "completed":
                raise LivePaperProtocolError("ranking requires six completed trials")
            start = _utc(trial.get("started_at_utc"), field="trial.started_at_utc")
            end = _utc(trial.get("ended_at_utc"), field="trial.ended_at_utc")
            if start < preregistered_at or end < start:
                raise LivePaperProtocolError("ranking trial predates preregistration")
            wall = _number(trial.get("wall_seconds"), field="trial.wall_seconds")
            if wall > (end - start).total_seconds() + 1.0:
                raise LivePaperProtocolError("trial wall time exceeds elapsed time")
            total_wall += wall
            rollout_ref = _text(trial.get("rollout_ref"), field="trial.rollout_ref")
            if rollout_ref in rollout_refs:
                raise LivePaperProtocolError("ranking trials cannot share rollout refs")
            rollout_refs.add(rollout_ref)
            score = _number(trial.get("score"), field="trial.score")
            if score > 1.0:
                raise LivePaperProtocolError("trial score must be in [0, 1]")
            converted_trials.append(
                {
                    "trial_id": _identifier(
                        trial.get("trial_id"), field="trial.trial_id"
                    ),
                    "candidate_id": prereg["candidate_id"],
                    "seed": seed,
                    "rollout_ref": rollout_ref,
                    "episode_status": "completed",
                    "score": score,
                }
            )
        if seen_seeds != set(prereg["seeds"]):
            raise LivePaperProtocolError("policy is missing an exact seed")
        converted.append(
            {
                "policy_id": policy_id,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "run_id": run_id,
                "trials": converted_trials,
            }
        )
    ranking = evaluate_policy_ranking(
        {
            "schema_version": 1,
            "protocol": "paper_claim_policy_ranking_v1",
            "evidence_source": "live_policy_rollout",
            "study_id": prereg["study_id"],
            "candidate_universe": [prereg["candidate_id"]],
            "seeds": prereg["seeds"],
            "reference_source_ref": prereg["reference_source_ref"],
            "reference_scores": prereg["reference_scores"],
            "policies": converted,
        }
    )
    ranking.update(
        {
            "preregistration_sha256": prereg["preregistration_sha256"],
            "exact_seed_pair": True,
            "exact_trials_per_policy": 3,
            "exact_total_policy_rollouts": 6,
            "measured_trial_wall_seconds_total": total_wall,
            "paper_table9_eligible": False,
            "scope_limitation": (
                "Two policies, one task, and three seeds; a tie leaves Spearman null."
            ),
        }
    )
    return ranking


def build_table3_codegen_preregistration(
    *, study_id: str, created_at_utc: str
) -> dict[str, Any]:
    _utc(created_at_utc, field="created_at_utc")
    cells = [
        {
            "cell_id": f"{proposal['proposal_id']}__{condition}",
            "proposal_id": proposal["proposal_id"],
            "condition": condition,
        }
        for proposal in TABLE3_PROPOSALS
        for condition in TABLE3_CONDITIONS
    ]
    body = {
        "schema_version": 1,
        "protocol": TABLE3_PROTOCOL,
        "study_id": _identifier(study_id, field="study_id"),
        "created_at_utc": created_at_utc,
        "unseen_proposals": deepcopy(list(TABLE3_PROPOSALS)),
        "conditions": list(TABLE3_CONDITIONS),
        "cells": cells,
        "required_downstream_stages": [
            "codegen",
            "compile",
            "render",
            "simulator",
            "oracle",
        ],
        "success_rule": "all_five_downstream_stages_pass",
        "oracle_fixture_minimum": {"positive": 1, "negative": 1},
        "act_rollout_budget": 0,
        "claim_scope": "five_unseen_proposals_per_condition_micro_ablation_not_table3",
    }
    return _seal(body, hash_field="preregistration_sha256")


def validate_table3_codegen_preregistration(value: Any) -> dict[str, Any]:
    row = _verify_seal(_object(value, field="table3 preregistration"), hash_field="preregistration_sha256")
    if row.get("schema_version") != 1 or row.get("protocol") != TABLE3_PROTOCOL:
        raise LivePaperProtocolError("unsupported Table 3 preregistration")
    rebuilt = build_table3_codegen_preregistration(
        study_id=row.get("study_id"), created_at_utc=row.get("created_at_utc")
    )
    if rebuilt != row:
        raise LivePaperProtocolError("Table 3 preregistration contract was modified")
    return row


def evaluate_table3_codegen(preregistration: Any, result_manifest: Any) -> dict[str, Any]:
    prereg = validate_table3_codegen_preregistration(preregistration)
    result = _object(result_manifest, field="table3 result")
    if result.get("schema_version") != 1 or result.get("protocol") != f"{TABLE3_PROTOCOL}_runs":
        raise LivePaperProtocolError("unsupported Table 3 result")
    if result.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise LivePaperProtocolError("Table 3 result is not bound to preregistration")
    raw_cells = _items(result.get("cells"), field="cells", minimum=25)
    expected = {cell["cell_id"]: cell for cell in prereg["cells"]}
    if len(raw_cells) != 25 or {cell.get("cell_id") for cell in raw_cells if isinstance(cell, Mapping)} != set(expected):
        raise LivePaperProtocolError("Table 3 requires the exact 5x5 cell grid")
    rows: list[dict[str, Any]] = []
    for raw in raw_cells:
        cell = _object(raw, field="cell")
        cell_id = cell.get("cell_id")
        frozen = expected[cell_id]
        if cell.get("proposal_id") != frozen["proposal_id"] or cell.get("condition") != frozen["condition"]:
            raise LivePaperProtocolError(f"cell identity differs from preregistration: {cell_id}")
        stages = _object(cell.get("stages"), field=f"{cell_id}.stages")
        if set(stages) != set(prereg["required_downstream_stages"]):
            raise LivePaperProtocolError(f"{cell_id} is missing downstream stages")
        codegen = _object(stages["codegen"], field=f"{cell_id}.codegen")
        if codegen.get("generated_by_provider") is not True:
            raise LivePaperProtocolError(f"{cell_id} is proposal-only, not real codegen")
        _text(codegen.get("artifact_ref"), field=f"{cell_id}.codegen.artifact_ref")
        _sha256(codegen.get("artifact_sha256"), field=f"{cell_id}.codegen.artifact_sha256")
        stage_pass = [True]
        for stage_name in ("compile", "render", "simulator", "oracle"):
            stage = _object(stages[stage_name], field=f"{cell_id}.{stage_name}")
            if not isinstance(stage.get("passed"), bool):
                raise LivePaperProtocolError(f"{cell_id}.{stage_name}.passed must be boolean")
            _text(stage.get("receipt_ref"), field=f"{cell_id}.{stage_name}.receipt_ref")
            _sha256(stage.get("receipt_sha256"), field=f"{cell_id}.{stage_name}.receipt_sha256")
            stage_pass.append(stage["passed"])
            if stage_name == "oracle":
                if _integer(stage.get("positive_fixture_count"), field="positive_fixture_count", minimum=1) < 1:
                    raise LivePaperProtocolError("oracle requires a positive fixture")
                if _integer(stage.get("negative_fixture_count"), field="negative_fixture_count", minimum=1) < 1:
                    raise LivePaperProtocolError("oracle requires a negative fixture")
        rows.append(
            {
                "cell_id": cell_id,
                "proposal_id": frozen["proposal_id"],
                "condition": frozen["condition"],
                "success": all(stage_pass),
            }
        )
    rates = {
        condition: sum(row["success"] for row in rows if row["condition"] == condition) / 5.0
        for condition in TABLE3_CONDITIONS
    }
    return {
        "schema_version": 1,
        "protocol": f"{TABLE3_PROTOCOL}_result",
        "study_id": prereg["study_id"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "cell_count": 25,
        "provider_generation_count": 25,
        "act_rollouts_started": 0,
        "rows": rows,
        "success_rates": rates,
        "paper_table3_eligible": False,
        "claim_scope": prereg["claim_scope"],
    }


def validate_proxy_gold_manifest(repo_root: str | Path, value: Any) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    row = _object(value, field="proxy manifest")
    if row.get("schema_version") != 1 or row.get("protocol") != PROXY_PROTOCOL:
        raise LivePaperProtocolError("unsupported proxy manifest")
    if row.get("annotator_kind") != "development_agent_proxy":
        raise LivePaperProtocolError("proxy manifest must not impersonate human gold")
    if row.get("human_reviewer_count") != 0 or row.get("paper_eligible") is not False:
        raise LivePaperProtocolError("development proxy must declare zero humans and paper_eligible=false")
    query_ref = _text(row.get("query_manifest_ref"), field="query_manifest_ref")
    query_path = (root / query_ref).resolve()
    if not query_path.is_relative_to(root) or not query_path.is_file():
        raise LivePaperProtocolError("query manifest ref is missing or outside repository")
    try:
        query_manifest = json.loads(query_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LivePaperProtocolError(f"cannot read query manifest: {exc}") from exc
    cases = _items(query_manifest.get("cases"), field="query cases", minimum=20)
    if not 20 <= len(cases) <= 30:
        raise LivePaperProtocolError("Plan proxy suite must contain 20-30 queries")
    for case in cases:
        annotation = _object(case.get("annotation"), field="query annotation")
        if (
            annotation.get("source") != "development_agent_proxy"
            or annotation.get("paper_eligible") is not False
            or annotation.get("human_votes") != []
        ):
            raise LivePaperProtocolError("query proxy labels must remain explicitly non-human")
    clips = _items(row.get("clip_slots"), field="clip_slots", minimum=8)
    expected = {(condition, polarity) for condition in PAPER_VQA_CONDITIONS for polarity in ("positive", "negative")}
    observed: set[tuple[str, str]] = set()
    materialized = 0
    for index, clip in enumerate(clips):
        clip = _object(clip, field=f"clip_slots[{index}]")
        pair = (clip.get("condition"), clip.get("polarity"))
        if pair not in expected or pair in observed:
            raise LivePaperProtocolError("clip slots must be the exact four-condition polarity grid")
        observed.add(pair)
        if not isinstance(clip.get("proxy_gold_observed"), bool):
            raise LivePaperProtocolError("clip proxy label must be boolean")
        if clip.get("label_source") != "development_agent_proxy":
            raise LivePaperProtocolError("clip label source must remain development proxy")
        is_materialized = clip.get("materialized")
        if not isinstance(is_materialized, bool):
            raise LivePaperProtocolError("clip materialized must be boolean")
        source_or_recipe_ref = _text(
            clip.get("source_or_recipe_ref"), field="source_or_recipe_ref"
        )
        if is_materialized:
            artifact = (root / source_or_recipe_ref).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise LivePaperProtocolError(
                    "materialized clip ref is missing or outside repository"
                )
        materialized += int(is_materialized)
    if observed != expected or len(clips) != 8:
        raise LivePaperProtocolError("clip slots must contain exactly 8 entries")
    return {
        "schema_version": 1,
        "protocol": f"{PROXY_PROTOCOL}_validation",
        "query_count": len(cases),
        "clip_slot_count": len(clips),
        "materialized_clip_count": materialized,
        "conditions": list(PAPER_VQA_CONDITIONS),
        "annotation_scope": "development_agent_proxy_not_human_gold",
        "human_reviewer_count": 0,
        "paper_plan_validity_eligible": False,
        "paper_vqa_robustness_eligible": False,
        "ready_for_proxy_smoke": materialized == len(clips),
    }
