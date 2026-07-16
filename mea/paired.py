"""Exact-seed manifests and deterministic two-condition paired statistics."""

from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL_ID = "exact_seed_paired_v1"
DEFAULT_CONDITIONS = (
    {"id": "easy", "task_config": "demo_clean"},
    {"id": "hard", "task_config": "demo_randomized"},
)
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_]*\Z")


class PairedProtocolError(ValueError):
    """Raised when a paired-evaluation artifact violates its contract."""


def _identifier(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise PairedProtocolError(f"{field} must be a lowercase identifier")
    return normalized


def _seeds(values: Any) -> list[int]:
    if not isinstance(values, list) or not values:
        raise PairedProtocolError("seeds must be a non-empty list")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PairedProtocolError("every seed must be a non-negative integer")
        if value in normalized:
            raise PairedProtocolError(f"duplicate seed: {value}")
        normalized.append(value)
    return normalized


def validate_seed_manifest(
    payload: Any,
    *,
    expected_task_name: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one exact-seed paired protocol manifest."""

    if not isinstance(payload, Mapping):
        raise PairedProtocolError("seed manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise PairedProtocolError("seed manifest schema_version must be 1")
    if payload.get("protocol") != PROTOCOL_ID:
        raise PairedProtocolError(f"seed manifest protocol must be {PROTOCOL_ID}")
    task_name = _identifier(payload.get("task_name"), field="task_name")
    if expected_task_name is not None and task_name != expected_task_name:
        raise PairedProtocolError(
            f"seed manifest task_name {task_name!r} does not match "
            f"{expected_task_name!r}"
        )
    seeds = _seeds(payload.get("seeds"))

    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list) or len(raw_conditions) != 2:
        raise PairedProtocolError("conditions must contain exactly two entries")
    conditions: list[dict[str, str]] = []
    condition_ids: set[str] = set()
    task_configs: set[str] = set()
    for raw in raw_conditions:
        if not isinstance(raw, Mapping):
            raise PairedProtocolError("each condition must be an object")
        condition_id = _identifier(raw.get("id"), field="condition.id")
        task_config = _identifier(
            raw.get("task_config"), field="condition.task_config"
        )
        if condition_id in condition_ids:
            raise PairedProtocolError(f"duplicate condition id: {condition_id}")
        if task_config in task_configs:
            raise PairedProtocolError(
                f"paired conditions must use distinct task configs: {task_config}"
            )
        condition_ids.add(condition_id)
        task_configs.add(task_config)
        conditions.append({"id": condition_id, "task_config": task_config})

    checkpoint_setting = _identifier(
        payload.get("checkpoint_setting", "demo_clean"),
        field="checkpoint_setting",
    )
    expert_data_num = payload.get("expert_data_num", 50)
    policy_seed = payload.get("policy_seed", 0)
    if (
        isinstance(expert_data_num, bool)
        or not isinstance(expert_data_num, int)
        or expert_data_num <= 0
    ):
        raise PairedProtocolError("expert_data_num must be a positive integer")
    if (
        isinstance(policy_seed, bool)
        or not isinstance(policy_seed, int)
        or policy_seed < 0
    ):
        raise PairedProtocolError("policy_seed must be a non-negative integer")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "task_name": task_name,
        "seeds": seeds,
        "conditions": conditions,
        "checkpoint_setting": checkpoint_setting,
        "expert_data_num": expert_data_num,
        "policy_seed": policy_seed,
    }


def build_seed_manifest(
    *,
    task_name: str,
    seeds: Sequence[int],
    conditions: Sequence[Mapping[str, Any]] = DEFAULT_CONDITIONS,
    checkpoint_setting: str = "demo_clean",
    expert_data_num: int = 50,
    policy_seed: int = 0,
) -> dict[str, Any]:
    """Build a normalized manifest from CLI-friendly values."""

    return validate_seed_manifest(
        {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "task_name": task_name,
            "seeds": list(seeds),
            "conditions": [dict(item) for item in conditions],
            "checkpoint_setting": checkpoint_setting,
            "expert_data_num": expert_data_num,
            "policy_seed": policy_seed,
        }
    )


def load_seed_manifest(
    path: str | Path,
    *,
    expected_task_name: str | None = None,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedProtocolError(f"cannot load seed manifest {source}: {exc}") from exc
    return validate_seed_manifest(payload, expected_task_name=expected_task_name)


def seed_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the normalized manifest using stable canonical JSON bytes."""

    normalized = validate_seed_manifest(payload)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measurement_map(
    condition: Mapping[str, Any],
    *,
    condition_id: str,
    requested_seeds: list[int],
) -> dict[int, dict[str, Any]]:
    measurements = condition.get("seed_measurements")
    if not isinstance(measurements, list):
        raise PairedProtocolError(
            f"condition {condition_id!r} seed_measurements must be a list"
        )
    result: dict[int, dict[str, Any]] = {}
    for raw in measurements:
        if not isinstance(raw, Mapping):
            raise PairedProtocolError(
                f"condition {condition_id!r} measurement must be an object"
            )
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PairedProtocolError(
                f"condition {condition_id!r} measurement seed must be an integer"
            )
        if seed in result:
            raise PairedProtocolError(
                f"condition {condition_id!r} has duplicate seed {seed}"
            )
        if seed not in requested_seeds:
            raise PairedProtocolError(
                f"condition {condition_id!r} has unrequested seed {seed}"
            )
        eligibility_status = str(raw.get("eligibility_status") or "")
        if eligibility_status not in {
            "passed",
            "unstable",
            "expert_failed",
            "error",
            "protocol_violation",
        }:
            raise PairedProtocolError(
                f"condition {condition_id!r} seed {seed} has invalid eligibility"
            )
        policy_executed = raw.get("policy_executed")
        policy_success = raw.get("policy_success")
        if not isinstance(policy_executed, bool):
            raise PairedProtocolError("policy_executed must be boolean")
        if policy_success is not None and not isinstance(policy_success, bool):
            raise PairedProtocolError("policy_success must be boolean or null")
        if policy_executed != (policy_success is not None):
            raise PairedProtocolError(
                "policy_success must be present exactly when policy was executed"
            )
        if policy_executed and eligibility_status != "passed":
            raise PairedProtocolError(
                "policy cannot be counted as executed unless eligibility passed"
            )
        time_to_success = raw.get("time_to_success")
        if time_to_success is not None:
            if (
                isinstance(time_to_success, bool)
                or not isinstance(time_to_success, (int, float))
                or not math.isfinite(float(time_to_success))
                or float(time_to_success) < 0
            ):
                raise PairedProtocolError(
                    "time_to_success must be a finite non-negative number or null"
                )
            time_to_success = float(time_to_success)
        if time_to_success is not None and policy_success is not True:
            raise PairedProtocolError(
                "time_to_success is valid only for a successful policy rollout"
            )
        normalized = dict(raw)
        normalized["time_to_success"] = time_to_success
        result[seed] = normalized
    missing = [seed for seed in requested_seeds if seed not in result]
    if missing:
        raise PairedProtocolError(
            f"condition {condition_id!r} is missing seeds: {missing}"
        )
    return result


def build_paired_summary(
    seed_manifest: Mapping[str, Any],
    condition_runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Join exact seed rows and compute deterministic paired outcomes."""

    manifest = validate_seed_manifest(seed_manifest)
    condition_specs = manifest["conditions"]
    first_id, second_id = [item["id"] for item in condition_specs]
    expected_ids = {first_id, second_id}
    if set(condition_runs) != expected_ids:
        raise PairedProtocolError(
            "condition run ids must exactly match the seed manifest"
        )
    by_condition = {
        condition_id: _measurement_map(
            condition_runs[condition_id],
            condition_id=condition_id,
            requested_seeds=manifest["seeds"],
        )
        for condition_id in (first_id, second_id)
    }
    eligibility_status_counts = {
        condition_id: {
            status: sum(
                row["eligibility_status"] == status
                for row in by_condition[condition_id].values()
            )
            for status in (
                "passed",
                "unstable",
                "expert_failed",
                "error",
                "protocol_violation",
            )
        }
        for condition_id in (first_id, second_id)
    }

    pairs: list[dict[str, Any]] = []
    outcomes = {
        "both_success": 0,
        f"{first_id}_only": 0,
        f"{second_id}_only": 0,
        "neither": 0,
    }
    paired_eligible_count = 0
    paired_evaluated_count = 0
    first_success_count = 0
    second_success_count = 0
    both_success_times: list[tuple[int, float, float]] = []
    for seed in manifest["seeds"]:
        first = by_condition[first_id][seed]
        second = by_condition[second_id][seed]
        eligible = all(
            item["eligibility_status"] == "passed" for item in (first, second)
        )
        evaluated = all(item["policy_executed"] for item in (first, second))
        if eligible:
            paired_eligible_count += 1
        outcome = None
        if evaluated:
            paired_evaluated_count += 1
            first_success = bool(first["policy_success"])
            second_success = bool(second["policy_success"])
            first_success_count += int(first_success)
            second_success_count += int(second_success)
            if first_success and second_success:
                outcome = "both_success"
            elif first_success:
                outcome = f"{first_id}_only"
            elif second_success:
                outcome = f"{second_id}_only"
            else:
                outcome = "neither"
            outcomes[outcome] += 1
            if (
                first_success
                and second_success
                and first.get("time_to_success") is not None
                and second.get("time_to_success") is not None
            ):
                both_success_times.append(
                    (
                        seed,
                        float(first["time_to_success"]),
                        float(second["time_to_success"]),
                    )
                )
        pairs.append(
            {
                "seed": seed,
                "paired_eligible": eligible,
                "paired_evaluated": evaluated,
                "outcome": outcome,
                "conditions": {
                    first_id: first,
                    second_id: second,
                },
            }
        )

    denominator = paired_evaluated_count
    first_rate = first_success_count / denominator if denominator else None
    second_rate = second_success_count / denominator if denominator else None
    time_rows = [
        {
            "seed": seed,
            first_id: first_time,
            second_id: second_time,
            f"{second_id}_minus_{first_id}": second_time - first_time,
        }
        for seed, first_time, second_time in both_success_times
    ]
    deltas = [row[f"{second_id}_minus_{first_id}"] for row in time_rows]
    protocol_violation_measurement_count = sum(
        counts["protocol_violation"]
        for counts in eligibility_status_counts.values()
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "status": "completed",
        "task_name": manifest["task_name"],
        "condition_order": [first_id, second_id],
        "requested_seed_count": len(manifest["seeds"]),
        "paired_eligible_count": paired_eligible_count,
        "paired_evaluated_count": paired_evaluated_count,
        "paired_not_evaluated_count": (
            len(manifest["seeds"]) - paired_evaluated_count
        ),
        "coverage_rate": (
            paired_evaluated_count / len(manifest["seeds"])
        ),
        "eligibility_status_counts": eligibility_status_counts,
        "protocol_violation_measurement_count": (
            protocol_violation_measurement_count
        ),
        "valid_for_comparison": (
            paired_evaluated_count > 0
            and protocol_violation_measurement_count == 0
        ),
        "success": {
            "denominator": denominator,
            first_id: {
                "count": first_success_count,
                "rate": first_rate,
            },
            second_id: {
                "count": second_success_count,
                "rate": second_rate,
            },
            f"{second_id}_minus_{first_id}": (
                second_rate - first_rate
                if first_rate is not None and second_rate is not None
                else None
            ),
            f"{first_id}_minus_{second_id}": (
                first_rate - second_rate
                if first_rate is not None and second_rate is not None
                else None
            ),
            "outcomes": outcomes,
        },
        "time_to_success": {
            "paired_both_success_count": len(time_rows),
            f"mean_{first_id}": (
                statistics.fmean(row[first_id] for row in time_rows)
                if time_rows
                else None
            ),
            f"mean_{second_id}": (
                statistics.fmean(row[second_id] for row in time_rows)
                if time_rows
                else None
            ),
            f"mean_{second_id}_minus_{first_id}": (
                statistics.fmean(deltas) if deltas else None
            ),
            "pairs": time_rows,
        },
        "pairs": pairs,
    }
