#!/usr/bin/env python3
"""Execute frozen live-paper commands and derive receipts from disk evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(root: Path, ref: str) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else root / path


def run_frozen_command(
    root: Path,
    command: dict[str, Any],
    *,
    log_path: Path,
) -> tuple[int, str, str, float]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in command.get("environment", {}).items()})
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise RuntimeError("command argv must be a non-empty string list")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()
    with log_path.open("wb") as log:
        completed = subprocess.run(
            argv,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_seconds = time.monotonic() - started
    ended_at = utc_now()
    return completed.returncode, started_at, ended_at, wall_seconds


def efficiency_attempt(
    root: Path,
    prereg: dict[str, Any],
    *,
    arm: str,
    arm_run_id: str,
    binding: dict[str, Any],
    ordinal: int,
) -> tuple[dict[str, Any], bool | None]:
    command_path = resolve(root, binding["command_ref"])
    if sha256(command_path) != binding["command_sha256"]:
        raise RuntimeError(f"command hash mismatch before execution: {command_path}")
    command = read_json(command_path)
    attempt_id = f"{arm}_attempt_{ordinal:02d}"
    log_path = command_path.parent / "live.log"
    returncode, started_at, ended_at, wall_seconds = run_frozen_command(
        root, command, log_path=log_path
    )
    receipt_path = resolve(root, binding["receipt_ref"])
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "click_bell_bound_live_rollout_receipt_v1",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "arm": arm,
        "arm_run_id": arm_run_id,
        "attempt_id": attempt_id,
        "candidate_id": binding["candidate_id"],
        "variant_id": binding["variant_id"],
        "variant_manifest_sha256": next(
            candidate["variant_binding"]["variant_manifest_sha256"]
            for candidate in prereg["candidate_universe"]
            if candidate["candidate_id"] == binding["candidate_id"]
        ),
        "command_sha256": binding["command_sha256"],
        "checkpoint_sha256": prereg["checkpoint"]["artifact_sha256"],
        "seed": prereg["seed"],
        "evidence_source": "live_policy_rollout",
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_seconds": wall_seconds,
        "status": "runtime_error" if returncode else "completed",
        "success": None,
        "seed_results_ref": None,
        "seed_results_sha256": None,
        "telemetry_episode_ref": None,
        "telemetry_episode_sha256": None,
    }
    success: bool | None = None
    if returncode == 0:
        seed_results_path = resolve(root, binding["expected_seed_results_ref"])
        telemetry_path = resolve(root, binding["expected_telemetry_episode_ref"])
        seed_results = read_json(seed_results_path)
        measurements = seed_results.get("seed_measurements")
        if not isinstance(measurements, list) or len(measurements) != 1:
            raise RuntimeError("live command did not produce exact N=1 seed results")
        success = measurements[0].get("policy_success")
        if not isinstance(success, bool):
            raise RuntimeError("live command did not produce boolean policy_success")
        receipt.update(
            {
                "success": success,
                "seed_results_ref": binding["expected_seed_results_ref"],
                "seed_results_sha256": sha256(seed_results_path),
                "telemetry_episode_ref": binding["expected_telemetry_episode_ref"],
                "telemetry_episode_sha256": sha256(telemetry_path),
            }
        )
    write_json(receipt_path, receipt)
    if returncode != 0:
        raise RuntimeError(
            f"{arm} {binding['candidate_id']} failed with returncode "
            f"{returncode}; see {log_path}"
        )
    attempt = {
        "attempt_id": attempt_id,
        "candidate_id": binding["candidate_id"],
        "receipt_ref": binding["receipt_ref"],
        "receipt_sha256": sha256(receipt_path),
    }
    return attempt, success


def run_efficiency(root: Path, prereg_path: Path, output_root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mea.live_paper_protocols import evaluate_click_bell_efficiency
    from mea.planner.query_contract import assess_query_sufficiency

    prereg = read_json(prereg_path)
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for arm in ("fixed", "adaptive"):
        arm_run_id = f"{prereg['study_id']}_{arm}_live"
        attempts: list[dict[str, Any]] = []
        observed: dict[str, bool | None] = {}
        query_assessment: dict[str, Any] | None = None
        schedule = prereg["execution_schedule"][arm]
        for ordinal, binding in enumerate(schedule, start=1):
            if arm == "adaptive" and ordinal > prereg["adaptive_contract"]["max_episode_starts"]:
                break
            attempt, success = efficiency_attempt(
                root,
                prereg,
                arm=arm,
                arm_run_id=arm_run_id,
                binding=binding,
                ordinal=ordinal,
            )
            attempts.append(attempt)
            observed[binding["candidate_id"]] = success
            if arm == "adaptive" and len(attempts) >= prereg["adaptive_contract"]["min_episode_starts"]:
                if prereg.get("query_sufficiency_contract") is not None:
                    query_assessment = assess_query_sufficiency(
                        prereg["query_sufficiency_contract"],
                        [
                            {
                                "candidate_id": candidate_id,
                                "outcome": "pass" if outcome is True else "fail",
                                "score": 1.0 if outcome is True else 0.0,
                                "diagnosis": None,
                            }
                            for candidate_id, outcome in observed.items()
                            if isinstance(outcome, bool)
                        ],
                        completed_rounds=len(attempts),
                    )
                    if query_assessment["evidence_sufficient"]:
                        break
                    continue
                paired_failure = any(
                    left in observed
                    and right in observed
                    and (observed[left] is False or observed[right] is False)
                    for left, right in (
                        (
                            "object_position.left_fixed",
                            "object_position.right_fixed",
                        ),
                        (
                            "object_instance.base0",
                            "object_instance.base1",
                        ),
                    )
                )
                if paired_failure:
                    break
        if arm == "fixed":
            stop_reason = "fixed_suite_complete"
        else:
            if prereg.get("query_sufficiency_contract") is not None:
                if query_assessment is None:
                    query_assessment = assess_query_sufficiency(
                        prereg["query_sufficiency_contract"],
                        [],
                        completed_rounds=len(attempts),
                    )
                stop_reason = (
                    "query_sufficient"
                    if query_assessment["evidence_sufficient"]
                    else "budget_exhausted"
                )
            else:
                position_pair_observed = {
                    "object_position.left_fixed",
                    "object_position.right_fixed",
                }.issubset(observed)
                position_failure = any(
                    observed.get(candidate) is False
                    for candidate in (
                        "object_position.left_fixed",
                        "object_position.right_fixed",
                    )
                )
                stop_reason = (
                    "query_sufficient"
                    if position_pair_observed and position_failure
                    else "budget_exhausted"
                )
        result = {
            "schema_version": 1,
            "protocol": f"{prereg['protocol']}_arm",
            "arm": arm,
            "arm_run_id": arm_run_id,
            "preregistration_sha256": prereg["preregistration_sha256"],
            "stop_reason": stop_reason,
            "attempts": attempts,
        }
        result_path = output_root / f"{arm}_result.json"
        write_json(result_path, result)
        results[arm] = result
    final = evaluate_click_bell_efficiency(
        prereg,
        results["fixed"],
        results["adaptive"],
        repo_root=root,
    )
    write_json(output_root / "efficiency_result.json", final)


def ranking_trial(
    root: Path,
    *,
    policy_id: str,
    binding: dict[str, Any],
    execute: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    command_path = resolve(root, binding["command_ref"])
    if sha256(command_path) != binding["command_sha256"]:
        raise RuntimeError(f"command hash mismatch before execution: {command_path}")
    command = read_json(command_path)
    log_path = command_path.parent / "live.log"
    if execute:
        returncode, _, _, _ = run_frozen_command(root, command, log_path=log_path)
        if returncode:
            return None, {
                "policy_id": policy_id,
                "seed": binding["seed"],
                "reason": "command_runtime_error",
                "returncode": returncode,
                "log_ref": log_path.relative_to(root).as_posix(),
            }
    seed_results_path = resolve(root, binding["expected_seed_results_ref"])
    telemetry_path = resolve(root, binding["expected_telemetry_episode_ref"])
    if not seed_results_path.is_file():
        return None, {
            "policy_id": policy_id,
            "seed": binding["seed"],
            "reason": "missing_seed_results",
            "seed_results_ref": binding["expected_seed_results_ref"],
        }
    seed_results = read_json(seed_results_path)
    measurements = seed_results.get("seed_measurements")
    measurement = (
        measurements[0]
        if isinstance(measurements, list)
        and len(measurements) == 1
        and isinstance(measurements[0], dict)
        else {}
    )
    success = measurement.get("policy_success")
    if (
        seed_results.get("task_name") != "beat_block_hammer"
        or seed_results.get("task_config") != "demo_clean"
        or seed_results.get("requested_seeds") != [binding["seed"]]
        or seed_results.get("requested_count") != 1
        or seed_results.get("eligible_count") != 1
        or seed_results.get("evaluated_count") != 1
        or seed_results.get("all_eligible") is not True
        or seed_results.get("no_seed_replacement") is not True
        or measurement.get("seed") != binding["seed"]
        or measurement.get("eligibility_status") != "passed"
        or measurement.get("execution_attempted") is not True
        or measurement.get("policy_executed") is not True
        or not isinstance(success, bool)
        or measurement.get("policy_status")
        != ("success" if success else "failure")
    ):
        return None, {
            "policy_id": policy_id,
            "seed": binding["seed"],
            "reason": "ineligible_or_not_executed",
            "eligibility_status": measurement.get("eligibility_status"),
            "policy_status": measurement.get("policy_status"),
            "seed_results_ref": binding["expected_seed_results_ref"],
            "seed_results_sha256": sha256(seed_results_path),
        }
    if not telemetry_path.is_file():
        return None, {
            "policy_id": policy_id,
            "seed": binding["seed"],
            "reason": "missing_telemetry_episode",
            "seed_results_ref": binding["expected_seed_results_ref"],
            "telemetry_episode_ref": binding["expected_telemetry_episode_ref"],
        }
    telemetry = read_json(telemetry_path)
    required_telemetry = {
        "task_name": "beat_block_hammer",
        "task_module": "envs.beat_block_hammer",
        "task_config": "demo_clean",
        "checkpoint_setting": "demo_clean",
        "policy_name": "ACT" if policy_id == "act" else "DP3",
        "seed": binding["seed"],
        "episode_index": 0,
        "success": success,
        "error": None,
    }
    if (
        any(
            telemetry.get(key) != expected
            for key, expected in required_telemetry.items()
        )
        or not isinstance(telemetry.get("policy_steps"), int)
        or isinstance(telemetry.get("policy_steps"), bool)
        or telemetry["policy_steps"] <= 0
    ):
        return None, {
            "policy_id": policy_id,
            "seed": binding["seed"],
            "reason": "telemetry_binding_mismatch",
            "telemetry_episode_ref": binding[
                "expected_telemetry_episode_ref"
            ],
            "telemetry_episode_sha256": sha256(telemetry_path),
        }
    trial = {
        "trial_id": f"{policy_id}_seed_{binding['seed']}",
        "seed": binding["seed"],
        "evidence_source": "live_policy_rollout",
        "status": "completed",
        "error": None,
        "seed_results_ref": binding["expected_seed_results_ref"],
        "seed_results_sha256": sha256(seed_results_path),
        "telemetry_episode_ref": binding["expected_telemetry_episode_ref"],
        "telemetry_episode_sha256": sha256(telemetry_path),
    }
    trial["_observed_success"] = success
    return trial, None


def run_ranking(
    root: Path,
    prereg_path: Path,
    output_root: Path,
    *,
    execute: bool,
) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mea.live_paper_protocols import evaluate_exact_seed_ranking

    prereg = read_json(prereg_path)
    output_root.mkdir(parents=True, exist_ok=True)
    policies = []
    issues: list[dict[str, Any]] = []
    observed: dict[str, list[dict[str, Any]]] = {"act": [], "dp3": []}
    for policy_id in ("act", "dp3"):
        trials: list[dict[str, Any]] = []
        for binding in prereg["execution_schedule"][policy_id]:
            trial, issue = ranking_trial(
                root,
                policy_id=policy_id,
                binding=binding,
                execute=execute,
            )
            if issue is not None:
                issues.append(issue)
                continue
            assert trial is not None
            observed[policy_id].append(
                {
                    "seed": trial["seed"],
                    "success": trial.pop("_observed_success"),
                    "telemetry_episode_ref": trial["telemetry_episode_ref"],
                }
            )
            trials.append(trial)
        policies.append(
            {
                "policy_id": policy_id,
                "checkpoint": prereg["policies"][policy_id],
                "run_id": f"{prereg['study_id']}_{policy_id}_live",
                "trials": trials,
            }
        )
    if issues:
        incomplete = {
            "schema_version": 1,
            "protocol": f"{prereg['protocol']}_incomplete_result",
            "status": "incomplete_exact_seed_contract",
            "preregistration_sha256": prereg["preregistration_sha256"],
            "requested_policy_rollouts": 6,
            "completed_policy_rollouts": sum(len(rows) for rows in observed.values()),
            "observed_policy_outcomes": observed,
            "issues": issues,
            "pair_order": None,
            "spearman": None,
            "paper_table9_eligible": False,
            "scope_limitation": (
                "The frozen exact-seed contract was not completed; no seed "
                "substitution or partial-ranking inference is allowed."
            ),
        }
        write_json(output_root / "ranking_incomplete_result.json", incomplete)
        return
    runs = {
        "schema_version": 1,
        "protocol": f"{prereg['protocol']}_runs",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "policies": policies,
    }
    write_json(output_root / "ranking_runs.json", runs)
    final = evaluate_exact_seed_ranking(prereg, runs, repo_root=root)
    write_json(output_root / "ranking_result.json", final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", choices=("efficiency", "ranking"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Inspect existing ranking outputs without starting policy commands.",
    )
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    prereg_path = resolve(root, str(args.preregistration))
    output_root = resolve(root, str(args.output_root))
    if args.protocol == "efficiency":
        if args.finalize_only:
            raise SystemExit("--finalize-only is currently ranking-only")
        run_efficiency(root, prereg_path, output_root)
    else:
        run_ranking(
            root,
            prereg_path,
            output_root,
            execute=not args.finalize_only,
        )


if __name__ == "__main__":
    main()
