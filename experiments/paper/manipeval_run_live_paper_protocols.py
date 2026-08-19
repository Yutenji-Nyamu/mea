#!/usr/bin/env python3
"""Execute paper-only live commands and derive receipts from disk evidence."""

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
    from experiments.paper.live_protocols import evaluate_click_bell_efficiency
    from experiments.paper.query_contract_compat import assess_query_sufficiency

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
    shared_eligibility: dict[str, Any],
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
        or seed_results.get("shared_eligibility_ref")
        != shared_eligibility["ref"]
        or seed_results.get("shared_eligibility_sha256")
        != shared_eligibility["sha256"]
        or seed_results.get("scene_signature_sha256")
        != shared_eligibility["scene_signature_sha256"]
        or measurement.get("shared_eligibility_ref")
        != shared_eligibility["ref"]
        or measurement.get("shared_eligibility_sha256")
        != shared_eligibility["sha256"]
        or measurement.get("scene_signature_sha256")
        != shared_eligibility["scene_signature_sha256"]
        or measurement.get("scene_signature_match") is not True
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
        "shared_eligibility_ref": shared_eligibility["ref"],
        "shared_eligibility_sha256": shared_eligibility["sha256"],
        "scene_signature_sha256": shared_eligibility[
            "scene_signature_sha256"
        ],
    }
    trial["_observed_success"] = success
    return trial, None


def ranking_shared_eligibility(
    root: Path,
    *,
    binding: dict[str, Any],
    execute: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from mea.paired import (
        load_shared_eligibility_manifest,
        shared_eligibility_sha256,
    )

    command_path = resolve(root, binding["command_ref"])
    if sha256(command_path) != binding["command_sha256"]:
        raise RuntimeError(
            f"expert command hash mismatch before execution: {command_path}"
        )
    command = read_json(command_path)
    log_path = command_path.parent / "expert.log"
    if execute:
        returncode, _, _, _ = run_frozen_command(
            root, command, log_path=log_path
        )
        if returncode:
            return None, {
                "seed": binding["seed"],
                "reason": "shared_expert_runtime_error",
                "returncode": returncode,
                "log_ref": log_path.relative_to(root).as_posix(),
            }
    eligibility_path = resolve(
        root, binding["expected_eligibility_ref"]
    )
    if not eligibility_path.is_file():
        return None, {
            "seed": binding["seed"],
            "reason": "missing_shared_eligibility",
            "shared_eligibility_ref": binding[
                "expected_eligibility_ref"
            ],
        }
    try:
        eligibility = load_shared_eligibility_manifest(
            eligibility_path,
            expected_task_name="beat_block_hammer",
            expected_task_module="envs.beat_block_hammer",
            expected_task_config="demo_clean",
            expected_seed=binding["seed"],
        )
    except Exception as exc:
        return None, {
            "seed": binding["seed"],
            "reason": "invalid_shared_eligibility",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "shared_eligibility_ref": binding[
                "expected_eligibility_ref"
            ],
        }
    return {
        "seed": binding["seed"],
        "ref": binding["expected_eligibility_ref"],
        "sha256": shared_eligibility_sha256(eligibility),
        "scene_signature_sha256": eligibility[
            "scene_signature_sha256"
        ],
        "exact_instruction": eligibility["exact_instruction"],
    }, None


def run_ranking(
    root: Path,
    prereg_path: Path,
    output_root: Path,
    *,
    execute: bool,
) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.paper.live_protocols import evaluate_exact_seed_ranking

    prereg = read_json(prereg_path)
    output_root.mkdir(parents=True, exist_ok=True)
    requested_shared_probes = len(prereg["eligibility_schedule"])
    requested_policy_rollouts = sum(
        len(rows) for rows in prereg["execution_schedule"].values()
    )
    shared_by_seed: dict[int, dict[str, Any]] = {}
    eligibility_issues: list[dict[str, Any]] = []
    for binding in prereg["eligibility_schedule"]:
        shared, issue = ranking_shared_eligibility(
            root,
            binding=binding,
            execute=execute,
        )
        if issue is not None:
            eligibility_issues.append(issue)
            continue
        assert shared is not None
        shared_by_seed[binding["seed"]] = shared
    if eligibility_issues:
        incomplete = {
            "schema_version": 1,
            "protocol": f"{prereg['protocol']}_incomplete_result",
            "status": "shared_expert_eligibility_incomplete",
            "preregistration_sha256": prereg["preregistration_sha256"],
            "requested_shared_expert_probes": requested_shared_probes,
            "completed_shared_expert_probes": len(shared_by_seed),
            "requested_policy_rollouts": requested_policy_rollouts,
            "completed_policy_rollouts": 0,
            "issues": eligibility_issues,
            "pair_order": None,
            "spearman": None,
            "paper_table9_eligible": False,
            "scope_limitation": (
                "At least one shared expert probe failed, so no policy "
                "rollout was allowed to start."
            ),
        }
        write_json(output_root / "ranking_incomplete_result.json", incomplete)
        return
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
                shared_eligibility=shared_by_seed[binding["seed"]],
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
            "requested_policy_rollouts": requested_policy_rollouts,
            "completed_policy_rollouts": sum(len(rows) for rows in observed.values()),
            "requested_shared_expert_probes": requested_shared_probes,
            "completed_shared_expert_probes": len(shared_by_seed),
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


def _table3_reviews(
    path: Path | None,
    *,
    expected_cell_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = read_json(path)
    if (
        value.get("schema_version") != 1
        or value.get("annotator_kind") != "development_agent_proxy"
        or value.get("human_reviewer_count") != 0
        or value.get("paper_eligible") is not False
    ):
        raise RuntimeError("invalid Table 3 development-proxy review manifest")
    reviews = value.get("reviews")
    if not isinstance(reviews, list):
        raise RuntimeError("Table 3 review manifest requires reviews")
    by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if (
            not isinstance(review, dict)
            or set(review)
            != {"cell_id", "passed", "blind_to_condition", "notes"}
            or review.get("cell_id") not in expected_cell_ids
            or not isinstance(review.get("passed"), bool)
            or not isinstance(review.get("blind_to_condition"), bool)
            or not isinstance(review.get("notes"), str)
            or not review["notes"].strip()
        ):
            raise RuntimeError("invalid Table 3 proxy review row")
        if review["cell_id"] in by_id:
            raise RuntimeError("duplicate Table 3 proxy review row")
        by_id[review["cell_id"]] = review
    if set(by_id) != expected_cell_ids:
        raise RuntimeError("Table 3 proxy review must cover the exact cell grid")
    return by_id


def _table3_completed_cell(
    root: Path,
    *,
    frozen: dict[str, Any],
    review: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    expected = frozen["expected_stage_receipts"]
    manifest_path = resolve(root, expected["codegen"]["manifest_ref"])
    task_path = resolve(root, expected["codegen"]["artifact_ref"])
    candidate_path = manifest_path.parent / "candidate_manifest.json"
    static_path = resolve(root, expected["compile"]["receipt_ref"])
    scene_path = resolve(root, expected["render"]["receipt_ref"])
    fixtures_path = resolve(root, expected["oracle"]["receipt_ref"])
    required = (
        manifest_path,
        task_path,
        candidate_path,
        static_path,
        scene_path,
        fixtures_path,
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(
            f"completed Table 3 cell is missing artifacts: {frozen['cell_id']}"
        )
    manifest = read_json(manifest_path)
    candidate = read_json(candidate_path)
    static = read_json(static_path)
    scene = read_json(scene_path)
    fixtures_value = json.loads(fixtures_path.read_text(encoding="utf-8"))
    if not isinstance(fixtures_value, list) or not fixtures_value:
        raise RuntimeError("Table 3 checker fixtures must be a non-empty list")
    provenance = candidate.get("codegen_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    prompt_components = manifest.get("taskgen_prompt_components")
    if not isinstance(prompt_components, dict):
        prompt_components = {}
    switches = frozen["module_switches"]
    switch_binding_passed = (
        manifest.get("taskgen_ablation_switches") == switches
        and provenance.get("taskgen_ablation_switches") == switches
        and prompt_components.get("rag_context") == switches["rag"]
        and prompt_components.get("readme_agent_context")
        == switches["readme_agent"]
        and prompt_components.get("visual_self_check")
        == switches["visual_self_check"]
        and (
            isinstance(manifest.get("visual_self_reflection"), dict)
            == switches["visual_self_check"]
        )
    )
    compile_passed = (
        static.get("variant_spec", {}).get("valid") is True
        and static.get("provider_scene_checker", {}).get("valid") is True
    )
    render_passed = (
        scene.get("setup_success") is True
        and scene.get("render_success") is True
    )
    simulator_passed = (
        render_passed
        and scene.get("rule_check", {}).get("passed") is True
        and scene.get("expert", {}).get("passed") is True
    )
    fixture_passed = all(
        isinstance(item, dict) and item.get("passed") is True
        for item in fixtures_value
    )
    positive_count = sum(
        isinstance(item, dict) and item.get("expected") is True
        for item in fixtures_value
    )
    negative_count = sum(
        isinstance(item, dict) and item.get("expected") is False
        for item in fixtures_value
    )
    codegen_passed = (
        manifest.get("status") == "completed_without_act"
        and manifest.get("provider", {}).get("called") is True
        and provenance.get("generated_by_model") is True
        and provenance.get("restricted_success_spec_compiler_used") is False
        and switch_binding_passed
    )
    execution.update(
        {
            "manifest_ref": expected["codegen"]["manifest_ref"],
            "manifest_sha256": sha256(manifest_path),
            "manifest_status": manifest.get("status"),
        }
    )
    return {
        "cell_id": frozen["cell_id"],
        "proposal_id": frozen["proposal_id"],
        "condition": frozen["condition"],
        "execution": execution,
        "stages": {
            "codegen": {
                "generated_by_provider": codegen_passed,
                "scene_generated_by_model": provenance.get(
                    "generated_by_model"
                )
                is True,
                "checker_generated_by_model": provenance.get(
                    "generated_by_model"
                )
                is True,
                "module_switches": manifest.get(
                    "taskgen_ablation_switches"
                ),
                "prompt_components": prompt_components,
                "artifact_ref": expected["codegen"]["artifact_ref"],
                "artifact_sha256": sha256(task_path),
            },
            "compile": {
                "passed": compile_passed,
                "receipt_ref": expected["compile"]["receipt_ref"],
                "receipt_sha256": sha256(static_path),
            },
            "render": {
                "passed": render_passed,
                "receipt_ref": expected["render"]["receipt_ref"],
                "receipt_sha256": sha256(scene_path),
            },
            "simulator": {
                "passed": simulator_passed,
                "receipt_ref": expected["simulator"]["receipt_ref"],
                "receipt_sha256": sha256(scene_path),
            },
            "oracle": {
                "passed": fixture_passed,
                "receipt_ref": expected["oracle"]["receipt_ref"],
                "receipt_sha256": sha256(fixtures_path),
                "positive_fixture_count": positive_count,
                "negative_fixture_count": negative_count,
            },
        },
        "blind_proxy_review": {
            "annotator_kind": "development_agent_proxy",
            "blind_to_condition": review["blind_to_condition"],
            "passed": review["passed"],
            "human_reviewer_count": 0,
            "notes": review["notes"],
        },
    }


def run_table3(
    root: Path,
    prereg_path: Path,
    output_root: Path,
    *,
    execute: bool,
    proxy_review_path: Path | None,
) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.paper.live_protocols import evaluate_table3_codegen

    prereg = read_json(prereg_path)
    output_root.mkdir(parents=True, exist_ok=True)
    expected_ids = {cell["cell_id"] for cell in prereg["cells"]}
    reviews = _table3_reviews(
        proxy_review_path, expected_cell_ids=expected_ids
    )
    existing_executions: dict[str, dict[str, Any]] = {}
    if not execute:
        receipt_manifest = read_json(
            output_root / "table3_execution_receipts.json"
        )
        raw_existing = receipt_manifest.get("cells")
        if not isinstance(raw_existing, list):
            raise RuntimeError("Table 3 execution receipts are missing cells")
        existing_executions = {
            str(item.get("cell_id")): item
            for item in raw_existing
            if isinstance(item, dict)
        }
        if set(existing_executions) != expected_ids:
            raise RuntimeError(
                "Table 3 execution receipts do not cover the preregistered grid"
            )
    raw_executions: list[dict[str, Any]] = []
    for frozen in prereg["cells"]:
        runner_path = resolve(root, frozen["runner_ref"])
        if sha256(runner_path) != frozen["runner_sha256"]:
            raise RuntimeError(f"runner hash mismatch: {runner_path}")
        runner = read_json(runner_path)
        log_path = output_root / "logs" / f"{frozen['cell_id']}.log"
        if execute:
            returncode, started_at, ended_at, wall_seconds = run_frozen_command(
                root, runner, log_path=log_path
            )
        else:
            raw_executions.append(
                existing_executions[frozen["cell_id"]]
            )
            continue
        manifest_path = resolve(
            root,
            frozen["expected_stage_receipts"]["codegen"]["manifest_ref"],
        )
        manifest = (
            read_json(manifest_path) if manifest_path.is_file() else None
        )
        execution = {
            "status": (
                "completed"
                if returncode == 0
                and isinstance(manifest, dict)
                and manifest.get("status") == "completed_without_act"
                else "failed"
            ),
            "returncode": returncode,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "wall_seconds": wall_seconds,
            "log_ref": log_path.relative_to(root).as_posix(),
            "log_sha256": sha256(log_path),
            "manifest_ref": (
                manifest_path.relative_to(root).as_posix()
                if manifest_path.is_file()
                else None
            ),
            "manifest_sha256": (
                sha256(manifest_path) if manifest_path.is_file() else None
            ),
            "manifest_status": (
                manifest.get("status") if isinstance(manifest, dict) else None
            ),
        }
        raw_executions.append(
            {
                "cell_id": frozen["cell_id"],
                "proposal_id": frozen["proposal_id"],
                "condition": frozen["condition"],
                "execution": execution,
            }
        )
    write_json(
        output_root / "table3_execution_receipts.json",
        {
            "schema_version": 1,
            "protocol": f"{prereg['protocol']}_execution_receipts",
            "preregistration_sha256": prereg["preregistration_sha256"],
            "cells": raw_executions,
            "review_required": sorted(
                cell["cell_id"]
                for cell in raw_executions
                if cell["execution"]["status"] == "completed"
            ),
        },
    )
    if not reviews:
        return
    cells: list[dict[str, Any]] = []
    by_frozen = {cell["cell_id"]: cell for cell in prereg["cells"]}
    for raw in raw_executions:
        frozen = by_frozen[raw["cell_id"]]
        if raw["execution"]["status"] == "completed":
            cells.append(
                _table3_completed_cell(
                    root,
                    frozen=frozen,
                    review=reviews[raw["cell_id"]],
                    execution=raw["execution"],
                )
            )
        else:
            cells.append(
                {
                    **raw,
                    "stages": {},
                    "blind_proxy_review": {
                        "annotator_kind": "development_agent_proxy",
                        "blind_to_condition": reviews[raw["cell_id"]][
                            "blind_to_condition"
                        ],
                        "passed": False,
                        "human_reviewer_count": 0,
                        "notes": "Generation failed before visual proxy review.",
                    },
                }
            )
    runs = {
        "schema_version": 1,
        "protocol": f"{prereg['protocol']}_runs",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "cells": cells,
    }
    write_json(output_root / "table3_runs.json", runs)
    final = evaluate_table3_codegen(prereg, runs, repo_root=root)
    write_json(output_root / "table3_result.json", final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "protocol", choices=("efficiency", "ranking", "table3")
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Inspect existing ranking/Table3 outputs without starting commands.",
    )
    parser.add_argument(
        "--proxy-review",
        type=Path,
        help="Development-agent proxy review manifest for Table3 finalization.",
    )
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    prereg_path = resolve(root, str(args.preregistration))
    output_root = resolve(root, str(args.output_root))
    if args.protocol == "efficiency":
        if args.finalize_only:
            raise SystemExit("--finalize-only is currently ranking-only")
        run_efficiency(root, prereg_path, output_root)
    elif args.protocol == "ranking":
        run_ranking(
            root,
            prereg_path,
            output_root,
            execute=not args.finalize_only,
        )
    else:
        review_path = (
            resolve(root, str(args.proxy_review))
            if args.proxy_review is not None
            else None
        )
        run_table3(
            root,
            prereg_path,
            output_root,
            execute=not args.finalize_only,
            proxy_review_path=review_path,
        )


if __name__ == "__main__":
    main()
