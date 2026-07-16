#!/usr/bin/env python3
"""Run a strict exact-seed Easy/Hard paired ACT evaluation.

This entry point deliberately has no provider or UIUI dependency.  It first
checks expert eligibility for every requested seed under both task configs,
freezes the ordered intersection, and only then runs the same ACT checkpoint
on that intersection.  A rejected seed is recorded; it is never replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.paired import (
    PairedProtocolError,
    build_paired_summary,
    build_seed_manifest,
    load_seed_manifest,
    seed_manifest_sha256,
)
from mea.toolkit import evaluate_telemetry_root


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run one auditable child command without leaking output into JSON."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    if environment:
        child_environment.update(environment)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=child_environment,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def checkpoint_preflight(
    repo_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    task_name = manifest["task_name"]
    setting = manifest["checkpoint_setting"]
    expert_data_num = manifest["expert_data_num"]
    checkpoint_dir = (
        repo_root
        / "policy/ACT/act_ckpt"
        / f"act-{task_name}"
        / f"{setting}-{expert_data_num}"
    )
    files = [
        checkpoint_dir / "policy_last.ckpt",
        checkpoint_dir / "dataset_stats.pkl",
    ]
    missing = [path for path in files if not path.is_file()]
    result = {
        "directory": str(checkpoint_dir),
        "required_files": [str(path) for path in files],
        "missing_files": [str(path) for path in missing],
        "passed": not missing,
    }
    if missing:
        raise RuntimeError(
            f"ACT checkpoint preflight failed for {task_name}: "
            + ", ".join(str(path) for path in missing)
            + ". Download it directly on the server with "
            f"`python scripts/download_act_checkpoint.py {task_name}`; "
            "do not relay routine checkpoints through a workstation."
        )
    return result


def probe_command(
    *,
    repo_root: Path,
    task_name: str,
    task_module: str,
    task_config: str,
    checkpoint_setting: str,
    seed: int,
    episode_index: int,
    image_path: Path,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mea.taskgen.probe",
        "--repo-root",
        str(repo_root),
        "--task-name",
        task_name,
        "--task-module",
        task_module,
        "--task-config",
        task_config,
        "--ckpt-setting",
        checkpoint_setting,
        "--seed",
        str(seed),
        "--episode-index",
        str(episode_index),
        "--image",
        str(image_path),
        "--output",
        str(result_path),
        "--expert",
        "--eval-mode",
    ]


def classify_probe(returncode: int, payload: Mapping[str, Any]) -> str:
    """Map the probe's fields and exit code into the paired protocol."""

    error = payload.get("error")
    if isinstance(error, Mapping) and error.get("type") == "UnStableError":
        return "unstable"
    complete = (
        returncode == 0
        and payload.get("setup_success") is True
        and payload.get("render_success") is True
        and payload.get("rule_check", {}).get("passed") is True
        and payload.get("expert", {}).get("passed") is True
    )
    if complete:
        return "passed"
    if returncode == 2 or payload.get("expert", {}).get("passed") is False:
        return "expert_failed"
    return "error"


def run_eligibility_condition(
    repo_root: Path,
    run_dir: Path,
    manifest: Mapping[str, Any],
    condition: Mapping[str, str],
    *,
    task_module: str,
) -> list[dict[str, Any]]:
    condition_id = condition["id"]
    condition_root = run_dir / "eligibility" / condition_id
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(manifest["seeds"]):
        seed_root = condition_root / f"seed_{seed}"
        result_path = seed_root / "probe.json"
        image_path = seed_root / "initial_head.png"
        log_path = seed_root / "probe.log"
        command = probe_command(
            repo_root=repo_root,
            task_name=manifest["task_name"],
            task_module=task_module,
            task_config=condition["task_config"],
            checkpoint_setting=manifest["checkpoint_setting"],
            seed=seed,
            episode_index=index,
            image_path=image_path,
            result_path=result_path,
        )
        returncode = run_command(command, cwd=repo_root, log_path=log_path)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            }
        rows.append(
            {
                "requested_index": index,
                "seed": seed,
                "eligibility_status": classify_probe(returncode, payload),
                "returncode": returncode,
                "task_config": condition["task_config"],
                "artifacts": {
                    "probe": str(result_path.relative_to(run_dir)),
                    "image": (
                        str(image_path.relative_to(run_dir))
                        if image_path.is_file()
                        else None
                    ),
                    "log": str(log_path.relative_to(run_dir)),
                },
                "probe": payload,
            }
        )
    write_json(condition_root / "eligibility.json", {"rows": rows})
    return rows


def act_command(
    *,
    manifest: Mapping[str, Any],
    condition: Mapping[str, str],
    task_module: str,
    gpu: int,
    telemetry_profile: str,
    selected_manifest_path: Path,
    result_path: Path,
    telemetry_root: Path,
    output_dir: Path,
) -> list[str]:
    seeds = manifest["seeds"]
    return [
        "bash",
        "policy/ACT/eval_mea.sh",
        manifest["task_name"],
        condition["task_config"],
        manifest["checkpoint_setting"],
        str(manifest["expert_data_num"]),
        str(manifest["policy_seed"]),
        str(gpu),
        str(len(seeds)),
        task_module,
        "",  # no task overlay: this protocol evaluates official tasks
        str(seeds[0]),  # audit-only in strict mode; manifest is authoritative
        str(telemetry_root),
        telemetry_profile,
        str(selected_manifest_path),
        str(result_path),
        str(output_dir),
    ]


def validate_exact_result(
    payload: Any,
    *,
    manifest: Mapping[str, Any],
    condition: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PairedProtocolError("exact evaluator result must be an object")
    if payload.get("task_name") != manifest["task_name"]:
        raise PairedProtocolError("exact evaluator task_name mismatch")
    if payload.get("task_config") != condition["task_config"]:
        raise PairedProtocolError("exact evaluator task_config mismatch")
    if payload.get("condition_id") != condition["id"]:
        raise PairedProtocolError("exact evaluator condition_id mismatch")
    if payload.get("requested_seeds") != manifest["seeds"]:
        raise PairedProtocolError("exact evaluator changed requested seed order")
    rows = payload.get("seed_measurements")
    if not isinstance(rows, list):
        raise PairedProtocolError("exact evaluator omitted seed_measurements")
    if [row.get("seed") for row in rows] != manifest["seeds"]:
        raise PairedProtocolError("exact evaluator replaced or reordered a seed")
    if payload.get("no_seed_replacement") is not True:
        raise PairedProtocolError("exact evaluator did not certify no replacement")
    return payload


def trusted_tools_by_seed(
    telemetry_root: Path,
    *,
    task_name: str,
    task_config: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    if not list(telemetry_root.glob("episode_*/episode.json")):
        return {}, None
    summary = evaluate_telemetry_root(
        telemetry_root,
        user_request="official success and time to success",
        task_name=task_name,
    )
    by_seed: dict[int, dict[str, Any]] = {}
    for episode in summary["episodes"]:
        metadata = episode["metadata"]
        if metadata.get("task_config") != task_config:
            raise PairedProtocolError(
                "telemetry task_config does not match its paired condition"
            )
        seed = int(metadata["seed"])
        if seed in by_seed:
            raise PairedProtocolError(f"duplicate telemetry seed: {seed}")
        tools = {item["tool"]: item for item in episode["tool_results"]}
        if "official_check_success" not in tools or "time_to_success" not in tools:
            raise PairedProtocolError(
                f"telemetry seed {seed} lacks generic trusted tools"
            )
        by_seed[seed] = {
            "episode": episode,
            "official_success": bool(tools["official_check_success"]["value"]),
            "time_to_success": tools["time_to_success"]["value"],
        }
    return by_seed, summary


def merge_condition_measurements(
    *,
    candidate_seeds: list[int],
    selected_seeds: list[int],
    probe_rows: list[dict[str, Any]],
    exact_result: Mapping[str, Any] | None,
    tools_by_seed: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    probes = {row["seed"]: row for row in probe_rows}
    evaluator_rows = {
        row["seed"]: row
        for row in (exact_result or {}).get("seed_measurements", [])
    }
    measurements: list[dict[str, Any]] = []
    for seed in candidate_seeds:
        probe = probes[seed]
        row: dict[str, Any] = {
            "seed": seed,
            "eligibility_status": probe["eligibility_status"],
            "policy_executed": False,
            "policy_success": None,
            "time_to_success": None,
            "selected_for_evaluation": seed in selected_seeds,
            "probe_returncode": probe["returncode"],
            "probe_artifacts": probe["artifacts"],
        }
        if seed not in selected_seeds:
            measurements.append(row)
            continue

        evaluator = evaluator_rows.get(seed)
        issues: list[str] = []
        tool_evidence = tools_by_seed.get(seed)
        if evaluator is None:
            issues.append("missing exact evaluator row")
        else:
            if evaluator.get("eligibility_status") != "passed":
                issues.append(
                    "eligibility changed between frozen probe and evaluator"
                )
            if evaluator.get("policy_executed") is not True:
                issues.append("policy did not produce a complete outcome")
        if tool_evidence is None:
            issues.append("missing trusted-tool telemetry")
        elif evaluator is not None:
            evaluator_success = evaluator.get("policy_success")
            metadata_success = bool(
                tool_evidence["episode"]["metadata"].get("success")
            )
            trusted_success = tool_evidence["official_success"]
            if not (
                evaluator_success is metadata_success is trusted_success
            ):
                issues.append(
                    "evaluator, recorder, and official trusted tool disagree"
                )

        if issues:
            row.update(
                {
                    "eligibility_status": "protocol_violation",
                    "protocol_issues": issues,
                    "evaluator_measurement": evaluator,
                }
            )
        else:
            row.update(
                {
                    "eligibility_status": "passed",
                    "policy_executed": True,
                    "policy_success": bool(tool_evidence["official_success"]),
                    "time_to_success": tool_evidence["time_to_success"],
                    "telemetry_episode": tool_evidence["episode"]["episode_dir"],
                    "evaluator_measurement": evaluator,
                }
            )
        measurements.append(row)
    return measurements


def _default_run_id() -> str:
    return "run_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def run_paired(arguments: argparse.Namespace) -> dict[str, Any]:
    repo_root = arguments.repo_root.expanduser().resolve()
    if arguments.manifest is not None:
        manifest = load_seed_manifest(arguments.manifest)
    else:
        manifest = build_seed_manifest(
            task_name=arguments.task_name,
            seeds=arguments.seeds,
        )
    task_module = arguments.task_module or f"envs.{manifest['task_name']}"
    run_id = arguments.run_id or _default_run_id()
    if not _RUN_ID.fullmatch(run_id):
        raise PairedProtocolError(
            "run_id must contain only letters, digits, dot, underscore, or dash"
        )
    run_dir = repo_root / "mea/paired_runs" / run_id
    checkpoint = checkpoint_preflight(repo_root, manifest)
    plan = {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "task_module": task_module,
        "gpu": arguments.gpu,
        "telemetry_profile": arguments.telemetry_profile,
        "seed_manifest": manifest,
        "seed_manifest_sha256": seed_manifest_sha256(manifest),
        "checkpoint": checkpoint,
        "requires_uiui": False,
        "allow_protocol_violations": bool(
            getattr(arguments, "allow_protocol_violations", False)
        ),
    }
    if arguments.dry_run:
        return {**plan, "status": "dry_run"}
    if run_dir.exists():
        raise RuntimeError(f"paired run already exists: {run_dir}")

    run_dir.mkdir(parents=True)
    write_json(run_dir / "seed_manifest.json", manifest)
    started_at = datetime.now().astimezone().isoformat()
    write_json(
        run_dir / "status.json",
        {**plan, "status": "running", "started_at": started_at},
    )

    try:
        eligibility: dict[str, list[dict[str, Any]]] = {}
        for condition in manifest["conditions"]:
            eligibility[condition["id"]] = run_eligibility_condition(
                repo_root,
                run_dir,
                manifest,
                condition,
                task_module=task_module,
            )
        eligibility_maps = {
            condition_id: {
                row["seed"]: row["eligibility_status"]
                for row in rows
            }
            for condition_id, rows in eligibility.items()
        }
        selected_seeds = [
            seed
            for seed in manifest["seeds"]
            if all(
                eligibility_maps[condition["id"]][seed] == "passed"
                for condition in manifest["conditions"]
            )
        ]
        write_json(
            run_dir / "eligibility" / "summary.json",
            {
                "requested_seeds": manifest["seeds"],
                "selected_seeds": selected_seeds,
                "selection": "ordered intersection of exact expert eligibility",
                "no_seed_replacement": True,
            },
        )

        condition_runs: dict[str, dict[str, Any]] = {}
        selected_manifest = None
        selected_manifest_path = run_dir / "selected_seed_manifest.json"
        if selected_seeds:
            selected_manifest = build_seed_manifest(
                task_name=manifest["task_name"],
                seeds=selected_seeds,
                conditions=manifest["conditions"],
                checkpoint_setting=manifest["checkpoint_setting"],
                expert_data_num=manifest["expert_data_num"],
                policy_seed=manifest["policy_seed"],
            )
            write_json(selected_manifest_path, selected_manifest)

        for condition in manifest["conditions"]:
            condition_id = condition["id"]
            condition_root = run_dir / "conditions" / condition_id
            exact_result = None
            tools_by_seed: dict[int, dict[str, Any]] = {}
            trusted_summary = None
            act_audit = None
            if selected_manifest is not None:
                telemetry_root = condition_root / "telemetry"
                result_path = condition_root / "seed_results.json"
                output_dir = condition_root / "eval_output"
                command = act_command(
                    manifest=selected_manifest,
                    condition=condition,
                    task_module=task_module,
                    gpu=arguments.gpu,
                    telemetry_profile=arguments.telemetry_profile,
                    selected_manifest_path=selected_manifest_path,
                    result_path=result_path,
                    telemetry_root=telemetry_root,
                    output_dir=output_dir,
                )
                returncode = run_command(
                    command,
                    cwd=repo_root,
                    log_path=condition_root / "act.log",
                    environment={"PYTHON_BIN": sys.executable},
                )
                if not result_path.is_file():
                    raise RuntimeError(
                        f"exact evaluator produced no result for {condition_id}; "
                        f"see {condition_root / 'act.log'}"
                    )
                exact_result = validate_exact_result(
                    json.loads(result_path.read_text(encoding="utf-8")),
                    manifest=selected_manifest,
                    condition=condition,
                )
                if returncode != 0:
                    raise RuntimeError(
                        f"exact evaluator failed for {condition_id} with "
                        f"return code {returncode}"
                    )
                tools_by_seed, trusted_summary = trusted_tools_by_seed(
                    telemetry_root,
                    task_name=manifest["task_name"],
                    task_config=condition["task_config"],
                )
                act_audit = {
                    "command": command,
                    "returncode": returncode,
                    "log": str((condition_root / "act.log").relative_to(run_dir)),
                    "exact_result": str(result_path.relative_to(run_dir)),
                    "telemetry_root": str(telemetry_root.relative_to(run_dir)),
                }

            measurements = merge_condition_measurements(
                candidate_seeds=manifest["seeds"],
                selected_seeds=selected_seeds,
                probe_rows=eligibility[condition_id],
                exact_result=exact_result,
                tools_by_seed=tools_by_seed,
            )
            condition_run = {
                "condition_id": condition_id,
                "task_config": condition["task_config"],
                "act": act_audit,
                "trusted_tools": trusted_summary,
                "seed_measurements": measurements,
            }
            write_json(condition_root / "condition_result.json", condition_run)
            condition_runs[condition_id] = condition_run

        summary = build_paired_summary(manifest, condition_runs)
        summary["selected_seed_count"] = len(selected_seeds)
        summary["selected_seeds"] = selected_seeds
        summary["seed_manifest_sha256"] = seed_manifest_sha256(manifest)
        summary["selected_seed_manifest_sha256"] = (
            seed_manifest_sha256(selected_manifest)
            if selected_manifest is not None
            else None
        )
        if not selected_seeds:
            summary["status"] = "completed_without_paired_evaluation"
        elif summary["protocol_violation_measurement_count"]:
            summary["status"] = "completed_with_protocol_violations"
        write_json(run_dir / "paired_summary.json", summary)
        finished = {
            **plan,
            "status": summary["status"],
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "selected_seeds": selected_seeds,
            "paired_summary": str(run_dir / "paired_summary.json"),
        }
        write_json(run_dir / "status.json", finished)
        if (
            summary["protocol_violation_measurement_count"]
            and not getattr(arguments, "allow_protocol_violations", False)
        ):
            raise PairedProtocolError(
                "paired run contains protocol violations and is not valid "
                "for comparison; inspect paired_summary.json or explicitly "
                "pass --allow-protocol-violations for diagnostic workflows"
            )
        return finished
    except Exception as exc:
        write_json(
            run_dir / "status.json",
            {
                **plan,
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict exact-seed paired Easy/Hard ACT evaluation"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seeds", nargs="+", type=int)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--task-name", default="click_bell")
    parser.add_argument("--task-module")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-protocol-violations",
        action="store_true",
        help=(
            "Return success despite protocol violations; diagnostic use only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    try:
        result = run_paired(parse_args())
    except (PairedProtocolError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
