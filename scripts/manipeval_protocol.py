#!/usr/bin/env python3
"""Repeat complete MEA Agent evaluations with a small ACT-only budget."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.protocol import (
    AGILE_BUDGETS,
    ProtocolError,
    build_expected_sample_identities,
    build_repetition_schedule,
    canonical_sha256,
    collect_evaluation_measurement,
    evaluation_id_for_attempt,
    now_iso,
    render_protocol_report,
    summarize_protocol,
    validate_budget,
    validate_run_id,
    write_json_atomic,
)
from mea.planner.click_bell import CLICK_BELL_TEMPLATE_IDS
from mea.providers import available_model_profiles
from mea.toolkit import load_task_schema


def _validate_robotwin_runtime() -> None:
    try:
        sapien_spec = importlib.util.find_spec("sapien")
    except (ImportError, AttributeError, ValueError) as exc:
        raise ProtocolError(
            f"cannot inspect the RoboTwin runtime under {sys.executable}: {exc}"
        ) from exc
    if sapien_spec is None:
        raise ProtocolError(
            "RoboTwin dependency 'sapien' is unavailable under "
            f"{sys.executable}; activate the RoboTwin environment before "
            "starting or resuming a protocol run"
        )


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _default_run_id() -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"protocol_{stamp}_{uuid.uuid4().hex[:8]}"


def _validate_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProtocolError("--base-url must be an absolute HTTP(S) endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProtocolError(
            "--base-url must not contain credentials, query parameters, or fragments"
        )
    return value


def _write_status(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_protocol(manifest)
    write_json_atomic(run_dir / "protocol_manifest.json", manifest)
    write_json_atomic(run_dir / "summary/protocol_summary.json", summary)
    (run_dir / "protocol_report.md").write_text(
        render_protocol_report(manifest, summary),
        encoding="utf-8",
    )
    return summary


def _acquire_lock(run_dir: Path) -> Path:
    path = run_dir / "run.lock"
    if path.exists():
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
            pid = int(lock.get("pid"))
            os.kill(pid, 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
        else:
            raise ProtocolError(f"protocol run is already active under pid {pid}")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_at": now_iso()}, handle)
        handle.write("\n")
    return path


def _agent_command(
    repo_root: Path,
    config: dict[str, Any],
    *,
    evaluation_id: str,
    start_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts/manipeval_agent.py"),
        "--repo-root",
        str(repo_root),
        "--request",
        config["request"],
        "--evaluation-id",
        evaluation_id,
        "--task-name",
        config["task_name"],
        "--execution-backend",
        "act",
        "--start-seed",
        str(start_seed),
        "--num-episodes",
        str(config["episodes"]),
        "--model-profile",
        config["model_profile"],
        "--telemetry-profile",
        config["telemetry_profile"],
        "--gpu",
        str(config["gpu"]),
        "--max-reflections",
        str(config["max_reflections"]),
        "--no-history",
    ]
    if config.get("task_profile", "official") != "official":
        command.extend(
            [
                "--task-profile",
                config["task_profile"],
                "--generated-rounds",
                str(config["generated_rounds"]),
            ]
        )
    if config.get("task_module"):
        command.extend(["--task-module", config["task_module"]])
    if config.get("base_url"):
        command.extend(["--base-url", config["base_url"]])
    return command


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    on_start=None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if on_start is not None:
            on_start(process.pid)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def _pid_matches_attempt(pid: Any, evaluation_id: str) -> bool:
    try:
        normalized = int(pid)
        os.kill(normalized, 0)
    except (OSError, TypeError, ValueError):
        return False
    command_line = Path(f"/proc/{normalized}/cmdline")
    if command_line.is_file():
        try:
            value = command_line.read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            return True
        return evaluation_id in value
    return True


def _new_manifest(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    if not args.request:
        raise ProtocolError("--request is required for a new protocol run")
    repetitions = validate_budget(args.repetitions, name="repetitions")
    episodes = validate_budget(args.episodes, name="episodes")
    run_id = validate_run_id(args.run_id or _default_run_id())
    if args.task_name == "beat_block_hammer":
        raise ProtocolError(
            "protocol_v1 supports schema-backed official ACT tasks only; "
            "beat_block_hammer keeps its generated-task route"
        )
    task_profile = str(args.task_profile)
    if task_profile == "position_lr":
        if args.task_name != "click_bell":
            raise ProtocolError("position_lr is only available for click_bell")
        if args.generated_rounds not in {1, 2}:
            raise ProtocolError("position_lr generated_rounds must be 1 or 2")
        expected_variant_ids = list(
            CLICK_BELL_TEMPLATE_IDS[: int(args.generated_rounds)]
        )
    elif task_profile == "official":
        expected_variant_ids = []
    else:
        raise ProtocolError("the protocol runner supports official or position_lr")
    try:
        load_task_schema(repo_root, args.task_name)
    except Exception as exc:
        raise ProtocolError(
            f"task {args.task_name!r} is not a schema-backed official task: {exc}"
        ) from exc
    if not os.getenv("UIUI_API_KEY"):
        raise ProtocolError(
            "UIUI_API_KEY is required for complete Agent VQA/Feedback evaluation"
        )
    checkpoint_dir = (
        repo_root
        / "policy/ACT/act_ckpt"
        / f"act-{args.task_name}"
        / "demo_clean-50"
    )
    missing_checkpoint_files = [
        path.name
        for path in (
            checkpoint_dir / "policy_last.ckpt",
            checkpoint_dir / "dataset_stats.pkl",
        )
        if not path.is_file()
    ]
    if missing_checkpoint_files:
        raise ProtocolError(
            f"ACT checkpoint is incomplete for {args.task_name}: "
            + ", ".join(missing_checkpoint_files)
        )
    config = {
        "request": str(args.request).strip(),
        "task_name": args.task_name,
        "task_module": args.task_module or f"envs.{args.task_name}",
        "policy": "ACT",
        "repetitions": repetitions,
        "episodes": episodes,
        "start_seed": int(args.start_seed),
        "model_profile": args.model_profile,
        "telemetry_profile": args.telemetry_profile,
        "gpu": int(args.gpu),
        "max_reflections": int(args.max_reflections),
        "base_url": _validate_base_url(args.base_url),
        "history": "disabled_for_repetition_comparability",
        "task_profile": task_profile,
        "generated_rounds": (
            int(args.generated_rounds) if task_profile == "position_lr" else None
        ),
        "expected_variant_ids": expected_variant_ids,
        "sample_identity_fields": (
            ["variant_id", "seed"] if expected_variant_ids else ["seed"]
        ),
    }
    if not config["request"]:
        raise ProtocolError("--request must be non-empty")
    return {
        "schema_version": 2 if expected_variant_ids else 1,
        "protocol": (
            "agent_act_generated_agile_v2"
            if expected_variant_ids
            else "agent_act_agile_v1"
        ),
        "run_id": run_id,
        "status": "created",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "base_commit": _git_head(repo_root),
        "config": config,
        "config_sha256": canonical_sha256(config),
        "repetitions": build_repetition_schedule(
            repetitions=repetitions,
            episodes=episodes,
            start_seed=args.start_seed,
        ),
    }


def _load_manifest(repo_root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    resolved = validate_run_id(run_id)
    run_dir = repo_root / "mea/protocol_runs" / resolved
    path = run_dir / "protocol_manifest.json"
    if not path.is_file():
        raise ProtocolError(f"protocol manifest does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != resolved:
        raise ProtocolError("protocol manifest identity mismatch")
    supported = {
        (1, "agent_act_agile_v1"),
        (2, "agent_act_generated_agile_v2"),
    }
    if (value.get("schema_version"), value.get("protocol")) not in supported:
        raise ProtocolError("unsupported protocol manifest schema")
    if value.get("config_sha256") != canonical_sha256(value.get("config") or {}):
        raise ProtocolError("protocol config hash mismatch")
    config = value.get("config") or {}
    expected = build_repetition_schedule(
        repetitions=config.get("repetitions"),
        episodes=config.get("episodes"),
        start_seed=config.get("start_seed"),
    )
    repetitions = value.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != len(expected):
        raise ProtocolError("protocol repetition schedule length mismatch")
    immutable_fields = ("index", "start_seed", "requested_episodes")
    for actual, scheduled in zip(repetitions, expected):
        if not isinstance(actual, dict) or any(
            actual.get(field) != scheduled[field] for field in immutable_fields
        ):
            raise ProtocolError("protocol repetition schedule was modified")
    current_head = _git_head(repo_root)
    if value.get("base_commit") and current_head != value.get("base_commit"):
        raise ProtocolError(
            "repository HEAD changed since protocol creation; start a new run"
        )
    return run_dir, value


def _expected_samples(
    config: dict[str, Any], repetition: dict[str, Any]
) -> list[dict[str, Any]] | None:
    variant_ids = list(config.get("expected_variant_ids") or [])
    if not variant_ids:
        return None
    return build_expected_sample_identities(
        variant_ids=variant_ids,
        episodes=repetition["requested_episodes"],
        start_seed=repetition["start_seed"],
    )


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    _validate_robotwin_runtime()
    repo_root = args.repo_root.expanduser().resolve()
    if args.resume_run:
        if args.run_id:
            raise ProtocolError("--resume-run and --run-id are mutually exclusive")
        run_dir, manifest = _load_manifest(repo_root, args.resume_run)
    else:
        manifest = _new_manifest(args, repo_root)
        run_dir = repo_root / "mea/protocol_runs" / manifest["run_id"]
        if run_dir.exists():
            raise ProtocolError(f"protocol run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        _write_status(run_dir, manifest)

    chunk_size = validate_budget(args.chunk_size, name="chunk_size")
    lock_path = _acquire_lock(run_dir)
    newly_executed = 0
    try:
        # Reaching this point means any prior wrapper lock was stale. Refuse to
        # duplicate a still-running Agent child; otherwise preserve/recover the
        # old attempt before creating an append-only retry.
        stale_running = False
        for repetition in manifest["repetitions"]:
            if repetition.get("status") != "running":
                continue
            attempts = repetition.get("attempts") or []
            attempt = attempts[-1] if attempts else None
            if attempt and _pid_matches_attempt(
                attempt.get("child_pid"), str(attempt.get("evaluation_id") or "")
            ):
                raise ProtocolError(
                    "the prior Agent child is still active under pid "
                    f"{attempt.get('child_pid')}"
                )
            recovered = None
            if attempt:
                recovered = collect_evaluation_measurement(
                    repo_root,
                    evaluation_id=attempt["evaluation_id"],
                    requested_episodes=repetition["requested_episodes"],
                    returncode=0,
                    agent_wall_duration_seconds=0.0,
                    expected_sample_identities=_expected_samples(
                        manifest["config"], repetition
                    ),
                )
            if attempt and recovered and recovered["completed"]:
                attempt["measurement"] = recovered
                attempt["status"] = "completed"
                attempt["finished_at"] = now_iso()
                repetition["status"] = "completed"
            else:
                if attempt:
                    attempt["measurement"] = recovered
                    attempt["status"] = "interrupted"
                    attempt["finished_at"] = now_iso()
                repetition["status"] = "interrupted"
            stale_running = True
        if stale_running:
            manifest["status"] = "in_progress"
            manifest["updated_at"] = now_iso()
            _write_status(run_dir, manifest)

        for repetition in manifest["repetitions"]:
            if newly_executed >= chunk_size:
                break
            if repetition["status"] == "completed":
                continue
            if repetition["status"] == "failed" and not args.retry_failed:
                continue
            if repetition["status"] not in {"pending", "failed", "interrupted"}:
                continue

            attempt_index = len(repetition["attempts"]) + 1
            evaluation_id = evaluation_id_for_attempt(
                manifest["run_id"], repetition["index"], attempt_index
            )
            attempt_dir = (
                run_dir
                / "repetitions"
                / f"rep_{repetition['index']:03d}"
                / f"attempt_{attempt_index:02d}"
            )
            command = _agent_command(
                repo_root,
                manifest["config"],
                evaluation_id=evaluation_id,
                start_seed=repetition["start_seed"],
            )
            attempt = {
                "attempt_index": attempt_index,
                "evaluation_id": evaluation_id,
                "status": "running",
                "started_at": now_iso(),
                "finished_at": None,
                "command_path": str(
                    (attempt_dir / "command.json").relative_to(run_dir)
                ),
                "log_path": str((attempt_dir / "agent.log").relative_to(run_dir)),
                "measurement": None,
                "child_pid": None,
                "command_sha256": hashlib.sha256(
                    json.dumps(command, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
            }
            repetition["attempts"].append(attempt)
            repetition["status"] = "running"
            manifest["status"] = "running"
            manifest["updated_at"] = now_iso()
            write_json_atomic(
                attempt_dir / "command.json",
                {
                    "command": command,
                    "policy": "ACT",
                    "credentials_recorded": False,
                },
            )
            _write_status(run_dir, manifest)

            started = time.perf_counter()

            def record_child_pid(pid: int) -> None:
                attempt["child_pid"] = int(pid)
                manifest["updated_at"] = now_iso()
                _write_status(run_dir, manifest)

            try:
                returncode = _run_logged(
                    command,
                    cwd=repo_root,
                    log_path=attempt_dir / "agent.log",
                    on_start=record_child_pid,
                )
                duration = time.perf_counter() - started
                measurement = collect_evaluation_measurement(
                    repo_root,
                    evaluation_id=evaluation_id,
                    requested_episodes=repetition["requested_episodes"],
                    returncode=returncode,
                    agent_wall_duration_seconds=duration,
                    expected_sample_identities=_expected_samples(
                        manifest["config"], repetition
                    ),
                )
            except BaseException as exc:
                duration = time.perf_counter() - started
                try:
                    measurement = collect_evaluation_measurement(
                        repo_root,
                        evaluation_id=evaluation_id,
                        requested_episodes=repetition["requested_episodes"],
                        returncode=130 if isinstance(exc, KeyboardInterrupt) else 1,
                        agent_wall_duration_seconds=duration,
                        expected_sample_identities=_expected_samples(
                            manifest["config"], repetition
                        ),
                    )
                except Exception:
                    measurement = {
                        "completed": False,
                        "failure_stage": "protocol_wrapper",
                        "agent_wall_duration_seconds": duration,
                        "artifact_issues": [],
                    }
                measurement["wrapper_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                attempt["measurement"] = measurement
                attempt["finished_at"] = now_iso()
                attempt["status"] = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                )
                repetition["status"] = attempt["status"]
                manifest["updated_at"] = now_iso()
                _write_status(run_dir, manifest)
                raise
            attempt["measurement"] = measurement
            attempt["finished_at"] = now_iso()
            attempt["status"] = "completed" if measurement["completed"] else "failed"
            repetition["status"] = attempt["status"]
            manifest["updated_at"] = now_iso()
            newly_executed += 1
            _write_status(run_dir, manifest)

        summary = _write_status(run_dir, manifest)
        manifest["status"] = summary["status"]
        manifest["updated_at"] = now_iso()
        summary = _write_status(run_dir, manifest)
        return summary
    finally:
        lock_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--request")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-run")
    parser.add_argument("--task-name", default="click_bell")
    parser.add_argument("--task-module")
    parser.add_argument(
        "--task-profile",
        choices=["official", "position_lr"],
        default="official",
    )
    parser.add_argument("--generated-rounds", type=int, choices=[1, 2], default=2)
    parser.add_argument("--repetitions", type=int, choices=AGILE_BUDGETS, default=1)
    parser.add_argument("--episodes", type=int, choices=AGILE_BUDGETS, default=1)
    parser.add_argument("--chunk-size", type=int, choices=AGILE_BUDGETS, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--start-seed", type=int, default=100401)
    parser.add_argument(
        "--model-profile", choices=available_model_profiles(), default="economy"
    )
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-reflections", type=int, default=0)
    parser.add_argument("--base-url")
    return parser.parse_args()


def main() -> None:
    try:
        summary = run_protocol(parse_args())
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("failed_repetitions") or summary.get("status") == (
        "completed_with_protocol_violation"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
