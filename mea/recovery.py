"""Bounded same-round Tool-orchestration recovery controller.

The controller retries only an explicitly classified unexpected Tool
orchestration exception. It never retries policy or simulator failures and
never launches a policy rollout itself. A retry may repeat provider/registry
work performed inside the orchestration callback, so this is deliberately not
described as a deterministic-analysis-only replay.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping


class UnexpectedToolExecutionError(RuntimeError):
    """An unexpected Tool orchestration exception eligible for one retry."""


class BoundedRecoveryError(RuntimeError):
    """Recovery exhausted or violated its immutable-input contract."""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise BoundedRecoveryError(f"append-only artifact already exists: {path}")
    _write(path, value)


def run_bounded_tool_recovery(
    attempt_root: str | Path,
    *,
    logical_round_id: str,
    execute: Callable[[Path, int], Mapping[str, Any]],
    telemetry_sha256: Callable[[], str],
    max_restarts: int = 1,
) -> dict[str, Any]:
    """Execute Tool orchestration with at most one immutable-telemetry retry."""

    if max_restarts not in {0, 1}:
        raise ValueError("max_restarts must be 0 or 1")
    if not logical_round_id:
        raise ValueError("logical_round_id must be non-empty")
    root = Path(attempt_root)
    if root.exists():
        raise BoundedRecoveryError(f"attempt root already exists: {root}")
    root.mkdir(parents=True)
    initial_hash = telemetry_sha256()
    if not isinstance(initial_hash, str) or not initial_hash:
        raise BoundedRecoveryError("telemetry_sha256 must return a non-empty string")
    attempts: list[dict[str, Any]] = []

    for zero_index in range(max_restarts + 1):
        index = zero_index + 1
        attempt_dir = root / f"attempt_{index:02d}"
        attempt_dir.mkdir()
        started_at = _now()
        current_hash = telemetry_sha256()
        _write_new(
            attempt_dir / "attempt_started.json",
            {
                "attempt_index": index,
                "status": "started",
                "started_at": started_at,
                "telemetry_sha256": current_hash,
            },
        )
        if current_hash != initial_hash:
            attempt = {
                "attempt_index": index,
                "status": "integrity_failure",
                "started_at": started_at,
                "finished_at": _now(),
                "telemetry_sha256": current_hash,
                "failure": {
                    "type": "TelemetryIntegrityError",
                    "message": "telemetry changed between Tool recovery attempts",
                },
            }
            attempts.append(attempt)
            _write_new(attempt_dir / "attempt_result.json", attempt)
            break
        attempt = {
            "attempt_index": index,
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "telemetry_sha256": current_hash,
            "failure": None,
        }
        try:
            value = execute(attempt_dir, index)
            if not isinstance(value, Mapping):
                raise TypeError("Tool execution result must be a mapping")
        except Exception as exc:
            retryable = isinstance(exc, UnexpectedToolExecutionError)
            attempt.update(
                {
                    "status": "failed_retryable" if retryable else "failed_terminal",
                    "finished_at": _now(),
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            attempts.append(attempt)
            _write_new(attempt_dir / "attempt_result.json", attempt)
            if not retryable or zero_index >= max_restarts:
                break
            continue
        attempt.update(
            {
                "status": "completed",
                "finished_at": _now(),
                "result": dict(value),
            }
        )
        attempts.append(attempt)
        _write_new(attempt_dir / "attempt_result.json", attempt)
        summary = {
            "schema_version": 2,
            "logical_round_id": logical_round_id,
            "recovery_scope": "tool_orchestration_substage",
            "status": "completed",
            "max_restarts": max_restarts,
            "attempt_count": len(attempts),
            "restarts_used": len(attempts) - 1,
            "same_telemetry_reused": all(
                item["telemetry_sha256"] == initial_hash for item in attempts
            ),
            "telemetry_sha256": initial_hash,
            "failure_class": (
                "unexpected_tool_execution_exception" if len(attempts) > 1 else None
            ),
            "action": (
                "retry_tool_orchestration_substage" if len(attempts) > 1 else "none"
            ),
            "additional_act_rollouts_started_by_recovery": 0,
            "policy_or_simulator_restarted": False,
            "provider_or_registry_work_may_repeat": True,
            "attempts": attempts,
            "result": dict(value),
        }
        _write(root / "recovery_summary.json", summary)
        return summary

    summary = {
        "schema_version": 2,
        "logical_round_id": logical_round_id,
        "recovery_scope": "tool_orchestration_substage",
        "status": "failed",
        "max_restarts": max_restarts,
        "attempt_count": len(attempts),
        "restarts_used": max(len(attempts) - 1, 0),
        "same_telemetry_reused": all(
            item["telemetry_sha256"] == initial_hash for item in attempts
        ),
        "telemetry_sha256": initial_hash,
        "failure_class": (
            "unexpected_tool_execution_exception"
            if attempts
            and attempts[-1]["failure"]["type"] == "UnexpectedToolExecutionError"
            else "terminal_or_integrity_failure"
        ),
        "action": "stop_round_and_record_failure",
        "additional_act_rollouts_started_by_recovery": 0,
        "policy_or_simulator_restarted": False,
        "provider_or_registry_work_may_repeat": True,
        "attempts": attempts,
    }
    _write(root / "recovery_summary.json", summary)
    raise BoundedRecoveryError(
        f"Tool recovery failed after {len(attempts)} attempt(s); "
        f"summary={root / 'recovery_summary.json'}"
    )


__all__ = [
    "BoundedRecoveryError",
    "UnexpectedToolExecutionError",
    "run_bounded_tool_recovery",
]
