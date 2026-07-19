"""Stage-aware, append-only whole-round recovery controller.

The central action table mirrors the paper's stage-specific behavior.  Local
planning/TaskGen/ToolGen regeneration remains the responsibility of those
stages.  Only an unexpected exception while executing the planned Tool is
eligible to restart the whole evaluation round.  Policy and simulator failures
are recorded as policy outcomes and are never retried here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping


class WholeRoundRecoveryError(RuntimeError):
    """The whole-round controller stopped without a successful round result."""

    def __init__(self, message: str, *, summary: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.summary = dict(summary or {})


@dataclass
class StageFailure(RuntimeError):
    """A typed stage failure classified by the central recovery table."""

    stage: str
    failure_kind: str
    message: str
    runtime: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


RESTART_PLANNING = "restart_planning_attempt"
REGENERATE_TASK = "regenerate_task"
REGENERATE_TOOL = "regenerate_tool"
RESTART_WHOLE_ROUND = "restart_whole_round"
RECORD_POLICY_FAILURE = "record_policy_failure"
FAIL_EVALUATION = "fail_evaluation"


_ACTION_TABLE: dict[tuple[str, str], str] = {
    ("planning", "ground_truth_disagreement"): RESTART_PLANNING,
    ("task_generation", "visual_self_check_failed"): REGENERATE_TASK,
    ("tool_generation", "unit_test_failed"): REGENERATE_TOOL,
    ("tool_execution", "unexpected_exception"): RESTART_WHOLE_ROUND,
    ("policy_execution", "policy_failure"): RECORD_POLICY_FAILURE,
    ("simulation", "engine_failure"): RECORD_POLICY_FAILURE,
}


def recovery_action(stage: str, failure_kind: str) -> str:
    """Return the only permitted action for a typed stage failure."""

    return _ACTION_TABLE.get((stage, failure_kind), FAIL_EVALUATION)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WholeRoundRecoveryError(
            f"round identity is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise WholeRoundRecoveryError(f"append-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _runtime(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    provider = raw.get("provider_called", False)
    simulator = raw.get("simulator_called", False)
    act = raw.get("act_rollouts_started", 0)
    if not isinstance(provider, bool) or not isinstance(simulator, bool):
        raise WholeRoundRecoveryError("runtime provider/simulator fields must be boolean")
    if isinstance(act, bool) or not isinstance(act, int) or act < 0:
        raise WholeRoundRecoveryError("runtime ACT count must be a non-negative integer")
    return {
        "provider_called": provider,
        "simulator_called": simulator,
        "act_rollouts_started": act,
    }


def _totals(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [attempt["runtime"] for attempt in attempts]
    return {
        "provider_called": any(row["provider_called"] for row in runtimes),
        "simulator_called": any(row["simulator_called"] for row in runtimes),
        "act_rollouts_started": sum(row["act_rollouts_started"] for row in runtimes),
    }


def _summary(
    *,
    logical_round_id: str,
    identity_sha256: str,
    status: str,
    max_restarts: int,
    attempts: list[dict[str, Any]],
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    restarts = max(len(attempts) - 1, 0)
    total_runtime = _totals(attempts)
    additional_act = sum(
        attempt["runtime"]["act_rollouts_started"] for attempt in attempts[1:]
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "recovery_scope": "whole_evaluation_round",
        "logical_round_id": logical_round_id,
        "round_identity_sha256": identity_sha256,
        "status": status,
        "max_restarts": max_restarts,
        "attempt_count": len(attempts),
        "restarts_used": restarts,
        "whole_round_restarted": restarts > 0,
        "policy_or_simulator_restarted": restarts > 0,
        "additional_round_attempts_started_by_recovery": restarts,
        "additional_act_rollouts_started_by_recovery": additional_act,
        "runtime": total_runtime,
        "attempts": attempts,
    }
    if result is not None:
        value["result"] = dict(result)
    return value


def run_stage_aware_round_recovery(
    attempt_root: str | Path,
    *,
    logical_round_id: str,
    round_identity: Mapping[str, Any],
    execute_round: Callable[[Path, int], Mapping[str, Any]],
    max_restarts: int = 1,
) -> dict[str, Any]:
    """Execute one logical round with at most one eligible whole-round restart."""

    if not isinstance(logical_round_id, str) or not logical_round_id:
        raise ValueError("logical_round_id must be non-empty")
    if not isinstance(round_identity, Mapping) or not round_identity:
        raise ValueError("round_identity must be a non-empty object")
    if max_restarts not in {0, 1}:
        raise ValueError("max_restarts must be 0 or 1")
    root = Path(attempt_root)
    if root.exists():
        raise WholeRoundRecoveryError(f"attempt root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    initial_identity_hash = _canonical_sha256(round_identity)
    attempts: list[dict[str, Any]] = []

    for zero_index in range(max_restarts + 1):
        attempt_index = zero_index + 1
        attempt_dir = root / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(exist_ok=False)
        current_identity_hash = _canonical_sha256(round_identity)
        started_at = _now()
        _write_new(
            attempt_dir / "attempt_started.json",
            {
                "schema_version": 1,
                "attempt_index": attempt_index,
                "status": "started",
                "started_at": started_at,
                "logical_round_id": logical_round_id,
                "round_identity_sha256": current_identity_hash,
            },
        )
        if current_identity_hash != initial_identity_hash:
            failure = StageFailure(
                "round_orchestration",
                "identity_changed",
                "round identity changed between attempts",
            )
            action = FAIL_EVALUATION
            attempt = {
                "attempt_index": attempt_index,
                "status": "failed_integrity",
                "started_at": started_at,
                "finished_at": _now(),
                "round_identity_sha256": current_identity_hash,
                "failure": {
                    "stage": failure.stage,
                    "failure_kind": failure.failure_kind,
                    "type": type(failure).__name__,
                    "message": failure.message,
                },
                "recovery_action": action,
                "runtime": _runtime(None),
            }
            attempts.append(attempt)
            _write_new(attempt_dir / "attempt_result.json", attempt)
            break
        try:
            value = execute_round(attempt_dir, attempt_index)
            if not isinstance(value, Mapping):
                raise TypeError("round result must be a mapping")
            if _canonical_sha256(round_identity) != initial_identity_hash:
                raise StageFailure(
                    "round_orchestration",
                    "identity_changed",
                    "round identity changed during execution",
                )
            runtime = _runtime(value.get("runtime") if isinstance(value, Mapping) else None)
        except Exception as exc:
            failure = (
                exc
                if isinstance(exc, StageFailure)
                else StageFailure(
                    "round_orchestration",
                    "unclassified_exception",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            action = recovery_action(failure.stage, failure.failure_kind)
            runtime = _runtime(failure.runtime)
            retryable = action == RESTART_WHOLE_ROUND and zero_index < max_restarts
            policy_failure = action == RECORD_POLICY_FAILURE
            attempt = {
                "attempt_index": attempt_index,
                "status": (
                    "failed_retryable"
                    if retryable
                    else "completed_policy_failure"
                    if policy_failure
                    else "failed_terminal"
                ),
                "started_at": started_at,
                "finished_at": _now(),
                "round_identity_sha256": current_identity_hash,
                "failure": {
                    "stage": failure.stage,
                    "failure_kind": failure.failure_kind,
                    "type": type(failure).__name__,
                    "message": failure.message,
                    "details": dict(failure.details),
                },
                "recovery_action": action,
                "runtime": runtime,
            }
            attempts.append(attempt)
            _write_new(attempt_dir / "attempt_result.json", attempt)
            if retryable:
                continue
            if policy_failure:
                result = {
                    "status": "policy_execution_failure",
                    "failure": attempt["failure"],
                }
                summary = _summary(
                    logical_round_id=logical_round_id,
                    identity_sha256=initial_identity_hash,
                    status="completed_with_policy_failure",
                    max_restarts=max_restarts,
                    attempts=attempts,
                    result=result,
                )
                _write_new(root / "recovery_summary.json", summary)
                return summary
            break
        attempt = {
            "attempt_index": attempt_index,
            "status": "completed",
            "started_at": started_at,
            "finished_at": _now(),
            "round_identity_sha256": current_identity_hash,
            "failure": None,
            "recovery_action": "none",
            "runtime": runtime,
            "result": dict(value),
        }
        attempts.append(attempt)
        _write_new(attempt_dir / "attempt_result.json", attempt)
        summary = _summary(
            logical_round_id=logical_round_id,
            identity_sha256=initial_identity_hash,
            status="completed",
            max_restarts=max_restarts,
            attempts=attempts,
            result=value,
        )
        _write_new(root / "recovery_summary.json", summary)
        return summary

    summary = _summary(
        logical_round_id=logical_round_id,
        identity_sha256=initial_identity_hash,
        status="failed",
        max_restarts=max_restarts,
        attempts=attempts,
    )
    _write_new(root / "recovery_summary.json", summary)
    raise WholeRoundRecoveryError(
        f"whole-round recovery failed after {len(attempts)} attempt(s); "
        f"summary={root / 'recovery_summary.json'}",
        summary=summary,
    )


__all__ = [
    "FAIL_EVALUATION",
    "RECORD_POLICY_FAILURE",
    "REGENERATE_TASK",
    "REGENERATE_TOOL",
    "RESTART_PLANNING",
    "RESTART_WHOLE_ROUND",
    "StageFailure",
    "WholeRoundRecoveryError",
    "recovery_action",
    "run_stage_aware_round_recovery",
]
