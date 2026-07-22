"""Bounded, stage-aware recovery for one TaskGen candidate.

This controller is deliberately narrower than whole-round recovery.  It may
repair or regenerate a task candidate before policy evaluation starts, but it
never treats an ACT failure as a reason to generate another task or to rerun
the policy.  The callback receives the typed action selected for the previous
failure, so the caller can route to the existing SuccessSpec or visual repair
implementation without duplicating those implementations here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping


class TaskGenerationRecoveryError(RuntimeError):
    """Raised when a candidate cannot be accepted within the fixed budget."""

    def __init__(self, message: str, *, summary: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.summary = dict(summary or {})


@dataclass
class TaskGenerationStageError(RuntimeError):
    """Typed TaskGen failure reported by a materializer or validation gate."""

    stage: str
    failure_kind: str
    message: str
    runtime: Mapping[str, Any] = field(default_factory=dict)
    diagnosis: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


TERMINAL = "terminal"
REPAIR_SUCCESS_SPEC = "repair_success_spec"
REGENERATE_CANDIDATE = "regenerate_candidate"
REPAIR_SCENE = "repair_scene"


_RECOVERY_ACTIONS: dict[tuple[str, str], str] = {
    ("success_spec", "invalid_spec"): REPAIR_SUCCESS_SPEC,
    ("scene_codegen", "invalid_candidate"): REGENERATE_CANDIDATE,
    ("static_validation", "failed"): REGENERATE_CANDIDATE,
    ("render_probe", "failed"): REPAIR_SCENE,
    ("vision_validation", "failed"): REPAIR_SCENE,
    ("expert_gate", "unsolvable"): REGENERATE_CANDIDATE,
}


def task_generation_recovery_action(stage: str, failure_kind: str) -> str:
    """Return the bounded local action; unknown or policy failures are terminal."""

    return _RECOVERY_ACTIONS.get((stage, failure_kind), TERMINAL)


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
        raise TaskGenerationRecoveryError(
            f"proposal identity is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise TaskGenerationRecoveryError(
            f"append-only TaskGen artifact already exists: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _runtime(value: Mapping[str, Any] | None) -> dict[str, int | bool]:
    raw = dict(value or {})
    result: dict[str, int | bool] = {}
    for field in ("provider_calls", "simulator_probes", "expert_probes"):
        item = raw.get(field, 0)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise TaskGenerationRecoveryError(f"{field} must be a non-negative integer")
        result[field] = item
    act = raw.get("act_rollouts_started", 0)
    if isinstance(act, bool) or not isinstance(act, int) or act < 0:
        raise TaskGenerationRecoveryError(
            "act_rollouts_started must be a non-negative integer"
        )
    result["act_rollouts_started"] = act
    return result


def _totals(attempts: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "provider_calls",
        "simulator_probes",
        "expert_probes",
        "act_rollouts_started",
    )
    return {
        field: sum(int(attempt["runtime"][field]) for attempt in attempts)
        for field in fields
    }


def _summary(
    *,
    identity_sha256: str,
    status: str,
    max_regenerations: int,
    attempts: list[dict[str, Any]],
    accepted_result: Mapping[str, Any] | None = None,
    execution_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "recovery_scope": "task_generation_before_policy",
        "proposal_identity_sha256": identity_sha256,
        "status": status,
        "max_regenerations": max_regenerations,
        "attempt_count": len(attempts),
        "regenerations_used": max(len(attempts) - 1, 0),
        "attempts": attempts,
        "runtime": _totals(attempts),
        "policy_retry_allowed": False,
    }
    if accepted_result is not None:
        result["accepted_result"] = dict(accepted_result)
    if execution_result is not None:
        result["post_acceptance_execution"] = dict(execution_result)
    return result


AttemptCallback = Callable[[Path, int, str | None], Mapping[str, Any]]
AcceptedCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_bounded_task_generation(
    attempt_root: str | Path,
    *,
    proposal_identity: Mapping[str, Any],
    execute_attempt: AttemptCallback,
    execute_after_acceptance: AcceptedCallback | None = None,
    max_regenerations: int = 1,
) -> dict[str, Any]:
    """Generate/repair at most twice, then optionally launch policy exactly once.

    ``execute_attempt`` must raise :class:`TaskGenerationStageError` for a
    classified gate failure.  On the next call it receives the selected action
    (for example ``repair_scene``).  This is the executable connection between
    a diagnosis and the caller's existing bounded repair callback.
    """

    if not isinstance(proposal_identity, Mapping) or not proposal_identity:
        raise ValueError("proposal_identity must be a non-empty object")
    if max_regenerations not in {0, 1}:
        raise ValueError("max_regenerations must be 0 or 1")
    root = Path(attempt_root)
    if root.exists():
        raise TaskGenerationRecoveryError(f"attempt root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    identity_sha256 = _canonical_sha256(proposal_identity)
    attempts: list[dict[str, Any]] = []
    next_action: str | None = None

    for zero_index in range(max_regenerations + 1):
        attempt_index = zero_index + 1
        attempt_dir = root / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(exist_ok=False)
        started_at = _now()
        _write_new(
            attempt_dir / "attempt_started.json",
            {
                "schema_version": 1,
                "attempt_index": attempt_index,
                "status": "started",
                "started_at": started_at,
                "proposal_identity_sha256": identity_sha256,
                "requested_action": next_action,
            },
        )
        if _canonical_sha256(proposal_identity) != identity_sha256:
            error = TaskGenerationStageError(
                "resolution", "proposal_identity_changed", "proposal identity changed"
            )
            action = TERMINAL
            runtime = _runtime(None)
        else:
            try:
                value = execute_attempt(attempt_dir, attempt_index, next_action)
                if not isinstance(value, Mapping):
                    raise TypeError("TaskGen attempt result must be a mapping")
                if value.get("status") != "accepted":
                    raise TaskGenerationStageError(
                        "acceptance", "not_accepted", "attempt did not return accepted"
                    )
                runtime = _runtime(
                    value.get("runtime") if isinstance(value.get("runtime"), Mapping) else None
                )
                if runtime["act_rollouts_started"] != 0:
                    raise TaskGenerationStageError(
                        "policy_execution",
                        "started_before_task_acceptance",
                        "TaskGen attempt started ACT before candidate acceptance",
                        runtime=runtime,
                    )
            except Exception as exc:
                error = (
                    exc
                    if isinstance(exc, TaskGenerationStageError)
                    else TaskGenerationStageError(
                        "task_generation",
                        "unclassified_exception",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                action = task_generation_recovery_action(
                    error.stage, error.failure_kind
                )
                runtime = _runtime(error.runtime)
            else:
                attempt = {
                    "attempt_index": attempt_index,
                    "status": "accepted",
                    "started_at": started_at,
                    "finished_at": _now(),
                    "requested_action": next_action,
                    "failure": None,
                    "recovery_action": "none",
                    "runtime": runtime,
                    "result": dict(value),
                }
                attempts.append(attempt)
                _write_new(attempt_dir / "attempt_result.json", attempt)
                execution_result = None
                if execute_after_acceptance is not None:
                    executed = execute_after_acceptance(value)
                    if not isinstance(executed, Mapping):
                        raise TaskGenerationRecoveryError(
                            "post-acceptance execution result must be a mapping"
                        )
                    execution_result = dict(executed)
                summary = _summary(
                    identity_sha256=identity_sha256,
                    status="accepted",
                    max_regenerations=max_regenerations,
                    attempts=attempts,
                    accepted_result=value,
                    execution_result=execution_result,
                )
                _write_new(root / "task_generation_attempt_summary.json", summary)
                return summary

        retryable = action != TERMINAL and zero_index < max_regenerations
        attempt = {
            "attempt_index": attempt_index,
            "status": "failed_retryable" if retryable else "failed_terminal",
            "started_at": started_at,
            "finished_at": _now(),
            "requested_action": next_action,
            "failure": {
                "stage": error.stage,
                "failure_kind": error.failure_kind,
                "type": type(error).__name__,
                "message": error.message,
                "diagnosis": dict(error.diagnosis),
            },
            "recovery_action": action,
            "runtime": runtime,
        }
        attempts.append(attempt)
        _write_new(attempt_dir / "attempt_result.json", attempt)
        if retryable:
            next_action = action
            continue
        break

    summary = _summary(
        identity_sha256=identity_sha256,
        status="failed",
        max_regenerations=max_regenerations,
        attempts=attempts,
    )
    _write_new(root / "task_generation_attempt_summary.json", summary)
    raise TaskGenerationRecoveryError(
        f"TaskGen recovery failed after {len(attempts)} attempt(s)",
        summary=summary,
    )


__all__ = [
    "REGENERATE_CANDIDATE",
    "REPAIR_SCENE",
    "REPAIR_SUCCESS_SPEC",
    "TERMINAL",
    "TaskGenerationRecoveryError",
    "TaskGenerationStageError",
    "run_bounded_task_generation",
    "task_generation_recovery_action",
]
