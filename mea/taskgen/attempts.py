"""One generation pass plus at most one targeted TaskGen repair.

The simulator, checker fixtures, visual diagnosis, and expert gate live in the
materializer.  This module only decides whether one reported pre-policy failure
may be returned to that same materializer for a focused repair.  It writes one
result instead of maintaining an append-only attempt ledger or proposal hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


class TaskGenerationRecoveryError(RuntimeError):
    """Raised when generation and its optional repair do not validate."""

    def __init__(self, message: str, *, result: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.result = dict(result or {})


class CandidateUnexecutableError(TaskGenerationRecoveryError):
    """The generated candidate failed a validated pre-policy expert gate."""


@dataclass
class TaskGenerationStageError(RuntimeError):
    """Typed failure reported by a TaskGen materializer or validation gate."""

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
    ("preservation_validation", "failed"): REGENERATE_CANDIDATE,
    ("expert_gate", "candidate_unexecutable"): REGENERATE_CANDIDATE,
    ("expert_gate", "unsolvable"): REGENERATE_CANDIDATE,
}


def task_generation_recovery_action(stage: str, failure_kind: str) -> str:
    """Return the one local repair action; unknown and policy failures stop."""

    return _RECOVERY_ACTIONS.get((stage, failure_kind), TERMINAL)


def _runtime(value: Mapping[str, Any] | None) -> dict[str, int]:
    raw = dict(value or {})
    result: dict[str, int] = {}
    for name in (
        "provider_calls",
        "simulator_probes",
        "expert_probes",
        "act_rollouts_started",
    ):
        item = raw.get(name, 0)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise TaskGenerationRecoveryError(
                f"{name} must be a non-negative integer"
            )
        result[name] = item
    return result


def _failure(error: TaskGenerationStageError) -> dict[str, Any]:
    return {
        "stage": error.stage,
        "failure_kind": error.failure_kind,
        "type": type(error).__name__,
        "message": error.message,
        "diagnosis": dict(error.diagnosis),
    }


def _run(
    execute: Callable[[Path, int, str | None], Mapping[str, Any]],
    directory: Path,
    index: int,
    action: str | None,
) -> tuple[dict[str, Any] | None, TaskGenerationStageError | None, dict[str, int]]:
    directory.mkdir(parents=True, exist_ok=True)
    try:
        value = execute(directory, index, action)
        if not isinstance(value, Mapping):
            raise TypeError("TaskGen result must be an object")
        if value.get("status") != "accepted":
            raise TaskGenerationStageError(
                "acceptance", "not_accepted", "TaskGen result was not accepted"
            )
        runtime = _runtime(
            value.get("runtime") if isinstance(value.get("runtime"), Mapping) else None
        )
        if runtime["act_rollouts_started"]:
            raise TaskGenerationStageError(
                "policy_execution",
                "started_before_task_acceptance",
                "TaskGen started policy execution before candidate acceptance",
                runtime=runtime,
            )
        return dict(value), None, runtime
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
        return None, error, _runtime(error.runtime)


def _write_result(root: Path, result: Mapping[str, Any]) -> None:
    (root / "task_generation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


AttemptCallback = Callable[[Path, int, str | None], Mapping[str, Any]]
AcceptedCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_task_generation(
    work_root: str | Path,
    *,
    execute: AttemptCallback,
    execute_after_acceptance: AcceptedCallback | None = None,
    allow_repair: bool = True,
) -> dict[str, Any]:
    """Generate, validate, optionally repair once, then return one result."""

    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    value, error, generation_runtime = _run(execute, root / "generation", 1, None)
    generation: dict[str, Any] = {
        "status": "accepted" if error is None else "failed",
        "failure": None if error is None else _failure(error),
        "runtime": generation_runtime,
    }
    repair: dict[str, Any] | None = None
    if error is not None:
        action = task_generation_recovery_action(error.stage, error.failure_kind)
        if allow_repair and action != TERMINAL:
            value, repair_error, repair_runtime = _run(
                execute, root / "repair", 2, action
            )
            repair = {
                "action": action,
                "status": "accepted" if repair_error is None else "failed",
                "failure": (
                    None if repair_error is None else _failure(repair_error)
                ),
                "runtime": repair_runtime,
            }
            error = repair_error

    records = [generation] + ([repair] if repair is not None else [])
    runtime = {
        name: sum(int(record["runtime"][name]) for record in records)
        for name in generation_runtime
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted" if error is None else "failed",
        "generation": generation,
        "repair": repair,
        "runtime": runtime,
    }
    if error is None and value is not None:
        result["accepted_result"] = value
        if execute_after_acceptance is not None:
            executed = execute_after_acceptance(value)
            if not isinstance(executed, Mapping):
                raise TaskGenerationRecoveryError(
                    "post-acceptance execution result must be an object",
                    result=result,
                )
            result["post_acceptance_execution"] = dict(executed)
    _write_result(root, result)
    if error is not None:
        raise TaskGenerationRecoveryError(
            "TaskGen generation and optional repair did not validate",
            result=result,
        )
    return result


__all__ = [
    "CandidateUnexecutableError",
    "REGENERATE_CANDIDATE",
    "REPAIR_SCENE",
    "REPAIR_SUCCESS_SPEC",
    "TERMINAL",
    "TaskGenerationRecoveryError",
    "TaskGenerationStageError",
    "run_task_generation",
    "task_generation_recovery_action",
]
