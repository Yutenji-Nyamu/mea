"""Internal contracts shared by one native RoboTwin method round."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import BackendTaskBinding, MaterializedCandidate, MethodRuntime


class NativeAgentRoundError(RuntimeError):
    """Raised when a native policy round exceeds validated capabilities."""


def build_native_run_id(
    evaluation_id: str,
    round_id: str,
    policy_backend: str,
) -> str:
    digest = hashlib.sha256(
        f"{evaluation_id}:{round_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"run_native_{policy_backend}_{digest}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NativeAgentRoundError(
            "TaskGen proposal identity is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def is_zero_rollout_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


@dataclass(frozen=True)
class NativeRoundPreparation:
    root: Path
    evaluation_root: Path
    contract: Mapping[str, Any]
    seed: int
    proposal: Mapping[str, Any] | None
    generated_task_required: bool
    run_id: str
    child_dir: Path
    rollout_dir: Path
    query: str
    runtime: MethodRuntime
    binding: BackendTaskBinding
    candidate: MaterializedCandidate
    taskgen_manifest: Mapping[str, Any] | None


@dataclass(frozen=True)
class NativeRoundEvaluation:
    authoritative_rollout: Any
    trusted_tool_evaluation: Mapping[str, Any]
    checker_result: Mapping[str, Any] | None
    evidence: Any
    execution_scope: str
    semantic_ready: bool
