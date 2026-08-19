"""Internal contracts shared by one native RoboTwin method round."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import BackendTaskBinding, MaterializedCandidate, MethodRuntime


class NativeAgentRoundError(RuntimeError):
    """Raised when a native policy round exceeds validated capabilities."""


_NATIVE_RUN_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_MAX_PATH_COMPONENT_LENGTH = 255


def _native_run_token(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeAgentRoundError(f"{field} must be a non-empty string")
    token = value.strip()
    if _NATIVE_RUN_TOKEN.fullmatch(token) is None:
        raise NativeAgentRoundError(
            f"{field} must contain only letters, digits, or underscore"
        )
    return token


def build_native_run_id(
    evaluation_id: str,
    round_id: str,
    policy_backend: str,
) -> str:
    run_id = "run_native_{}_{}_{}".format(
        _native_run_token(policy_backend, "policy_backend"),
        _native_run_token(evaluation_id, "evaluation_id"),
        _native_run_token(round_id, "round_id"),
    )
    if len(run_id) > _MAX_PATH_COMPONENT_LENGTH:
        raise NativeAgentRoundError(
            "native run_id exceeds the filesystem path-component limit"
        )
    return run_id


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
