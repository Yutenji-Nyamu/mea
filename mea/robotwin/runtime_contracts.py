"""Shared contracts for the RoboTwin MethodRuntime backend."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from mea.method_runtime import MaterializedCandidate, RolloutRequest
from mea.taskgen.generic_backend import GenericRoboTwinTaskAdapter

from .task_identity import RoboTwinTaskIdentity

RuntimeTaskIdentity = GenericRoboTwinTaskAdapter | RoboTwinTaskIdentity
TaskAdapterFactory = Callable[[str], RuntimeTaskIdentity]
TaskContextProbeRunner = Callable[..., Mapping[str, Any]]
AcceptedTaskGenMaterializer = Callable[..., Mapping[str, Any]]


class RoboTwinRolloutRunner(Protocol):
    """Translate one materialized candidate into a native policy episode."""

    def __call__(
        self,
        *,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class _RoboTwinNativeCandidate:
    adapter: RuntimeTaskIdentity
    experiment_candidate: Mapping[str, Any]
    taskgen_resolution: Mapping[str, Any]
    rollout_manifest: Mapping[str, Any]


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _json_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return deepcopy(dict(value))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
