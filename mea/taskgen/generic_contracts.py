"""Typed contracts shared by generic TaskGen generation stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


class GenericTaskGenError(RuntimeError):
    """Raised when a dynamic candidate cannot be reused or generated."""

    def __init__(
        self,
        message: str,
        *,
        runtime: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.runtime = dict(runtime or {})


ValidateMethods = Callable[
    [Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]
]
BuildModule = Callable[[Mapping[str, str], Mapping[str, Any]], str]
PreflightCandidate = Callable[
    [Path, str, Mapping[str, Any]], Mapping[str, Any]
]
ResolveMetric = Callable[[Mapping[str, Any]], str]
ResolveCheckerContract = Callable[
    [Mapping[str, Any]], Mapping[str, Any]
]
ExactTaskLookup = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
CheckerFixtureValidator = Callable[
    [Mapping[str, str], Mapping[str, Any]], list[Mapping[str, Any]]
]


@dataclass(frozen=True)
class GenericTaskGenHooks:
    """Simulator hooks shared by every semantic candidate for a task."""

    validate_methods: ValidateMethods
    build_module: BuildModule
    preflight_candidate: PreflightCandidate
    resolve_metric: ResolveMetric
    resolve_checker_contract: ResolveCheckerContract
    prompt_constraints: str = ""


@dataclass(frozen=True)
class GenericRoboTwinTaskAdapter:
    """Thin description of one policy-compatible official RoboTwin task."""

    task_name: str
    official_source: str
    official_class: str
    task_schema: Mapping[str, Any]
    documentation_paths: tuple[str, ...]
    asset_paths: tuple[str, ...]
    hooks: GenericTaskGenHooks
    task_context: Mapping[str, Any] | None = None


__all__ = [
    "BuildModule",
    "CheckerFixtureValidator",
    "ExactTaskLookup",
    "GenericRoboTwinTaskAdapter",
    "GenericTaskGenError",
    "GenericTaskGenHooks",
    "PreflightCandidate",
    "ResolveCheckerContract",
    "ResolveMetric",
    "ValidateMethods",
]
