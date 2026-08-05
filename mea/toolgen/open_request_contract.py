"""Shared public contracts for open Tool requests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class OpenToolRequestError(ValueError):
    """Raised when a Query-induced Tool request is malformed."""


class OpenToolRequestUnsupported(OpenToolRequestError):
    """Raised when the runtime exposes no valid measurement path."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = deepcopy(dict(artifact))
        super().__init__(str(self.artifact["reason"]))


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenToolRequestError(f"{field} must be a non-empty string")
    return value.strip()
