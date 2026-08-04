"""Shared errors for the production Plan Agent runtime."""

from __future__ import annotations


class PlanAgentSessionError(ValueError):
    """Raised when semantic planning cannot be bound to trusted evidence."""


# Historical name retained for readers and callers of pre-rename artifacts.
ClaimFirstRuntimeError = PlanAgentSessionError


__all__ = ["PlanAgentSessionError", "ClaimFirstRuntimeError"]
