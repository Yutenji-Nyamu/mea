"""Shared errors for the production Plan Agent runtime."""

from __future__ import annotations


class PlanAgentSessionError(ValueError):
    """Raised when semantic planning cannot be bound to trusted evidence."""


__all__ = ["PlanAgentSessionError"]
