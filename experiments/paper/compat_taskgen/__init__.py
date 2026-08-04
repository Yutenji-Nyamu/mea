"""Frozen TaskGen compatibility surface for paper protocols.

Production RoboTwin rounds use :mod:`mea.taskgen.runtime` through the shared
method runtime.  Importing this package must not eagerly import the historical
task-specific CLI; the public dispatcher loads it only for an explicit generic
standalone invocation or a compatibility caller.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def load_legacy_cli() -> ModuleType:
    """Load the historical standalone implementation on demand."""

    return import_module("experiments.paper.compat_taskgen.legacy_cli")


__all__ = ["load_legacy_cli"]
