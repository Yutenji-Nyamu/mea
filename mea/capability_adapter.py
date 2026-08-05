"""Deprecated compatibility import for frozen paper protocols.

Production code must use ``RuntimeTaskBinding`` for execution identity and
``ArtifactRetrievalIndex`` for optional reviewed Task/Tool/VQA hints.
"""

from experiments.paper.compat_capability_adapter import *  # noqa: F401,F403
from experiments.paper.compat_capability_adapter import __all__
