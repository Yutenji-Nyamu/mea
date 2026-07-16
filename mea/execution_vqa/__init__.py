"""Execution-time visual evidence for RoboTwin policy rollouts.

This package is deliberately separate from TaskGen's scene-level visual
self-reflection.  It observes an already completed rollout and never changes
simulator-derived Tool results.
"""

from .prototype import (
    ExecutionVQAError,
    analyze_execution_montage,
    build_execution_montage,
    read_contact_events,
    read_semantic_timeline,
    run_execution_vqa,
    select_keyframes,
    validate_execution_vqa_response,
)
from .query import (
    ALL_PHENOMENON_IDS,
    ANSWER_CONTRACT,
    LEGACY_PHENOMENON_IDS,
    QUESTION_CATALOG,
    ExecutionVQAQueryError,
    build_execution_vqa_query,
    validate_execution_vqa_query,
)

__all__ = [
    "ExecutionVQAError",
    "analyze_execution_montage",
    "build_execution_montage",
    "read_contact_events",
    "read_semantic_timeline",
    "run_execution_vqa",
    "select_keyframes",
    "validate_execution_vqa_response",
    "ALL_PHENOMENON_IDS",
    "ANSWER_CONTRACT",
    "LEGACY_PHENOMENON_IDS",
    "QUESTION_CATALOG",
    "ExecutionVQAQueryError",
    "build_execution_vqa_query",
    "validate_execution_vqa_query",
]
