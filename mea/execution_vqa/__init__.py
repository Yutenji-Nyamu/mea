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

__all__ = [
    "ExecutionVQAError",
    "analyze_execution_montage",
    "build_execution_montage",
    "read_contact_events",
    "read_semantic_timeline",
    "run_execution_vqa",
    "select_keyframes",
    "validate_execution_vqa_response",
]
