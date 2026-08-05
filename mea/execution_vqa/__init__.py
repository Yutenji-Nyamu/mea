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
    RUN_LOCAL_NUMERIC_AUTHORITIES,
    RUN_LOCAL_QUESTION_MAX_CHARS,
    RUN_LOCAL_QUESTION_TYPES,
    RUN_LOCAL_TARGET_ROLES,
    RUN_LOCAL_VISUAL_SCOPES,
    ExecutionVQAQueryError,
    build_execution_vqa_query,
    is_run_local_phenomenon_id,
    validate_execution_vqa_query,
    validate_run_local_question_spec,
    vqa_need_semantic_key,
)
from .reviewed_generated_questions import (
    build_generated_vqa_question_review_template,
    find_reviewed_generated_vqa_question,
    install_reviewed_generated_vqa_question,
    load_reviewed_generated_vqa_questions,
    validate_generated_vqa_question_review,
)
from .reviewed_registry import (
    ReviewedVQAQuerySpecError,
    load_reviewed_vqa_query_specs,
    match_reviewed_vqa_query_spec,
    validate_vqa_query_review,
    validate_vqa_query_spec,
)
from .open_question import (
    OpenVQAQuestionAgent,
    OpenVQAQuestionError,
    load_run_local_vqa_questions,
    materialize_open_execution_vqa_query,
    register_run_local_vqa_question,
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
    "RUN_LOCAL_NUMERIC_AUTHORITIES",
    "RUN_LOCAL_QUESTION_MAX_CHARS",
    "RUN_LOCAL_QUESTION_TYPES",
    "RUN_LOCAL_TARGET_ROLES",
    "RUN_LOCAL_VISUAL_SCOPES",
    "ExecutionVQAQueryError",
    "build_execution_vqa_query",
    "is_run_local_phenomenon_id",
    "validate_execution_vqa_query",
    "validate_run_local_question_spec",
    "vqa_need_semantic_key",
    "ReviewedVQAQuerySpecError",
    "build_generated_vqa_question_review_template",
    "find_reviewed_generated_vqa_question",
    "install_reviewed_generated_vqa_question",
    "load_reviewed_generated_vqa_questions",
    "load_reviewed_vqa_query_specs",
    "match_reviewed_vqa_query_spec",
    "validate_vqa_query_review",
    "validate_vqa_query_spec",
    "validate_generated_vqa_question_review",
    "OpenVQAQuestionAgent",
    "OpenVQAQuestionError",
    "load_run_local_vqa_questions",
    "materialize_open_execution_vqa_query",
    "register_run_local_vqa_question",
]
