"""Evidence-grounded user feedback for completed MEA evaluations."""

from .prototype import (
    apply_deterministic_consistency_guard,
    FeedbackAgent,
    FeedbackAgentError,
    render_evaluation_report,
    validate_feedback,
)
from .evidence_report import EvidenceReportError, write_evidence_report
from .answer_scope import (
    AnswerScopeError,
    build_answer_scope,
    project_answer_scope,
    validate_answer_scope,
    validate_answer_scope_projection,
)

__all__ = [
    "apply_deterministic_consistency_guard",
    "FeedbackAgent",
    "FeedbackAgentError",
    "render_evaluation_report",
    "validate_feedback",
    "EvidenceReportError",
    "write_evidence_report",
    "AnswerScopeError",
    "build_answer_scope",
    "project_answer_scope",
    "validate_answer_scope",
    "validate_answer_scope_projection",
]
