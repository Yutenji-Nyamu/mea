"""Public MetricSpec API; implementation is split by method responsibility."""

from .metric_codegen import build_task_code_context, compile_metric_spec_source
from .metric_evaluation import _event_matches, evaluate_metric_spec
from .metric_runtime import execute_metric_spec
from .metric_schema import (
    MetricSpecError,
    metric_spec_tool_spec,
    validate_metric_spec,
)

__all__ = [
    "MetricSpecError",
    "build_task_code_context",
    "compile_metric_spec_source",
    "evaluate_metric_spec",
    "execute_metric_spec",
    "metric_spec_tool_spec",
    "validate_metric_spec",
]
