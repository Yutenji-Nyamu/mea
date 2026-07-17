"""Bounded generation of offline tools over recorded trajectories."""

from .prototype import (
    ToolGenError,
    ToolGenPrototype,
    execute_generated_tool,
    extract_generated_tool,
    validate_generated_tool,
)
from .orchestration import (
    ToolOrchestrationError,
    bell_active_tcp_min_xy_error_tool_request,
    contact_tool_request,
    contact_tool_spec,
    execute_tool_request,
    execute_tool_spec,
    official_success_tool_request,
    pickup_to_contact_tool_request,
    pickup_to_contact_tool_spec,
    validate_tool_spec,
)
from .router import (
    ToolRouterError,
    catalog_snapshot,
    route_tool_request,
    validate_tool_request,
)
from .registry import (
    RunLocalRegistryError,
    find_run_local_registration,
    infer_registry_dir,
    load_registry,
    request_candidate_promotion,
)
from .targets import (
    BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
    PICKUP_TO_CONTACT_METRIC,
    evaluate_target_oracle,
)

__all__ = [
    "ToolGenError",
    "ToolGenPrototype",
    "execute_generated_tool",
    "extract_generated_tool",
    "validate_generated_tool",
    "ToolOrchestrationError",
    "bell_active_tcp_min_xy_error_tool_request",
    "ToolRouterError",
    "RunLocalRegistryError",
    "contact_tool_request",
    "contact_tool_spec",
    "execute_tool_request",
    "execute_tool_spec",
    "official_success_tool_request",
    "pickup_to_contact_tool_request",
    "pickup_to_contact_tool_spec",
    "catalog_snapshot",
    "route_tool_request",
    "validate_tool_request",
    "validate_tool_spec",
    "find_run_local_registration",
    "infer_registry_dir",
    "load_registry",
    "request_candidate_promotion",
    "PICKUP_TO_CONTACT_METRIC",
    "BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC",
    "evaluate_target_oracle",
]
