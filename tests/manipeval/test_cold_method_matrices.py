"""Cold compatibility, prompt-wording, and repeated schema matrices.

The default mainline retains one representative for each paper-method
boundary. These cases remain available to the full server regression without
inflating the routine method loop.
"""

from __future__ import annotations

import unittest

from tests.mainline.test_claim_first_runtime import (
    ClaimFirstRuntimeTests as _PlanAgentCases,
)
from tests.mainline.test_generic_taskgen_backend import (
    GenericTaskGenBackendTests as _TaskGenCases,
)
from tests.mainline.test_open_task_resolver import (
    QueryInterpretationTests as _QueryCases,
)
from tests.mainline.test_open_tool_request import (
    OpenToolRequestTest as _ToolGenCases,
)
from tests.mainline.test_production_cli_boundary import (
    ProductionCliBoundaryTests as _CliCases,
)


class ColdTaskGenMatrixTests(unittest.TestCase):
    test_semantic_field_access_guide_exposes_exact_read_only_apis = (
        _TaskGenCases._cold_semantic_field_access_guide_exposes_exact_read_only_apis
    )
    test_safe_ast_allows_conventional_discard_loop_target = (
        _TaskGenCases._cold_safe_ast_allows_conventional_discard_loop_target
    )
    test_compound_position_and_orientation_checks_both = (
        _TaskGenCases._cold_compound_position_and_orientation_checks_both
    )
    test_chinese_goal_and_contact_geometry_preservation_is_typed = (
        _TaskGenCases._cold_chinese_goal_and_contact_geometry_preservation_is_typed
    )
    test_height_preservation_ignores_requested_xy_offset = (
        _TaskGenCases._cold_height_preservation_ignores_requested_xy_offset
    )
    test_legacy_visual_color_preservation_still_passes = (
        _TaskGenCases._cold_legacy_visual_color_preservation_still_passes
    )
    test_pose_property_item_assignment_is_rejected = (
        _TaskGenCases._cold_pose_property_item_assignment_is_rejected
    )
    test_scale_gate_defers_nonliteral_or_irrelevant_changes = (
        _TaskGenCases._cold_scale_gate_defers_nonliteral_or_irrelevant_changes
    )
    test_ablation_condition_never_reuses_complete_task_artifact = (
        _TaskGenCases._cold_ablation_condition_never_reuses_complete_task_artifact
    )
    test_unexpected_preflight_failure_is_terminal_and_counted = (
        _TaskGenCases._cold_unexpected_preflight_failure_is_terminal_and_counted
    )
    test_unavailable_review_is_terminal_not_a_checker_repair = (
        _TaskGenCases._cold_unavailable_review_is_terminal_not_a_checker_repair
    )
    test_checker_repair_diagnosis_includes_terminal_xyz_state = (
        _TaskGenCases._cold_checker_repair_diagnosis_includes_terminal_xyz_state
    )


class ColdToolGenSchemaMatrixTests(unittest.TestCase):
    test_agent_exposes_and_accepts_terminal_signal_component = (
        _ToolGenCases._cold_agent_exposes_and_accepts_terminal_signal_component
    )
    test_agent_exposes_and_accepts_terminal_signal_difference = (
        _ToolGenCases._cold_agent_exposes_and_accepts_terminal_signal_difference
    )
    test_agent_exposes_terminal_minimum_distance_for_candidate_tcps = (
        _ToolGenCases._cold_agent_exposes_terminal_minimum_distance_for_candidate_tcps
    )
    test_lift_height_difference_aligns_operation_signals_and_component = (
        _ToolGenCases._cold_lift_height_difference_aligns_operation_signals_and_component
    )
    test_terminal_semantic_need_rejects_event_metric_and_wrong_component = (
        _ToolGenCases._cold_terminal_semantic_need_rejects_event_metric_and_wrong_component
    )
    test_absolute_terminal_component_requires_absolute_flag = (
        _ToolGenCases._cold_absolute_terminal_component_requires_absolute_flag
    )
    test_multi_component_terminal_need_cannot_escape_to_event_metric = (
        _ToolGenCases._cold_multi_component_terminal_need_cannot_escape_to_event_metric
    )
    test_event_count_and_time_between_event_shapes_are_executable = (
        _ToolGenCases._cold_event_count_and_time_between_event_shapes_are_executable
    )
    test_active_arm_need_rejects_fixed_side_metric_spec = (
        _ToolGenCases._cold_active_arm_need_rejects_fixed_side_metric_spec
    )
    test_click_point_accuracy_rejects_target_to_target_distance = (
        _ToolGenCases._cold_click_point_accuracy_rejects_target_to_target_distance
    )


class ColdQueryInterpretationMatrixTests(unittest.TestCase):
    Provider = _QueryCases.Provider
    test_experimental_success_wording_requires_checker = (
        _QueryCases._cold_experimental_success_wording_requires_checker
    )
    test_generated_checker_requirement_is_not_treated_as_optional = (
        _QueryCases._cold_generated_checker_requirement_is_not_treated_as_optional
    )
    test_imperative_generated_round_checker_is_not_optional = (
        _QueryCases._cold_imperative_generated_round_checker_is_not_optional
    )
    test_agent_can_be_frozen_to_one_attempt = (
        _QueryCases._cold_agent_can_be_frozen_to_one_attempt
    )
    test_historical_free_concern_class_name_remains_readable = (
        _QueryCases._cold_historical_free_concern_class_name_remains_readable
    )
    test_single_task_rejects_clearly_wrong_cross_task_even_with_margin = (
        _QueryCases._cold_single_task_rejects_clearly_wrong_cross_task_even_with_margin
    )


class ColdPlanAgentCompatibilityTests(unittest.TestCase):
    test_plan_agent_session_is_canonical_with_legacy_aliases = (
        _PlanAgentCases._cold_plan_agent_session_is_canonical_with_legacy_aliases
    )
    test_provider_incidental_catalog_words_do_not_hide_external_mass_concern = (
        _PlanAgentCases._cold_provider_incidental_catalog_words_do_not_hide_external_mass_concern
    )
    test_catalog_external_detail_not_grounded_in_query_stays_discoverable = (
        _PlanAgentCases._cold_catalog_external_detail_not_grounded_in_query_stays_discoverable
    )
    test_tied_registered_concern_enters_candidate_discovery = (
        _PlanAgentCases._cold_tied_registered_concern_enters_candidate_discovery
    )
    test_explicit_evidence_artifact_paths_override_shared_round_directory = (
        _PlanAgentCases._cold_explicit_evidence_artifact_paths_override_shared_round_directory
    )
    test_flat_compact_tool_value_reaches_next_planner_evidence = (
        _PlanAgentCases._cold_flat_compact_tool_value_reaches_next_planner_evidence
    )
    test_legacy_control_flag_cannot_override_query_contract = (
        _PlanAgentCases._cold_legacy_control_flag_cannot_override_query_contract
    )


class ColdCliCompatibilityTests(unittest.TestCase):
    test_legacy_factory_still_loads_compatibility_planners = (
        _CliCases._cold_legacy_factory_still_loads_compatibility_planners
    )
    test_historical_claim_first_value_normalizes_to_plan_agent = (
        _CliCases._cold_historical_claim_first_value_normalizes_to_plan_agent
    )


if __name__ == "__main__":
    unittest.main()
