"""Lazy public facade for TaskGen contracts.

The production method runtime imports concrete modules directly.  This facade
keeps historical ``from mea.taskgen import ...`` callers compatible without
eagerly importing the BBH and ClickBell dialects, SuccessSpec compiler, visual
reflection stack, and registries on every production import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    # Generic prototype contracts.
    "TaskGenError": (".prototype", "TaskGenError"),
    "TaskGenPrototype": (".prototype", "TaskGenPrototype"),
    "extract_json_response": (".prototype", "extract_json_response"),
    "extract_load_actors": (".prototype", "extract_load_actors"),
    "validate_load_actors": (".prototype", "validate_load_actors"),
    "validate_variant_spec": (".prototype", "validate_variant_spec"),
    # Visual reflection compatibility.
    "VisualReflectionError": (".reflection", "VisualReflectionError"),
    "execute_reflection_loop": (".reflection", "execute_reflection_loop"),
    "expected_color_name": (".reflection", "expected_color_name"),
    "repair_generated_method": (".reflection", "repair_generated_method"),
    "validate_bbh_distractor_vision_observation": (
        ".reflection",
        "validate_bbh_distractor_vision_observation",
    ),
    "validate_click_bell_vision_observation": (
        ".reflection",
        "validate_click_bell_vision_observation",
    ),
    "validate_vision_observation": (".reflection", "validate_vision_observation"),
    # Artifact and scene contracts.
    "SceneCheckSpecError": (".scene_checks", "SceneCheckSpecError"),
    "build_scene_check_spec": (".scene_checks", "build_scene_check_spec"),
    "validate_scene_check_spec": (".scene_checks", "validate_scene_check_spec"),
    "TaskArtifactBundleError": (".artifacts", "TaskArtifactBundleError"),
    "validate_task_artifact_bundle": (".artifacts", "validate_task_artifact_bundle"),
    "write_task_artifact_bundle": (".artifacts", "write_task_artifact_bundle"),
    "OfficialTaskRunError": (".official", "OfficialTaskRunError"),
    "create_official_task_run": (".official", "create_official_task_run"),
    # Legacy ClickBell overlay.
    "ClickBellTaskGenError": (".click_bell", "ClickBellTaskGenError"),
    "compile_click_bell_overlay": (".click_bell", "compile_click_bell_overlay"),
    "create_click_bell_variant_run": (".click_bell", "create_click_bell_variant_run"),
    "validate_click_bell_variant_hint": (
        ".click_bell",
        "validate_click_bell_variant_hint",
    ),
    # Legacy capability and SuccessSpec contracts.
    "EXPERIMENTAL_SUCCESS_PRESERVE_MARKER": (
        ".capabilities",
        "EXPERIMENTAL_SUCCESS_PRESERVE_MARKER",
    ),
    "TASK_CAPABILITIES": (".capabilities", "TASK_CAPABILITIES"),
    "CapabilityError": (".capabilities", "CapabilityError"),
    "build_variant_spec": (".capabilities", "build_variant_spec"),
    "capability_card": (".capabilities", "capability_card"),
    "get_capability": (".capabilities", "get_capability"),
    "load_legacy_variant_spec": (".capabilities", "load_legacy_variant_spec"),
    "validate_variant_spec_envelope": (
        ".capabilities",
        "validate_variant_spec_envelope",
    ),
    "DEFAULT_BBH_SUCCESS_SPEC": (".success_spec", "DEFAULT_BBH_SUCCESS_SPEC"),
    "SUCCESS_SPEC_V2_DEVELOPMENT_ENVELOPE": (
        ".success_spec",
        "SUCCESS_SPEC_V2_DEVELOPMENT_ENVELOPE",
    ),
    "SUCCESS_SPEC_V2_EXPERIMENTAL_ACT_ENVELOPE": (
        ".success_spec",
        "SUCCESS_SPEC_V2_EXPERIMENTAL_ACT_ENVELOPE",
    ),
    "SUCCESS_SPEC_V2_EXPERIMENTAL_MAX_THRESHOLD_M": (
        ".success_spec",
        "SUCCESS_SPEC_V2_EXPERIMENTAL_MAX_THRESHOLD_M",
    ),
    "SUCCESS_SPEC_V2_EXPERIMENTAL_MIN_THRESHOLD_M": (
        ".success_spec",
        "SUCCESS_SPEC_V2_EXPERIMENTAL_MIN_THRESHOLD_M",
    ),
    "SUCCESS_SPEC_V2_MAX_THRESHOLD_M": (
        ".success_spec",
        "SUCCESS_SPEC_V2_MAX_THRESHOLD_M",
    ),
    "SUCCESS_SPEC_V2_OFFICIAL_ENVELOPE": (
        ".success_spec",
        "SUCCESS_SPEC_V2_OFFICIAL_ENVELOPE",
    ),
    "SuccessSpecError": (".success_spec", "SuccessSpecError"),
    "SuccessSpecRepairError": (".success_spec", "SuccessSpecRepairError"),
    "compile_success_spec": (".success_spec", "compile_success_spec"),
    "default_bbh_success_spec": (".success_spec", "default_bbh_success_spec"),
    "default_bbh_success_spec_v2": (".success_spec", "default_bbh_success_spec_v2"),
    "development_bbh_success_spec_v2": (
        ".success_spec",
        "development_bbh_success_spec_v2",
    ),
    "experimental_bbh_success_spec_v2": (
        ".success_spec",
        "experimental_bbh_success_spec_v2",
    ),
    "repair_success_spec": (".success_spec", "repair_success_spec"),
    "success_spec_validation_report": (
        ".success_spec",
        "success_spec_validation_report",
    ),
    "validate_compiled_success_method": (
        ".success_spec",
        "validate_compiled_success_method",
    ),
    "validate_success_spec": (".success_spec", "validate_success_spec"),
    # Bounded recovery contracts used by the generic backend.
    "CandidateUnexecutableError": (".attempts", "CandidateUnexecutableError"),
    "REGENERATE_CANDIDATE": (".attempts", "REGENERATE_CANDIDATE"),
    "REPAIR_SCENE": (".attempts", "REPAIR_SCENE"),
    "REPAIR_SUCCESS_SPEC": (".attempts", "REPAIR_SUCCESS_SPEC"),
    "TASKGEN_TERMINAL": (".attempts", "TERMINAL"),
    "TaskGenerationRecoveryError": (".attempts", "TaskGenerationRecoveryError"),
    "TaskGenerationStageError": (".attempts", "TaskGenerationStageError"),
    "run_bounded_task_generation": (".attempts", "run_bounded_task_generation"),
    "task_generation_recovery_action": (
        ".attempts",
        "task_generation_recovery_action",
    ),
    # Frozen BBH/ClickBell provider dialects.
    "BBHDistractorTaskGenError": (
        ".bbh_distractor",
        "BBHDistractorTaskGenError",
    ),
    "bbh_distractor_proposal_from_task_proposal": (
        ".bbh_distractor",
        "bbh_distractor_proposal_from_task_proposal",
    ),
    "bbh_distractor_rollout_execution": (
        ".bbh_distractor",
        "bbh_distractor_rollout_execution",
    ),
    "build_bbh_distractor_module": (
        ".bbh_distractor",
        "build_bbh_distractor_module",
    ),
    "default_bbh_distractor_proposal": (
        ".bbh_distractor",
        "default_bbh_distractor_proposal",
    ),
    "materialize_bbh_distractor_candidate": (
        ".bbh_distractor",
        "materialize_bbh_distractor_candidate",
    ),
    "reference_bbh_distractor_methods": (
        ".bbh_distractor",
        "reference_bbh_distractor_methods",
    ),
    "run_bbh_distractor_checker_fixtures": (
        ".bbh_distractor",
        "run_bbh_distractor_checker_fixtures",
    ),
    "validate_bbh_distractor_manifest": (
        ".bbh_distractor",
        "validate_bbh_distractor_manifest",
    ),
    "validate_bbh_distractor_methods": (
        ".bbh_distractor",
        "validate_bbh_distractor_methods",
    ),
    "validate_bbh_distractor_proposal": (
        ".bbh_distractor",
        "validate_bbh_distractor_proposal",
    ),
    "ClickBellDistractorTaskGenError": (
        ".click_bell_distractor",
        "ClickBellDistractorTaskGenError",
    ),
    "click_bell_distractor_rollout_execution": (
        ".click_bell_distractor",
        "click_bell_distractor_rollout_execution",
    ),
    "click_bell_distractor_from_task_proposal": (
        ".click_bell_distractor",
        "click_bell_distractor_from_task_proposal",
    ),
    "materialize_click_bell_distractor_candidate": (
        ".click_bell_distractor",
        "materialize_click_bell_distractor_candidate",
    ),
}

_MODULE_EXPORTS = frozenset({"round_materialization", "visual_validation"})


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        value = import_module(f"{__name__}.{name}")
    else:
        try:
            module_name, attribute = _EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | set(_MODULE_EXPORTS))


__all__ = sorted(_EXPORTS)
