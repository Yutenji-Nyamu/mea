"""Task generation prototype for promptable RoboTwin evaluation."""

from .prototype import (
    TaskGenError,
    TaskGenPrototype,
    extract_json_response,
    extract_load_actors,
    validate_load_actors,
    validate_variant_spec,
)
from .reflection import (
    VisualReflectionError,
    execute_reflection_loop,
    expected_color_name,
    inject_oversized_block_fixture,
    inject_wrong_color_fixture,
    repair_generated_method,
    validate_click_bell_vision_observation,
    validate_vision_observation,
)
from .scene_checks import (
    SceneCheckSpecError,
    build_scene_check_spec,
    validate_scene_check_spec,
)
from .artifacts import (
    TaskArtifactBundleError,
    validate_task_artifact_bundle,
    write_task_artifact_bundle,
)
from .official import (
    OfficialTaskRunError,
    create_official_task_run,
)
from .click_bell import (
    ClickBellTaskGenError,
    compile_click_bell_overlay,
    create_click_bell_variant_run,
    validate_click_bell_variant_hint,
)
from .capabilities import (
    TASK_CAPABILITIES,
    CapabilityError,
    build_variant_spec,
    capability_card,
    get_capability,
    load_legacy_variant_spec,
    validate_variant_spec_envelope,
)
from .acceptance import (
    DEFAULT_ACCEPTANCE_RUNS,
    TaskGenAcceptanceError,
    build_cached_taskgen_acceptance,
)
from .success_spec import (
    DEFAULT_BBH_SUCCESS_SPEC,
    SuccessSpecError,
    SuccessSpecRepairError,
    compile_success_spec,
    default_bbh_success_spec,
    repair_success_spec,
    validate_compiled_success_method,
    validate_success_spec,
)

__all__ = [
    "TaskGenError",
    "TaskGenPrototype",
    "extract_json_response",
    "extract_load_actors",
    "validate_load_actors",
    "validate_variant_spec",
    "VisualReflectionError",
    "execute_reflection_loop",
    "expected_color_name",
    "inject_oversized_block_fixture",
    "inject_wrong_color_fixture",
    "repair_generated_method",
    "validate_click_bell_vision_observation",
    "validate_vision_observation",
    "SceneCheckSpecError",
    "build_scene_check_spec",
    "validate_scene_check_spec",
    "TaskArtifactBundleError",
    "validate_task_artifact_bundle",
    "write_task_artifact_bundle",
    "OfficialTaskRunError",
    "create_official_task_run",
    "ClickBellTaskGenError",
    "compile_click_bell_overlay",
    "create_click_bell_variant_run",
    "validate_click_bell_variant_hint",
    "TASK_CAPABILITIES",
    "CapabilityError",
    "build_variant_spec",
    "capability_card",
    "get_capability",
    "load_legacy_variant_spec",
    "validate_variant_spec_envelope",
    "DEFAULT_ACCEPTANCE_RUNS",
    "TaskGenAcceptanceError",
    "build_cached_taskgen_acceptance",
    "DEFAULT_BBH_SUCCESS_SPEC",
    "SuccessSpecError",
    "SuccessSpecRepairError",
    "compile_success_spec",
    "default_bbh_success_spec",
    "repair_success_spec",
    "validate_compiled_success_method",
    "validate_success_spec",
]
