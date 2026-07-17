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
]
