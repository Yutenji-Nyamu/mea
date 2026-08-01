"""Visual and simulator-state validation used by the TaskGen runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mea.providers import OpenAICompatibleProvider
from mea.taskgen.click_bell import (
    ClickBellTaskGenError,
    validate_click_bell_variant_hint,
)
from mea.taskgen.prototype import extract_json_response
from mea.taskgen.reflection import (
    VisualReflectionError,
    validate_bbh_distractor_vision_observation,
    validate_click_bell_vision_observation,
    validate_vision_observation,
)
from mea.taskgen.scene_checks import (
    build_scene_check_spec,
    validate_scene_check_spec,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_vision_check(
    provider: OpenAICompatibleProvider,
    run_dir: Path,
    spec: dict[str, Any],
    *,
    model: str,
    image_path: Path | None = None,
    prompt_path: Path | None = None,
    response_path: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    image_path = image_path or run_dir / "evidence/initial_head.png"
    prompt_path = prompt_path or run_dir / "validation/vision_prompt.md"
    response_path = response_path or run_dir / "validation/vision_response.txt"
    result_path = result_path or run_dir / "validation/vision.json"
    scene_check_path = run_dir / "generation/scene_check_spec.json"
    if scene_check_path.is_file():
        scene_check = validate_scene_check_spec(
            json.loads(scene_check_path.read_text(encoding="utf-8"))
        )
    else:
        scene_check = build_scene_check_spec(spec)
        _write_json(scene_check_path, scene_check)
    scene_check_text = json.dumps(scene_check, ensure_ascii=False, indent=2)
    is_provider_distractor = (
        scene_check.get("success_semantics") == "provider_generated_python"
    )
    if is_provider_distractor:
        task_proposal_path = run_dir / "generation/task_proposal.json"
        if not task_proposal_path.is_file():
            raise VisualReflectionError(
                "provider distractor visual check requires TaskProposal"
            )
        task_label = str(spec.get("task_name"))
        target_label = "bell" if task_label == "click_bell" else "block"
        task_proposal_text = task_proposal_path.read_text(encoding="utf-8")
        prompt = f"""This is the initial rendered frame of a proposal-derived
RoboTwin {task_label} task. The proposal intentionally adds a lookalike physical
distractor beside the target.

TASK PROPOSAL:
{task_proposal_text}

SCENE CHECK SPEC:
{scene_check_text}

Check only proposal-derived visual facts: the target and lookalike distractor
are separately visible, and the initial scene is physically plausible.
Do not infer exact actor identity, exact offset, contact latches, or success
from RGB.

Return JSON only:
{{
  "aligned": true,
  "target_actor": "{target_label}",
  "target_visible": true,
  "lookalike_distractor_visible": true,
  "scene_physically_plausible": true,
  "unexpected_changes": [],
  "diagnosis": "Whether both intended objects are visible and plausible.",
  "suggestions": [],
  "confidence": 0.8
}}
"""
    elif spec.get("task_name") == "click_bell":
        bell_change = spec["changes"].get("bell")
        randomization_change = spec["changes"].get("domain_randomization") or {}
        clutter_change = (
            randomization_change
            if "cluttered_table" in randomization_change
            else None
        )
        background_change = (
            randomization_change
            if "random_background" in randomization_change
            else None
        )
        lighting_change = (
            randomization_change if "random_light" in randomization_change else None
        )
        if clutter_change is not None:
            contract_description = (
                "This round intentionally enables RoboTwin's simulator-native "
                "cluttered_table with clean_background_rate=0. The extra tabletop "
                "objects are expected and must not be reported as an unexpected "
                "change. Check that the target bell remains visible and the "
                "physical scene is plausible; exact clutter count is checked from "
                "simulator task state."
            )
        elif background_change is not None:
            contract_description = (
                "This round enables RoboTwin's simulator-native random_background "
                "with clean_background_rate=0 and eval_mode=true. Both table and "
                "wall therefore use the upstream unseen texture split. Check only "
                "that the bell remains visible and the rendered scene is plausible; "
                "exact texture ids and split are checked from simulator task info."
            )
        elif lighting_change is not None:
            contract_description = (
                "This round enables RoboTwin's simulator-native random_light with "
                "crazy_random_light_rate=0, so point and directional light colors "
                "are randomized once at setup without per-frame flicker. Check only "
                "that the bell remains visible and illumination is usable; the "
                "configuration branch is checked from simulator task attributes."
            )
        elif bell_change and bell_change.get("instance_mode") == "fixed":
            bell_id = int(bell_change["bell_id"])
            visual_description = (
                "白色 dome、黑色底座、较大实例" if bell_id == 0 else "蓝色 dome、棕色底座、较小实例"
            )
            contract_description = (
                f"本轮固定官方 bell base{bell_id}（{visual_description}），位置保持官方随机。"
                "精确 bell_id 已由 simulator task attribute 检查负责。"
            )
        elif bell_change:
            expected_xy = bell_change["xy"]
            contract_description = (
                f"本轮固定 workspace xy={expected_xy}，bell 实例保持官方随机。"
                "精确 XY 已由 simulator tracked actor 检查负责。"
            )
        prompt = f"""这是 RoboTwin click_bell 受限单轴变式的初始场景首帧。
请只检查目标 bell 是否清晰可见、场景是否物理合理、是否存在明显多余或缺失物体。
{contract_description}
不能仅凭 RGB 宣称精确坐标或实例 ID 是否正确。

PROPOSAL-DERIVED SCENE CHECK SPEC:
{scene_check_text}

只输出 JSON：
{{
  "aligned": true,
  "target_actor": "bell",
  "bell_visible": true,
  "unexpected_changes": [],
  "diagnosis": "目标铃是否可见以及场景是否存在明显异常",
  "suggestions": [],
  "confidence": 0.8
}}
"""
    else:
        expected_half_size = 0.025 * float(spec["changes"]["block"]["scale"])
        prompt = f"""这是 RoboTwin beat_block_hammer 的初始场景首帧。
请检查被锤子敲击的方块是否符合下面的 VariantSpec，并检查场景是否有明显异常：
{json.dumps(spec, ensure_ascii=False, indent=2)}

PROPOSAL-DERIVED SCENE CHECK SPEC:
{scene_check_text}

If position_mode or yaw_mode is official_random, one sampled RGB frame cannot
prove that the distribution or strike-path alignment is wrong. Simulator state,
the rule gate, and the expert gate own that judgment. Do not report a sampled
legal pose as an unexpected position change unless the actor is visibly missing,
overlapping another object, or outside the workspace.

官方 scale=1.0 的方块 half_size 是 (0.025, 0.025, 0.025) 米；本次预期
half_size 是 ({expected_half_size:.6f}, {expected_half_size:.6f}, {expected_half_size:.6f}) 米。
请结合方块与锤子的相对尺寸判断是否明显偏大或偏小。

只输出 JSON：
{{
  "aligned": true,
  "target_actor": "block",
  "observed_color": "blue",
  "unexpected_changes": [],
  "diagnosis": "场景与需求是否一致，以及不一致的具体原因",
  "suggestions": ["若不一致，给出只修改 load_actors() 的具体建议"],
  "confidence": 0.8
}}
"""
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    response = provider.vision(
        prompt,
        image_path,
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    response_path.write_text(response + "\n", encoding="utf-8")
    parsed = extract_json_response(response)
    result = (
        validate_bbh_distractor_vision_observation(parsed)
        if is_provider_distractor
        else validate_click_bell_vision_observation(parsed)
        if spec.get("task_name") == "click_bell"
        else validate_vision_observation(parsed, spec)
    )
    result["provider_metadata"] = dict(provider.last_metadata)
    _write_json(result_path, result)
    return result


def validate_click_bell_scene_contract(
    scene: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Validate the controlled axis from simulator state, never from RGB."""

    if not isinstance(spec, dict) or spec.get("task_name") != "click_bell":
        raise ClickBellTaskGenError("scene contract requires a click_bell variant spec")
    normalized = validate_click_bell_variant_hint(spec.get("changes"))
    bell_change = normalized.get("bell")
    randomization_change = normalized.get("domain_randomization") or {}
    clutter_change = (
        randomization_change if "cluttered_table" in randomization_change else None
    )
    background_change = (
        randomization_change if "random_background" in randomization_change else None
    )
    lighting_change = (
        randomization_change if "random_light" in randomization_change else None
    )
    expected_axis = (
        "robustness.scene_clutter"
        if clutter_change is not None
        else "scene_background_texture"
        if background_change is not None
        else "scene_lighting"
        if lighting_change is not None
        else "object_instance"
        if bell_change and bell_change.get("instance_mode") == "fixed"
        else "object_position"
    )
    declared_axis = spec.get("controlled_axis")
    if declared_axis is not None and declared_axis != expected_axis:
        raise ClickBellTaskGenError(
            "variant spec controlled_axis does not match its strict bell contract"
        )
    bell = next(
        (
            actor
            for actor in scene.get("tracked_actors", [])
            if actor.get("id") == "bell"
        ),
        None,
    )
    actual_xy = (
        [float(value) for value in bell.get("position", [])[:2]]
        if isinstance(bell, dict)
        else []
    )
    if bell_change and bell_change.get("position_mode") == "fixed":
        expected_xy = [float(value) for value in bell_change["xy"]]
        position_passed = len(actual_xy) == 2 and all(
            abs(left - right) <= 1e-6 for left, right in zip(actual_xy, expected_xy)
        )
        position = {
            "status": "passed" if position_passed else "failed",
            "passed": position_passed,
            "expected_xy": expected_xy,
            "actual_xy": actual_xy,
            "tolerance": 1e-6,
            "authority": "simulator_tracked_actor_xy",
        }
    else:
        position = {
            "status": "not_applicable",
            "passed": True,
            "expected_xy": None,
            "actual_xy": actual_xy,
            "tolerance": None,
            "authority": "simulator_tracked_actor_xy",
        }

    if bell_change and bell_change.get("instance_mode") == "fixed":
        expected_bell_id = int(bell_change["bell_id"])
        actual_bell_id = (scene.get("task_attributes") or {}).get("bell_id")
        instance_passed = (
            not isinstance(actual_bell_id, bool)
            and isinstance(actual_bell_id, int)
            and actual_bell_id == expected_bell_id
        )
        instance = {
            "status": "passed" if instance_passed else "failed",
            "passed": instance_passed,
            "expected_bell_id": expected_bell_id,
            "actual_bell_id": actual_bell_id,
            "authority": "simulator_task_attribute:bell_id",
        }
    else:
        instance = {
            "status": "not_applicable",
            "passed": True,
            "expected_bell_id": None,
            "actual_bell_id": (scene.get("task_attributes") or {}).get("bell_id"),
            "authority": "simulator_task_attribute:bell_id",
        }

    randomization = scene.get("domain_randomization") or {}
    actual_clutter_enabled = randomization.get("cluttered_table")
    actual_objects = randomization.get("cluttered_objects")
    if not isinstance(actual_objects, list):
        actual_objects = []
    actual_count = randomization.get("cluttered_object_count")
    if isinstance(actual_count, bool) or not isinstance(actual_count, int):
        actual_count = len(actual_objects)
    if clutter_change is not None:
        clutter_passed = bool(
            actual_clutter_enabled is True
            and randomization.get("clean_background_rate") == 0.0
            and actual_count >= 1
        )
        clutter = {
            "status": "passed" if clutter_passed else "failed",
            "passed": clutter_passed,
            "expected_enabled": True,
            "expected_clean_background_rate": 0.0,
            "minimum_object_count": 1,
            "actual_enabled": actual_clutter_enabled,
            "actual_clean_background_rate": randomization.get("clean_background_rate"),
            "actual_count": actual_count,
            "actual_objects": actual_objects,
            "authority": "simulator_task_info:cluttered_table_info",
        }
    else:
        clutter = {
            "status": "not_applicable",
            "passed": True,
            "expected_enabled": None,
            "expected_clean_background_rate": None,
            "minimum_object_count": 0,
            "actual_enabled": actual_clutter_enabled,
            "actual_clean_background_rate": randomization.get("clean_background_rate"),
            "actual_count": actual_count,
            "actual_objects": actual_objects,
            "authority": "simulator_task_info:cluttered_table_info",
        }

    actual_random_background = randomization.get("random_background")
    actual_wall_texture = randomization.get("wall_texture")
    actual_table_texture = randomization.get("table_texture")
    actual_texture_split = randomization.get("texture_split")
    if background_change is not None:
        background_passed = bool(
            scene.get("eval_mode") is True
            and actual_random_background is True
            and randomization.get("clean_background_rate") == 0.0
            and isinstance(actual_wall_texture, str)
            and actual_wall_texture.startswith("unseen/")
            and isinstance(actual_table_texture, str)
            and actual_table_texture.startswith("unseen/")
            and actual_texture_split == "unseen"
            and randomization.get("background_authority")
            == "simulator_task_info:texture_info"
        )
        background_texture = {
            "status": "passed" if background_passed else "failed",
            "passed": background_passed,
            "expected_random_background": True,
            "expected_clean_background_rate": 0.0,
            "expected_eval_mode": True,
            "expected_split": "unseen",
            "actual_random_background": actual_random_background,
            "actual_clean_background_rate": randomization.get(
                "clean_background_rate"
            ),
            "actual_eval_mode": scene.get("eval_mode"),
            "actual_split": actual_texture_split,
            "actual_wall_texture": actual_wall_texture,
            "actual_table_texture": actual_table_texture,
            "authority": "simulator_task_info:texture_info",
        }
    else:
        background_texture = {
            "status": "not_applicable",
            "passed": True,
            "expected_random_background": None,
            "expected_clean_background_rate": None,
            "expected_eval_mode": None,
            "expected_split": None,
            "actual_random_background": actual_random_background,
            "actual_clean_background_rate": randomization.get(
                "clean_background_rate"
            ),
            "actual_eval_mode": scene.get("eval_mode"),
            "actual_split": actual_texture_split,
            "actual_wall_texture": actual_wall_texture,
            "actual_table_texture": actual_table_texture,
            "authority": "simulator_task_info:texture_info",
        }

    actual_random_light = randomization.get("random_light")
    actual_crazy_rate = randomization.get("crazy_random_light_rate")
    actual_crazy_light = randomization.get("crazy_random_light")
    direction_light_count = randomization.get("direction_light_count")
    point_light_count = randomization.get("point_light_count")
    direction_light_colors = randomization.get("direction_light_colors")
    point_light_colors = randomization.get("point_light_colors")

    def valid_light_colors(colors: Any, count: Any) -> bool:
        return bool(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 1
            and isinstance(colors, list)
            and len(colors) == count
            and all(
                isinstance(color, list)
                and len(color) == 3
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in color
                )
                for color in colors
            )
        )

    if lighting_change is not None:
        lighting_passed = bool(
            actual_random_light is True
            and actual_crazy_rate == 0.0
            and actual_crazy_light is False
            and valid_light_colors(direction_light_colors, direction_light_count)
            and valid_light_colors(point_light_colors, point_light_count)
            and randomization.get("lighting_authority")
            == (
                "simulator_task_attributes:random_light,crazy_random_light_rate,"
                "crazy_random_light;simulator_light_components:get_color"
            )
        )
        lighting = {
            "status": "passed" if lighting_passed else "failed",
            "passed": lighting_passed,
            "expected_random_light": True,
            "expected_crazy_random_light_rate": 0.0,
            "expected_temporal_flicker": False,
            "actual_random_light": actual_random_light,
            "actual_crazy_random_light_rate": actual_crazy_rate,
            "actual_crazy_random_light": actual_crazy_light,
            "direction_light_count": direction_light_count,
            "point_light_count": point_light_count,
            "direction_light_colors": direction_light_colors,
            "point_light_colors": point_light_colors,
            "authority": randomization.get("lighting_authority"),
        }
    else:
        lighting = {
            "status": "not_applicable",
            "passed": True,
            "expected_random_light": None,
            "expected_crazy_random_light_rate": None,
            "expected_temporal_flicker": None,
            "actual_random_light": actual_random_light,
            "actual_crazy_random_light_rate": actual_crazy_rate,
            "actual_crazy_random_light": actual_crazy_light,
            "direction_light_count": direction_light_count,
            "point_light_count": point_light_count,
            "direction_light_colors": direction_light_colors,
            "point_light_colors": point_light_colors,
            "authority": randomization.get("lighting_authority"),
        }

    passed = bool(
        position["passed"]
        and instance["passed"]
        and clutter["passed"]
        and background_texture["passed"]
        and lighting["passed"]
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "actor_id": "bell",
        "controlled_axis": expected_axis,
        "position": position,
        "instance": instance,
        "clutter": clutter,
        "background_texture": background_texture,
        "lighting": lighting,
        "authorities": [
            position["authority"],
            instance["authority"],
            clutter["authority"],
            background_texture["authority"],
            lighting["authority"],
        ],
    }
