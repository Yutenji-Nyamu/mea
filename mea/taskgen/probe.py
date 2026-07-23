"""Setup-only render and optional expert-solvability probe for TaskGen output."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import traceback
import math
from pathlib import Path
from typing import Any

import yaml

from mea.execution_receipt import (
    load_execution_receipt,
    validate_execution_invocation,
    validate_frozen_candidate_source,
    validate_imported_task_binding,
)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def light_component_colors(lights: Any) -> list[list[float] | None]:
    """Read RGB values from SAPIEN light components for simulator-side evidence."""

    colors: list[list[float] | None] = []
    for light in lights if isinstance(lights, (list, tuple)) else []:
        try:
            getter = getattr(light, "get_color", None)
            color = getter() if callable(getter) else getattr(light, "color", None)
            values = [float(value) for value in color]
            colors.append(values if len(values) == 3 else None)
        except (TypeError, ValueError, AttributeError):
            colors.append(None)
    return colors


def load_task_args(
    repo_root: Path,
    *,
    task_name: str,
    task_config: str,
    ckpt_setting: str,
    overlay_path: Path | None,
    eval_mode: bool = False,
) -> dict[str, Any]:
    # Simulator dependencies are needed only when a probe actually executes.
    from envs import CONFIGS_PATH

    with (repo_root / "task_config" / f"{task_config}.yml").open(
        "r", encoding="utf-8"
    ) as handle:
        args = yaml.safe_load(handle) or {}

    if overlay_path:
        with overlay_path.open("r", encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        if not isinstance(overlay, dict):
            raise ValueError("overlay 必须是 YAML mapping")
        deep_update(args, overlay)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["policy_name"] = "ACT"
    args["eval_mode"] = bool(eval_mode)
    args["eval_video_save_dir"] = None

    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8"
    ) as handle:
        embodiment_types = yaml.safe_load(handle)

    embodiment = args["embodiment"]

    def embodiment_file(name: str) -> str:
        path = embodiment_types[name]["file_path"]
        if path is None:
            raise ValueError(f"embodiment 没有 file_path: {name}")
        return path

    if len(embodiment) == 1:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment 必须包含 1 或 3 项")

    def read_embodiment(path: str) -> dict[str, Any]:
        with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    args["left_embodiment_config"] = read_embodiment(args["left_robot_file"])
    args["right_embodiment_config"] = read_embodiment(args["right_robot_file"])
    return args


def actor_summary(task: Any) -> list[dict[str, Any]]:
    actors = []
    for actor in task.scene.get_all_actors():
        pose = actor.get_pose()
        actors.append(
            {
                "name": actor.get_name(),
                "position": [float(value) for value in pose.p],
                "quaternion": [float(value) for value in pose.q],
            }
        )
    return actors


def _pose_summary(pose: Any) -> dict[str, list[float]]:
    return {
        "position": [float(value) for value in pose.p],
        "quaternion": [float(value) for value in pose.q],
    }


def tracked_actor_summary(
    task: Any,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read only actors and semantic points declared by the TaskSchema."""

    summaries: list[dict[str, Any]] = []
    for actor_spec in schema["tracked_actors"]:
        actor = getattr(task, actor_spec["task_attribute"])
        summary: dict[str, Any] = {
            "id": actor_spec["id"],
            "task_attribute": actor_spec["task_attribute"],
            "scene_name": actor_spec["scene_name"],
            **_pose_summary(actor.get_pose()),
            "functional_points": {},
            "contact_points": {},
        }
        for point_id in actor_spec.get("functional_points", []):
            point = actor.get_functional_point(point_id, "pose")
            summary["functional_points"][str(point_id)] = _pose_summary(point)
        for point_id in actor_spec.get("contact_points", []):
            point = [float(value) for value in actor.get_contact_point(point_id)]
            summary["contact_points"][str(point_id)] = {
                "position": point[:3],
                "raw": point,
            }
        summaries.append(summary)
    return summaries


def task_attribute_summary(task: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """Snapshot trusted scalar task attributes declared by TaskSchema."""

    result: dict[str, Any] = {}
    for attribute in schema.get("probe_task_attributes", []):
        if not hasattr(task, attribute):
            raise AttributeError(
                f"declared probe task attribute is missing: {attribute}"
            )
        value = getattr(task, attribute)
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise TypeError(f"probe task attribute must be a JSON scalar: {attribute}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"probe task attribute must be finite: {attribute}")
        result[attribute] = value
    return result


def task_schema_rule_check(
    task: Any,
    schema: dict[str, Any],
    *,
    scene_actors: list[dict[str, Any]],
    tracked_actors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Setup-only structural checks shared by every schema-backed task."""

    scene_names = {actor["name"] for actor in scene_actors}
    declared_scene_names = {actor["scene_name"] for actor in schema["tracked_actors"]}
    numeric_values: list[float] = []
    for actor in tracked_actors:
        numeric_values.extend(actor["position"])
        numeric_values.extend(actor["quaternion"])
        for point in actor["functional_points"].values():
            numeric_values.extend(point["position"])
            numeric_values.extend(point["quaternion"])
        for point in actor["contact_points"].values():
            numeric_values.extend(point["raw"])
    checks = {
        "all_tracked_actor_attributes_present": all(
            hasattr(task, actor["task_attribute"]) for actor in schema["tracked_actors"]
        ),
        "declared_scene_names_present": declared_scene_names.issubset(scene_names),
        "finite_tracked_actor_state": bool(numeric_values)
        and all(math.isfinite(value) and abs(value) < 100 for value in numeric_values),
        "official_check_success_callable": callable(
            getattr(task, "check_success", None)
        ),
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "tracked_actor_ids": [actor["id"] for actor in tracked_actors],
        "declared_scene_names": sorted(declared_scene_names),
    }


def run_probe(arguments: argparse.Namespace) -> dict[str, Any]:
    repo_root = arguments.repo_root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    image = arguments.image.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "task_name": arguments.task_name,
        "task_module": arguments.task_module,
        "seed": arguments.seed,
        "setup_success": False,
        "render_success": False,
        "expert_requested": arguments.expert,
        "telemetry_requested": arguments.telemetry_dir is not None,
        "eval_mode": bool(getattr(arguments, "eval_mode", False)),
        "execution_receipt_requested": (
            getattr(arguments, "execution_receipt", None) is not None
        ),
    }
    task = None
    recorder = None
    recorder_started = False
    try:
        from mea.toolkit import load_task_schema

        schema = load_task_schema(repo_root, arguments.task_name)
        result["task_schema"] = {
            "schema_version": schema["schema_version"],
            "task_name": schema["task_name"],
            "task_family": schema.get("task_family"),
            "trusted_tool_profile": schema.get("trusted_tool_profile"),
        }
        args = load_task_args(
            repo_root,
            task_name=arguments.task_name,
            task_config=arguments.task_config,
            ckpt_setting=arguments.ckpt_setting,
            overlay_path=arguments.overlay,
            eval_mode=bool(getattr(arguments, "eval_mode", False)),
        )
        execution_receipt_path = getattr(
            arguments,
            "execution_receipt",
            None,
        )
        execution_receipt = (
            load_execution_receipt(
                execution_receipt_path,
                verify_checkpoint_files=True,
            )
            if execution_receipt_path is not None
            else None
        )
        if execution_receipt is not None:
            if arguments.telemetry_dir is None:
                raise ValueError(
                    "execution_receipt requires telemetry_dir"
                )
            receipt_policy = (
                "expert" if arguments.expert else "setup_probe"
            )
            validate_execution_invocation(
                execution_receipt,
                task_name=arguments.task_name,
                task_module=arguments.task_module,
                task_config=arguments.task_config,
                checkpoint_setting=arguments.ckpt_setting,
                policy_name=receipt_policy,
                seed=arguments.seed,
                episode_index=arguments.episode_index,
                checkpoint_dir=None,
                verify_checkpoint_files=True,
            )
            validate_frozen_candidate_source(execution_receipt)
        module = importlib.import_module(arguments.task_module)
        task_class = getattr(module, arguments.task_name)
        task = task_class()
        if execution_receipt is not None:
            validate_imported_task_binding(execution_receipt, task)
        task.setup_demo(now_ep_num=0, seed=arguments.seed, is_test=True, **args)
        result["setup_success"] = True
        task_info = getattr(task, "info", {}) or {}
        cluttered_objects = task_info.get("cluttered_table_info", [])
        if not isinstance(cluttered_objects, list):
            cluttered_objects = []
        texture_info = task_info.get("texture_info", {})
        if not isinstance(texture_info, dict):
            texture_info = {}
        wall_texture = texture_info.get("wall_texture")
        table_texture = texture_info.get("table_texture")
        texture_prefixes = {
            value.split("/", 1)[0]
            for value in (wall_texture, table_texture)
            if isinstance(value, str) and "/" in value
        }
        texture_split = (
            next(iter(texture_prefixes)) if len(texture_prefixes) == 1 else None
        )
        direction_lights = getattr(task, "direction_light_lst", [])
        point_lights = getattr(task, "point_light_lst", [])
        direction_light_colors = light_component_colors(direction_lights)
        point_light_colors = light_component_colors(point_lights)
        result["domain_randomization"] = {
            "cluttered_table": bool(getattr(task, "cluttered_table", False)),
            "clean_background_rate": float(getattr(task, "clean_background_rate", 1.0)),
            "cluttered_object_count": len(cluttered_objects),
            "cluttered_objects": cluttered_objects,
            "authority": "simulator_task_info:cluttered_table_info",
            "random_background": bool(getattr(task, "random_background", False)),
            "wall_texture": wall_texture,
            "table_texture": table_texture,
            "texture_split": texture_split,
            "background_authority": "simulator_task_info:texture_info",
            "random_light": bool(getattr(task, "random_light", False)),
            "crazy_random_light_rate": float(
                getattr(task, "crazy_random_light_rate", 0.0)
            ),
            "crazy_random_light": bool(
                getattr(task, "crazy_random_light", False)
            ),
            "direction_light_count": len(direction_lights),
            "point_light_count": len(point_lights),
            "direction_light_colors": direction_light_colors,
            "point_light_colors": point_light_colors,
            "lighting_authority": (
                "simulator_task_attributes:random_light,crazy_random_light_rate,"
                "crazy_random_light;simulator_light_components:get_color"
            ),
        }
        result["actors"] = actor_summary(task)
        result["tracked_actors"] = tracked_actor_summary(task, schema)
        result["task_attributes"] = task_attribute_summary(task, schema)
        tracked_by_id = {actor["id"]: actor for actor in result["tracked_actors"]}
        # Preserve the original BBH probe contract for existing reports/tests.
        if "block" in tracked_by_id:
            result["block_pose"] = {
                key: tracked_by_id["block"][key] for key in ("position", "quaternion")
            }
        if "hammer" in tracked_by_id:
            result["hammer_pose"] = {
                key: tracked_by_id["hammer"][key] for key in ("position", "quaternion")
            }
        task.save_camera_rgb(str(image), camera_name="head_camera")
        result["render_success"] = image.is_file() and image.stat().st_size > 0
        result["image"] = str(image)

        result["rule_check"] = task_schema_rule_check(
            task,
            schema,
            scene_actors=result["actors"],
            tracked_actors=result["tracked_actors"],
        )
        if arguments.task_name == "beat_block_hammer":
            actor_names = {actor["name"] for actor in result["actors"]}
            result["rule_check"].update(
                {
                    "has_hammer": "020_hammer" in actor_names,
                    "has_block": "box" in actor_names,
                    "finite_block_pose": all(
                        abs(value) < 100 for value in result["block_pose"]["position"]
                    ),
                }
            )
            result["rule_check"]["passed"] = all(
                result["rule_check"][name]
                for name in (
                    "all_tracked_actor_attributes_present",
                    "declared_scene_names_present",
                    "finite_tracked_actor_state",
                    "official_check_success_callable",
                    "has_hammer",
                    "has_block",
                    "finite_block_pose",
                )
            )

        if arguments.telemetry_dir is not None:
            from mea.toolkit import EpisodeRecorder

            telemetry_dir = arguments.telemetry_dir.expanduser().resolve()
            recorder = EpisodeRecorder(
                repo_root,
                telemetry_dir,
                task_name=arguments.task_name,
                seed=arguments.seed,
                episode_index=arguments.episode_index,
                policy_name="expert" if arguments.expert else "setup_probe",
                task_module=arguments.task_module,
                task_config=arguments.task_config,
                checkpoint_setting=arguments.ckpt_setting,
                telemetry_profile_id=arguments.telemetry_profile,
                visual_capture_profile_id=getattr(
                    arguments, "visual_capture_profile", None
                ),
                execution_receipt=execution_receipt,
            )
            task._mea_recorder = recorder
            try:
                recorder.start(task)
            except Exception:
                task._mea_recorder = None
                raise
            recorder_started = True

        if arguments.expert:
            if recorder is not None:
                recorder.on_policy_action_start(
                    task,
                    action=[],
                    action_type="expert_plan",
                )
            task.play_once()
            result["expert"] = {
                "plan_success": bool(task.plan_success),
                "check_success": bool(task.check_success()),
            }
            result["expert"]["passed"] = all(result["expert"].values())
            if recorder is not None:
                recorder.on_policy_action_end(
                    task,
                    success=bool(result["expert"]["passed"]),
                )

        if recorder is not None:
            task._mea_recorder = None
            telemetry_success = (
                bool(result.get("expert", {}).get("passed"))
                if arguments.expert
                else bool(result["rule_check"]["passed"])
            )
            metadata = recorder.finish(task, success=telemetry_success)
            recorder_started = False
            result["telemetry"] = {
                "episode_dir": str(recorder.output_dir),
                "metadata": metadata,
            }
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if recorder is not None and recorder_started and not recorder.finished:
            recorder.record_error(exc)
            if task is not None:
                task._mea_recorder = None
                try:
                    metadata = recorder.finish(
                        task,
                        success=False,
                        error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                    result["telemetry"] = {
                        "episode_dir": str(recorder.output_dir),
                        "metadata": metadata,
                    }
                except Exception as recorder_exc:
                    result["telemetry_error"] = {
                        "type": type(recorder_exc).__name__,
                        "message": str(recorder_exc),
                    }
    finally:
        if task is not None:
            task._mea_recorder = None
            try:
                task.close_env(clear_cache=True)
            except Exception as close_exc:
                result["close_error"] = str(close_exc)

    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-module", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--ckpt-setting", default="demo_clean")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", action="store_true")
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Use the evaluator distribution (including unseen randomization).",
    )
    parser.add_argument("--telemetry-dir", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument(
        "--visual-capture-profile",
        choices=["event_keyframes_v1"],
    )
    return parser.parse_args()


def main() -> None:
    result = run_probe(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not (
        result.get("setup_success")
        and result.get("render_success")
        and result.get("rule_check", {}).get("passed")
    ):
        raise SystemExit(1)
    if result.get("expert_requested") and not result.get("expert", {}).get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
