"""Setup-only render and optional expert-solvability probe for TaskGen output."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import yaml

from envs import CONFIGS_PATH


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_task_args(
    repo_root: Path,
    *,
    task_name: str,
    task_config: str,
    ckpt_setting: str,
    overlay_path: Path | None,
) -> dict[str, Any]:
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
    args["eval_mode"] = False
    args["eval_video_save_dir"] = None

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as handle:
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
    }
    task = None
    try:
        args = load_task_args(
            repo_root,
            task_name=arguments.task_name,
            task_config=arguments.task_config,
            ckpt_setting=arguments.ckpt_setting,
            overlay_path=arguments.overlay,
        )
        module = importlib.import_module(arguments.task_module)
        task_class = getattr(module, arguments.task_name)
        task = task_class()
        task.setup_demo(now_ep_num=0, seed=arguments.seed, is_test=True, **args)
        result["setup_success"] = True
        result["actors"] = actor_summary(task)
        result["block_pose"] = {
            "position": [float(value) for value in task.block.get_pose().p],
            "quaternion": [float(value) for value in task.block.get_pose().q],
        }
        result["hammer_pose"] = {
            "position": [float(value) for value in task.hammer.get_pose().p],
            "quaternion": [float(value) for value in task.hammer.get_pose().q],
        }
        task.save_camera_rgb(str(image), camera_name="head_camera")
        result["render_success"] = image.is_file() and image.stat().st_size > 0
        result["image"] = str(image)

        actor_names = {actor["name"] for actor in result["actors"]}
        result["rule_check"] = {
            "has_hammer": "020_hammer" in actor_names,
            "has_block": "box" in actor_names,
            "finite_block_pose": all(
                abs(value) < 100 for value in result["block_pose"]["position"]
            ),
        }
        result["rule_check"]["passed"] = all(result["rule_check"].values())

        if arguments.expert:
            task.play_once()
            result["expert"] = {
                "plan_success": bool(task.plan_success),
                "check_success": bool(task.check_success()),
            }
            result["expert"]["passed"] = all(result["expert"].values())
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if task is not None:
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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", action="store_true")
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
