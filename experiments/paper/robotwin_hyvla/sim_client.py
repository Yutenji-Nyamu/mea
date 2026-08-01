from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import yaml

from envs import CONFIGS_PATH
from transport import request


def _demo_clean_args(task_name: str) -> dict:
    with open(os.path.join(CONFIGS_PATH, "demo_clean.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8"
    ) as f:
        embodiments = yaml.safe_load(f)
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) != 1:
        raise RuntimeError(f"expected one dual-arm embodiment, got {embodiment}")
    robot_file = embodiments[embodiment[0]]["file_path"]
    with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as f:
        robot_config = yaml.safe_load(f)
    args.update(
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
    )
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        cameras = yaml.safe_load(f)
    head_type = args["camera"]["head_camera_type"]
    args.update(
        head_camera_h=cameras[head_type]["h"],
        head_camera_w=cameras[head_type]["w"],
        task_name=task_name,
        task_config="demo_clean",
        render_freq=0,
        eval_mode=True,
        save_data=False,
        eval_video_save_dir=None,
    )
    return args


def _request_payload(observation: dict, instruction: str) -> dict:
    camera_observation = observation["observation"]
    images = {
        name: np.ascontiguousarray(camera_observation[name]["rgb"], dtype=np.uint8)
        for name in ("head_camera", "left_camera", "right_camera")
    }
    endpose = observation["endpose"]
    return {
        "command": "act",
        "images": {
            key: {"shape": list(value.shape), "data": value.tobytes()}
            for key, value in images.items()
        },
        "endpose": {
            "left_endpose": np.asarray(endpose["left_endpose"]).tolist(),
            "left_gripper": float(endpose["left_gripper"]),
            "right_endpose": np.asarray(endpose["right_endpose"]).tolist(),
            "right_gripper": float(endpose["right_gripper"]),
        },
        "task": instruction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic RoboTwin client for the isolated Hy-VLA server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18781)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_module = importlib.import_module(f"envs.{args.task}")
    task_class = getattr(task_module, args.task)
    task = task_class()
    started = time.perf_counter()
    action_latencies: list[float] = []
    network_forwards = 0
    actions_executed = 0
    client = socket.create_connection((args.host, args.port), timeout=30)
    client.settimeout(180)

    try:
        request(client, {"command": "reset"})
        task.setup_demo(
            now_ep_num=0,
            seed=args.seed,
            is_test=True,
            **_demo_clean_args(args.task),
        )
        observation = task.get_obs()
        initial_head = observation["observation"]["head_camera"]["rgb"]
        iio.imwrite(args.output_dir / "initial_head.png", initial_head)

        while actions_executed < int(task.step_lim) and not task.eval_success:
            response = request(
                client,
                _request_payload(observation, str(task.get_instruction())),
            )
            action = np.asarray(response["action"], dtype=np.float32)
            if action.shape != (16,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid server action: {action}")
            task.take_action(action, action_type="ee")
            actions_executed += 1
            action_latencies.append(float(response["latency_seconds"]))
            network_forwards += int(response["network_forward"])
            observation = task.get_obs()

        official_checker = bool(task.check_success())
        success = bool(task.eval_success or official_checker)
        final_head = observation["observation"]["head_camera"]["rgb"]
        iio.imwrite(args.output_dir / "final_head.png", final_head)
        result = {
            "schema_version": 1,
            "task": args.task,
            "task_config": "demo_clean",
            "seed": args.seed,
            "policy": "tencent/Hy-Embodied-0.5-VLA-RoboTwin",
            "protocol": "official Hy-VLA encode_obs/wrapper with isolated localhost transport",
            "action_semantics": "dual-arm 16D EE wxyz",
            "step_limit": int(task.step_lim),
            "actions_executed": actions_executed,
            "network_forwards": network_forwards,
            "action_latencies_seconds": action_latencies,
            "eval_success": bool(task.eval_success),
            "official_check_success": official_checker,
            "success": success,
            "wall_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("ROLLOUT_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    finally:
        try:
            request(client, {"command": "close"})
        except Exception:
            pass
        client.close()
        task.close_env(clear_cache=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
