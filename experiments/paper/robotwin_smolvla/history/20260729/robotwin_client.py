from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import socket
import struct
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import yaml

from envs import CONFIGS_PATH


HEADER = struct.Struct("!Q")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("peer closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket):
    size = HEADER.unpack(recv_exact(sock, HEADER.size))[0]
    return pickle.loads(recv_exact(sock, size))


def send_message(sock: socket.socket, value) -> None:
    payload = pickle.dumps(value, protocol=5)
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def request(sock: socket.socket, value):
    send_message(sock, value)
    response = recv_message(sock)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "policy server request failed"))
    return response


def resolved_demo_clean_args(task_name: str) -> dict:
    with open(os.path.join(CONFIGS_PATH, "demo_clean.yml"), encoding="utf-8") as stream:
        args = yaml.safe_load(stream)
    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8"
    ) as stream:
        embodiment_types = yaml.safe_load(stream)
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) != 1:
        raise RuntimeError(f"expected one dual-arm embodiment, got {embodiment}")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as stream:
        robot_config = yaml.safe_load(stream)
    args["left_embodiment_config"] = robot_config
    args["right_embodiment_config"] = robot_config
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as stream:
        cameras = yaml.safe_load(stream)
    head_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head_type]["h"]
    args["head_camera_w"] = cameras[head_type]["w"]
    args.update(
        task_name=task_name,
        task_config="demo_clean",
        render_freq=0,
        eval_mode=True,
        save_data=False,
        eval_video_save_dir=None,
    )
    return args


def encode_observation(observation: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    camera_observation = observation["observation"]
    images = {
        name: np.ascontiguousarray(camera_observation[name]["rgb"], dtype=np.uint8)
        for name in ("head_camera", "left_camera", "right_camera")
    }
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,):
        raise RuntimeError(f"expected state shape (14,), got {state.shape}")
    return images, state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18771)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_module = importlib.import_module(f"envs.{args.task}")
    task_class = getattr(task_module, args.task)
    task = task_class()
    started = time.perf_counter()
    chunk_latencies: list[float] = []
    actions_executed = 0
    chunk_count = 0
    final_success = False
    final_checker = False
    client = socket.create_connection((args.host, args.port), timeout=30)
    client.settimeout(120)

    try:
        request(client, {"command": "reset"})
        task.setup_demo(
            now_ep_num=0,
            seed=args.seed,
            is_test=True,
            **resolved_demo_clean_args(args.task),
        )
        observation = task.get_obs()
        initial_images, initial_state = encode_observation(observation)
        iio.imwrite(output_dir / "initial_head.png", initial_images["head_camera"])

        while actions_executed < int(task.step_lim) and not task.eval_success:
            images, state = encode_observation(observation)
            encoded_images = {
                key: {"shape": list(value.shape), "data": value.tobytes()}
                for key, value in images.items()
            }
            response = request(
                client,
                {
                    "command": "act_chunk",
                    "images": encoded_images,
                    "state": state.tolist(),
                    "task": args.task.replace("_", " "),
                },
            )
            if response["state_dim"] != 14:
                raise RuntimeError(f"server processed state_dim={response['state_dim']}")
            chunk_count += 1
            chunk_latencies.append(float(response["latency_seconds"]))
            for action in np.asarray(response["actions"], dtype=np.float32):
                if actions_executed >= int(task.step_lim) or task.eval_success:
                    break
                task.take_action(action)
                actions_executed += 1
            observation = task.get_obs()

        final_checker = bool(task.check_success())
        final_success = bool(task.eval_success or final_checker)
        final_images, final_state = encode_observation(observation)
        iio.imwrite(output_dir / "final_head.png", final_images["head_camera"])
        result = {
            "task": args.task,
            "task_config": "demo_clean",
            "seed": args.seed,
            "policy": "lerobot/smolvla_robotwin",
            "step_limit": int(task.step_lim),
            "actions_executed": actions_executed,
            "chunk_count": chunk_count,
            "chunk_latencies_seconds": chunk_latencies,
            "success": final_success,
            "eval_success": bool(task.eval_success),
            "official_check_success": final_checker,
            "initial_state": initial_state.tolist(),
            "final_state": final_state.tolist(),
            "wall_seconds": time.perf_counter() - started,
            "protocol": "localhost length-prefixed pickle; one 50-action chunk per request",
        }
        (output_dir / "result.json").write_text(
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
