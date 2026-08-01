from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

from transport import recv_message, send_message


def _decode_observation(request, np):
    encoded_images = request["images"]
    images = {
        key: np.frombuffer(encoded_images[key]["data"], dtype=np.uint8)
        .reshape(tuple(encoded_images[key]["shape"]))
        .copy()
        for key in ("head_camera", "left_camera", "right_camera")
    }
    endpose = request["endpose"]
    return {
        "endpose": {
            "left_endpose": list(endpose["left_endpose"]),
            "left_gripper": float(endpose["left_gripper"]),
            "right_endpose": list(endpose["right_endpose"]),
            "right_gripper": float(endpose["right_gripper"]),
        },
        "observation": {
            key: {"rgb": images[key]}
            for key in ("head_camera", "left_camera", "right_camera")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolated official Hy-VLA policy process for RoboTwin."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18781)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--max-clients", type=int, default=1)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("--host must be loopback")
    if args.max_clients < 1:
        parser.error("--max-clients must be positive")

    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    for required in (
        source / "robotwin_eval" / "deploy_policy.py",
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "norm_stats.pkl",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(source))

    import numpy as np
    import torch
    from robotwin_eval.deploy_policy import encode_obs
    from robotwin_eval.policy_wrapper import build_policy

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    wrapper = build_policy(
        {
            "ckpt_path": str(checkpoint),
            "norm_path": str(checkpoint / "norm_stats.pkl"),
            "blend_mode": "rel_abs",
            "exc_action_size": 7,
            "img_history_size": 6,
            "img_history_interval": 5,
        }
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.unlink(missing_ok=True)
    ready = {
        "pid": os.getpid(),
        "host": args.host,
        "port": args.port,
        "source": str(source),
        "checkpoint": str(checkpoint),
        "load_seconds": load_seconds,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "action_semantics": "dual-arm 16D EE wxyz",
    }

    action_latencies: list[float] = []
    network_forwards = 0
    request_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        args.ready_file.write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("SERVER_READY_JSON=" + json.dumps(ready, sort_keys=True), flush=True)

        for _ in range(args.max_clients):
            connection, address = listener.accept()
            print(f"CLIENT_CONNECTED address={address}", flush=True)
            with connection:
                while True:
                    request = recv_message(connection)
                    command = request.get("command")
                    if command == "close":
                        send_message(connection, {"ok": True})
                        break
                    if command == "reset":
                        wrapper.reset()
                        send_message(connection, {"ok": True})
                        continue
                    if command != "act":
                        send_message(
                            connection,
                            {"ok": False, "error": f"unknown command {command!r}"},
                        )
                        continue
                    try:
                        batch = encode_obs(
                            _decode_observation(request, np), str(request["task"])
                        )
                        invokes_model = len(wrapper.action_cache) == 0
                        started = time.perf_counter()
                        with torch.inference_mode():
                            action = np.asarray(wrapper.get_action(batch))
                        torch.cuda.synchronize()
                        latency = time.perf_counter() - started
                        if action.shape != (16,) or not np.isfinite(action).all():
                            raise RuntimeError(
                                f"invalid action shape/values: {action.shape}, "
                                f"finite={np.isfinite(action).all()}"
                            )
                        request_count += 1
                        network_forwards += int(invokes_model)
                        action_latencies.append(latency)
                        send_message(
                            connection,
                            {
                                "ok": True,
                                "action": action.tolist(),
                                "latency_seconds": latency,
                                "network_forward": invokes_model,
                            },
                        )
                    except Exception as exc:
                        send_message(
                            connection,
                            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                        )
                        raise

    summary = {
        **ready,
        "request_count": request_count,
        "network_forwards": network_forwards,
        "action_latencies_seconds": action_latencies,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    args.summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SERVER_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
