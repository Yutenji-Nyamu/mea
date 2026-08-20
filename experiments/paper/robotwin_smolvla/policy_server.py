from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-metadata", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18771)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--max-clients", type=int, default=1)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("--host must be an IPv4 loopback address")
    if args.max_clients < 1:
        parser.error("--max-clients must be positive")

    checkpoint = Path(args.checkpoint)
    ready_file = Path(args.ready_file)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.unlink(missing_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.vlm_model_name = args.backbone_metadata
    config.load_vlm_weights = False
    config.device = "cuda"

    rename_map = {
        "observation.images.head_camera": "observation.images.camera1",
        "observation.images.left_camera": "observation.images.camera2",
        "observation.images.right_camera": "observation.images.camera3",
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": args.backbone_metadata},
        },
    )

    load_start = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    chunk_latencies: list[float] = []
    request_count = 0
    trial_seeds: list[int] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)

        ready = {
            "host": args.host,
            "port": args.port,
            "pid": __import__("os").getpid(),
            "load_seconds": load_seconds,
            "checkpoint": str(checkpoint),
            "state_config_shape": list(config.input_features["observation.state"].shape),
            "action_shape": list(config.output_features["action"].shape),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        }
        ready_file.write_text(
            json.dumps(ready, indent=2) + "\n", encoding="utf-8"
        )
        print("SERVER_READY_JSON=" + json.dumps(ready, sort_keys=True), flush=True)

        for client_index in range(args.max_clients):
            connection, address = listener.accept()
            print(
                f"CLIENT_CONNECTED index={client_index} address={address}",
                flush=True,
            )
            with connection:
                while True:
                    request = recv_message(connection)
                    command = request.get("command")
                    if command == "close":
                        send_message(connection, {"ok": True})
                        break
                    if command == "reset":
                        trial_seed = request.get("seed")
                        if (
                            isinstance(trial_seed, bool)
                            or not isinstance(trial_seed, int)
                            or trial_seed < 0
                        ):
                            send_message(
                                connection,
                                {
                                    "ok": False,
                                    "error": (
                                        "reset requires a non-negative int seed"
                                    ),
                                },
                            )
                            continue
                        torch.manual_seed(trial_seed)
                        torch.cuda.manual_seed_all(trial_seed)
                        np.random.seed(trial_seed)
                        policy.reset()
                        trial_seeds.append(trial_seed)
                        send_message(connection, {"ok": True})
                        continue
                    if command != "act_chunk":
                        send_message(
                            connection,
                            {"ok": False, "error": f"unknown command {command!r}"},
                        )
                        continue
                    try:
                        encoded_images = request["images"]
                        images = {
                            key: np.frombuffer(
                                encoded_images[key]["data"], dtype=np.uint8
                            )
                            .reshape(tuple(encoded_images[key]["shape"]))
                            .copy()
                            for key in (
                                "head_camera",
                                "left_camera",
                                "right_camera",
                            )
                        }
                        state = np.asarray(
                            request["state"], dtype=np.float32
                        ).reshape(1, 14)
                        raw = {
                            "pixels": {
                                key: images[key][None, ...]
                                for key in (
                                    "head_camera",
                                    "left_camera",
                                    "right_camera",
                                )
                            },
                            "agent_pos": state,
                        }
                        batch = preprocess_observation(raw)
                        batch["task"] = [str(request["task"])]
                        batch = preprocessor(batch)
                        started = time.perf_counter()
                        with torch.inference_mode():
                            action_chunk = policy.predict_action_chunk(batch)
                        torch.cuda.synchronize()
                        latency = time.perf_counter() - started
                        action_chunk = (
                            postprocessor(action_chunk).detach().cpu().numpy()[0]
                        )
                        if action_chunk.shape != (50, 14) or not np.isfinite(
                            action_chunk
                        ).all():
                            raise RuntimeError(
                                "invalid action chunk shape/values: "
                                f"{action_chunk.shape}, "
                                f"finite={np.isfinite(action_chunk).all()}"
                            )
                        chunk_latencies.append(latency)
                        request_count += 1
                        send_message(
                            connection,
                            {
                                "ok": True,
                                # Never pickle an ndarray across NumPy 1.26/2.x.
                                "actions": action_chunk.tolist(),
                                "latency_seconds": latency,
                                "state_dim": int(
                                    batch["observation.state"].shape[-1]
                                ),
                            },
                        )
                    except Exception as exc:
                        send_message(
                            connection,
                            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                        )
                        raise

    summary = {
        "request_count": request_count,
        "trial_seeds": trial_seeds,
        "chunk_latencies_seconds": chunk_latencies,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print("SERVER_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
