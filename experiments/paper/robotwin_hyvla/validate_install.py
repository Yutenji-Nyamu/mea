from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _official_revision(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def _synthetic_observation(height: int, width: int):
    import numpy as np

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    left = [0.35, 0.20, 0.25, 1.0, 0.0, 0.0, 0.0]
    right = [0.35, -0.20, 0.25, 1.0, 0.0, 0.0, 0.0]
    return {
        "endpose": {
            "left_endpose": left,
            "left_gripper": 0.0,
            "right_endpose": right,
            "right_gripper": 0.0,
        },
        "observation": {
            "head_camera": {"rgb": rgb.copy()},
            "left_camera": {"rgb": rgb.copy()},
            "right_camera": {"rgb": rgb.copy()},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline load and one-action validation for official Hy-VLA."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", default="pick up the bottle")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()

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

    observation = _synthetic_observation(480, 640)
    encoded = encode_obs(observation, args.instruction)
    encode_shapes = {
        key: list(value.shape)
        for key, value in encoded.items()
        if isinstance(value, np.ndarray)
    }
    expected_shapes = {
        "observation.images.top_head": [1, 3, 480, 640],
        "observation.images.hand_left": [1, 3, 480, 640],
        "observation.images.hand_right": [1, 3, 480, 640],
        "observation.state": [1, 32],
        "raw_images.top_head": [480, 640, 3],
        "raw_images.hand_left": [480, 640, 3],
        "raw_images.hand_right": [480, 640, 3],
    }
    if encode_shapes != expected_shapes:
        raise RuntimeError(f"unexpected encode_obs shapes: {encode_shapes}")
    if not all(
        np.isfinite(value).all()
        for value in encoded.values()
        if isinstance(value, np.ndarray)
    ):
        raise RuntimeError("encode_obs produced non-finite values")

    summary: dict[str, object] = {
        "schema_version": 1,
        "source": str(source),
        "source_revision": _official_revision(source),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "encode_obs_shapes": encode_shapes,
        "instruction": args.instruction,
        "seed": args.seed,
        "offline": True,
    }

    if not args.encode_only:
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

        forward_started = time.perf_counter()
        with torch.inference_mode():
            action = wrapper.get_action(encoded)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_started
        action = np.asarray(action)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"invalid Hy-VLA action: shape={action.shape}, "
                f"finite={np.isfinite(action).all()}"
            )
        summary.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "config": {
                    "chunk_size": wrapper.config.chunk_size,
                    "n_action_steps": wrapper.config.n_action_steps,
                    "max_state_dim": wrapper.config.max_state_dim,
                    "max_action_dim": wrapper.config.max_action_dim,
                    "use_video_encoder": wrapper.config.use_video_encoder,
                    "img_history_size": wrapper.img_history_size,
                    "img_history_interval": wrapper.img_history_interval,
                    "exc_action_size": wrapper.exc_action_size,
                },
                "load_seconds": load_seconds,
                "forward_seconds": forward_seconds,
                "action_shape": list(action.shape),
                "action_finite": True,
                "action": action.tolist(),
                "cached_actions": len(wrapper.action_cache),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("HYVLA_VALIDATION_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
