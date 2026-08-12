from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline X-VLA RoboTwin checkpoint load and bounded action inference."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", default="press the stapler")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    for required in (
        source / "models" / "modeling_xvla.py",
        source / "models" / "processing_xvla.py",
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(source))

    import numpy as np
    import torch
    from PIL import Image

    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    processor = XVLAProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = XVLA.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to("cuda", dtype=torch.float32)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    images = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(3)]
    inputs = processor(images, args.instruction)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prepared = {
        key: value.to(device=device, dtype=dtype)
        if value.is_floating_point()
        else value.to(device=device)
        for key, value in inputs.items()
    }
    prepared.update(
        proprio=torch.zeros((1, 20), device=device, dtype=dtype),
        domain_id=torch.tensor([6], device=device, dtype=torch.long),
    )

    infer_started = time.perf_counter()
    with torch.inference_mode():
        action = model.generate_actions(**prepared, steps=args.steps)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - infer_started
    action_cpu = action.float().cpu().numpy()
    result = {
        "source": str(source),
        "checkpoint": str(checkpoint),
        "instruction": args.instruction,
        "denoising_steps": args.steps,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_dtype": str(dtype),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "action_shape": list(action_cpu.shape),
        "action_finite": bool(np.isfinite(action_cpu).all()),
        "action_min": float(np.nanmin(action_cpu)),
        "action_max": float(np.nanmax(action_cpu)),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if result["action_shape"] != [1, 30, 20] or not result["action_finite"]:
        raise RuntimeError(f"invalid action output: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
