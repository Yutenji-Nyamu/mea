#!/usr/bin/env python3
"""Validate the minimal SmolVLA-to-LIBERO adapter contract.

This is an independent feasibility smoke.  It is not RoboTwin evidence and is
not part of the ManipEvalAgent policy-ranking protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


EXPECTED_INPUTS = {
    "observation.images.image": ("VISUAL", [3, 256, 256]),
    "observation.images.image2": ("VISUAL", [3, 256, 256]),
    "observation.state": ("STATE", [8]),
}
EXPECTED_OUTPUTS = {"action": ("ACTION", [7])}


def _feature_contract(features: dict[str, dict]) -> dict[str, tuple[str, list[int]]]:
    return {
        key: (str(value.get("type")), list(value.get("shape", [])))
        for key, value in features.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]

    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    input_contract = _feature_contract(config.get("input_features", {}))
    output_contract = _feature_contract(config.get("output_features", {}))

    checks = {
        "required_files": not missing,
        "policy_type_smolvla": config.get("type") == "smolvla",
        "libero_observation_contract": input_contract == EXPECTED_INPUTS,
        "libero_action_contract": output_contract == EXPECTED_OUTPUTS,
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "independent_libero_adapter_feasibility",
        "paper_eligible": False,
        "checkpoint": checkpoint.as_posix(),
        "model_bytes": (
            (checkpoint / "model.safetensors").stat().st_size
            if (checkpoint / "model.safetensors").is_file()
            else 0
        ),
        "missing_files": missing,
        "checks": checks,
        "input_contract": input_contract,
        "output_contract": output_contract,
        "recommended_eval": {
            "env_type": "libero",
            "suite": "libero_object",
            "task_ids": [0],
            "episodes": 1,
            "batch_size": 1,
            "control_mode": "relative",
        },
        "limitation": (
            "This validates the checkpoint/adapter contract and, when requested, "
            "weight loading. It is not a rollout, a RoboTwin result, or "
            "ManipEvalAgent Table 9 evidence."
        ),
    }

    if args.load_model and result["status"] == "passed":
        started = time.monotonic()
        try:
            import torch
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.smolvla import SmolVLAPolicy

            policy_config = PreTrainedConfig.from_pretrained(checkpoint)
            policy_config.device = args.device
            # The complete fine-tuned state dict is local. Initializing the
            # architecture from config avoids downloading duplicate base VLM weights.
            policy_config.load_vlm_weights = False
            policy = SmolVLAPolicy.from_pretrained(
                checkpoint,
                config=policy_config,
                local_files_only=True,
            )
            result["model_load"] = {
                "status": "passed",
                "device": args.device,
                "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            result["status"] = "failed"
            result["model_load"] = {
                "status": "failed",
                "device": args.device,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
