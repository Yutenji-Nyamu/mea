"""LeRobot/SmolVLA policy adapter shared by official and custom LIBERO tasks."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .benchmark import EpisodeRecord, TaskContract


class LeRobotPolicyAdapter:
    """Load one local checkpoint and run bounded rollouts through LeRobot processors."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str = "cuda",
        n_action_steps: int = 10,
        horizon_steps: int = 100,
    ):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self.n_action_steps = n_action_steps
        self.horizon_steps = horizon_steps
        self.policy: Any | None = None
        self.env_config: Any | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        self.env_preprocessor: Any | None = None
        self.env_postprocessor: Any | None = None

    def load(self) -> dict[str, Any]:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors

        started = time.monotonic()
        policy_config = PreTrainedConfig.from_pretrained(self.checkpoint)
        policy_config.pretrained_path = self.checkpoint
        policy_config.device = self.device
        policy_config.load_vlm_weights = False
        policy_config.n_action_steps = self.n_action_steps
        env_config = LiberoEnvConfig(
            task="libero_object",
            task_ids=[0],
            episode_length=self.horizon_steps,
            control_mode="relative",
            observation_height=256,
            observation_width=256,
            max_parallel_tasks=1,
        )
        policy = make_policy(cfg=policy_config, env_cfg=env_config)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_config,
            policy_cfg=policy_config,
        )
        self.policy = policy
        self.env_config = env_config
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        return {
            "status": "passed",
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "n_action_steps": self.n_action_steps,
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "torch_version": torch.__version__,
        }

    def run(
        self,
        *,
        env_factory: Callable[[], Any],
        seed: int,
        output_dir: str | Path,
        task_id: str,
        task_contract_path: str | Path,
        bddl_path: str | Path,
        provenance: dict[str, Any],
    ) -> EpisodeRecord:
        if self.policy is None:
            raise RuntimeError("LeRobotPolicyAdapter.load() must be called first")

        import gymnasium as gym
        import imageio.v3 as iio
        import torch
        from PIL import Image
        from lerobot.scripts.lerobot_eval import rollout

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        frames: list[np.ndarray] = []

        def render_callback(vector_env: Any) -> None:
            rendered = vector_env.call("render")[0]
            frames.append(np.asarray(rendered, dtype=np.uint8))

        env = gym.vector.SyncVectorEnv([env_factory])
        started = time.monotonic()
        precision_context = (
            torch.autocast(device_type="cuda")
            if bool(getattr(self.policy.config, "use_amp", False))
            else nullcontext()
        )
        with torch.no_grad(), precision_context:
            data = rollout(
                env=env,
                policy=self.policy,
                env_preprocessor=self.env_preprocessor,
                env_postprocessor=self.env_postprocessor,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                seeds=[seed],
                return_observations=False,
                render_callback=render_callback,
            )
        elapsed = time.monotonic() - started
        env.close()

        actions = data["action"][0].detach().cpu().numpy()
        rewards = data["reward"][0].detach().cpu().numpy()
        successes = data["success"][0].detach().cpu().numpy().astype(bool)
        done = data["done"][0].detach().cpu().numpy().astype(bool)
        executed_steps = int(np.argmax(done) + 1) if bool(done.any()) else len(actions)
        actions = actions[:executed_steps]
        rewards = rewards[:executed_steps]
        success = bool(successes[:executed_steps].any())

        actions_path = output / "actions.npy"
        np.save(actions_path, actions)
        if frames:
            for name, index in (
                ("first", 0),
                ("middle", len(frames) // 2),
                ("last", len(frames) - 1),
            ):
                Image.fromarray(frames[index]).save(output / f"{name}_frame.png")
        video_path: Path | None = None
        if frames:
            try:
                video_path = output / "episode.mp4"
                iio.imwrite(video_path, np.stack(frames), fps=30)
            except Exception as exc:
                (output / "video_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
                video_path = None

        record = EpisodeRecord(
            schema_version=1,
            benchmark="libero",
            policy_name="smolvla",
            checkpoint=str(self.checkpoint),
            suite="libero_object",
            task_id=task_id,
            seed=seed,
            horizon_steps=self.horizon_steps,
            executed_steps=executed_steps,
            success=success,
            reward_sum=float(rewards.sum()),
            goal_predicate_satisfied=success,
            elapsed_seconds=round(elapsed, 3),
            bddl_path=str(Path(bddl_path).expanduser().resolve()),
            video_path=str(video_path) if video_path else None,
            task_contract_path=str(Path(task_contract_path).expanduser().resolve()),
            actions_path=str(actions_path),
            provenance=provenance,
        )
        (output / "episode.json").write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return record

    def unload(self) -> None:
        if self.policy is None:
            return
        import torch

        del self.policy
        self.policy = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
