"""LeRobot/SmolVLA policy adapter shared by official and custom LIBERO tasks."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .benchmark import (
    BATCH23_PARITY_ACTION_STEPS,
    BATCH23_PARITY_HORIZON_STEPS,
    BATCH23_PARITY_OBSERVATION_SIZE,
    EpisodeRecord,
    TaskContract,
)


class LeRobotPolicyAdapter:
    """Load one local checkpoint and run bounded rollouts through LeRobot processors."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str = "cuda",
        n_action_steps: int = BATCH23_PARITY_ACTION_STEPS,
        horizon_steps: int = BATCH23_PARITY_HORIZON_STEPS,
        observation_size: int = BATCH23_PARITY_OBSERVATION_SIZE,
        suite_name: str = "libero_object",
        task_id: int = 0,
        backbone_metadata: str | Path | None = None,
    ):
        if not suite_name.strip():
            raise ValueError("suite_name cannot be empty")
        if int(task_id) < 0:
            raise ValueError("task_id must be non-negative")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self.n_action_steps = n_action_steps
        self.horizon_steps = horizon_steps
        self.observation_size = observation_size
        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.backbone_metadata = (
            Path(backbone_metadata).expanduser().resolve()
            if backbone_metadata is not None
            else None
        )
        self.policy: Any | None = None
        self.env_config: Any | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        self.env_preprocessor: Any | None = None
        self.env_postprocessor: Any | None = None
        self.seed_contract: dict[str, Any] | None = None
        self._official_vector_env: Any | None = None
        self._paired_rollout_rng_state: dict[str, Any] | None = None
        self._rollouts_started = 0

    def load(self, *, seed: int) -> dict[str, Any]:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.random_utils import get_rng_state, set_seed

        started = time.monotonic()
        # Match lerobot-eval's process-level setup before policy construction.
        # SmolVLA samples flow-matching noise whenever its action queue is empty;
        # seeding only env.reset() does not make policy inference reproducible.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        set_seed(seed)
        self.seed_contract = {
            "schema_version": 1,
            "global_seed": int(seed),
            "environment_reset_seed": int(seed),
            "policy_rng_origin_seed": int(seed),
            "seed_scope": "once_before_env_and_policy_construction",
            "policy_rng_first_rollout_state": (
                "continuation_after_env_policy_and_processor_construction"
            ),
            "first_rollout_integer_reseed": False,
            "paired_round_rng_restore": True,
            "paired_rng_capture_point": (
                "after_env_policy_and_processors_before_first_rollout"
            ),
            "implementation": "lerobot.utils.random_utils.set_seed",
            "cudnn_benchmark": True,
            "cuda_matmul_allow_tf32": True,
        }
        env_config = LiberoEnvConfig(
            task=self.suite_name,
            task_ids=[self.task_id],
            episode_length=self.horizon_steps,
            control_mode="relative",
            observation_height=self.observation_size,
            observation_width=self.observation_size,
            max_parallel_tasks=1,
        )
        if (
            self.backbone_metadata is not None
            and not (self.backbone_metadata / "config.json").is_file()
        ):
            raise FileNotFoundError(
                "LIBERO SmolVLA backbone metadata is missing config.json: "
                f"{self.backbone_metadata}"
            )
        # Stock evaluator order is set_seed -> make_env -> make_policy -> rollout.
        envs = make_env(
            env_config,
            n_envs=1,
            use_async_envs=True,
        )
        self._official_vector_env = envs[self.suite_name][self.task_id]
        policy_config = PreTrainedConfig.from_pretrained(self.checkpoint)
        policy_config.pretrained_path = self.checkpoint
        policy_config.device = self.device
        policy_config.load_vlm_weights = False
        if self.backbone_metadata is not None:
            policy_config.vlm_model_name = str(self.backbone_metadata)
        policy_config.n_action_steps = self.n_action_steps
        policy = make_policy(
            cfg=policy_config,
            env_cfg=env_config,
            rename_map={},
        )
        policy.eval()
        preprocessor_overrides = {
            "device_processor": {"device": self.device},
            "rename_observations_processor": {"rename_map": {}},
        }
        if self.backbone_metadata is not None:
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(self.backbone_metadata)
            }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides=preprocessor_overrides,
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
        self._paired_rollout_rng_state = get_rng_state()
        self._rollouts_started = 0
        return {
            "status": "passed",
            "checkpoint": str(self.checkpoint),
            "backbone_metadata": (
                str(self.backbone_metadata)
                if self.backbone_metadata is not None
                else None
            ),
            "device": self.device,
            "n_action_steps": self.n_action_steps,
            "horizon_steps": self.horizon_steps,
            "observation_size": self.observation_size,
            "suite": self.suite_name,
            "task_id": self.task_id,
            "control_mode": "relative",
            "seed_contract": dict(self.seed_contract),
            "processor_artifacts": {
                "preprocessor": str(self.checkpoint / "policy_preprocessor.json"),
                "postprocessor": str(self.checkpoint / "policy_postprocessor.json"),
            },
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "torch_version": torch.__version__,
        }

    def make_stock_official_vector_env(self) -> Any:
        """Use the same public environment factory as ``lerobot-eval``."""
        if self.env_config is None or self._official_vector_env is None:
            raise RuntimeError("LeRobotPolicyAdapter.load() must be called first")
        env = self._official_vector_env
        self._official_vector_env = None
        return env

    def prepare_policy_rng_for_rollout(self) -> tuple[int, str]:
        """Keep the first rollout stock-compatible and pair later variants."""
        from lerobot.utils.random_utils import set_rng_state

        if self._rollouts_started == 0:
            mode = "stock_first_rollout_continuation"
        else:
            if self._paired_rollout_rng_state is None:
                raise RuntimeError("paired rollout RNG state is missing")
            set_rng_state(self._paired_rollout_rng_state)
            mode = "restored_pre_first_rollout_state"
        self._rollouts_started += 1
        return self._rollouts_started, mode

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
        use_stock_official_env: bool = False,
    ) -> EpisodeRecord:
        if self.policy is None:
            raise RuntimeError("LeRobotPolicyAdapter.load() must be called first")
        if self.seed_contract is None:
            raise RuntimeError("policy seed contract is missing")
        if int(seed) != int(self.seed_contract["environment_reset_seed"]):
            raise ValueError(
                "rollout seed disagrees with the seed fixed before policy construction"
            )

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

        if use_stock_official_env:
            env = self.make_stock_official_vector_env()
            env_route = "lerobot_make_env_single_task"
        else:
            env = gym.vector.SyncVectorEnv([env_factory])
            env_route = "custom_sync_vector_env"
        rollout_index, policy_rng_mode = self.prepare_policy_rng_for_rollout()
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
            suite=self.suite_name,
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
            provenance={
                **provenance,
                "env_route": env_route,
                "rollout_index": rollout_index,
                "policy_rng_mode": policy_rng_mode,
                "seed_contract": dict(self.seed_contract),
            },
        )
        (output / "episode.json").write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return record

    def unload(self) -> None:
        if self._official_vector_env is not None:
            self._official_vector_env.close()
            self._official_vector_env = None
        if self.policy is None:
            return
        import torch

        del self.policy
        self.policy = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
