"""Schema-driven, multi-rate RoboTwin episode recorder.

The public recorder coordinates lifecycle state. Telemetry normalization,
events, visual capture, and artifact publication are owned by focused mixins.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.execution_receipt import (
    validate_execution_invocation,
    validate_imported_task_binding,
)

from .profiles import load_telemetry_profile, telemetry_profile_sha256
from .recorder_artifacts import RecorderArtifactMixin
from .recorder_contracts import (
    RecorderError,
    extend_task_schema_with_generated_actors,
)
from .recorder_events import RecorderEventMixin
from .recorder_telemetry import RecorderTelemetryMixin
from .recorder_visual import RecorderVisualMixin, VISUAL_CAPTURE_PROFILES
from .schema import load_task_schema, validate_task_schema


class EpisodeRecorder(
    RecorderVisualMixin,
    RecorderTelemetryMixin,
    RecorderEventMixin,
    RecorderArtifactMixin,
):
    """Collect one episode without changing policy or task semantics."""

    def __init__(
        self,
        repo_root: str | Path,
        output_dir: str | Path,
        *,
        task_name: str,
        seed: int,
        episode_index: int,
        policy_name: str,
        task_module: str | None = None,
        task_config: str | None = None,
        checkpoint_setting: str | None = None,
        telemetry_profile_id: str = "balanced_v1",
        visual_capture_profile_id: str | None = None,
        execution_receipt: Mapping[str, Any] | None = None,
        task_schema: Mapping[str, Any] | None = None,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema = (
            validate_task_schema(
                deepcopy(dict(task_schema)),
                expected_task_name=task_name,
            )
            if task_schema is not None
            else load_task_schema(self.repo_root, task_name)
        )
        self.task_name = task_name
        self.seed = int(seed)
        self.episode_index = int(episode_index)
        self.policy_name = policy_name
        self.task_module = task_module
        self.task_config = task_config
        self.checkpoint_setting = checkpoint_setting
        self.execution_receipt = (
            validate_execution_invocation(
                execution_receipt,
                task_name=task_name,
                task_module=task_module,
                task_config=task_config,
                checkpoint_setting=checkpoint_setting,
                policy_name=policy_name,
                seed=self.seed,
                episode_index=self.episode_index,
                checkpoint_dir=(
                    execution_receipt.get("checkpoint", {}).get("root")
                    if execution_receipt.get("checkpoint", {}).get("kind")
                    == "act_checkpoint_bundle"
                    else None
                ),
                verify_checkpoint_files=True,
            )
            if execution_receipt is not None
            else None
        )
        self.executed_binding: dict[str, Any] | None = None
        self.telemetry_profile = load_telemetry_profile(telemetry_profile_id)
        self.telemetry_profile_id = telemetry_profile_id
        self.telemetry_profile_hash = telemetry_profile_sha256(
            self.telemetry_profile
        )
        if (
            visual_capture_profile_id is not None
            and visual_capture_profile_id not in VISUAL_CAPTURE_PROFILES
        ):
            raise ValueError(
                "unknown visual capture profile: "
                f"{visual_capture_profile_id!r}"
            )
        self.visual_capture_profile_id = visual_capture_profile_id
        dynamics_stream = self.telemetry_profile.get("streams", {}).get(
            "dynamics_trace"
        )
        self.dynamics_period = (
            int(dynamics_stream["every_physics_steps"])
            if dynamics_stream is not None
            else None
        )
        if self.dynamics_period is not None and self.dynamics_period <= 0:
            raise RecorderError("dynamics sampling period must be positive")
        self.physics_dt = float(self.schema.get("physics_timestep_seconds", 0.004))
        self.action_dimension = int(self.schema.get("action_dimension", 0))
        self.policy_rows: list[dict[str, Any]] = []
        self.semantic_rows: list[dict[str, Any]] = []
        self.dynamics_rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.physics_step = 0
        self.policy_step = -1
        self.pending_action: list[float] | None = None
        self.pending_action_type: str | None = None
        self.active_contacts: dict[tuple[str, str], dict[str, Any]] = {}
        self.success_seen = False
        self.started_at = time.time()
        self.finished = False
        self._task: Any = None
        self.visual_frames: list[dict[str, Any]] = []
        self.temporal_visual_frame_count = 0
        self.visual_capture_errors: list[dict[str, str]] = []
        self.first_physical_contact_seen = False
        self.initial_physical_contacts: set[tuple[str, str]] = set()
        self.visual_keyframe_dir = self.output_dir / "visual_keyframes"
        if self.visual_capture_profile_id is not None:
            self._prepare_visual_capture()

        (self.output_dir / "schema.json").write_text(
            json.dumps(self.schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "telemetry_profile.json").write_text(
            json.dumps(self.telemetry_profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.execution_receipt is not None:
            (self.output_dir / "execution_receipt.json").write_text(
                json.dumps(
                    self.execution_receipt,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def start(self, task: Any) -> None:
        self._task = task
        self.schema = extend_task_schema_with_generated_actors(
            self.schema,
            task,
        )
        self._validate_task(task)
        # ``schema.json`` is an executed-episode contract. Generated tasks may
        # add public tracked actors only after construction, so overwrite the
        # initial official snapshot once that extension has been validated.
        (self.output_dir / "schema.json").write_text(
            json.dumps(self.schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.execution_receipt is not None:
            self.executed_binding = validate_imported_task_binding(
                self.execution_receipt,
                task,
            )
        if self.visual_capture_profile_id is not None:
            # Resting/support contacts exist before expert motion and must not
            # consume the first action-induced contact keyframe.
            self.initial_physical_contacts = {
                pair
                for pair, sample in self._contact_samples(task).items()
                if sample["physical_contact"]
            }
        self._capture_visual_keyframe(task, reason="initial")
        self.policy_rows.append(self._full_state(task, phase="initial", action=None))
        self.semantic_rows.append(self._semantic_state(task))
        self._record_dynamics(task, force=True)


__all__ = [
    "EpisodeRecorder",
    "RecorderError",
    "extend_task_schema_with_generated_actors",
]
