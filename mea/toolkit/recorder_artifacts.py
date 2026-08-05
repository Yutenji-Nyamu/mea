"""Episode artifact encoding, receipt projection, and final manifest."""

from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from typing import Any

import numpy as np

from .recorder_contracts import RecorderError
from .recorder_visual import VISUAL_CAPTURE_FPS


class RecorderArtifactMixin:
    def finish(
        self,
        task: Any,
        *,
        success: bool,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.finished:
            raise RecorderError("EpisodeRecorder.finish() 不能重复调用")
        self.finished = True
        for pair in list(self.active_contacts):
            self._close_contact(pair, reason="episode_end")
        if success:
            self._record_success(task)
        self._capture_visual_keyframe(task, reason="final")
        self.policy_rows.append(
            self._full_state(task, phase="final", action=None)
        )
        self._record_dynamics(task, force=True, success_override=success)
        generated_checker_success: bool | None = None
        official_core_predicate_satisfied: bool | None = None
        official_core_checker = getattr(
            task,
            "mea_official_check_success",
            None,
        )
        if callable(official_core_checker):
            try:
                generated_checker_success = bool(task.check_success())
                official_core_predicate_satisfied = bool(
                    official_core_checker()
                )
            except Exception as exc:
                self.events.append(
                    {
                        "type": "outcome_semantics_error",
                        "policy_step": self.policy_step,
                        "physics_step": self.physics_step,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        self._write_policy_csv()
        semantic_arrays = self._write_semantic_npz()
        dynamics_arrays = self._write_dynamics_npz()
        self._write_events()
        try:
            visual_capture = self._finalize_visual_capture()
        except Exception as exc:  # visual evidence is always best-effort
            self._visual_capture_error("finalize", exc)
            # A container produced before the manifest failed is not a
            # contract-complete artifact. Remove it so path-only consumers
            # cannot mistake best-effort output for approved visual evidence.
            for path in (
                self.output_dir / "video.mp4",
                self.output_dir / "video.partial.mp4",
                self.output_dir / "visual_keyframes.json",
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    self._visual_capture_error("cleanup", cleanup_exc)
            visual_capture = {
                "schema_version": 1,
                "profile_id": self.visual_capture_profile_id,
                "status": "failed",
                "camera": "head_camera",
                "frame_count": len(self.visual_frames),
                "nominal_frame_rate_hz": VISUAL_CAPTURE_FPS,
                "frames": self.visual_frames,
                "errors": self.visual_capture_errors,
            }
        stream_metadata: dict[str, Any] = {
            "policy_state": {
                "artifact": "states.csv",
                "sampling": "policy_boundary",
                "rows": len(self.policy_rows),
            },
            "semantic_trace": {
                "artifact": "semantic_trace.npz",
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "rows": len(self.semantic_rows),
                "arrays": semantic_arrays,
            },
            "contact_events": {
                "artifact": "events.jsonl",
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "mode": "interval_summary",
                "rows": len(self.events),
            },
        }
        if self.dynamics_period is not None:
            stream_metadata["dynamics_trace"] = {
                "artifact": "dynamics_trace.npz",
                "sampling": "physics_period",
                "every_physics_steps": self.dynamics_period,
                "force_initial_sample": True,
                "force_final_sample": True,
                "rows": len(self.dynamics_rows),
                "arrays": dynamics_arrays,
            }
        visual_completed = bool(
            visual_capture and visual_capture.get("status") == "completed"
        )
        metadata = {
            "schema_version": 1,
            "recorder_schema_version": 2,
            "task_name": self.task_name,
            "task_module": self.task_module,
            "task_config": self.task_config,
            "checkpoint_setting": self.checkpoint_setting,
            "policy_name": self.policy_name,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "success": bool(success),
            **(
                {
                    "generated_checker_success": (
                        generated_checker_success
                    ),
                    "official_core_predicate_satisfied": (
                        official_core_predicate_satisfied
                    ),
                }
                if generated_checker_success is not None
                and official_core_predicate_satisfied is not None
                else {}
            ),
            "policy_steps": max(self.policy_step + 1, 0),
            "physics_steps": self.physics_step,
            "physics_timestep_seconds": self.physics_dt,
            "simulation_duration_seconds": self.physics_step * self.physics_dt,
            "wall_duration_seconds": time.time() - self.started_at,
            "policy_state_rows": len(self.policy_rows),
            "semantic_trace_rows": len(self.semantic_rows),
            "dynamics_trace_rows": len(self.dynamics_rows),
            "telemetry_profile_id": self.telemetry_profile_id,
            "telemetry_profile_sha256": self.telemetry_profile_hash,
            "telemetry": {
                "profile_id": self.telemetry_profile_id,
                "profile_sha256": self.telemetry_profile_hash,
                "profile_artifact": "telemetry_profile.json",
                "streams": stream_metadata,
            },
            "contact_interval_count": sum(
                item.get("type") == "contact_interval" for item in self.events
            ),
            "error": error,
            **(
                {
                    "execution_receipt": deepcopy(self.execution_receipt),
                    "execution_receipt_sha256": self.execution_receipt[
                        "receipt_sha256"
                    ],
                    "executed_binding": deepcopy(self.executed_binding),
                    "executed_task_module_sha256": self.executed_binding[
                        "task_source_sha256"
                    ],
                    "executed_checkpoint_bundle_sha256": (
                        self.executed_binding["checkpoint_bundle_sha256"]
                    ),
                }
                if self.execution_receipt is not None
                and self.executed_binding is not None
                else {}
            ),
            **(
                {"visual_capture": visual_capture}
                if visual_capture is not None
                else {}
            ),
            **(
                {
                    "video_alignment": {
                        "schema_version": 1,
                        "mode": "event_keyframes",
                        "nominal_frame_rate_hz": VISUAL_CAPTURE_FPS,
                        "frame_manifest": "visual_keyframes.json",
                        "frame_semantics": (
                            "ordered sparse event evidence; not continuous-time "
                            "video"
                        ),
                    }
                }
                if visual_completed
                else {}
            ),
            "artifacts": {
                "policy_states": "states.csv",
                "semantic_trace": "semantic_trace.npz",
                **(
                    {"dynamics_trace": "dynamics_trace.npz"}
                    if self.dynamics_period is not None
                    else {}
                ),
                "events": "events.jsonl",
                "task_schema": "schema.json",
                "telemetry_profile": "telemetry_profile.json",
                **(
                    {"execution_receipt": "execution_receipt.json"}
                    if self.execution_receipt is not None
                    else {}
                ),
                **(
                    {"visual_keyframes": "visual_keyframes.json"}
                    if (self.output_dir / "visual_keyframes.json").is_file()
                    else {}
                ),
                **({"video": "video.mp4"} if visual_completed else {}),
            },
        }
        (self.output_dir / "episode.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return metadata

    def _write_policy_csv(self) -> None:
        columns = [
            "phase",
            "policy_step",
            "physics_step",
            "simulation_time_seconds",
            "wall_time_seconds",
            "video_frame_index",
            "success",
            "action_type",
        ]
        remaining = sorted(
            set().union(*(row.keys() for row in self.policy_rows)) - set(columns)
        )
        with (self.output_dir / "states.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns + remaining)
            writer.writeheader()
            writer.writerows(self.policy_rows)

    @staticmethod
    def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        }

    @staticmethod
    def _typed_arrays(
        rows: list[dict[str, Any]],
        *,
        step_dtype: Any = np.int64,
    ) -> dict[str, np.ndarray]:
        if not rows:
            return {}
        expected = set(rows[0])
        for index, row in enumerate(rows[1:], start=1):
            if set(row) != expected:
                missing = sorted(expected - set(row))
                extra = sorted(set(row) - expected)
                raise RecorderError(
                    f"telemetry row {index} schema drift: missing={missing}, extra={extra}"
                )
        arrays: dict[str, np.ndarray] = {}
        for key in sorted(expected):
            if key in {"physics_step", "policy_step"}:
                dtype: Any = step_dtype
            elif key == "success":
                dtype = np.bool_
            elif key == "simulation_time_seconds":
                dtype = np.float64
            else:
                dtype = np.float32
            try:
                arrays[key] = np.asarray([row[key] for row in rows], dtype=dtype)
            except (TypeError, ValueError) as exc:
                raise RecorderError(f"cannot encode telemetry array {key!r}") from exc
        return arrays

    def _write_semantic_npz(self) -> dict[str, Any]:
        # Keep the legacy semantic axis dtype byte-compatible with recorder v1.
        arrays = self._typed_arrays(self.semantic_rows, step_dtype=np.float64)
        np.savez_compressed(self.output_dir / "semantic_trace.npz", **arrays)
        return self._array_manifest(arrays)

    def _write_dynamics_npz(self) -> dict[str, Any]:
        if self.dynamics_period is None:
            return {}
        arrays = self._typed_arrays(self.dynamics_rows)
        np.savez_compressed(self.output_dir / "dynamics_trace.npz", **arrays)
        return self._array_manifest(arrays)

    def _write_events(self) -> None:
        with (self.output_dir / "events.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")


__all__ = ["RecorderArtifactMixin"]

