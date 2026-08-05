"""Sparse visual evidence capture for one recorded episode."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from mea.visual_capture import (
    VISUAL_CAPTURE_PROFILE_CONFIGS,
    VISUAL_CAPTURE_PROFILES,
)

from .recorder_contracts import RecorderError


VISUAL_CAPTURE_FPS = 2


class RecorderVisualMixin:
    def _visual_capture_error(self, stage: str, error: Exception) -> None:
        self.visual_capture_errors.append(
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    def _prepare_visual_capture(self) -> None:
        """Remove stale visual artifacts before a retried expert probe."""

        try:
            self.visual_keyframe_dir.mkdir(parents=True, exist_ok=True)
            for path in self.visual_keyframe_dir.glob("frame_*.png"):
                path.unlink()
            for path in (
                self.output_dir / "visual_keyframes.json",
                self.output_dir / "video.mp4",
                self.output_dir / "video.partial.mp4",
            ):
                if path.exists():
                    path.unlink()
        except Exception as exc:
            self._visual_capture_error("prepare", exc)

    def _capture_visual_keyframe(
        self,
        task: Any,
        *,
        reason: str,
    ) -> int | None:
        """Capture one sparse head-camera frame without affecting telemetry."""

        if self.visual_capture_profile_id is None:
            return None
        if (
            self.visual_frames
            and self.visual_frames[-1]["physics_step"] == self.physics_step
        ):
            reasons = self.visual_frames[-1]["reasons"]
            if reason not in reasons:
                reasons.append(reason)
            return int(self.visual_frames[-1]["frame_index"])
        frame_index = len(self.visual_frames)
        relative_image = Path("visual_keyframes") / f"frame_{frame_index:03d}.png"
        destination = self.output_dir / relative_image
        try:
            save_camera_rgb = getattr(task, "save_camera_rgb")
            save_camera_rgb(str(destination), camera_name="head_camera")
            if not destination.is_file() or destination.stat().st_size <= 0:
                raise RecorderError(
                    f"head-camera keyframe was not written: {destination}"
                )
        except Exception as exc:
            self._visual_capture_error(f"capture:{reason}", exc)
            return None
        self.visual_frames.append(
            {
                "frame_index": frame_index,
                "policy_step": self.policy_step,
                "physics_step": self.physics_step,
                "simulation_time_seconds": self.physics_step * self.physics_dt,
                "reasons": [reason],
                "image": relative_image.as_posix(),
            }
        )
        return frame_index

    def _capture_temporal_keyframe_if_due(self, task: Any) -> int | None:
        """Capture a bounded fixed-policy-step sample for temporal VQA."""

        if self.visual_capture_profile_id is None:
            return None
        profile = VISUAL_CAPTURE_PROFILE_CONFIGS[
            self.visual_capture_profile_id
        ]
        period = profile["policy_step_period"]
        limit = int(profile["max_periodic_frames"])
        completed_policy_steps = self.policy_step + 1
        if (
            period is None
            or completed_policy_steps <= 0
            or completed_policy_steps % int(period) != 0
            or self.temporal_visual_frame_count >= limit
        ):
            return None
        frame_index = self._capture_visual_keyframe(
            task,
            reason="periodic_policy_step",
        )
        if frame_index is not None:
            self.temporal_visual_frame_count += 1
        return frame_index

    def _finalize_visual_capture(self) -> dict[str, Any] | None:
        if self.visual_capture_profile_id is None:
            return None

        video = self.output_dir / "video.mp4"
        partial_video = self.output_dir / "video.partial.mp4"
        if not self.visual_frames:
            self.visual_capture_errors.append(
                {
                    "stage": "encode",
                    "type": "RecorderError",
                    "message": "no visual keyframes were captured",
                }
            )
        if not self.visual_capture_errors:
            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(VISUAL_CAPTURE_FPS),
                "-start_number",
                "0",
                "-i",
                str(self.visual_keyframe_dir / "frame_%03d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(partial_video),
            ]
            try:
                process = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=60,
                )
                if process.returncode != 0:
                    raise RecorderError(
                        "ffmpeg failed with return code "
                        f"{process.returncode}: {process.stderr.strip()}"
                    )
                if not partial_video.is_file() or partial_video.stat().st_size <= 0:
                    raise RecorderError("ffmpeg did not produce a non-empty video")
                partial_video.replace(video)
            except Exception as exc:
                self._visual_capture_error("encode", exc)
        if partial_video.exists():
            try:
                partial_video.unlink()
            except OSError:
                pass

        completed = not self.visual_capture_errors and video.is_file()
        result = {
            "schema_version": 1,
            "profile_id": self.visual_capture_profile_id,
            "status": "completed" if completed else "failed",
            "camera": "head_camera",
            "frame_count": len(self.visual_frames),
            "nominal_frame_rate_hz": VISUAL_CAPTURE_FPS,
            "sampling": dict(
                VISUAL_CAPTURE_PROFILE_CONFIGS[
                    self.visual_capture_profile_id
                ]
            ),
            "frames": self.visual_frames,
            "errors": self.visual_capture_errors,
        }
        self._write_visual_manifest(result)
        return result

    def _write_visual_manifest(self, result: dict[str, Any]) -> None:
        (self.output_dir / "visual_keyframes.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )



__all__ = [
    "RecorderVisualMixin",
    "VISUAL_CAPTURE_FPS",
    "VISUAL_CAPTURE_PROFILES",
]
