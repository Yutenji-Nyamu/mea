"""Bounded click_bell property variants preserving official task behavior."""

from __future__ import annotations

import numpy as np
import sapien
import math

from envs.click_bell import click_bell as OfficialClickBell
from envs.utils import create_actor, rand_pose


class click_bell(OfficialClickBell):
    """Expose one trusted position or instance axis through ``mea.bell``."""

    def setup_demo(self, **kwargs):
        self._mea_config = kwargs.get("mea") or {}
        super().setup_demo(**kwargs)

    @staticmethod
    def _sample_official_pose():
        pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        while abs(pose.p[0]) < 0.05:
            pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
        return pose

    def load_actors(self):
        bell_config = self._mea_config.get("bell") or {}
        if not self._mea_config.get("enabled", False):
            return super().load_actors()

        position_mode = bell_config.get("position_mode", "official_random")
        if position_mode not in {"official_random", "fixed"}:
            raise ValueError(f"Unsupported bell position_mode: {position_mode!r}")

        # Always consume the official pose RNG first.  A fixed overlay changes
        # only XY and therefore preserves downstream RNG ordering for model id.
        official_pose = self._sample_official_pose()
        if position_mode == "fixed":
            xy = bell_config.get("xy")
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError("mea.bell.xy must contain exactly two numbers")
            x, y = (float(xy[0]), float(xy[1]))
            if not all(math.isfinite(value) for value in (x, y)):
                raise ValueError("mea.bell.xy values must be finite")
            if not (-0.25 <= x <= 0.25 and abs(x) >= 0.05):
                raise ValueError("mea.bell.xy[0] is outside the official safe range")
            if not (-0.2 <= y <= 0.0):
                raise ValueError("mea.bell.xy[1] is outside the official range")
            bell_pose = sapien.Pose(
                [x, y, float(official_pose.p[2])], official_pose.q
            )
        else:
            bell_pose = official_pose

        # Always consume the official instance RNG, even when a trusted
        # instance overlay replaces its value.  This keeps all downstream RNG
        # ordering identical to the upstream task.
        official_bell_id = np.random.choice([0, 1], 1)[0]
        instance_mode = bell_config.get("instance_mode", "official_random")
        if instance_mode not in {"official_random", "fixed"}:
            raise ValueError(f"Unsupported bell instance_mode: {instance_mode!r}")
        if instance_mode == "fixed":
            requested_bell_id = bell_config.get("bell_id")
            if (
                isinstance(requested_bell_id, bool)
                or not isinstance(requested_bell_id, int)
                or requested_bell_id not in {0, 1}
            ):
                raise ValueError("mea.bell.bell_id must be integer 0 or 1")
            self.bell_id = requested_bell_id
        else:
            self.bell_id = official_bell_id
        self.bell = create_actor(
            scene=self,
            pose=bell_pose,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = (
            self.is_left_gripper_close
            if self.bell.get_pose().p[0] < 0
            else self.is_right_gripper_close
        )
