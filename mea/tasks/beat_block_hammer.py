"""Configurable BeatBlockHammer variant with official behavior as the default."""

from math import cos, sin

import numpy as np
import sapien

from envs.beat_block_hammer import beat_block_hammer as OfficialBeatBlockHammer
from envs.utils import create_actor, create_box, rand_pose


class beat_block_hammer(OfficialBeatBlockHammer):
    """Expose block appearance and pose through the ``mea.block`` config."""

    def setup_demo(self, **kwargs):
        self._mea_config = kwargs.get("mea") or {}
        super().setup_demo(**kwargs)

    @staticmethod
    def _sample_official_block_pose():
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
        while abs(block_pose.p[0]) < 0.05 or np.sum(block_pose.p[:2] ** 2) < 0.001:
            block_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.15],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )
        return block_pose

    def load_actors(self):
        if not self._mea_config.get("enabled", False):
            return super().load_actors()

        block_config = self._mea_config.get("block") or {}
        position_mode = block_config.get("position_mode", "official_random")
        yaw_mode = block_config.get("yaw_mode", "official_random")
        valid_modes = {"fixed", "official_random"}
        if position_mode not in valid_modes:
            raise ValueError(f"Unsupported block position_mode: {position_mode!r}")
        if yaw_mode not in valid_modes:
            raise ValueError(f"Unsupported block yaw_mode: {yaw_mode!r}")

        official_pose = None
        if "official_random" in {position_mode, yaw_mode}:
            official_pose = self._sample_official_block_pose()

        if position_mode == "fixed":
            xy = block_config.get("xy", [0.15, 0.05])
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError("mea.block.xy must contain exactly two numbers")
            position = [float(xy[0]), float(xy[1]), 0.76]
        else:
            position = official_pose.p

        if yaw_mode == "fixed":
            yaw = float(block_config.get("yaw", 0.0))
            quaternion = [cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)]
        else:
            quaternion = official_pose.q

        scale = float(block_config.get("scale", 1.0))
        if scale <= 0:
            raise ValueError(f"mea.block.scale must be positive, got {scale}")

        color = block_config.get("color", [1.0, 0.0, 0.0])
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ValueError("mea.block.color must contain exactly three channels")
        color = tuple(float(channel) for channel in color)
        if any(channel < 0 or channel > 1 for channel in color):
            raise ValueError("mea.block.color channels must be in [0, 1]")

        block_pose = sapien.Pose(position, quaternion)
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025 * scale,) * 3,
            color=color,
            name="box",
            is_static=True,
        )
        self.hammer.set_mass(0.001)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append(
            [
                block_pose.p[0] - 0.05,
                block_pose.p[1] - 0.05,
                block_pose.p[0] + 0.05,
                block_pose.p[1] + 0.05,
            ]
        )

        print(
            "[MEAEval] block variant: "
            f"position={block_pose.p.tolist()}, "
            f"quaternion={block_pose.q.tolist()}, scale={scale}, color={color}"
        )
