"""Lightweight RoboTwin-to-DP3 observation normalization.

This module intentionally avoids Hydra, SAPIEN, Torch, and the DP3 model
package so its input contract can be tested in the main RoboTwin environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _float32_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    """Normalize one DP3 input while failing clearly on missing observations."""

    if all(hasattr(value, name) for name in ("detach", "cpu", "numpy")):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(
            f"{field} must be a non-empty {ndim}D array, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must contain numeric values")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return array


def encode_obs(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert RoboTwin observations to the released DP3 checkpoint contract."""

    agent_pos = _float32_array(
        observation["joint_action"]["vector"],
        field="joint_action.vector",
        ndim=1,
    )
    if agent_pos.shape != (14,):
        raise ValueError(
            "joint_action.vector must have shape (14,) for the released "
            f"dual-arm DP3 checkpoint, got {agent_pos.shape}"
        )

    point_cloud = _float32_array(
        observation["pointcloud"],
        field="pointcloud",
        ndim=2,
    )
    if point_cloud.shape[1] < 3:
        raise ValueError(
            "pointcloud must provide at least XYZ columns, got "
            f"shape {point_cloud.shape}"
        )
    return {"agent_pos": agent_pos, "point_cloud": point_cloud}


def ensure_pointcloud_observation(
    task_env: Any,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Enable RoboTwin's real point-cloud sensor and refresh an empty first obs."""

    if np.asarray(observation.get("pointcloud")).size:
        return observation

    data_type = getattr(task_env, "data_type", None)
    if not isinstance(data_type, dict):
        raise ValueError(
            "DP3 requires task_env.data_type so point-cloud collection can "
            "be enabled"
        )
    data_type["pointcloud"] = True
    refreshed = task_env.get_obs()
    if np.asarray(refreshed.get("pointcloud")).size == 0:
        raise ValueError(
            "DP3 enabled point-cloud collection but the refreshed "
            "observation is still empty"
        )
    return refreshed


__all__ = ["encode_obs", "ensure_pointcloud_observation"]
