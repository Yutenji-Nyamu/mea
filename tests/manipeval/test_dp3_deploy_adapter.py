import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

_ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "policy"
    / "DP3"
    / "observation_adapter.py"
)
_SPEC = spec_from_file_location("robotwin_dp3_observation_adapter", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ADAPTER = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ADAPTER)
encode_obs = _ADAPTER.encode_obs
ensure_pointcloud_observation = _ADAPTER.ensure_pointcloud_observation


def _observation(pointcloud):
    return {
        "joint_action": {"vector": list(range(14))},
        "pointcloud": pointcloud,
    }


class DP3ObservationAdapterTests(unittest.TestCase):
    def test_encode_obs_normalizes_numeric_lists(self):
        encoded = encode_obs(
            _observation([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        )

        self.assertEqual(encoded["agent_pos"].shape, (14,))
        self.assertEqual(encoded["point_cloud"].shape, (2, 6))
        self.assertEqual(encoded["agent_pos"].dtype, np.float32)
        self.assertEqual(encoded["point_cloud"].dtype, np.float32)

    def test_encode_obs_rejects_empty_pointcloud(self):
        with self.assertRaisesRegex(ValueError, "pointcloud must be a non-empty"):
            encode_obs(_observation([]))

    def test_ensure_pointcloud_refreshes_real_sensor_observation(self):
        class FakeTask:
            def __init__(self):
                self.data_type = {"pointcloud": False}
                self.calls = 0

            def get_obs(self):
                self.calls += 1
                self.assert_sensor_enabled = self.data_type["pointcloud"]
                return _observation(np.ones((1024, 6), dtype=np.float32))

        task = FakeTask()
        refreshed = ensure_pointcloud_observation(task, _observation([]))

        self.assertTrue(task.data_type["pointcloud"])
        self.assertTrue(task.assert_sensor_enabled)
        self.assertEqual(task.calls, 1)
        self.assertEqual(np.asarray(refreshed["pointcloud"]).shape, (1024, 6))

    def test_ensure_pointcloud_does_not_refresh_existing_observation(self):
        class FailIfCalled:
            data_type = {"pointcloud": True}

            def get_obs(self):
                raise AssertionError("existing point cloud must be reused")

        observation = _observation(np.ones((1024, 6), dtype=np.float32))
        self.assertIs(
            ensure_pointcloud_observation(FailIfCalled(), observation),
            observation,
        )


if __name__ == "__main__":
    unittest.main()
