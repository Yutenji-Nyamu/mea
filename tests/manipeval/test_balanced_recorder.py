import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mea.toolkit import EpisodeRecorder, TrajectoryView


class _Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class _Entity:
    def __init__(self, offset=0.0):
        self.offset = float(offset)

    def get_qpos(self):
        return np.asarray([self.offset, self.offset + 1.0])

    def get_qvel(self):
        return np.asarray([self.offset + 2.0, self.offset + 3.0])

    def get_linear_velocity(self):
        return np.asarray([self.offset, 0.1, 0.2])

    def get_angular_velocity(self):
        return np.asarray([0.3, self.offset, 0.4])


class _Actor:
    def __init__(self):
        self.step = 0
        self.actor = _Entity()

    def get_pose(self):
        return _Pose([self.step / 100.0, 0.2, 0.3])

    def get_functional_point(self, point_id, ret="list"):
        pose = _Pose([self.step / 100.0, point_id, 0.4])
        return pose if ret == "pose" else list(pose.p) + list(pose.q)

    def get_contact_point(self, point_id, ret="list"):
        pose = _Pose([self.step / 100.0, point_id, 0.5])
        return pose if ret == "pose" else list(pose.p) + list(pose.q)


class _Robot:
    def __init__(self):
        self.left_entity = _Entity(1.0)
        self.right_entity = _Entity(2.0)

    def get_left_ee_pose(self):
        return [1, 2, 3, 1, 0, 0, 0]

    def get_right_ee_pose(self):
        return [4, 5, 6, 1, 0, 0, 0]

    def get_left_tcp_pose(self):
        return [1.1, 2.1, 3.1, 1, 0, 0, 0]

    def get_right_tcp_pose(self):
        return [4.1, 5.1, 6.1, 1, 0, 0, 0]

    def get_left_gripper_val(self):
        return 0.25

    def get_right_gripper_val(self):
        return 0.75


class BalancedRecorderTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        schema_dir = self.root / "mea/toolkit/schemas"
        schema_dir.mkdir(parents=True)
        schema = {
            "schema_version": 1,
            "task_name": "generic_task",
            "physics_timestep_seconds": 0.004,
            "action_dimension": 2,
            "tracked_actors": [
                {
                    "id": "target",
                    "task_attribute": "target",
                    "scene_name": "target",
                    "functional_points": [0],
                    "contact_points": [1],
                }
            ],
            "contact_focus_actor_ids": ["target"],
            "semantic_fields": [
                {
                    "name": "target_position",
                    "source": "actor_position",
                    "actor_id": "target",
                },
                {
                    "name": "target_functional_position",
                    "source": "actor_functional_position",
                    "actor_id": "target",
                    "point_id": 0,
                },
                {
                    "name": "target_contact_position",
                    "source": "actor_contact_position",
                    "actor_id": "target",
                    "point_id": 1,
                },
                {
                    "name": "left_tcp_position",
                    "source": "robot_tcp_position",
                    "side": "left",
                },
                {
                    "name": "right_tcp_position",
                    "source": "robot_tcp_position",
                    "side": "right",
                },
            ],
        }
        (schema_dir / "generic_task.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )
        self.task = SimpleNamespace(
            target=_Actor(),
            robot=_Robot(),
            scene=SimpleNamespace(get_contacts=lambda: []),
            eval_success=False,
            check_success=lambda: False,
        )
        self.episode = self.root / "episode"

    def tearDown(self):
        self._temporary.cleanup()

    def _record(self, *, profile="balanced_v1", steps=12):
        recorder = EpisodeRecorder(
            self.root,
            self.episode,
            task_name="generic_task",
            seed=7,
            episode_index=0,
            policy_name="test",
            telemetry_profile_id=profile,
        )
        recorder.start(self.task)
        for step in range(1, steps + 1):
            self.task.target.step = step
            recorder.on_physics_step(self.task)
        return recorder.finish(self.task, success=False)

    def test_balanced_profile_adds_typed_50hz_dynamics_and_forced_final(self):
        metadata = self._record(steps=12)

        self.assertTrue((self.episode / "states.csv").is_file())
        self.assertTrue((self.episode / "semantic_trace.npz").is_file())
        self.assertTrue((self.episode / "events.jsonl").is_file())
        self.assertTrue((self.episode / "dynamics_trace.npz").is_file())
        self.assertTrue((self.episode / "telemetry_profile.json").is_file())

        with np.load(self.episode / "semantic_trace.npz") as archive:
            self.assertEqual(archive["physics_step"].tolist(), list(range(13)))
            self.assertEqual(archive["physics_step"].dtype, np.dtype("float64"))
            self.assertIn("target_position", archive.files)
            self.assertIn("target_contact_position", archive.files)

        with np.load(self.episode / "dynamics_trace.npz") as archive:
            self.assertEqual(archive["physics_step"].tolist(), [0, 5, 10, 12])
            self.assertEqual(archive["physics_step"].dtype, np.dtype("int64"))
            self.assertEqual(archive["success"].dtype, np.dtype("bool"))
            self.assertEqual(
                archive["actor.target.position"].dtype,
                np.dtype("float32"),
            )
            self.assertEqual(archive["robot.left.qpos"].shape, (4, 2))
            self.assertEqual(archive["robot.right.tcp_pose"].shape, (4, 7))
            self.assertIn("actor.target.functional.0.quaternion", archive.files)
            self.assertIn("actor.target.contact.1.position", archive.files)
            self.assertAlmostEqual(
                float(archive["actor.target.position"][-1, 0]), 0.12
            )

        profile = json.loads(
            (self.episode / "telemetry_profile.json").read_text(encoding="utf-8")
        )
        self.assertEqual(profile["profile_id"], "balanced_v1")
        self.assertEqual(metadata["telemetry_profile_id"], "balanced_v1")
        self.assertRegex(metadata["telemetry_profile_sha256"], r"^[0-9a-f]{64}$")
        dynamics_metadata = metadata["telemetry"]["streams"]["dynamics_trace"]
        self.assertEqual(dynamics_metadata["every_physics_steps"], 5)
        self.assertEqual(dynamics_metadata["rows"], 4)

        trajectory = TrajectoryView(self.episode)
        self.assertEqual(
            trajectory.dynamics["physics_step"].tolist(), [0, 5, 10, 12]
        )

    def test_final_regular_boundary_is_replaced_not_duplicated(self):
        self._record(steps=10)
        with np.load(self.episode / "dynamics_trace.npz") as archive:
            self.assertEqual(archive["physics_step"].tolist(), [0, 5, 10])

    def test_legacy_profile_preserves_old_artifacts_without_dynamics(self):
        metadata = self._record(profile="legacy_v1", steps=2)
        self.assertFalse((self.episode / "dynamics_trace.npz").exists())
        self.assertEqual(metadata["dynamics_trace_rows"], 0)
        trajectory = TrajectoryView(self.episode)
        self.assertEqual(trajectory.dynamics, {})

    def test_unknown_profile_is_rejected_before_runtime_attachment(self):
        with self.assertRaisesRegex(ValueError, "unknown telemetry profile"):
            EpisodeRecorder(
                self.root,
                self.episode,
                task_name="generic_task",
                seed=7,
                episode_index=0,
                policy_name="test",
                telemetry_profile_id="agent_supplied_python",
            )


if __name__ == "__main__":
    unittest.main()
