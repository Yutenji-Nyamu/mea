import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mea.execution_receipt import (
    build_checkpoint_bundle,
    canonical_sha256,
    file_sha256,
    seal_execution_receipt,
    validate_recorded_execution_metadata,
)
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


class _Task(SimpleNamespace):
    pass


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
        self.task = _Task(
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

    def test_execution_receipt_records_actual_imported_source_binding(self):
        module_origin = Path(__file__).resolve()
        identity = {
            "task_name": "generic_task",
            "task_class": "_Task",
            "task_module": _Task.__module__,
            "task_source_sha256": file_sha256(module_origin),
            "proposal_sha256": "1" * 64,
            "scene_method_sha256": "2" * 64,
            "success_method_sha256": "3" * 64,
        }
        receipt = seal_execution_receipt(
            {
                "schema_version": 1,
                "receipt_type": "mea_task_execution_preflight",
                "candidate": {
                    **identity,
                    "module_origin": str(module_origin),
                    "candidate_manifest_sha256": "4" * 64,
                    "execution_module_sha256": canonical_sha256(identity),
                },
                "episode": {
                    "task_name": "generic_task",
                    "task_module": _Task.__module__,
                    "task_config": "demo_clean",
                    "checkpoint_setting": "demo_clean",
                    "policy_name": "expert",
                    "seed": 7,
                    "episode_index": 0,
                },
                "checkpoint": build_checkpoint_bundle(
                    None,
                    kind="expert_no_checkpoint",
                ),
            }
        )
        recorder = EpisodeRecorder(
            self.root,
            self.episode,
            task_name="generic_task",
            seed=7,
            episode_index=0,
            policy_name="expert",
            task_module=_Task.__module__,
            task_config="demo_clean",
            checkpoint_setting="demo_clean",
            execution_receipt=receipt,
        )
        receipt["candidate"]["proposal_sha256"] = "f" * 64
        recorder.start(self.task)
        metadata = recorder.finish(self.task, success=False)
        self.assertEqual(
            metadata["executed_task_module_sha256"],
            identity["task_source_sha256"],
        )
        self.assertEqual(
            metadata["execution_receipt"]["candidate"]["proposal_sha256"],
            "1" * 64,
        )
        validate_recorded_execution_metadata(
            metadata,
            metadata["execution_receipt"],
        )
        self.assertEqual(
            metadata["artifacts"]["execution_receipt"],
            "execution_receipt.json",
        )

    def test_event_keyframes_dedupe_contact_and_success_on_same_step(self):
        success = {"value": False}
        contacts = {"value": []}
        self.task.check_success = lambda: success["value"]
        self.task.scene.get_contacts = lambda: contacts["value"]

        def save_camera_rgb(path, *, camera_name):
            self.assertEqual(camera_name, "head_camera")
            Path(path).write_bytes(b"png")

        self.task.save_camera_rgb = save_camera_rgb
        target_body = SimpleNamespace(entity=SimpleNamespace(name="target"))
        gripper_body = SimpleNamespace(entity=SimpleNamespace(name="gripper"))
        table_body = SimpleNamespace(entity=SimpleNamespace(name="table"))
        contact_point = SimpleNamespace(
            impulse=np.asarray([0.1, 0.0, 0.0]),
            separation=-0.001,
            position=np.asarray([0.0, 0.0, 0.0]),
            normal=np.asarray([0.0, 0.0, 1.0]),
        )
        contact = SimpleNamespace(
            bodies=[target_body, gripper_body],
            points=[contact_point],
        )
        support_contact = SimpleNamespace(
            bodies=[target_body, table_body],
            points=[contact_point],
        )
        contacts["value"] = [support_contact]
        recorder = EpisodeRecorder(
            self.root,
            self.episode,
            task_name="generic_task",
            seed=7,
            episode_index=0,
            policy_name="expert",
            visual_capture_profile_id="event_keyframes_v1",
        )

        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b"mp4")
            return SimpleNamespace(returncode=0, stderr="")

        with patch("mea.toolkit.recorder.subprocess.run", side_effect=encode):
            recorder.start(self.task)
            contacts["value"] = [support_contact, contact]
            success["value"] = True
            recorder.on_physics_step(self.task)
            contacts["value"] = []
            recorder.on_physics_step(self.task)
            metadata = recorder.finish(self.task, success=True)

        manifest = json.loads(
            (self.episode / "visual_keyframes.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["frame_count"], 3)
        self.assertEqual(manifest["frames"][0]["reasons"], ["initial"])
        self.assertEqual(
            set(manifest["frames"][1]["reasons"]),
            {"success_transition", "first_physical_contact"},
        )
        self.assertEqual(manifest["frames"][2]["reasons"], ["final"])
        self.assertTrue((self.episode / "video.mp4").is_file())
        self.assertEqual(metadata["artifacts"]["video"], "video.mp4")
        self.assertEqual(metadata["video_alignment"]["mode"], "event_keyframes")

        events = [
            json.loads(line)
            for line in (self.episode / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        success_event = next(
            item for item in events if item["type"] == "success_transition"
        )
        contact_event = next(
            item
            for item in events
            if item["type"] == "contact_interval"
            and "gripper" in item["actors"]
        )
        support_event = next(
            item
            for item in events
            if item["type"] == "contact_interval"
            and "table" in item["actors"]
        )
        self.assertEqual(success_event["video_frame_index"], 1)
        self.assertEqual(contact_event["first_physical_video_frame_index"], 1)
        self.assertIsNone(support_event["first_physical_video_frame_index"])
        with np.load(self.episode / "semantic_trace.npz") as archive:
            self.assertEqual(archive["video_frame_index"].tolist(), [0, 1, 1])

    def test_visual_capture_failure_preserves_numeric_telemetry(self):
        def fail_capture(path, *, camera_name):
            raise RuntimeError("camera unavailable")

        self.task.save_camera_rgb = fail_capture
        stale_keyframes = self.episode / "visual_keyframes"
        stale_keyframes.mkdir(parents=True)
        (stale_keyframes / "frame_099.png").write_bytes(b"stale")
        (self.episode / "visual_keyframes.json").write_text(
            "{}", encoding="utf-8"
        )
        (self.episode / "video.mp4").write_bytes(b"stale")
        recorder = EpisodeRecorder(
            self.root,
            self.episode,
            task_name="generic_task",
            seed=7,
            episode_index=0,
            policy_name="expert",
            visual_capture_profile_id="event_keyframes_v1",
        )
        recorder.start(self.task)
        recorder.on_physics_step(self.task)
        metadata = recorder.finish(self.task, success=False)

        self.assertEqual(metadata["visual_capture"]["status"], "failed")
        self.assertNotIn("video", metadata["artifacts"])
        self.assertFalse((self.episode / "video.mp4").exists())
        self.assertFalse((stale_keyframes / "frame_099.png").exists())
        self.assertTrue((self.episode / "states.csv").is_file())
        self.assertTrue((self.episode / "semantic_trace.npz").is_file())
        self.assertTrue((self.episode / "events.jsonl").is_file())

    def test_ffmpeg_failure_preserves_keyframes_and_numeric_telemetry(self):
        self.task.save_camera_rgb = lambda path, *, camera_name: Path(
            path
        ).write_bytes(b"png")
        recorder = EpisodeRecorder(
            self.root,
            self.episode,
            task_name="generic_task",
            seed=7,
            episode_index=0,
            policy_name="expert",
            visual_capture_profile_id="event_keyframes_v1",
        )
        with patch(
            "mea.toolkit.recorder.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stderr="encoder failed"),
        ):
            recorder.start(self.task)
            metadata = recorder.finish(self.task, success=False)

        self.assertEqual(metadata["visual_capture"]["status"], "failed")
        self.assertEqual(metadata["visual_capture"]["frame_count"], 1)
        self.assertNotIn("video", metadata["artifacts"])
        self.assertTrue((self.episode / "visual_keyframes/frame_000.png").is_file())
        self.assertTrue((self.episode / "visual_keyframes.json").is_file())
        self.assertTrue((self.episode / "dynamics_trace.npz").is_file())

    def test_visual_manifest_failure_cannot_abort_numeric_telemetry(self):
        self.task.save_camera_rgb = lambda path, *, camera_name: Path(
            path
        ).write_bytes(b"png")
        recorder = EpisodeRecorder(
            self.root,
            self.episode,
            task_name="generic_task",
            seed=7,
            episode_index=0,
            policy_name="expert",
            visual_capture_profile_id="event_keyframes_v1",
        )

        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b"mp4")
            return SimpleNamespace(returncode=0, stderr="")

        recorder.start(self.task)
        with patch(
            "mea.toolkit.recorder.subprocess.run", side_effect=encode
        ), patch.object(
            EpisodeRecorder,
            "_write_visual_manifest",
            side_effect=OSError("manifest unavailable"),
        ):
            metadata = recorder.finish(self.task, success=False)

        self.assertEqual(metadata["visual_capture"]["status"], "failed")
        self.assertEqual(
            metadata["visual_capture"]["errors"][-1]["stage"], "finalize"
        )
        self.assertNotIn("visual_keyframes", metadata["artifacts"])
        self.assertNotIn("video", metadata["artifacts"])
        self.assertFalse((self.episode / "video.mp4").exists())
        self.assertFalse((self.episode / "visual_keyframes.json").exists())
        self.assertTrue((self.episode / "states.csv").is_file())
        self.assertTrue((self.episode / "semantic_trace.npz").is_file())
        self.assertTrue((self.episode / "events.jsonl").is_file())
        self.assertTrue((self.episode / "episode.json").is_file())

    def test_unknown_visual_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown visual capture profile"):
            EpisodeRecorder(
                self.root,
                self.episode,
                task_name="generic_task",
                seed=7,
                episode_index=0,
                policy_name="expert",
                visual_capture_profile_id="model_supplied_capture_code",
            )


if __name__ == "__main__":
    unittest.main()
