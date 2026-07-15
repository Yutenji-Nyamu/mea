import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mea.toolkit import (
    TOOL_CATALOG,
    EpisodeRecorder,
    TrajectoryView,
    TrustedToolRetriever,
    evaluate_telemetry_root,
    run_trusted_tools,
)


class ToolkitTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.episode_dir = self.root / "act/episode_000_seed_100000"
        self.episode_dir.mkdir(parents=True)
        self._write_episode(self.episode_dir)

    def tearDown(self):
        self._temporary.cleanup()

    @staticmethod
    def _write_episode(episode_dir):
        schema = {
            "schema_version": 1,
            "task_name": "beat_block_hammer",
            "physics_timestep_seconds": 0.004,
            "pickup_height_threshold_m": 0.03,
            "success_contract": {"xy_tolerance_m": [0.02, 0.02]},
        }
        episode = {
            "schema_version": 1,
            "task_name": "beat_block_hammer",
            "policy_name": "ACT",
            "seed": 100000,
            "episode_index": 0,
            "success": True,
            "policy_steps": 3,
            "physics_steps": 30,
        }
        (episode_dir / "schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )
        (episode_dir / "episode.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
        with (episode_dir / "states.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["phase", "policy_step", "physics_step"]
            )
            writer.writeheader()
            writer.writerow(
                {"phase": "initial", "policy_step": -1, "physics_step": 0}
            )

        physics_step = np.asarray([0, 10, 20, 30], dtype=np.int64)
        policy_step = np.asarray([-1, 0, 1, 2], dtype=np.int64)
        np.savez_compressed(
            episode_dir / "semantic_trace.npz",
            physics_step=physics_step,
            policy_step=policy_step,
            simulation_time_seconds=physics_step * 0.004,
            success=np.asarray([False, False, False, True]),
            hammer_position=np.asarray(
                [[0.0, 0.0, 0.78], [0.1, 0.01, 0.80],
                 [0.145, 0.049, 0.84], [0.151, 0.051, 0.83]],
                dtype=np.float32,
            ),
            block_position=np.asarray(
                [[0.15, 0.05, 0.76]] * 4, dtype=np.float32
            ),
            hammer_functional_position=np.asarray(
                [[0.0, 0.0, 0.78], [0.1, 0.01, 0.80],
                 [0.145, 0.049, 0.84], [0.151, 0.051, 0.83]],
                dtype=np.float32,
            ),
            block_functional_position=np.asarray(
                [[0.15, 0.05, 0.76]] * 4, dtype=np.float32
            ),
            left_tcp_position=np.asarray(
                [[0.0, 0.0, 0.0]] * 4, dtype=np.float32
            ),
            right_tcp_position=np.asarray(
                [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0],
                 [0.03, 0.04, 0.0], [0.03, 0.04, 0.12]],
                dtype=np.float32,
            ),
        )
        events = [
            {
                "type": "contact_interval",
                "actors": ["020_hammer", "box"],
                "physical_contact": True,
                "first_physical_policy_step": 1,
                "first_physical_physics_step": 20,
                "first_physical_simulation_time_seconds": 0.08,
                "max_impulse": 0.42,
                "min_separation": -0.001,
                "peak_policy_step": 1,
                "peak_physics_step": 20,
            },
            {
                "type": "success_transition",
                "policy_step": 2,
                "physics_step": 30,
                "simulation_time_seconds": 0.12,
                "video_frame_index": 2,
            },
        ]
        (episode_dir / "events.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in events),
            encoding="utf-8",
        )

    def test_all_eight_trusted_tools_on_synthetic_trajectory(self):
        trajectory = TrajectoryView(self.episode_dir)
        results = run_trusted_tools(trajectory, list(TOOL_CATALOG))
        by_name = {item["tool"]: item for item in results}

        self.assertEqual(set(by_name), set(TOOL_CATALOG))
        self.assertAlmostEqual(by_name["hammer_pickup_height"]["value"], 0.06)
        self.assertTrue(by_name["hammer_pickup_height"]["passed"])
        self.assertAlmostEqual(
            by_name["hammer_block_min_xy_error"]["value"], 0.001,
            places=5,
        )
        self.assertTrue(by_name["hammer_block_contact_ever"]["value"])
        self.assertEqual(by_name["first_contact_step"]["value"], 20)
        self.assertAlmostEqual(by_name["max_contact_impulse"]["value"], 0.42)
        self.assertAlmostEqual(by_name["ee_path_length"]["value"], 0.19)
        self.assertTrue(by_name["official_check_success"]["value"])
        self.assertAlmostEqual(by_name["time_to_success"]["value"], 0.12)

        for result in results:
            self.assertRegex(result["tool_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("evidence", result)
        self.assertEqual(
            by_name["hammer_block_contact_ever"]["evidence"][0][
                "physics_step"
            ],
            20,
        )

    def test_tool_retriever_and_runner_write_auditable_results(self):
        selection = TrustedToolRetriever().select(
            "Report contact force, active-arm path, and time to success.",
            task_name="beat_block_hammer",
        )
        self.assertEqual(selection["selected_tools"], list(TOOL_CATALOG))

        summary = evaluate_telemetry_root(
            self.root,
            user_request=(
                "Report contact force, active-arm path, and time to success."
            ),
        )
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(
            len(summary["episodes"][0]["tool_results"]), len(TOOL_CATALOG)
        )
        self.assertTrue((self.root / "tool_results.json").is_file())
        self.assertTrue((self.episode_dir / "tool_results.json").is_file())
        self.assertEqual(
            set(summary["episodes"][0]["artifact_sha256"]),
            {"episode.json", "states.csv", "semantic_trace.npz", "events.jsonl"},
        )

    def test_contact_samples_keep_physical_evidence_and_peak(self):
        def point(impulse, separation, position, normal):
            return SimpleNamespace(
                impulse=impulse,
                separation=separation,
                position=position,
                normal=normal,
            )

        hammer_body = SimpleNamespace(
            entity=SimpleNamespace(get_name=lambda: "020_hammer")
        )
        block_body = SimpleNamespace(
            entity=SimpleNamespace(get_name=lambda: "box")
        )
        contacts = [
            SimpleNamespace(
                bodies=[hammer_body, block_body],
                points=[point([0.3, 0.4, 0.0], -0.002, [1, 2, 3], [0, 0, 1])],
            ),
            SimpleNamespace(
                bodies=[block_body, hammer_body],
                points=[point([0.8, 0.0, 0.0], 0.001, [4, 5, 6], [1, 0, 0])],
            ),
        ]
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.schema = {
            "tracked_actors": [
                {"id": "hammer", "scene_name": "020_hammer"},
                {"id": "block", "scene_name": "box"},
            ],
            "contact_focus_actor_ids": ["hammer", "block"],
        }
        task = SimpleNamespace(
            scene=SimpleNamespace(get_contacts=lambda: contacts)
        )

        sample = recorder._contact_samples(task)[("020_hammer", "box")]
        self.assertEqual(sample["point_count"], 2)
        self.assertAlmostEqual(sample["max_impulse"], 0.8)
        self.assertAlmostEqual(sample["min_separation"], -0.002)
        self.assertTrue(sample["physical_contact"])
        self.assertEqual(sample["peak_position"], [4.0, 5.0, 6.0])
        self.assertEqual(sample["peak_normal"], [1.0, 0.0, 0.0])

    def test_contact_interval_updates_and_closes_without_state_shadowing(self):
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.active_contacts = {}
        recorder.events = []
        recorder.policy_step = 3
        recorder.physics_step = 10
        recorder.physics_dt = 0.004
        pair = ("020_hammer", "box")
        samples = iter(
            [
                {
                    pair: {
                        "point_count": 1,
                        "max_impulse": 0.1,
                        "min_separation": 0.001,
                        "physical_contact": False,
                        "peak_position": [1.0, 2.0, 3.0],
                        "peak_normal": [0.0, 0.0, 1.0],
                    }
                },
                {
                    pair: {
                        "point_count": 2,
                        "max_impulse": 0.5,
                        "min_separation": -0.002,
                        "physical_contact": True,
                        "peak_position": [4.0, 5.0, 6.0],
                        "peak_normal": [1.0, 0.0, 0.0],
                    }
                },
                {},
            ]
        )
        recorder._contact_samples = lambda task: next(samples)
        task = SimpleNamespace()

        recorder._update_contact_events(task)
        recorder.physics_step = 11
        recorder._update_contact_events(task)
        recorder.physics_step = 12
        recorder._update_contact_events(task)

        self.assertFalse(recorder.active_contacts)
        self.assertEqual(len(recorder.events), 1)
        interval = recorder.events[0]
        self.assertTrue(interval["physical_contact"])
        self.assertEqual(interval["first_physical_physics_step"], 11)
        self.assertAlmostEqual(interval["min_separation"], -0.002)
        self.assertAlmostEqual(interval["max_impulse"], 0.5)
        self.assertEqual(interval["end_physics_step"], 12)


if __name__ == "__main__":
    unittest.main()
