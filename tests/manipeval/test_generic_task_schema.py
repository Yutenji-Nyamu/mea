import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mea.toolkit import (
    TaskSchemaError,
    TrajectoryView,
    TrustedToolRetriever,
    evaluate_telemetry_root,
    list_task_schemas,
    load_task_schema,
    required_trace_keys,
    run_trusted_tools,
    validate_task_schema,
)
from mea.toolkit.tools import TrajectoryError
from mea.toolgen import execute_tool_request, official_success_tool_request
from mea.toolgen.router import route_tool_request


REPO_ROOT = Path(__file__).resolve().parents[2]


class GenericTaskSchemaTests(unittest.TestCase):
    def test_registry_discovers_all_onboarded_tasks(self):
        summaries = {item["task_name"]: item for item in list_task_schemas(REPO_ROOT)}
        self.assertIn("beat_block_hammer", summaries)
        self.assertIn("click_bell", summaries)
        self.assertIn("adjust_bottle", summaries)
        self.assertIn("grab_roller", summaries)
        self.assertEqual(summaries["click_bell"]["tracked_actor_ids"], ["bell"])
        self.assertEqual(
            summaries["click_bell"]["trusted_tool_profile"],
            "generic_success",
        )
        self.assertEqual(summaries["adjust_bottle"]["tracked_actor_ids"], ["bottle"])
        self.assertEqual(summaries["grab_roller"]["tracked_actor_ids"], ["roller"])

    def test_new_task_schemas_use_only_generic_recorder_sources(self):
        adjust = load_task_schema(REPO_ROOT, "adjust_bottle")
        roller = load_task_schema(REPO_ROOT, "grab_roller")
        self.assertEqual(adjust["trusted_tool_profile"], "generic_success")
        self.assertEqual(adjust["tracked_actors"][0]["functional_points"], [0])
        self.assertEqual(roller["trusted_tool_profile"], "generic_success")
        self.assertEqual(roller["tracked_actors"][0]["contact_points"], [0, 1])
        self.assertIn("bottle_functional_position", required_trace_keys(adjust))
        self.assertIn("roller_left_contact_position", required_trace_keys(roller))

    def test_click_bell_declares_exact_trace_contract(self):
        schema = load_task_schema(REPO_ROOT, "click_bell")
        self.assertEqual(schema["tracked_actors"][0]["task_attribute"], "bell")
        self.assertEqual(schema["tracked_actors"][0]["contact_points"], [0])
        self.assertEqual(
            required_trace_keys(schema),
            {
                "physics_step",
                "policy_step",
                "simulation_time_seconds",
                "success",
                "bell_position",
                "bell_contact_position",
                "left_tcp_position",
                "right_tcp_position",
            },
        )

    def test_validator_rejects_undeclared_point_and_reserved_name(self):
        schema = load_task_schema(REPO_ROOT, "click_bell")
        bad_point = json.loads(json.dumps(schema))
        bad_point["semantic_fields"][1]["point_id"] = 99
        with self.assertRaisesRegex(TaskSchemaError, "point_id"):
            validate_task_schema(bad_point, expected_task_name="click_bell")

        reserved = json.loads(json.dumps(schema))
        reserved["semantic_fields"][0]["name"] = "physics_step"
        with self.assertRaisesRegex(TaskSchemaError, "保留字段"):
            validate_task_schema(reserved, expected_task_name="click_bell")


class GenericTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.episode = Path(self.temporary.name)
        schema = load_task_schema(REPO_ROOT, "click_bell")
        (self.episode / "schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )
        metadata = {
            "schema_version": 1,
            "task_name": "click_bell",
            "policy_name": "expert",
            "seed": 100000,
            "episode_index": 0,
            "success": True,
            "policy_steps": 2,
            "physics_steps": 10,
            "semantic_trace_rows": 3,
        }
        (self.episode / "episode.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with (self.episode / "states.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["phase", "physics_step"])
            writer.writeheader()
            writer.writerow({"phase": "initial", "physics_step": 0})
        step = np.asarray([0, 5, 10], dtype=np.int64)
        position = np.asarray([[0.1, -0.1, 0.76]] * 3, dtype=np.float32)
        np.savez_compressed(
            self.episode / "semantic_trace.npz",
            physics_step=step,
            policy_step=np.asarray([-1, 0, 1], dtype=np.int64),
            simulation_time_seconds=step * 0.004,
            success=np.asarray([False, False, True]),
            bell_position=position,
            bell_contact_position=np.asarray(
                [[0.1, -0.1, 0.82]] * 3, dtype=np.float32
            ),
            left_tcp_position=np.zeros((3, 3), dtype=np.float32),
            right_tcp_position=np.asarray(
                [[0.1, -0.1, 0.95], [0.1, -0.1, 0.84], [0.1, -0.1, 0.82]],
                dtype=np.float32,
            ),
        )
        events = {
            "type": "success_transition",
            "policy_step": 1,
            "physics_step": 10,
            "simulation_time_seconds": 0.04,
            "video_frame_index": 1,
        }
        (self.episode / "events.jsonl").write_text(
            json.dumps(events) + "\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_generic_retrieval_and_tools_run_on_click_bell(self):
        selection = TrustedToolRetriever(REPO_ROOT).select(
            "评估是否成功按响铃以及成功耗时",
            task_name="click_bell",
        )
        self.assertEqual(
            selection["selected_tools"],
            ["official_check_success", "time_to_success"],
        )
        trajectory = TrajectoryView(self.episode)
        results = run_trusted_tools(trajectory, selection["selected_tools"])
        self.assertTrue(results[0]["value"])
        self.assertAlmostEqual(results[1]["value"], 0.04)
        summary = evaluate_telemetry_root(
            self.episode,
            user_request="评估是否成功按响铃以及成功耗时",
            task_name="click_bell",
        )
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(
            [item["tool"] for item in summary["episodes"][0]["tool_results"]],
            ["official_check_success", "time_to_success"],
        )

    def test_auto_tool_router_reuses_generic_success_for_click_bell(self):
        request = official_success_tool_request("click_bell")
        routing = route_tool_request(request)
        self.assertEqual(routing["route_decision"]["resolved_route"], "reuse")

        child_run = self.episode / "child_run"
        runtime_episode = (
            child_run / "evaluation/telemetry/expert/episode_000_seed_100000"
        )
        runtime_episode.mkdir(parents=True)
        for name in (
            "schema.json",
            "episode.json",
            "states.csv",
            "semantic_trace.npz",
            "events.jsonl",
        ):
            shutil.copy2(self.episode / name, runtime_episode / name)
        result = execute_tool_request(
            REPO_ROOT,
            child_run,
            self.episode / "tool_execution",
            request,
        )
        self.assertEqual(result["route"], "reuse")
        self.assertEqual(result["tool_spec"]["task_name"], "click_bell")
        self.assertEqual(result["episodes"][0]["role"], "expert_validation")
        self.assertTrue(result["episodes"][0]["result"]["value"])

    def test_auto_tool_router_rejects_bbh_only_metric_for_click_bell(self):
        routing = route_tool_request(
            {
                "schema_version": 1,
                "task_name": "click_bell",
                "metric": "hammer_block_contact_ever",
                "question": "Did the hammer contact the block?",
            }
        )
        self.assertEqual(routing["route_decision"]["status"], "unsupported")
        self.assertIsNone(routing["route_decision"]["resolved_route"])

    def test_task_specific_tool_is_rejected_for_click_bell(self):
        trajectory = TrajectoryView(self.episode)
        with self.assertRaisesRegex(TrajectoryError, "不兼容"):
            run_trusted_tools(trajectory, ["hammer_pickup_height"])

    def test_runner_rejects_mixed_requested_task(self):
        with self.assertRaisesRegex(RuntimeError, "混入其他任务"):
            evaluate_telemetry_root(
                self.episode,
                user_request="评估任务结果",
                task_name="beat_block_hammer",
            )

    def test_metadata_schema_task_mismatch_is_rejected(self):
        metadata = json.loads(
            (self.episode / "episode.json").read_text(encoding="utf-8")
        )
        metadata["task_name"] = "beat_block_hammer"
        (self.episode / "episode.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with self.assertRaisesRegex(TrajectoryError, "task_name"):
            TrajectoryView(self.episode)


class ProbeSchemaHelperTests(unittest.TestCase):
    def test_click_bell_actor_summary_and_rule_check(self):
        from mea.taskgen.probe import (
            task_schema_rule_check,
            tracked_actor_summary,
        )

        pose = SimpleNamespace(
            p=np.asarray([0.1, -0.1, 0.76]),
            q=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        bell = SimpleNamespace(
            get_pose=lambda: pose,
            get_contact_point=lambda point_id: np.asarray(
                [0.1, -0.1, 0.82, 1.0, 0.0, 0.0, 0.0]
            ),
        )
        task = SimpleNamespace(bell=bell, check_success=lambda: False)
        schema = load_task_schema(REPO_ROOT, "click_bell")
        tracked = tracked_actor_summary(task, schema)
        self.assertEqual(tracked[0]["id"], "bell")
        self.assertEqual(
            tracked[0]["contact_points"]["0"]["position"],
            [0.1, -0.1, 0.82],
        )
        result = task_schema_rule_check(
            task,
            schema,
            scene_actors=[{"name": "050_bell"}],
            tracked_actors=tracked,
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
