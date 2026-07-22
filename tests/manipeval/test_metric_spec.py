import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen import (
    MetricSpecError,
    compile_metric_spec_source,
    evaluate_metric_spec,
    execute_tool_request,
    execute_metric_spec,
    metric_spec_tool_spec,
    route_tool_request,
    validate_generated_tool,
    validate_metric_spec,
)
from mea.toolkit.tools import TrajectoryView
from mea.proposals import tool_request_from_proposal, validate_tool_proposal
from tests.manipeval.test_tool_orchestration import write_episode


SPEC = {
    "schema_version": 1,
    "operation": "minimum_distance",
    "left_signal": "right_tcp_position",
    "right_signal": "block_position",
    "dimensions": ["x", "y"],
    "unit": "m",
    "null_semantics": "null_if_no_finite_sample",
}
CONTACT_SELECTOR = {
    "event_type": "contact_interval",
    "actors": ["020_hammer", "box"],
    "physical_only": True,
}
SUCCESS_SELECTOR = {
    "event_type": "success_transition",
    "actors": None,
    "physical_only": False,
}
EVENT_COUNT_SPEC = {
    "schema_version": 1,
    "operation": "event_count",
    "event": CONTACT_SELECTOR,
    "unit": "count",
    "null_semantics": "zero_if_absent",
}
TIME_BETWEEN_EVENTS_SPEC = {
    "schema_version": 1,
    "operation": "time_between_events",
    "start_event": CONTACT_SELECTOR,
    "end_event": SUCCESS_SELECTOR,
    "unit": "s",
    "null_semantics": "null_if_missing_or_reversed",
}


class MetricSpecTests(unittest.TestCase):
    def test_strict_validation_compilation_and_router(self):
        self.assertEqual(validate_metric_spec(SPEC), SPEC)
        source = compile_metric_spec_source(SPEC)
        self.assertTrue(validate_generated_tool(source)["valid"])
        tool_spec = metric_spec_tool_spec(
            task_name="beat_block_hammer",
            metric="query_min_tcp_block_xy",
            question="How close did the TCP get to the block?",
            metric_spec=SPEC,
        )
        self.assertEqual(
            tool_spec["validation_requirements"],
            {
                "min_episodes": 1,
                "distinct_reference_values": False,
                "required_reference_values": [],
            },
        )
        routing = route_tool_request(
            {
                "schema_version": 2,
                "task_name": "beat_block_hammer",
                "metric": "query_min_tcp_block_xy",
                "question": tool_spec["question"],
                "metric_spec": SPEC,
            }
        )
        self.assertEqual(
            routing["route_decision"]["resolved_route"],
            "typed_metric_spec_compile",
        )
        self.assertFalse(routing["route_decision"]["provider_required"])
        self.assertEqual(
            routing["catalog_snapshot"]["typed_metric_spec"]["operations"],
            ["event_count", "minimum_distance", "time_between_events"],
        )

    def test_unbounded_operator_and_registry_collision_are_rejected(self):
        with self.assertRaisesRegex(MetricSpecError, "operation"):
            validate_metric_spec({**SPEC, "operation": "eval_python"})
        with self.assertRaisesRegex(MetricSpecError, "operation"):
            validate_metric_spec({**SPEC, "operation": ["minimum_distance"]})
        with self.assertRaisesRegex(RuntimeError, "cannot override"):
            route_tool_request(
                {
                    "schema_version": 2,
                    "task_name": "beat_block_hammer",
                    "metric": "official_check_success",
                    "question": "Override it?",
                    "metric_spec": SPEC,
                }
            )

        invalid_selector = {
            **EVENT_COUNT_SPEC,
            "event": {**CONTACT_SELECTOR, "actors": ["../hammer", "box"]},
        }
        with self.assertRaisesRegex(MetricSpecError, "actor ids"):
            validate_metric_spec(invalid_selector)

    def test_tool_proposal_v3_carries_the_typed_metric(self):
        proposal = validate_tool_proposal(
            {
                "schema_version": 3,
                "proposal_id": "query_metric.tool",
                "task_name": "beat_block_hammer",
                "aspect_id": "object_appearance.color",
                "evaluation_goal": "Measure query-specific geometric progress.",
                "metric": "query_min_tcp_block_xy",
                "question": "How close did the TCP get to the block?",
                "vqa_phenomenon_ids": [
                    "block_visibly_displaced",
                    "run_local.bbh.query_metric",
                ],
                "vqa_question_specs": [
                    {
                        "id": "run_local.bbh.query_metric",
                        "question_type": "visible_state_change",
                        "target_role": "task_target",
                        "question": "Does the rollout visibly show task-relevant contact progress?",
                        "visual_scope": "rollout_change",
                        "numeric_authority": "official_check_success_is_authoritative",
                    }
                ],
                "reuse_first": True,
                "metric_spec": SPEC,
            }
        )
        request = tool_request_from_proposal(proposal)
        self.assertEqual(request["schema_version"], 2)
        self.assertEqual(request["metric_spec"], SPEC)

    def test_agent_tool_boundary_executes_v3_metric_on_cached_telemetry(self):
        """The normal Proposal -> Router -> Orchestrator path accepts v3."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "generated_tasks/round_1"
            (child / "evaluation/telemetry/act").mkdir(parents=True)
            (child / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "task_name": "beat_block_hammer",
                        "task_module": "beat_block_hammer",
                        "generation_kind": "generated",
                    }
                ),
                encoding="utf-8",
            )
            act = child / "evaluation/telemetry/act/episode_000_seed_100000"
            expert = child / "evaluation/telemetry/expert/episode_000_seed_100000"
            write_episode(act, policy_name="ACT", physical_contact=False)
            write_episode(expert, policy_name="expert", physical_contact=True)
            proposal = validate_tool_proposal(
                {
                    "schema_version": 3,
                    "proposal_id": "query_contact_count.tool",
                    "task_name": "beat_block_hammer",
                    "aspect_id": "performance.pickup_to_contact_timing",
                    "evaluation_goal": "Count strict task contact intervals.",
                    "metric": "query_hammer_block_contact_count",
                    "question": "How many physical task contacts occurred?",
                    "vqa_phenomenon_ids": [
                        "block_visibly_displaced",
                        "run_local.bbh.contact_count",
                    ],
                    "vqa_question_specs": [
                        {
                            "id": "run_local.bbh.contact_count",
                            "question_type": "visible_state_change",
                            "target_role": "task_target",
                            "question": "Does the rollout show task-relevant contact?",
                            "visual_scope": "rollout_change",
                            "numeric_authority": "official_check_success_is_authoritative",
                        }
                    ],
                    "reuse_first": True,
                    "metric_spec": EVENT_COUNT_SPEC,
                }
            )
            request = tool_request_from_proposal(proposal)
            self.assertEqual(
                route_tool_request(request)["route_decision"]["resolved_route"],
                "typed_metric_spec_compile",
            )

            output = root / "evaluation/execution/round_1/planned_tool"
            result = execute_tool_request(
                Path(__file__).resolve().parents[2],
                child,
                output,
                request,
                task_proposal={"proposal_id": "round_1.task"},
            )
            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertFalse(result["validation"]["provider_called"])
            self.assertTrue(result["validation"]["task_code_context_consumed"])
            self.assertEqual(
                [item["result"]["value"] for item in result["episodes"]],
                [0, 1],
            )
            self.assertTrue((output / "tool_execution.json").is_file())

            paraphrase = {**request, "question": "Count strict contact intervals."}
            replay = execute_tool_request(
                Path(__file__).resolve().parents[2],
                child,
                root / "evaluation/execution/round_2/planned_tool",
                paraphrase,
                task_proposal={"proposal_id": "round_1.task"},
            )
            self.assertEqual(replay["route"], "run_local_reuse")
            self.assertEqual(
                replay["route_decision"]["matched_registry"],
                "evaluation_local_tool_registry",
            )

    def test_compile_validate_register_and_semantic_question_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "episode_a"
            second = root / "episode_b"
            write_episode(first, policy_name="ACT", physical_contact=False)
            write_episode(second, policy_name="expert", physical_contact=True)
            payload = dict(np.load(second / "semantic_trace.npz"))
            payload["right_tcp_position"] = (
                np.asarray(payload["right_tcp_position"]) + [0.08, 0.0, 0.0]
            )
            np.savez_compressed(second / "semantic_trace.npz", **payload)
            oracle_a = evaluate_metric_spec(SPEC, TrajectoryView(first))
            oracle_b = evaluate_metric_spec(SPEC, TrajectoryView(second))
            self.assertNotEqual(oracle_a["value"], oracle_b["value"])

            registry = root / "registry"
            result = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_min_tcp_block_xy",
                question="How close did the TCP get to the block?",
                metric_spec=SPEC,
                episode_dirs=[first, second],
                output_dir=root / "first",
                task_code_context={
                    "schema_version": 1,
                    "task_name": "beat_block_hammer",
                },
                registry_dir=registry,
            )
            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertFalse(result["provider_called"])
            self.assertTrue(result["task_code_context_consumed"])
            self.assertEqual(result["registration"]["scope"], "run_local")

            replay = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_min_tcp_block_xy",
                question="What was the minimum XY separation?",
                metric_spec=SPEC,
                episode_dirs=[first, second],
                output_dir=root / "second",
                registry_dir=registry,
            )
            self.assertEqual(replay["route"], "run_local_reuse")
            self.assertFalse(replay["provider_called"])
            self.assertEqual(
                replay["registration"]["registration_id"],
                result["registration"]["registration_id"],
            )

    def test_event_count_compiles_and_differentially_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "episode_no_contact"
            second = root / "episode_contact"
            write_episode(first, policy_name="ACT", physical_contact=False)
            write_episode(second, policy_name="expert", physical_contact=True)

            self.assertEqual(
                evaluate_metric_spec(
                    EVENT_COUNT_SPEC, TrajectoryView(first)
                )["value"],
                0,
            )
            self.assertEqual(
                evaluate_metric_spec(
                    EVENT_COUNT_SPEC, TrajectoryView(second)
                )["value"],
                1,
            )
            source = compile_metric_spec_source(EVENT_COUNT_SPEC)
            self.assertTrue(validate_generated_tool(source)["valid"])
            registry = root / "registry"
            result = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_hammer_block_contact_count",
                question=(
                    "How many physical hammer-block contact intervals occurred?"
                ),
                metric_spec=EVENT_COUNT_SPEC,
                episode_dirs=[first, second],
                output_dir=root / "event_count",
                registry_dir=registry,
            )
            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertEqual(
                [
                    item["oracle_projection"]["value"]
                    for item in result["episodes"]
                ],
                [0, 1],
            )
            replay = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_hammer_block_contact_count",
                question=(
                    "Count physical contact intervals between the task actors."
                ),
                metric_spec=EVENT_COUNT_SPEC,
                episode_dirs=[first, second],
                output_dir=root / "event_count_reuse",
                registry_dir=registry,
            )
            self.assertEqual(replay["route"], "run_local_reuse")
            self.assertEqual(
                replay["registration"]["registration_id"],
                result["registration"]["registration_id"],
            )

    def test_single_safe_rollout_can_validate_a_zero_event_metric(self):
        """A correct zero count must not require an artificial live collision."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "safe_act"
            write_episode(episode, policy_name="ACT", physical_contact=False)
            camera_spec = {
                **EVENT_COUNT_SPEC,
                "event": {
                    **CONTACT_SELECTOR,
                    "actors": ["020_hammer", "left_camera"],
                },
            }
            result = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_hammer_left_camera_contact_count",
                question="How many hammer-left_camera contacts occurred?",
                metric_spec=camera_spec,
                episode_dirs=[episode],
                output_dir=root / "single_zero",
                registry_dir=root / "registry",
            )
            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertEqual(result["episodes"][0]["generated_result"]["value"], 0)
            self.assertEqual(
                result["registration"]["scope"], "run_local"
            )

    def test_missing_trace_signal_is_reported_as_metric_spec_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "act"
            write_episode(episode, policy_name="ACT", physical_contact=False)
            invalid = {**SPEC, "left_signal": "invented_position"}
            with self.assertRaisesRegex(
                MetricSpecError, "absent from TaskSchema telemetry"
            ):
                execute_metric_spec(
                    task_name="beat_block_hammer",
                    metric="query_invalid_signal",
                    question="Use an unavailable signal?",
                    metric_spec=invalid,
                    episode_dirs=[episode],
                    output_dir=root / "invalid",
                )

    def test_time_between_events_compiles_and_handles_missing_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "episode_short"
            second = root / "episode_long"
            missing = root / "episode_missing"
            for path in (first, second):
                write_episode(path, policy_name="expert", physical_contact=True)
            write_episode(missing, policy_name="ACT", physical_contact=False)

            for path, contact_step in ((first, 1), (second, 0)):
                events_path = path / "events.jsonl"
                contact = json.loads(events_path.read_text(encoding="utf-8"))
                contact["first_physical_physics_step"] = contact_step
                contact["first_physical_simulation_time_seconds"] = (
                    contact_step * 0.004
                )
                success = {
                    "type": "success_transition",
                    "policy_step": 0,
                    "physics_step": 2,
                    "simulation_time_seconds": 0.008,
                    "video_frame_index": 0,
                }
                events_path.write_text(
                    json.dumps(contact) + "\n" + json.dumps(success) + "\n",
                    encoding="utf-8",
                )

            source = compile_metric_spec_source(TIME_BETWEEN_EVENTS_SPEC)
            self.assertTrue(validate_generated_tool(source)["valid"])
            missing_result = evaluate_metric_spec(
                TIME_BETWEEN_EVENTS_SPEC, TrajectoryView(missing)
            )
            self.assertIsNone(missing_result["value"])
            self.assertEqual(
                missing_result["details"]["reason"], "start_event_missing"
            )

            result = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_contact_to_success_time",
                question="How long passed from contact to success?",
                metric_spec=TIME_BETWEEN_EVENTS_SPEC,
                episode_dirs=[first, second],
                output_dir=root / "time_between_events",
            )
            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertEqual(
                [
                    item["oracle_projection"]["value"]
                    for item in result["episodes"]
                ],
                [0.004, 0.008],
            )


if __name__ == "__main__":
    unittest.main()
