import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen import (
    MetricSpecError,
    compatible_run_local_tool_requests,
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
from tests.mainline.test_tool_orchestration import write_episode


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
TERMINAL_Z_SPEC = {
    "schema_version": 1,
    "operation": "terminal_signal_component",
    "signal": "bottle_functional_position",
    "component": "z",
    "unit": "m",
    "null_semantics": "null_if_terminal_not_finite",
}
TERMINAL_DIFFERENCE_SPEC = {
    "schema_version": 1,
    "operation": "terminal_signal_difference",
    "left_signal": "right_tcp_position",
    "right_signal": "block_position",
    "component": "x",
    "unit": "m",
    "null_semantics": "null_if_terminal_not_finite",
}


class MetricSpecTests(unittest.TestCase):
    @staticmethod
    def _write_terminal_difference_episode(episode: Path) -> None:
        write_episode(episode, policy_name="ACT", physical_contact=False)
        trace = dict(np.load(episode / "semantic_trace.npz"))
        trace["right_tcp_position"] = np.asarray(
            [
                [0.10, 0.00, 0.78],
                [0.18, 0.00, 0.82],
                [0.25, 0.00, 0.84],
            ],
            dtype=np.float32,
        )
        trace["block_position"] = np.asarray(
            [
                [0.40, 0.05, 0.76],
                [0.40, 0.05, 0.76],
                [0.40, 0.05, 0.76],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(episode / "semantic_trace.npz", **trace)

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
            [
                "derived_observable",
                "event_count",
                "minimum_distance",
                "terminal_signal_component",
                "terminal_signal_difference",
                "time_between_events",
            ],
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
            reusable = compatible_run_local_tool_requests(
                registry,
                task_name="beat_block_hammer",
                episode_dirs=[first, second],
            )
            self.assertEqual(len(reusable), 1)
            self.assertEqual(
                reusable[0]["request"]["metric_spec"],
                SPEC,
            )
            self.assertEqual(
                reusable[0]["registration_id"],
                result["registration"]["registration_id"],
            )

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

    def test_terminal_signal_component_normalizes_and_differentially_executes(self):
        normalized = validate_metric_spec(TERMINAL_Z_SPEC)
        self.assertFalse(normalized["absolute"])
        self.assertTrue(
            validate_generated_tool(
                compile_metric_spec_source(TERMINAL_Z_SPEC)
            )["valid"]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "adjust_bottle_act"
            write_episode(episode, policy_name="ACT", physical_contact=False)
            for filename in ("episode.json", "schema.json"):
                path = episode / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["task_name"] = "adjust_bottle"
                path.write_text(json.dumps(payload), encoding="utf-8")
            trace = dict(np.load(episode / "semantic_trace.npz"))
            trace["bottle_functional_position"] = np.asarray(
                [
                    [-0.04, 0.02, 0.78],
                    [-0.12, 0.02, 0.86],
                    [-0.19, 0.02, 0.93],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(episode / "semantic_trace.npz", **trace)

            terminal_z = evaluate_metric_spec(
                TERMINAL_Z_SPEC, TrajectoryView(episode)
            )
            self.assertAlmostEqual(terminal_z["value"], 0.93, places=6)
            self.assertEqual(terminal_z["evidence_steps"], [2])

            absolute_x_spec = {
                **TERMINAL_Z_SPEC,
                "component": "x",
                "absolute": True,
            }
            absolute_x = evaluate_metric_spec(
                absolute_x_spec, TrajectoryView(episode)
            )
            self.assertAlmostEqual(absolute_x["value"], 0.19, places=6)

            result = execute_metric_spec(
                task_name="adjust_bottle",
                metric="query_terminal_bottle_height",
                question="What was the final bottle functional-point height?",
                metric_spec=TERMINAL_Z_SPEC,
                episode_dirs=[episode],
                output_dir=root / "terminal_z",
                registry_dir=root / "registry",
            )
            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertAlmostEqual(
                result["episodes"][0]["generated_result"]["value"],
                0.93,
                places=6,
            )
            self.assertEqual(
                result["tool_spec"]["required_signals"],
                [
                    "semantic_trace.bottle_functional_position",
                    "semantic_trace.physics_step",
                ],
            )

    def test_terminal_signal_component_rejects_invalid_shape_and_contract(self):
        with self.assertRaisesRegex(MetricSpecError, "component=x, y, or z"):
            validate_metric_spec({**TERMINAL_Z_SPEC, "component": "height"})
        with self.assertRaisesRegex(MetricSpecError, "absolute must be a boolean"):
            validate_metric_spec({**TERMINAL_Z_SPEC, "absolute": 1})
        with self.assertRaisesRegex(
            MetricSpecError, "null_if_terminal_not_finite"
        ):
            validate_metric_spec(
                {
                    **TERMINAL_Z_SPEC,
                    "null_semantics": "null_if_no_finite_sample",
                }
            )

    def test_terminal_signal_difference_preserves_signed_direction(self):
        normalized = validate_metric_spec(TERMINAL_DIFFERENCE_SPEC)
        self.assertFalse(normalized["absolute"])

        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "signed_difference"
            self._write_terminal_difference_episode(episode)
            result = evaluate_metric_spec(
                TERMINAL_DIFFERENCE_SPEC,
                TrajectoryView(episode),
            )
            self.assertAlmostEqual(result["value"], -0.15, places=6)
            self.assertEqual(result["evidence_steps"], [2])
            self.assertEqual(
                result["details"]["reason"],
                "measured",
            )

    def test_terminal_signal_difference_supports_absolute_distance(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "absolute_difference"
            self._write_terminal_difference_episode(episode)
            result = evaluate_metric_spec(
                {**TERMINAL_DIFFERENCE_SPEC, "absolute": True},
                TrajectoryView(episode),
            )
            self.assertAlmostEqual(result["value"], 0.15, places=6)
            self.assertTrue(result["details"]["absolute"])

    def test_terminal_signal_difference_rejects_invalid_contracts(self):
        with self.assertRaisesRegex(MetricSpecError, "signals must be distinct"):
            validate_metric_spec(
                {
                    **TERMINAL_DIFFERENCE_SPEC,
                    "right_signal": "right_tcp_position",
                }
            )
        with self.assertRaisesRegex(MetricSpecError, "component=x, y, or z"):
            validate_metric_spec(
                {**TERMINAL_DIFFERENCE_SPEC, "component": "heading"}
            )
        with self.assertRaisesRegex(MetricSpecError, "absolute must be a boolean"):
            validate_metric_spec(
                {**TERMINAL_DIFFERENCE_SPEC, "absolute": 1}
            )
        with self.assertRaisesRegex(
            MetricSpecError, "null_if_terminal_not_finite"
        ):
            validate_metric_spec(
                {
                    **TERMINAL_DIFFERENCE_SPEC,
                    "null_semantics": "null_if_no_finite_sample",
                }
            )

    def test_terminal_signal_difference_differentially_executes(self):
        source = compile_metric_spec_source(TERMINAL_DIFFERENCE_SPEC)
        self.assertTrue(validate_generated_tool(source)["valid"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "difference_episode"
            self._write_terminal_difference_episode(episode)
            result = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_terminal_tcp_block_x_difference",
                question=(
                    "What was the signed terminal TCP-minus-block "
                    "x displacement?"
                ),
                metric_spec=TERMINAL_DIFFERENCE_SPEC,
                episode_dirs=[episode],
                output_dir=root / "terminal_difference",
                registry_dir=root / "registry",
            )
            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertTrue(result["episodes"][0]["oracle_agreement"])
            self.assertAlmostEqual(
                result["episodes"][0]["generated_result"]["value"],
                -0.15,
                places=6,
            )
            self.assertEqual(
                result["tool_spec"]["required_signals"],
                [
                    "semantic_trace.right_tcp_position",
                    "semantic_trace.block_position",
                    "semantic_trace.physics_step",
                ],
            )
            self.assertEqual(
                result["limitations"][0],
                "typed semantic oracle: terminal_signal_difference",
            )


if __name__ == "__main__":
    unittest.main()
