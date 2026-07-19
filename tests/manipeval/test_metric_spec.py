import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen import (
    MetricSpecError,
    compile_metric_spec_source,
    evaluate_metric_spec,
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

    def test_unbounded_operator_and_registry_collision_are_rejected(self):
        with self.assertRaisesRegex(MetricSpecError, "operation"):
            validate_metric_spec({**SPEC, "operation": "eval_python"})
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


if __name__ == "__main__":
    unittest.main()
