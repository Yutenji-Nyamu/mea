import unittest
from pathlib import Path

import numpy as np

from mea.toolgen import (
    BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
    bell_active_tcp_min_xy_error_tool_request,
    route_tool_request,
)
from mea.toolgen.targets import evaluate_target_oracle
from mea.toolgen.prototype import validate_generated_tool
from mea.toolgen.prototype import _prompt


class FakeTrajectory:
    def __init__(self, *, bell_x: float):
        self.schema = {"physics_timestep_seconds": 0.004}
        self.trace = {
            "physics_step": np.asarray([0, 4, 8]),
            "simulation_time_seconds": np.asarray([0.0, 0.016, 0.032]),
            "bell_position": np.asarray(
                [[bell_x, -0.08, 0.78]] * 3, dtype=float
            ),
            "bell_contact_position": np.asarray(
                [[bell_x, -0.08, 0.80]] * 3, dtype=float
            ),
            "left_tcp_position": np.asarray(
                [[bell_x + 0.2, -0.08, 0.8], [bell_x + 0.03, -0.08, 0.8], [bell_x + 0.1, -0.08, 0.8]]
            ),
            "right_tcp_position": np.asarray(
                [[bell_x - 0.2, -0.08, 0.8], [bell_x - 0.1, -0.08, 0.8], [bell_x - 0.02, -0.08, 0.8]]
            ),
        }


class ClickBellToolGenMetricTests(unittest.TestCase):
    def test_oracle_selects_official_active_arm(self):
        left = evaluate_target_oracle(
            BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
            FakeTrajectory(bell_x=-0.2),
        )
        right = evaluate_target_oracle(
            BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
            FakeTrajectory(bell_x=0.2),
        )
        self.assertEqual(left["details"]["active_arm"], "left")
        self.assertAlmostEqual(left["value"], 0.03)
        self.assertEqual(left["evidence_steps"], [4])
        self.assertEqual(right["details"]["active_arm"], "right")
        self.assertAlmostEqual(right["value"], 0.02)
        self.assertEqual(right["evidence_steps"], [8])
        self.assertIsNone(right["passed"])

    def test_router_is_task_scoped_and_requests_codegen(self):
        request = bell_active_tcp_min_xy_error_tool_request()
        routed = route_tool_request(request)
        self.assertEqual(routed["route_decision"]["resolved_route"], "force_codegen")
        incompatible = dict(request, task_name="beat_block_hammer")
        self.assertEqual(
            route_tool_request(incompatible)["route_decision"]["status"],
            "unsupported",
        )

    def test_safe_numpy_sqrt_is_available_to_generated_metric(self):
        source = '''def generated_tool(trajectory):
    delta = trajectory.trace["left_tcp_position"][:, :2] - trajectory.trace["bell_contact_position"][:, :2]
    distance = np.sqrt(np.sum(delta * delta, axis=1))
    distance = np.where(np.isfinite(distance), distance, np.inf)
    index = int(np.argmin(distance))
    step = int(trajectory.trace["physics_step"][index])
    return {"value": float(distance[index]), "unit": "m", "passed": None, "evidence_steps": [step], "details": {"active_arm": "left", "min_error_physics_step": step, "simulation_time_seconds": float(trajectory.trace["simulation_time_seconds"][index])}}
'''
        self.assertTrue(validate_generated_tool(source)["valid"])
        self.assertTrue(
            validate_generated_tool(source.replace("np.argmin", "np.nanargmin"))[
                "valid"
            ]
        )

    def test_prompt_forbids_nonexistent_semantic_trace_alias(self):
        prompt = _prompt(
            Path(__file__).resolve().parents[2],
            "measure active TCP error",
            BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
            None,
            [],
            None,
        )
        self.assertIn('trajectory.trace["field_name"]', prompt)
        self.assertIn("trajectory.semantic_trace` does not exist", prompt)


if __name__ == "__main__":
    unittest.main()
