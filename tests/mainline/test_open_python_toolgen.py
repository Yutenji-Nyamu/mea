import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen import (
    MetricSpecError,
    compile_metric_spec_source,
    compatible_run_local_tool_requests,
    execute_metric_spec,
    execute_tool_request,
    validate_metric_spec,
)
from mea.toolgen.prototype import ToolGenError, validate_generated_tool
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

DERIVED_SPEC = {
    "schema_version": 2,
    "operation": "derived_observable",
    "observable_id": "query_peak_hammer_motion",
    "description": (
        "Maximum Euclidean displacement per positive physics step between "
        "consecutive hammer samples."
    ),
    "required_signals": ["hammer_position"],
    "unit": "m_per_step",
    "null_semantics": "null_if_no_finite_sample",
}

TERMINAL_MINIMUM_DISTANCE_SPEC = {
    "schema_version": 1,
    "operation": "terminal_minimum_distance",
    "left_signals": ["left_tcp_position", "right_tcp_position"],
    "right_signal": "block_position",
    "dimensions": ["x", "y", "z"],
    "unit": "m",
    "null_semantics": "null_if_terminal_not_finite",
}

DERIVED_SOURCE = """def generated_tool(trajectory):
    positions = np.asarray(trajectory.trace["hammer_position"], dtype=float)
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    delta = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    step_delta = np.diff(physics)
    finite = np.isfinite(delta) & (step_delta > 0)
    rate = np.where(finite, delta / step_delta, -np.inf)
    index = int(np.argmax(rate)) if np.any(finite) else None
    value = float(rate[index]) if index is not None else None
    steps = [int(physics[index]), int(physics[index + 1])] if index is not None else []
    return {
        "value": value,
        "unit": "m_per_step",
        "passed": None,
        "evidence_steps": steps,
        "details": {
            "operation": "derived_observable",
            "reason": "measured" if value is not None else "no_finite_sample",
        },
    }
"""
DERIVED_REVIEW = """{
  "schema_version": 1,
  "status": "approved",
  "checks": {
    "implements_metric_description": true,
    "uses_only_declared_signals": true,
    "preserves_requested_unit": true,
    "returns_diagnostic_not_success": true
  },
  "reason": "The implementation matches the declared trajectory metric."
}"""


def peak_hammer_motion_oracle(trajectory):
    positions = np.asarray(trajectory.trace["hammer_position"], dtype=float)
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    step_delta = np.diff(physics)
    motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    finite = np.isfinite(motion) & (step_delta > 0)
    denominator = np.where(step_delta > 0, step_delta, 1)
    rate = np.where(finite, motion / denominator, -np.inf)
    peak_index = int(np.argmax(rate)) if np.any(finite) else None
    value = float(rate[peak_index]) if peak_index is not None else None
    return {
        "value": value,
        "unit": "m_per_step",
        "passed": None,
        "evidence_steps": (
            [int(physics[peak_index]), int(physics[peak_index + 1])]
            if peak_index is not None
            else []
        ),
        "details": {
            "operation": "derived_observable",
            "reason": "measured" if value is not None else "no_finite_sample",
        },
    }


class SequencedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_metadata = {}
        self.prompts = []

    def text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        self.last_metadata = {"model": kwargs.get("model")}
        return response


class OpenPythonToolGenTests(unittest.TestCase):
    def test_terminal_minimum_distance_has_independent_compiler_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "episode"
            write_episode(episode, policy_name="SmolVLA", physical_contact=True)

            result = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="terminal_minimum_tcp_to_block_distance",
                question="Which TCP ended closest to the block?",
                metric_spec=TERMINAL_MINIMUM_DISTANCE_SPEC,
                episode_dirs=[episode],
                output_dir=root / "tool",
            )

            self.assertEqual(result["route"], "typed_metric_spec_compile")
            self.assertEqual(
                result["validation_authority"],
                "typed_metric_spec_interpreter",
            )
            oracle = result["episodes"][0]["oracle_projection"]
            self.assertAlmostEqual(oracle["value"], 0.08, places=5)
            self.assertEqual(
                oracle["details"]["operation"],
                "terminal_minimum_distance",
            )

    def test_bounded_for_loop_is_valid_generated_python(self):
        report = validate_generated_tool(
            """def generated_tool(trajectory):
    total = 0.0
    evidence_steps = []
    positions = trajectory.trace["hammer_position"]
    physics = trajectory.trace["physics_step"]
    for index in range(positions.shape[0]):
        value = positions[index]
        total += float(np.sum(np.abs(value)))
        evidence_steps.append(int(physics[index]))
    return {
        "value": total,
        "unit": "m",
        "passed": None,
        "evidence_steps": evidence_steps,
        "details": {"operation": "derived_observable", "reason": "measured"},
    }
"""
        )
        self.assertTrue(report["valid"])
        with self.assertRaisesRegex(ToolGenError, "TrajectoryView"):
            validate_generated_tool(
                """def generated_tool(trajectory):
    trajectory.events.append({})
    return {
        "value": 0.0,
        "unit": "m",
        "passed": None,
        "evidence_steps": [],
        "details": {"operation": "derived_observable", "reason": "measured"},
    }
"""
            )

    def test_orchestration_labels_semantic_review_without_numeric_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "generated_tasks/round_1"
            episode = (
                child
                / "evaluation/telemetry/act/episode_000_seed_100000"
            )
            write_episode(
                episode,
                policy_name="SmolVLA",
                physical_contact=False,
            )
            (child / "manifest.json").write_text(
                """{
  "schema_version": 1,
  "task_name": "beat_block_hammer",
  "task_module": "beat_block_hammer",
  "generation_kind": "generated"
}""",
                encoding="utf-8",
            )
            result = execute_tool_request(
                Path(__file__).resolve().parents[2],
                child,
                root / "evaluation/execution/round_1/planned_tool",
                {
                    "schema_version": 2,
                    "task_name": "beat_block_hammer",
                    "metric": "query_peak_hammer_motion",
                    "question": "Where does hammer motion peak?",
                    "metric_spec": DERIVED_SPEC,
                },
                provider=SequencedProvider(
                    [f"```python\n{DERIVED_SOURCE}\n```", DERIVED_REVIEW]
                ),
                model="test-model",
            )

            validation = result["validation"]
            self.assertTrue(validation["validation_gates_passed"])
            self.assertFalse(validation["independent_numeric_oracle"])
            self.assertIsNone(validation["oracle_agreement"])
            self.assertNotIn("differential_gates_passed", validation)

    def test_derived_observable_uses_semantic_review_then_exact_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "episode"
            write_episode(
                episode,
                policy_name="SmolVLA",
                physical_contact=False,
            )
            provider = SequencedProvider(
                [f"```python\n{DERIVED_SOURCE}\n```", DERIVED_REVIEW]
            )
            registry = root / "registry"
            generated = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_peak_hammer_motion",
                question="Where does hammer motion peak?",
                metric_spec=DERIVED_SPEC,
                episode_dirs=[episode],
                output_dir=root / "generated",
                registry_dir=registry,
                provider=provider,
                model="test-model",
            )
            self.assertEqual(
                generated["validation_authority"],
                "toolgen_semantic_review_runtime",
            )
            self.assertEqual(provider.calls, 2)
            self.assertEqual(
                generated["semantic_review"]["status"],
                "approved",
            )
            self.assertIsNone(
                generated["episodes"][0]["oracle_agreement"]
            )
            self.assertTrue(
                generated["episodes"][0]["semantic_contract_valid"]
            )

            replay_provider = SequencedProvider([])
            replay = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_peak_hammer_motion",
                question="Reuse the same metric.",
                metric_spec=DERIVED_SPEC,
                episode_dirs=[episode],
                output_dir=root / "replay",
                registry_dir=registry,
                provider=replay_provider,
                model="test-model",
            )
            self.assertEqual(replay["route"], "run_local_reuse")
            self.assertFalse(replay["provider_called"])
            self.assertEqual(replay_provider.calls, 0)

    def test_registry_miss_generates_repairs_gates_and_reuses_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "episode_a"
            second = root / "episode_b"
            write_episode(first, policy_name="ACT", physical_contact=False)
            write_episode(second, policy_name="expert", physical_contact=True)
            trace = dict(np.load(second / "semantic_trace.npz"))
            trace["right_tcp_position"] = (
                np.asarray(trace["right_tcp_position"]) + [0.08, 0.0, 0.0]
            )
            np.savez_compressed(second / "semantic_trace.npz", **trace)

            valid_source = compile_metric_spec_source(SPEC).replace(
                '"reason": "measured",',
                '"reason": "measured",\n'
                '            "implementation": "provider_equivalent",',
            )
            provider = SequencedProvider(
                [
                    "```python\ndef generated_tool(trajectory):\n    return {}\n```",
                    f"```python\n{valid_source}\n```",
                ]
            )
            registry = root / "registry"
            generated = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_min_tcp_block_xy",
                question="How close did the TCP get to the block?",
                metric_spec=SPEC,
                episode_dirs=[first, second],
                output_dir=root / "generated",
                task_code_context={
                    "schema_version": 1,
                    "task_name": "beat_block_hammer",
                },
                registry_dir=registry,
                provider=provider,
                model="test-model",
                max_attempts=2,
            )
            self.assertEqual(generated["route"], "provider_python_codegen")
            self.assertTrue(generated["provider_called"])
            self.assertEqual(provider.calls, 2)
            self.assertIn("plain Python int physics steps", provider.prompts[0])
            self.assertIn('"passed": null', provider.prompts[0])
            self.assertIn(
                '"details.operation": "minimum_distance"',
                provider.prompts[0],
            )
            self.assertIn('"details.reason": "measured"', provider.prompts[0])
            self.assertNotIn(
                "details.reason_on_measurement",
                provider.prompts[0],
            )
            self.assertEqual(generated["generation"]["successful_attempt"], 1)
            self.assertTrue(
                (root / "generated/attempts/attempt_1/prompt.md").is_file()
            )
            repaired_prompt = provider.prompts[1]
            self.assertIn("PREVIOUS FUNCTION", repaired_prompt)
            self.assertIn("def generated_tool(trajectory)", repaired_prompt)
            self.assertIn("generated Tool validation/oracle", repaired_prompt)
            self.assertEqual(generated["registration"]["scope"], "run_local")

            replay = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_min_tcp_block_xy",
                question="Report the same minimum separation.",
                metric_spec=SPEC,
                episode_dirs=[first, second],
                output_dir=root / "replay",
                registry_dir=registry,
                provider=SequencedProvider([]),
                model="test-model",
            )
            self.assertEqual(replay["route"], "run_local_reuse")
            self.assertFalse(replay["provider_called"])
            self.assertEqual(
                replay["registration"]["registration_id"],
                generated["registration"]["registration_id"],
            )

    def test_provider_defines_new_derived_observable_then_exactly_reuses_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "episode_a"
            second = root / "episode_b"
            write_episode(first, policy_name="ACT", physical_contact=False)
            write_episode(second, policy_name="expert", physical_contact=True)
            valid_source = """def generated_tool(trajectory):
    positions = np.asarray(trajectory.trace["hammer_position"], dtype=float)
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    step_delta = np.diff(physics)
    motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    finite = np.isfinite(motion) & (step_delta > 0)
    denominator = np.where(step_delta > 0, step_delta, 1)
    rate = np.where(finite, motion / denominator, -np.inf)
    peak_index = int(np.argmax(rate)) if np.any(finite) else None
    value = float(rate[peak_index]) if peak_index is not None else None
    steps = (
        [int(physics[peak_index]), int(physics[peak_index + 1])]
        if peak_index is not None else []
    )
    return {
        "value": value,
        "unit": "m_per_step",
        "passed": None,
        "evidence_steps": steps,
        "details": {
            "operation": "derived_observable",
            "reason": "measured" if value is not None else "no_finite_sample",
        },
    }
"""
            provider = SequencedProvider(
                [
                    """```python
def generated_tool(trajectory):
    return {
        "value": 0.0,
        "unit": "m_per_step",
        "passed": None,
        "evidence_steps": [],
        "details": {
            "operation": "derived_observable",
            "reason": "measured",
        },
    }
```""",
                    f"```python\n{valid_source}\n```",
                ]
            )
            registry = root / "registry"
            generated = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_peak_hammer_motion",
                question="Where does pre-contact hammer motion peak?",
                metric_spec=DERIVED_SPEC,
                episode_dirs=[first],
                fixture_episode_dirs=[second],
                oracle_evaluator=peak_hammer_motion_oracle,
                output_dir=root / "generated",
                registry_dir=registry,
                provider=provider,
                model="test-model",
                max_attempts=2,
            )
            self.assertEqual(generated["route"], "provider_python_codegen")
            self.assertEqual(provider.calls, 2)
            self.assertIn(
                "not a pre-registered metric operator",
                provider.prompts[0],
            )
            self.assertTrue(
                generated["episodes"][0]["oracle_agreement"]
            )
            self.assertEqual(len(generated["fixtures"]), 1)
            self.assertEqual(
                generated["limitations"][0],
                "provider-defined derived observable over declared telemetry",
            )
            self.assertEqual(
                compatible_run_local_tool_requests(
                    registry,
                    task_name="beat_block_hammer",
                    episode_dirs=[first],
                ),
                [],
            )
            advertised = compatible_run_local_tool_requests(
                registry,
                task_name="beat_block_hammer",
                episode_dirs=[first],
                include_derived_observables=True,
            )
            self.assertEqual(len(advertised), 1)
            self.assertEqual(
                advertised[0]["request"]["metric_spec"],
                DERIVED_SPEC,
            )

            replay_provider = SequencedProvider([])
            replay = execute_metric_spec(
                task_name="beat_block_hammer",
                metric="query_peak_hammer_motion",
                question="Reuse the same peak-motion observable.",
                metric_spec=DERIVED_SPEC,
                episode_dirs=[first],
                fixture_episode_dirs=[second],
                oracle_evaluator=peak_hammer_motion_oracle,
                output_dir=root / "replay",
                registry_dir=registry,
                provider=replay_provider,
                model="test-model",
            )
            self.assertEqual(replay["route"], "run_local_reuse")
            self.assertEqual(replay_provider.calls, 0)
            self.assertEqual(
                replay["registration"]["registration_id"],
                generated["registration"]["registration_id"],
            )

    def test_derived_observable_declares_signals_without_a_known_operator(self):
        self.assertEqual(
            validate_metric_spec(DERIVED_SPEC)["operation"],
            "derived_observable",
        )
        invalid = {
            **DERIVED_SPEC,
            "required_signals": ["../hammer_position"],
        }
        with self.assertRaisesRegex(MetricSpecError, "safe trace names"):
            validate_metric_spec(invalid)


if __name__ == "__main__":
    unittest.main()
