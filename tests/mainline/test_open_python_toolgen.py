import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen import compile_metric_spec_source, execute_metric_spec
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
            self.assertEqual(generated["generation"]["successful_attempt"], 1)
            self.assertTrue(
                (root / "generated/attempts/attempt_1/prompt.md").is_file()
            )
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


if __name__ == "__main__":
    unittest.main()
