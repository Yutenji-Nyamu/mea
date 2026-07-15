import csv
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen.examples import hammer_block_contact_example
from mea.toolgen.prototype import (
    ToolGenError,
    ToolGenPrototype,
    execute_generated_tool,
    retrieve_examples,
    validate_generated_tool,
)


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_metadata = {}

    def text(self, prompt, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        self.last_metadata = {
            "model": kwargs.get("model"),
            "usage": {"prompt_tokens": 100, "completion_tokens": 100},
        }
        return response


def generated_contact_source():
    source = inspect.getsource(hammer_block_contact_example)
    return source.replace(
        "def hammer_block_contact_example(trajectory):",
        "def generated_tool(trajectory):",
        1,
    )


class ToolGenTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.positive = self.root / "expert/episode_000_seed_100000"
        self.negative = self.root / "act/episode_000_seed_100000"
        self.reported_only = self.root / "reported/episode_000_seed_100000"
        self._write_episode(self.positive, physical_contact=True)
        self._write_episode(self.negative, physical_contact=False)
        self._write_episode(
            self.reported_only,
            physical_contact=False,
            reported_contact=True,
        )
        self.repo_root = Path(__file__).resolve().parents[2]

    def tearDown(self):
        self._temporary.cleanup()

    @staticmethod
    def _write_episode(
        episode_dir,
        *,
        physical_contact,
        reported_contact=None,
    ):
        if reported_contact is None:
            reported_contact = physical_contact
        episode_dir.mkdir(parents=True)
        schema = {
            "schema_version": 1,
            "task_name": "beat_block_hammer",
            "physics_timestep_seconds": 0.004,
            "pickup_height_threshold_m": 0.03,
            "success_contract": {"xy_tolerance_m": [0.02, 0.02]},
        }
        metadata = {
            "schema_version": 1,
            "task_name": "beat_block_hammer",
            "policy_name": "expert" if physical_contact else "ACT",
            "seed": 100000,
            "success": physical_contact,
            "physics_steps": 2,
            "semantic_trace_rows": 3,
        }
        (episode_dir / "schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )
        (episode_dir / "episode.json").write_text(
            json.dumps(metadata), encoding="utf-8"
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
        step = np.asarray([0, 1, 2], dtype=np.int64)
        positions = np.asarray(
            [[0.0, 0.0, 0.78], [0.1, 0.0, 0.80], [0.15, 0.05, 0.82]],
            dtype=np.float32,
        )
        np.savez_compressed(
            episode_dir / "semantic_trace.npz",
            physics_step=step,
            policy_step=np.asarray([-1, 0, 0], dtype=np.int64),
            simulation_time_seconds=step * 0.004,
            success=np.asarray([False, False, physical_contact]),
            hammer_position=positions,
            block_position=np.asarray([[0.15, 0.05, 0.76]] * 3),
            hammer_functional_position=positions,
            block_functional_position=np.asarray([[0.15, 0.05, 0.76]] * 3),
            left_tcp_position=np.zeros((3, 3), dtype=np.float32),
            right_tcp_position=positions,
        )
        events = []
        if reported_contact:
            events.append(
                {
                    "type": "contact_interval",
                    "actors": ["020_hammer", "box"],
                    "physical_contact": physical_contact,
                    "first_physical_policy_step": 0 if physical_contact else None,
                    "first_physical_physics_step": 1 if physical_contact else None,
                    "first_physical_simulation_time_seconds": (
                        0.004 if physical_contact else None
                    ),
                    "max_impulse": 0.4 if physical_contact else 0.0,
                    "min_separation": -0.001 if physical_contact else 0.001,
                    "peak_policy_step": 0,
                    "peak_physics_step": 2,
                }
            )
        (episode_dir / "events.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in events),
            encoding="utf-8",
        )

    def test_validator_rejects_file_access(self):
        source = """def generated_tool(trajectory):
    value = open('secret').read()
    return {'value': value, 'unit': None, 'passed': None, 'evidence_steps': [], 'details': {}}
"""
        with self.assertRaisesRegex(ToolGenError, "open|allowlist"):
            validate_generated_tool(source)

    def test_validator_rejects_numpy_escape_chain(self):
        source = """def generated_tool(trajectory):
    value = np.lib.npyio.os.system('echo no')
    return {'value': value, 'unit': None, 'passed': None, 'evidence_steps': [], 'details': {}}
"""
        with self.assertRaisesRegex(ToolGenError, "NumPy attribute chain"):
            validate_generated_tool(source)

    def test_differential_gate_rejects_duplicate_oracle_input(self):
        provider = FakeProvider([f"```python\n{generated_contact_source()}```"])
        with self.assertRaisesRegex(ToolGenError, "重复 episode path"):
            ToolGenPrototype(
                self.repo_root,
                provider,
                model="fake-model",
            ).generate(
                "判断锤子是否接触方块",
                reference_tool="hammer_block_contact_ever",
                episode_dirs=[self.negative, self.negative],
                output_dir=self.root / "duplicate",
            )
        self.assertEqual(provider.calls, 0)

    def test_worker_runs_when_caller_is_outside_repo(self):
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            result = execute_generated_tool(
                generated_contact_source(),
                self.positive,
                tool_name="generated_contact_external_cwd",
            )
        finally:
            os.chdir(previous)
        self.assertTrue(result["value"])
        self.assertEqual(result["evidence_steps"], [1])

    def test_gate_rejects_incomplete_episode_artifacts(self):
        (self.negative / "events.jsonl").unlink()
        provider = FakeProvider([f"```python\n{generated_contact_source()}```"])
        with self.assertRaisesRegex(ToolGenError, "缺少 core artifacts"):
            ToolGenPrototype(
                self.repo_root,
                provider,
                model="fake-model",
            ).generate(
                "判断锤子是否接触方块",
                reference_tool="hammer_block_contact_ever",
                episode_dirs=[self.negative, self.positive],
                output_dir=self.root / "incomplete",
            )
        self.assertEqual(provider.calls, 0)

    def test_retrieval_returns_three_compact_contact_examples(self):
        examples = retrieve_examples(
            "判断锤子是否接触方块，并给出首次接触和冲量",
            "hammer_block_contact_ever",
        )
        self.assertEqual(examples[0]["name"], "hammer_block_contact_ever")
        self.assertEqual(len(examples), 3)
        self.assertEqual(
            {item["name"] for item in examples},
            {
                "hammer_block_contact_ever",
                "first_contact_step",
                "max_contact_impulse",
            },
        )

    def test_force_codegen_matches_positive_and_negative_oracle(self):
        source = generated_contact_source()
        provider = FakeProvider([f"```python\n{source}```"])
        output_dir = self.root / "generated"
        result = ToolGenPrototype(
            self.repo_root,
            provider,
            model="fake-model",
        ).generate(
            "生成一个判断锤子是否真正接触方块的离线工具",
            reference_tool="hammer_block_contact_ever",
            episode_dirs=[self.negative, self.reported_only, self.positive],
            output_dir=output_dir,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["successful_attempt"], 0)
        self.assertRegex(result["generator_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(provider.calls, 1)
        execution = json.loads(
            (output_dir / "execution_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [item["generated_result"]["value"] for item in execution],
            [False, False, True],
        )
        self.assertTrue(
            execution[1]["generated_result"]["details"]["reported_contact"]
        )
        self.assertFalse(
            execution[1]["generated_result"]["details"]["physical_contact"]
        )
        self.assertTrue(all(item["deterministic"] for item in execution))
        self.assertTrue(all(item["oracle_agreement"] for item in execution))
        self.assertTrue(all(item["artifacts_unchanged"] for item in execution))
        self.assertTrue((output_dir / "generated_tool.py").is_file())
        registration = json.loads(
            (output_dir / "registration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(registration["scope"], "run_local")
        self.assertEqual(registration["status"], "validated")

    def test_failed_static_check_is_repaired_once(self):
        bad = """```python
def generated_tool(trajectory):
    return open('x')
```"""
        good = f"```python\n{generated_contact_source()}```"
        provider = FakeProvider([bad, good])
        result = ToolGenPrototype(
            self.repo_root,
            provider,
            model="fake-model",
        ).generate(
            "判断锤子是否接触方块",
            reference_tool="hammer_block_contact_ever",
            episode_dirs=[self.negative, self.positive],
            output_dir=self.root / "repaired",
            max_attempts=2,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["successful_attempt"], 1)
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(provider.calls, 2)


if __name__ == "__main__":
    unittest.main()
