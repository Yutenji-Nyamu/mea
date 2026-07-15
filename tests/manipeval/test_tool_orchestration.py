import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen import (
    ToolOrchestrationError,
    contact_tool_spec,
    execute_tool_spec,
    validate_tool_spec,
)
from mea.toolgen.examples import hammer_block_contact_example


class NeverCalledProvider:
    def __init__(self):
        self.calls = 0

    def text(self, prompt, **kwargs):
        self.calls += 1
        raise AssertionError("reuse route must not call the provider")


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_metadata = {}

    def text(self, prompt, **kwargs):
        self.calls += 1
        self.last_metadata = {
            "model": kwargs.get("model"),
            "usage": {"prompt_tokens": 20, "completion_tokens": 20},
        }
        return self.response


def generated_contact_source():
    source = inspect.getsource(hammer_block_contact_example)
    return source.replace(
        "def hammer_block_contact_example(trajectory):",
        "def generated_tool(trajectory):",
        1,
    )


def write_episode(episode_dir, *, policy_name, physical_contact):
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
        "policy_name": policy_name,
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

    steps = np.asarray([0, 1, 2], dtype=np.int64)
    hammer = np.asarray(
        [[0.0, 0.0, 0.78], [0.1, 0.0, 0.80], [0.15, 0.05, 0.82]],
        dtype=np.float32,
    )
    block = np.asarray([[0.15, 0.05, 0.76]] * 3, dtype=np.float32)
    np.savez_compressed(
        episode_dir / "semantic_trace.npz",
        physics_step=steps,
        policy_step=np.asarray([-1, 0, 0], dtype=np.int64),
        simulation_time_seconds=steps * 0.004,
        success=np.asarray([False, False, physical_contact]),
        hammer_position=hammer,
        block_position=block,
        hammer_functional_position=hammer,
        block_functional_position=block,
        left_tcp_position=np.zeros((3, 3), dtype=np.float32),
        right_tcp_position=hammer,
    )
    events = []
    if physical_contact:
        events.append(
            {
                "type": "contact_interval",
                "actors": ["020_hammer", "box"],
                "physical_contact": True,
                "first_physical_policy_step": 0,
                "first_physical_physics_step": 1,
                "first_physical_simulation_time_seconds": 0.004,
                "max_impulse": 0.4,
                "min_separation": -0.001,
                "peak_policy_step": 0,
                "peak_physics_step": 2,
            }
        )
    (episode_dir / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
    )


class ToolOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.child_run = self.root / "run_blue"
        self.act_episode = (
            self.child_run
            / "evaluation/telemetry/act/episode_000_seed_100000"
        )
        self.expert_episode = (
            self.child_run
            / "evaluation/telemetry/expert/episode_000_seed_100000"
        )
        write_episode(
            self.act_episode, policy_name="ACT", physical_contact=False
        )
        write_episode(
            self.expert_episode, policy_name="expert", physical_contact=True
        )

    def tearDown(self):
        self._temporary.cleanup()

    def test_tool_spec_validation_is_exact_and_route_aware(self):
        force = contact_tool_spec("force_codegen")
        self.assertEqual(
            validate_tool_spec(force, expected_route="force_codegen"), force
        )
        self.assertEqual(
            force["required_signals"],
            ["hammer_block_contact_intervals", "physics_step_index"],
        )
        self.assertEqual(
            force["validation_requirements"]["required_reference_values"],
            [False, True],
        )
        self.assertEqual(
            contact_tool_spec("reuse")["validation_requirements"][
                "required_reference_values"
            ],
            [],
        )

        extra = json.loads(json.dumps(force))
        extra["unexpected"] = True
        with self.assertRaisesRegex(ToolOrchestrationError, "fields"):
            validate_tool_spec(extra)

        reordered_signals = json.loads(json.dumps(force))
        reordered_signals["required_signals"].reverse()
        with self.assertRaisesRegex(
            ToolOrchestrationError, "required_signals"
        ):
            validate_tool_spec(reordered_signals)

        with self.assertRaisesRegex(ToolOrchestrationError, "reuse"):
            validate_tool_spec(force, expected_route="reuse")

    def test_reuse_executes_trusted_catalog_without_calling_provider(self):
        provider = NeverCalledProvider()
        output_dir = self.root / "reuse_output"
        result = execute_tool_spec(
            self.repo_root,
            self.child_run,
            output_dir,
            contact_tool_spec("reuse"),
            provider=provider,
            model="must-not-be-used",
        )

        self.assertEqual(provider.calls, 0)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["route"], "reuse")
        self.assertEqual(result["tool_spec"], contact_tool_spec("reuse"))
        self.assertEqual(result["source"]["scope"], "trusted_catalog")
        self.assertFalse(result["validation"]["provider_called"])
        self.assertEqual(
            [item["role"] for item in result["episodes"]],
            ["policy_under_evaluation", "expert_validation"],
        )
        self.assertEqual(
            [item["result"]["value"] for item in result["episodes"]],
            [False, True],
        )
        self.assertTrue((output_dir / "tool_spec.json").is_file())
        self.assertTrue((output_dir / "resolved_tool_spec.json").is_file())
        self.assertTrue((output_dir / "tool_execution.json").is_file())

    def test_force_codegen_uses_act_false_and_expert_true_contrast(self):
        source = generated_contact_source()
        provider = FakeProvider(f"```python\n{source}```")
        output_dir = self.root / "force_output"
        result = execute_tool_spec(
            self.repo_root,
            self.child_run,
            output_dir,
            contact_tool_spec("force_codegen"),
            provider=provider,
            model="fake-toolgen",
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["route"], "force_codegen")
        self.assertEqual(
            result["tool_spec"], contact_tool_spec("force_codegen")
        )
        self.assertEqual(result["source"]["scope"], "run_local_generated")
        self.assertTrue(result["validation"]["provider_called"])
        self.assertTrue(result["validation"]["all_gates_passed"])
        self.assertEqual(
            [item["role"] for item in result["episodes"]],
            ["policy_under_evaluation", "expert_validation"],
        )
        self.assertEqual(
            [item["result"]["value"] for item in result["episodes"]],
            [False, True],
        )
        self.assertTrue((output_dir / "generated/generated_tool.py").is_file())
        self.assertTrue((output_dir / "generated/registration.json").is_file())

    def test_reuse_does_not_require_false_true_contrast(self):
        child_run = self.root / "run_all_contact"
        write_episode(
            child_run / "evaluation/telemetry/act/episode_000_seed_1",
            policy_name="ACT",
            physical_contact=True,
        )
        write_episode(
            child_run / "evaluation/telemetry/expert/episode_000_seed_1",
            policy_name="expert",
            physical_contact=True,
        )
        result = execute_tool_spec(
            self.repo_root,
            child_run,
            self.root / "reuse_all_contact",
            contact_tool_spec("reuse"),
        )
        self.assertEqual(
            [item["result"]["value"] for item in result["episodes"]],
            [True, True],
        )

    def test_output_directory_must_not_preexist(self):
        output_dir = self.root / "already_exists"
        output_dir.mkdir()
        with self.assertRaisesRegex(ToolOrchestrationError, "已存在"):
            execute_tool_spec(
                self.repo_root,
                self.child_run,
                output_dir,
                contact_tool_spec("reuse"),
            )


if __name__ == "__main__":
    unittest.main()
