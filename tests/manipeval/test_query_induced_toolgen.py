"""Fixture tests for the bounded query-induced ToolGen v2 path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.toolgen.query_induced import (
    QueryInducedToolError,
    compile_metric_source,
    query_fingerprint,
    run_query_induced_toolgen,
    validate_compiled_source,
    validate_precontact_motion_oracles,
    validate_query_metric_proposal,
)


def _proposal(
    *,
    metric_id: str = "precontact_tcp_jerk_peak",
    order: int = 3,
    rationale: str = "measure abrupt active-TCP motion before target contact",
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "metric_id": metric_id,
        "finite_difference_order": order,
        "window_seconds": 0.12,
        "reducer": "peak_l2",
        "time_normalization": "physical_seconds",
        "threshold": 50.0 if order == 3 else 5.0,
        "unit": (
            "m_per_second_cubed"
            if order == 3
            else "m_per_second_squared"
        ),
        "null_semantics": (
            "null_if_no_target_contact_or_insufficient_precontact_samples"
        ),
        "rationale": rationale,
    }


class _Provider:
    def __init__(self, *responses: dict[str, object]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []
        self.last_metadata = {"id": "fixture-provider-call"}

    def text(self, prompt: str, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        self.prompts.append(prompt)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return json.dumps(response)


def _episode(root: Path, *, block_x: float = -0.2) -> Path:
    episode = root / ("episode_left" if block_x < 0.0 else "episode_right")
    episode.mkdir()
    count = 12
    times = np.arange(count, dtype=float) * 0.02
    left = np.zeros((count, 3), dtype=float)
    right = np.asarray(
        [
            [0.0 if index % 2 == 0 else 0.02, 0.0, 0.0]
            for index in range(count)
        ],
        dtype=float,
    )
    trace = {
        "physics_step": np.arange(count, dtype=np.int64),
        "policy_step": np.arange(count, dtype=np.int64),
        "simulation_time_seconds": times,
        "success": np.zeros(count, dtype=bool),
        "block_position": np.asarray(
            [[block_x, 0.0, 0.0] for _ in range(count)],
            dtype=float,
        ),
        "left_tcp_position": left,
        "right_tcp_position": right,
    }
    np.savez(episode / "semantic_trace.npz", **trace)
    (episode / "episode.json").write_text(
        json.dumps({"task_name": "toy_motion", "physics_steps": count - 1}),
        encoding="utf-8",
    )
    (episode / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": "toy_motion",
                "signals": sorted(trace),
            }
        ),
        encoding="utf-8",
    )
    (episode / "states.csv").write_text(
        "policy_step\n0\n",
        encoding="utf-8",
    )
    (episode / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "contact_interval",
                "physical_contact": True,
                "first_physical_physics_step": count - 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return episode


class QueryInducedToolGenTests(unittest.TestCase):
    def test_compositional_dsl_oracles_and_exact_ast(self) -> None:
        for order in (2, 3):
            proposal = _proposal(order=order)
            validated = validate_query_metric_proposal(proposal)
            source = compile_metric_source(validated)
            self.assertEqual(
                validate_compiled_source(source, validated),
                validate_compiled_source(source, validated),
            )
            oracle = validate_precontact_motion_oracles(
                proposal=validated,
                source=source,
            )
            self.assertEqual(
                [item["fixture"] for item in oracle],
                ["smooth", "oscillatory", "missing_target_contact"],
            )
            self.assertTrue(all(item["passed"] for item in oracle))
            with self.assertRaises(QueryInducedToolError):
                validate_compiled_source(source + "\nimport os\n", validated)

        wrong_unit = _proposal(order=2)
        wrong_unit["unit"] = "m_per_second_cubed"
        with self.assertRaises(QueryInducedToolError):
            validate_query_metric_proposal(wrong_unit)
        provider_owned_signal = _proposal()
        provider_owned_signal["signal"] = "left_tcp_position"
        with self.assertRaises(QueryInducedToolError):
            validate_query_metric_proposal(provider_owned_signal)

    def test_runtime_derives_active_arm_from_initial_block_x(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = _Provider(_proposal())
            left = run_query_induced_toolgen(
                query="Is motion abrupt immediately before target contact?",
                episode_dir=_episode(root, block_x=-0.2),
                output_dir=root / "left_output",
                registry_dir=root / "registry",
                provider=provider,
                model="fixture-model",
            )
            self.assertEqual(left["tool_result"]["active_arm"], "left")
            self.assertEqual(
                left["tool_result"]["signal"],
                "left_tcp_position",
            )
            self.assertEqual(
                left["tool_result"]["active_arm_rule"],
                "initial_block_position_x_lt_0_left_else_right",
            )
            self.assertTrue(left["tool_result"]["passed"])
            self.assertNotIn("signal", left["proposal"])
            right = run_query_induced_toolgen(
                query="Is motion abrupt immediately before target contact?",
                episode_dir=_episode(root, block_x=0.2),
                output_dir=root / "right_output",
                registry_dir=root / "registry",
                provider=None,
                model="fixture-model",
            )
            self.assertEqual(right["route"], "exact_query_registry_reuse")
            self.assertEqual(right["tool_result"]["active_arm"], "right")
            self.assertEqual(
                right["tool_result"]["signal"],
                "right_tcp_position",
            )
            self.assertFalse(right["tool_result"]["passed"])

    def test_reported_nonphysical_interval_is_not_target_contact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = _episode(root, block_x=0.2)
            (episode / "events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "contact_interval",
                        "actors": ["020_hammer", "box"],
                        "physical_contact": False,
                        "start_physics_step": 3,
                        "first_physical_physics_step": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = run_query_induced_toolgen(
                query="Is motion abrupt immediately before target contact?",
                episode_dir=episode,
                output_dir=root / "output",
                registry_dir=root / "registry",
                provider=_Provider(_proposal()),
                model="fixture-model",
            )
            self.assertIsNone(result["tool_result"]["value"])
            self.assertEqual(
                result["tool_result"]["null_reason"],
                "no_target_contact_event",
            )
            self.assertIsNone(
                result["tool_result"]["first_target_contact_trace_index"]
            )

    def test_exact_query_reuses_without_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = _Provider(_proposal())
            common = {
                "query": "接触前是否有抖动或急动",
                "episode_dir": _episode(root),
                "registry_dir": root / "registry",
                "model": "fixture-model",
            }
            first = run_query_induced_toolgen(
                output_dir=root / "first",
                provider=provider,
                **common,
            )
            second = run_query_induced_toolgen(
                output_dir=root / "second",
                provider=None,
                **common,
            )
            self.assertEqual(
                first["route"],
                "provider_generate_validate_register",
            )
            self.assertEqual(second["route"], "exact_query_registry_reuse")
            self.assertEqual(provider.calls, 1)
            self.assertTrue(first["provider_called"])
            self.assertFalse(second["provider_called"])
            self.assertTrue(first["codegen_performed"])
            self.assertFalse(second["codegen_performed"])
            self.assertFalse(
                first["live_telemetry"]["synthetic_fallback_used"]
            )

    def test_paraphrase_provider_can_reuse_without_codegen_or_register(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_proposal = _proposal()
            paraphrase_proposal = _proposal(
                rationale="reuse the validated metric for this paraphrase"
            )
            provider = _Provider(first_proposal, paraphrase_proposal)
            episode = _episode(root)
            first = run_query_induced_toolgen(
                query="Does the active TCP jerk before contact?",
                episode_dir=episode,
                output_dir=root / "first",
                registry_dir=root / "registry",
                provider=provider,
                model="fixture-model",
            )
            second = run_query_induced_toolgen(
                query="Is there abrupt end-effector motion just before touch?",
                episode_dir=episode,
                output_dir=root / "second",
                registry_dir=root / "registry",
                provider=provider,
                model="fixture-model",
            )
            self.assertNotEqual(
                first["query_fingerprint"],
                second["query_fingerprint"],
            )
            self.assertEqual(
                second["route"],
                "provider_semantic_registry_reuse",
            )
            self.assertEqual(provider.calls, 2)
            self.assertTrue(second["provider_called"])
            self.assertFalse(second["codegen_performed"])
            self.assertFalse(second["oracle_validation_performed"])
            self.assertFalse(second["registration_performed"])
            self.assertEqual(
                first["registration_id"],
                second["registration_id"],
            )
            self.assertGreaterEqual(
                second["registered_summary_count_presented"],
                1,
            )
            self.assertIn(
                '"metric_id": "precontact_tcp_jerk_peak"',
                provider.prompts[1],
            )
            self.assertIn(
                '"finite_difference_order": 3',
                provider.prompts[1],
            )
            self.assertNotIn(
                '"finite_difference_order": "2 or 3"',
                provider.prompts[1],
            )
            self.assertFalse((root / "second" / "generated_tool.py").exists())
            registry_index = json.loads(
                (root / "registry" / "index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(registry_index["entries"]), 1)

    def test_metric_id_collision_and_unseen_reuse_only_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _proposal()
            collision = _proposal(order=2)
            provider = _Provider(first, collision)
            episode = _episode(root)
            run_query_induced_toolgen(
                query="first query",
                episode_dir=episode,
                output_dir=root / "first",
                registry_dir=root / "registry",
                provider=provider,
            )
            with self.assertRaisesRegex(
                QueryInducedToolError,
                "collides",
            ):
                run_query_induced_toolgen(
                    query="different semantics with a reused name",
                    episode_dir=episode,
                    output_dir=root / "collision",
                    registry_dir=root / "registry",
                    provider=provider,
                )
            with self.assertRaisesRegex(
                QueryInducedToolError,
                "provider is required",
            ):
                run_query_induced_toolgen(
                    query="unseen exact language",
                    episode_dir=episode,
                    output_dir=root / "unseen",
                    registry_dir=root / "registry",
                    provider=None,
                )

    def test_query_fingerprint_has_no_semantic_alias_table(self) -> None:
        exact = query_fingerprint(" Contact before jitter ")
        same_normalized = query_fingerprint("contact   before JITTER")
        paraphrase = query_fingerprint("before contact jitter")
        self.assertEqual(exact, same_normalized)
        self.assertNotEqual(exact, paraphrase)


if __name__ == "__main__":
    unittest.main()
