"""Tests for the bounded Fig. 3/4 target/distractor candidate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mea.taskgen.bbh_distractor import (
    BBHDistractorTaskGenError,
    bbh_distractor_rollout_execution,
    default_bbh_distractor_proposal,
    materialize_bbh_distractor_candidate,
    reference_bbh_distractor_methods,
    run_bbh_distractor_checker_fixtures,
    validate_bbh_distractor_methods,
    validate_bbh_distractor_proposal,
)
from mea.toolgen.query_induced import (
    QueryInducedToolError,
    query_induced_result_to_tool_execution,
)
from mea.toolkit.aggregate import aggregate_tool_executions


class _Provider:
    def __init__(self, response: dict[str, str]) -> None:
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []
        self.last_metadata = {"id": "fixture-codegen-call"}

    def text(self, prompt: str, **_kwargs: object) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return json.dumps(self.response)


def _episode(
    root: Path,
    *,
    task_module: str,
    success: bool = True,
) -> Path:
    episode = root / "episode_seed_17"
    episode.mkdir(parents=True)
    count = 3
    vector = np.zeros((count, 3), dtype=float)
    trace = {
        "physics_step": np.arange(count, dtype=np.int64),
        "policy_step": np.arange(count, dtype=np.int64),
        "simulation_time_seconds": np.arange(count, dtype=float) * 0.02,
        "success": np.asarray([False, False, success], dtype=bool),
        "hammer_position": vector,
        "block_position": vector,
        "hammer_functional_position": vector,
        "block_functional_position": vector,
        "left_tcp_position": vector,
        "right_tcp_position": vector,
    }
    np.savez(episode / "semantic_trace.npz", **trace)
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "task_name": "beat_block_hammer",
                "task_module": task_module,
                "physics_steps": count - 1,
                "semantic_trace_rows": count,
                "success": success,
                "policy_name": "ACT",
                "seed": 17,
            }
        ),
        encoding="utf-8",
    )
    (episode / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": "beat_block_hammer",
                "signals": sorted(trace),
            }
        ),
        encoding="utf-8",
    )
    (episode / "states.csv").write_text(
        "policy_step\n0\n1\n2\n", encoding="utf-8"
    )
    events = (
        [
            {
                "type": "success_transition",
                "physics_step": 2,
                "policy_step": 2,
                "simulation_time_seconds": 0.04,
            }
        ]
        if success
        else []
    )
    (episode / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
    )
    return episode


def _jerk_proposal() -> dict[str, object]:
    return {
        "schema_version": 2,
        "metric_id": "precontact_tcp_jerk_peak",
        "finite_difference_order": 3,
        "window_seconds": 0.12,
        "reducer": "peak_l2",
        "time_normalization": "physical_seconds",
        "threshold": 50.0,
        "unit": "m_per_second_cubed",
        "null_semantics": (
            "null_if_no_target_contact_or_insufficient_precontact_samples"
        ),
        "rationale": "query asks about abrupt pre-contact motion",
    }


def _query_result(value: float | None, reason: str | None) -> dict[str, object]:
    return {
        "schema_version": 2,
        "query": "Is there abrupt motion before target contact?",
        "route": "provider_generate_validate_register",
        "proposal": _jerk_proposal(),
        "registration_id": "run_local_fixture",
        "live_telemetry": {
            "episode_dir": "/recorded/episode_seed_17",
            "task_name": "beat_block_hammer",
            "synthetic_fallback_used": False,
        },
        "tool_result": {
            "tool": "precontact_tcp_jerk_peak",
            "value": value,
            "unit": "m_per_second_cubed",
            "passed": value is not None and value <= 50.0,
            "null_semantics": (
                "null_if_no_target_contact_or_insufficient_precontact_samples"
            ),
            "null_reason": reason,
            "active_arm": "left",
            "signal": "left_tcp_position",
            "first_target_contact_trace_index": (
                9 if value is not None else None
            ),
            "evidence_steps": [9] if value is not None else [],
        },
    }


class BBHDistractorTaskGenTests(unittest.TestCase):
    def test_proposal_and_ast_policy_are_fail_closed(self) -> None:
        proposal = default_bbh_distractor_proposal()
        validated = validate_bbh_distractor_proposal(proposal)
        methods = reference_bbh_distractor_methods(validated)
        report = validate_bbh_distractor_methods(methods, validated)
        self.assertTrue(report["valid"])
        self.assertTrue(report["model_written_python"])
        self.assertFalse(report["restricted_success_spec_compiler_used"])

        too_far = default_bbh_distractor_proposal()
        too_far["scene"]["distractor_offset_xy_m"] = [0.3, 0.0]
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError, "separation"
        ):
            validate_bbh_distractor_proposal(too_far)

        unsafe = dict(methods)
        unsafe["check_success"] = "import os\n" + unsafe["check_success"]
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError, "exact proposal-derived AST"
        ):
            validate_bbh_distractor_methods(unsafe, validated)

        accepts_distractor = dict(methods)
        accepts_distractor["check_success"] = accepts_distractor[
            "check_success"
        ].replace("and not self._mea_distractor_contact_seen", "")
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError, "exact proposal-derived AST"
        ):
            validate_bbh_distractor_methods(accepts_distractor, validated)

    def test_three_checker_fixtures_include_latched_mishit(self) -> None:
        proposal = default_bbh_distractor_proposal()
        methods = reference_bbh_distractor_methods(proposal)
        fixtures = run_bbh_distractor_checker_fixtures(
            methods["check_success"], proposal
        )
        self.assertEqual(
            [item["fixture"] for item in fixtures],
            [
                "target_contact",
                "distractor_contact_latched",
                "no_contact",
            ],
        )
        self.assertTrue(all(item["passed"] for item in fixtures))
        self.assertEqual(fixtures[1]["calls"], [False, False])

    def test_model_provenance_rollout_binding_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = default_bbh_distractor_proposal()
            provider = _Provider(reference_bbh_distractor_methods(proposal))
            manifest = materialize_bbh_distractor_candidate(
                repo_root=root,
                run_id="run_fixture_distractor",
                proposal=proposal,
                provider=provider,
                model="fixture-model",
            )
            self.assertEqual(provider.calls, 1)
            self.assertIn(
                '"proposal_id": "bbh.lookalike_distractor.v1"',
                provider.prompts[0],
            )
            provenance = manifest["codegen_provenance"]
            self.assertEqual(
                provenance["source_kind"], "provider_response_python"
            )
            self.assertTrue(provenance["generated_by_model"])
            self.assertFalse(
                provenance["restricted_success_spec_compiler_used"]
            )
            self.assertEqual(
                manifest["live_boundary"]["act_rollouts_completed"], 0
            )
            candidate = (
                root
                / "mea/generated_tasks/run_fixture_distractor"
            )
            episode = _episode(
                root,
                task_module=manifest["task_module"],
            )
            execution = bbh_distractor_rollout_execution(
                episode_dir=episode,
                candidate_dir=candidate,
            )
            aggregate = aggregate_tool_executions([execution])
            metric = aggregate["metrics"][0]
            self.assertEqual(
                metric["metric"],
                "bbh_target_without_distractor_success",
            )
            cohort = metric["cohorts"][0]
            self.assertEqual(
                cohort["summary"]["statistics"]["true_rate"]["value"],
                1.0,
            )
            details = execution["episodes"][0]["result"]["details"]
            self.assertFalse(details["official_success"])
            self.assertEqual(
                details["authority"],
                "llm_generated_python_ast_validated",
            )

            wrong = _episode(
                root / "wrong",
                task_module="envs.beat_block_hammer",
            )
            with self.assertRaisesRegex(
                BBHDistractorTaskGenError, "task_module differs"
            ):
                bbh_distractor_rollout_execution(
                    episode_dir=wrong,
                    candidate_dir=candidate,
                )

            metadata_path = episode / "episode.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["success"] = 1
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                BBHDistractorTaskGenError,
                "JSON boolean",
            ):
                bbh_distractor_rollout_execution(
                    episode_dir=episode,
                    candidate_dir=candidate,
                )

    def test_query_tool_bridge_preserves_numeric_and_null_evidence(self) -> None:
        numeric = query_induced_result_to_tool_execution(
            _query_result(12.5, None),
            seed=17,
        )
        missing = query_induced_result_to_tool_execution(
            _query_result(None, "no_target_contact_event"),
            seed=18,
        )
        aggregate = aggregate_tool_executions([numeric, missing])
        cohort = aggregate["metrics"][0]["cohorts"][0]
        self.assertEqual(cohort["summary"]["quality"]["valid"]["value"], 1)
        self.assertEqual(cohort["summary"]["quality"]["missing"]["value"], 1)
        self.assertEqual(
            cohort["summary"]["statistics"]["mean"]["value"], 12.5
        )
        self.assertEqual(
            missing["episodes"][0]["result"]["details"]["reason"],
            "no_target_contact_event",
        )
        self.assertFalse(
            missing["episodes"][0]["result"]["details"][
                "synthetic_fallback_used"
            ]
        )

        invalid = _query_result(None, "no_target_contact_event")
        invalid["live_telemetry"]["synthetic_fallback_used"] = True
        with self.assertRaisesRegex(
            QueryInducedToolError, "real recorded telemetry"
        ):
            query_induced_result_to_tool_execution(invalid)


if __name__ == "__main__":
    unittest.main()
