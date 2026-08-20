import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mea.agent_evidence import build_evidence_bundle, compact_aggregate_result
from mea.execution_vqa.runtime import (
    _episode_numeric_tool_results,
    _same_telemetry_episode,
    run_round_execution_vqa,
)
from mea.planner.evidence_policy import (
    RoundEvidenceError,
    build_round_evidence,
    validate_round_evidence,
)
from mea.planner.plan_agent_evidence import (
    build_plan_agent_evidence_record,
    render_query_answer,
)
from mea.toolkit import aggregate_tool_executions
from mea.feedback.answer_scope import build_answer_scope


def completed_round(
    *,
    semantics_status: str = "official_only",
    vqa_conflict: bool = False,
    metric: str = "official_check_success",
    authority: str = "official_check_success",
    official_equivalent: bool = True,
) -> tuple[dict, dict]:
    round_plan = {
        "round_id": "round_1",
        "candidate_id": "dynamic.tested_candidate",
        "sub_aspect": "object_position",
        "task_instruction": "Evaluate the bounded candidate.",
        "observations": ["execution_vqa"] if vqa_conflict else [],
        "semantic_need_execution": {
            "rule_tool": {"requested": False},
            "vqa_tool": {"requested": vqa_conflict},
        },
    }
    round_summary = {
        "round_id": "round_1",
        "pipeline_passed": True,
        "observations": {
            "actual_seeds": [7],
            "policy_success": 1.0,
            "policy_outcome": {
                "metric": metric,
                "authority": authority,
                "official_equivalent": official_equivalent,
                "execution_scope": metric,
            },
            "outcome_semantics": {
                "status": semantics_status,
                "evidence_conflict": semantics_status == "conflict",
            },
            "execution_vqa": {
                "status": "passed" if vqa_conflict else "missing",
                "evidence_conflict": vqa_conflict,
                "observation": (
                    "The visual trace contradicts the numeric result."
                    if vqa_conflict
                    else None
                ),
            },
        },
    }
    round_summary["observations"]["round_evidence"] = build_round_evidence(
        round_plan,
        round_summary,
    )
    return round_plan, round_summary


class AgentEvidenceIntegrationTests(unittest.TestCase):
    def test_vqa_normalizes_episode_identity_and_reads_typed_tool_result(self):
        base = {"tool": "official_check_success", "value": True}
        planned = {"tool": "terminal_object_height", "value": 0.12}
        episode = {
            "tool_results": [base],
            "result": planned,
        }

        self.assertEqual(
            _episode_numeric_tool_results(episode),
            [base, planned],
        )
        self.assertTrue(
            _same_telemetry_episode(
                {
                    "episode_dir": (
                        "mea/generated_tasks/run/evaluation/telemetry/"
                        "act/episode_000"
                    )
                },
                {"episode_dir": "act/episode_000"},
            )
        )

    def test_bundle_reuses_the_single_validated_round_protocol(self):
        round_plan, round_summary = completed_round()
        # The old round-summary envelope may still contain execution-local
        # fields, but the final bundle must not reconstruct facts from it.
        round_summary["observations"]["policy_success"] = 0.0
        round_summary["observations"]["execution_vqa"][
            "evidence_conflict"
        ] = True
        with tempfile.TemporaryDirectory() as temporary:
            evidence = build_evidence_bundle(
                Path(temporary),
                "eval_single_protocol",
                "Where is the bounded weakness?",
                {
                    "max_rounds": 1,
                    "planning_state": "stopped_after_round_1",
                },
                [
                    {
                        "round_plan": round_plan,
                        "round_summary": round_summary,
                    }
                ],
                evaluation_aggregate={"large": "must not be transported"},
            )

        self.assertEqual(
            set(evidence),
            {
                "schema_version",
                "evaluation_id",
                "query",
                "plan",
                "rounds",
                "total_policy_episodes",
                "artifacts",
            },
        )
        self.assertEqual(evidence["schema_version"], 3)
        self.assertEqual(
            evidence["rounds"],
            [round_summary["observations"]["round_evidence"]],
        )
        self.assertEqual(evidence["rounds"][0]["policy"]["success_rate"], 1.0)
        self.assertFalse(evidence["rounds"][0]["vqa"]["evidence_conflict"])
        self.assertEqual(evidence["total_policy_episodes"], 1)
        self.assertNotIn("observations", evidence)
        self.assertNotIn("history_retrieval", evidence)
        self.assertNotIn("global_query_route", evidence)
        self.assertEqual(
            set(evidence["artifacts"]),
            {
                "evaluation_plan",
                "summary",
                "aggregate",
                "round_evidence",
            },
        )
        scope = build_answer_scope(evidence)
        self.assertEqual(scope["sample_count"], 1)
        self.assertEqual(scope["seeds"], [7])
        self.assertEqual(scope["termination"], "budget_exhausted")

    def test_bundle_rejects_missing_round_evidence(self):
        round_plan, round_summary = completed_round()
        round_summary["observations"].pop("round_evidence")

        with self.assertRaisesRegex(
            ValueError,
            "observations.round_evidence",
        ):
            build_evidence_bundle(
                Path("."),
                "eval_missing_round_evidence",
                "Where is the bounded weakness?",
                {"max_rounds": 1},
                [
                    {
                        "round_plan": round_plan,
                        "round_summary": round_summary,
                    }
                ],
            )

    def test_compact_aggregate_preserves_group_statistics(self):
        aggregate = aggregate_tool_executions(
            [
                {
                    "tool_execution": {
                        "status": "passed",
                        "tool_spec": {"metric": "contact"},
                        "episodes": [
                            {
                                "episode_dir": "act/episode_0",
                                "policy_name": "ACT",
                                "seed": 7,
                                "role": "policy_under_evaluation",
                                "result": {
                                    "tool": "contact",
                                    "value": True,
                                    "evidence_steps": [42],
                                },
                            }
                        ],
                    },
                    "context": {
                        "round_id": "round_1",
                        "variant": "blue_block",
                    },
                }
            ]
        )
        compact = compact_aggregate_result(aggregate)
        cohort = compact["metrics"][0]["cohorts"][0]
        self.assertEqual(
            set(cohort["groups"]),
            {"seed", "round_id", "variant", "policy_name"},
        )
        self.assertEqual(cohort["groups"]["seed"][0]["value"], 7)
        self.assertNotIn(
            "provenance",
            cohort["groups"]["seed"][0]["summary"]["statistics"][
                "true_rate"
            ],
        )

    def test_execution_vqa_uses_generated_result_from_same_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            episode_dir = child_dir / "evaluation/telemetry/act/episode_0"
            episode_dir.mkdir(parents=True)
            (episode_dir / "video.mp4").write_bytes(b"video")
            (episode_dir / "episode.json").write_text(
                json.dumps({"artifacts": {"video": "video.mp4"}}),
                encoding="utf-8",
            )
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            child_manifest = {
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "episode_dir": "act/episode_0",
                            "policy_name": "ACT",
                            "seed": 1,
                            "tool_results": [
                                {
                                    "tool": "official_check_success",
                                    "value": False,
                                    "evidence_steps": [],
                                }
                            ],
                        }
                    ]
                }
            }
            generated = {
                "episodes": [
                    {
                        "episode_dir": "act/episode_1",
                        "policy_name": "ACT",
                        "seed": 2,
                        "role": "policy_under_evaluation",
                        "result": {
                            "tool": "duration",
                            "value": 2.0,
                            "evidence_steps": [20],
                        },
                    },
                    {
                        "episode_dir": "act/episode_0",
                        "policy_name": "ACT",
                        "seed": 1,
                        "role": "policy_under_evaluation",
                        "result": {
                            "tool": "duration",
                            "value": 1.0,
                            "evidence_steps": [10],
                        },
                    },
                ]
            }

            captured = {}

            def fake_vqa(**kwargs):
                captured["tools"] = kwargs["numeric_tool_results"]
                captured["reference_scene"] = kwargs["reference_scene"]
                return {
                    "schema_version": 1,
                    "observation": {},
                    "evidence_conflict": False,
                    "artifacts": {},
                }

            with patch(
                "mea.execution_vqa.runtime.run_execution_vqa",
                side_effect=fake_vqa,
            ):
                result = run_round_execution_vqa(
                    repo_root=repo_root,
                    child_manifest=child_manifest,
                    child_dir=child_dir,
                    tool_evaluation=generated,
                    execution_dir=execution_dir,
                    provider=object(),
                    model="vision",
                )

            self.assertEqual(result["status"], "passed")
            duration = next(
                item for item in captured["tools"] if item["tool"] == "duration"
            )
            self.assertEqual(duration["value"], 1.0)
            self.assertIsNone(captured["reference_scene"])

    def test_execution_vqa_abstains_when_visual_evidence_is_insufficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            episode_dir = child_dir / "evaluation/telemetry/act/episode_0"
            episode_dir.mkdir(parents=True)
            (episode_dir / "video.mp4").write_bytes(b"video")
            (episode_dir / "episode.json").write_text(
                json.dumps({"artifacts": {"video": "video.mp4"}}),
                encoding="utf-8",
            )
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            manifest = {
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "episode_dir": "act/episode_0",
                            "policy_name": "ACT",
                            "seed": 1,
                            "tool_results": [],
                        }
                    ]
                }
            }

            with patch(
                "mea.execution_vqa.runtime.run_execution_vqa",
                return_value={
                    "schema_version": 1,
                    "observation": {
                        "phenomena": [
                            {"phenomenon_id": "contact", "observed": None}
                        ],
                        "numeric_consistency": "uncertain",
                    },
                    "evidence_conflict": False,
                    "artifacts": {},
                },
            ):
                result = run_round_execution_vqa(
                    repo_root=repo_root,
                    child_manifest=manifest,
                    child_dir=child_dir,
                    tool_evaluation=None,
                    execution_dir=execution_dir,
                    provider=object(),
                    model="vision",
                )

            self.assertEqual(result["status"], "abstained")
            self.assertIn("insufficient evidence", result["reason"])

    def test_round_evidence_keeps_required_vqa_abstention_as_a_fact(self):
        round_plan = {
            "round_id": "round_1",
            "template_id": "dynamic.visual_check",
            "observations": ["aggregate", "execution_vqa"],
            "execution": {"num_episodes": 1},
            "tool_request": {"metric": "official_check_success"},
            "semantic_need_execution": {
                "rule_tool": {"requested": False},
                "vqa_tool": {"requested": True},
            },
        }
        summary = {
            "round_id": "round_1",
            "pipeline_passed": True,
            "observations": {
                "actual_seeds": [7],
                "policy_success": 1.0,
                "policy_outcome": {
                    "metric": "official_check_success",
                    "authority": "official_check_success",
                    "official_equivalent": True,
                    "execution_scope": "official_equivalent",
                },
                "outcome_semantics": {
                    "status": "official_only",
                    "evidence_conflict": False,
                },
                "execution_vqa": {
                    "status": "abstained",
                    "evidence_conflict": False,
                    "observation": "The video does not show the target.",
                },
            },
        }

        evidence = build_round_evidence(round_plan, summary)

        self.assertEqual(evidence["vqa"]["status"], "abstained")
        self.assertTrue(evidence["vqa"]["required"])
        self.assertEqual(evidence["policy"]["success_rate"], 1.0)
        self.assertNotIn("evidence_strength", evidence)
        self.assertNotIn("coverage", evidence)

    def test_round_evidence_rejects_broken_policy_identity_and_rate_lineage(self):
        _, summary = completed_round()
        evidence = summary["observations"]["round_evidence"]
        broken_cases = (
            (
                "actual seed",
                {
                    **deepcopy(evidence),
                    "policy": {**evidence["policy"], "seeds": []},
                },
            ),
            (
                "official authority",
                {
                    **deepcopy(evidence),
                    "policy": {
                        **evidence["policy"],
                        "authority": "llm_generated_python_ast_validated",
                    },
                },
            ),
            (
                "unsupported outcome_semantics.status",
                {
                    **deepcopy(evidence),
                    "outcome_semantics": {
                        **evidence["outcome_semantics"],
                        "status": "surprising",
                    },
                },
            ),
        )

        for error, broken in broken_cases:
            with self.subTest(error=error), self.assertRaisesRegex(
                RoundEvidenceError,
                error,
            ):
                validate_round_evidence(broken)

    def test_non_comparable_completed_rate_remains_candidate_unknown(self):
        round_plan, summary = completed_round(
            semantics_status="non_comparable"
        )

        record = build_plan_agent_evidence_record(round_plan, summary)

        self.assertEqual(record["candidate_evidence"]["outcome"], "unknown")
        self.assertEqual(record["open_query_evidence"]["outcome"], "ambiguous")

    def test_bound_generated_extension_remains_experimentally_decidable(self):
        round_plan, summary = completed_round(
            semantics_status="expected_semantic_extension",
            metric="generated_check_success",
            authority="llm_generated_python_ast_validated",
            official_equivalent=False,
        )

        record = build_plan_agent_evidence_record(round_plan, summary)

        self.assertEqual(record["candidate_evidence"]["outcome"], "pass")
        self.assertTrue(
            any(
                "not an official RoboTwin success result" in item
                for item in record["open_query_evidence"]["limitations"]
            )
        )

    def test_vqa_conflict_reaches_candidate_and_query_answer(self):
        round_plan, summary = completed_round(vqa_conflict=True)
        record = build_plan_agent_evidence_record(round_plan, summary)

        answer = render_query_answer(
            "Does this bounded candidate expose a weakness?",
            {
                "evidence_sufficient": False,
                "stop_reason": "continue",
                "claim_verdict": "inconclusive",
                "observed_candidate_ids": [record["candidate_id"]],
                "limitations": [],
            },
            [record],
            baseline_valid=True,
        )

        self.assertEqual(record["candidate_evidence"]["outcome"], "conflict")
        self.assertIn(
            "VQA evidence_conflict=True",
            record["open_query_evidence"]["evidence_summary"],
        )
        self.assertTrue(answer["evidence_conflict"])
        self.assertTrue(
            any("VQA evidence" in item for item in answer["limitations"])
        )

    def test_completed_act_without_video_is_failed_not_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            episode_dir = child_dir / "evaluation/telemetry/act/episode_0"
            episode_dir.mkdir(parents=True)
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            manifest = {
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "episode_dir": "act/episode_0",
                            "policy_name": "ACT",
                            "seed": 1,
                            "tool_results": [],
                        }
                    ]
                }
            }
            result = run_round_execution_vqa(
                repo_root=repo_root,
                child_manifest=manifest,
                child_dir=child_dir,
                tool_evaluation=None,
                execution_dir=execution_dir,
                provider=object(),
                model="vision",
                round_plan={
                    "route": "official",
                    "task_name": "click_bell",
                    "template_id": "task_execution.official_baseline",
                    "sub_aspect": "task_execution.official_baseline",
                    "execution": {"backend": "act"},
                    "tool_request": {
                        "task_name": "click_bell",
                        "metric": "official_check_success",
                    },
                },
            )
            self.assertEqual(result["status"], "failed")
            saved = json.loads(
                (execution_dir / "execution_vqa_error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("missing video.mp4", saved["reason"])

    def test_typed_pre_policy_observation_is_raw_n_zero_round_evidence(self):
        round_plan = {
            "round_id": "round_2",
            "candidate_id": "generated_candidate",
            "execution": {"seeds": [1000]},
            "semantic_need_execution": {
                "rule_tool": {"requested": False},
                "vqa_tool": {"requested": False},
            },
        }
        summary = {
            "round_id": "round_2",
            "pipeline_passed": False,
            "failure_stage": "taskgen_expert_gate",
            "observations": {
                "actual_seeds": [],
                "planning_observation": {
                    "kind": "expert_oracle_unavailable",
                    "reason_code": (
                        "taskgen_expert_gate_official_baseline_unsolvable"
                    ),
                    "policy_rollouts_started": 0,
                    "policy_sample_count": 0,
                },
                "policy_success": None,
                "policy_outcome": {
                    "metric": None,
                    "authority": None,
                    "official_equivalent": None,
                    "execution_scope": "not_executed",
                },
                "outcome_semantics": {
                    "status": "non_comparable",
                    "evidence_conflict": False,
                },
            },
        }

        evidence = build_round_evidence(round_plan, summary)

        self.assertEqual(evidence["schema_version"], 1)
        self.assertFalse(evidence["pipeline"]["passed"])
        self.assertEqual(
            evidence["planning_observation"]["reason_code"],
            "taskgen_expert_gate_official_baseline_unsolvable",
        )
        self.assertEqual(validate_round_evidence(evidence), evidence)
        self.assertNotIn("reason_codes", evidence)

    def test_official_act_without_act_candidate_is_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            child_dir.mkdir(parents=True)
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            result = run_round_execution_vqa(
                repo_root=repo_root,
                child_manifest={"trusted_tool_evaluation": {"episodes": []}},
                child_dir=child_dir,
                tool_evaluation=None,
                execution_dir=execution_dir,
                provider=object(),
                model="vision",
                round_plan={
                    "route": "official",
                    "task_name": "click_bell",
                    "execution": {"backend": "act"},
                },
            )
            self.assertEqual(result["status"], "failed")
            self.assertTrue(
                (execution_dir / "execution_vqa_error.json").is_file()
            )

    def test_official_both_uses_act_video_without_visual_capture_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            expert_dir = child_dir / "evaluation/telemetry/expert/episode_0"
            act_dir = child_dir / "evaluation/telemetry/act/episode_0"
            expert_dir.mkdir(parents=True)
            act_dir.mkdir(parents=True)
            (expert_dir / "video.mp4").write_bytes(b"expert video")
            (expert_dir / "episode.json").write_text(
                json.dumps(
                    {
                        "visual_capture": {"status": "completed"},
                        "artifacts": {"video": "video.mp4"},
                    }
                ),
                encoding="utf-8",
            )
            # ACT produces a continuous rollout video. It deliberately has no
            # expert-only event-keyframe visual_capture declaration.
            (act_dir / "video.mp4").write_bytes(b"act video")
            (act_dir / "episode.json").write_text(
                json.dumps({"artifacts": {"video": "video.mp4"}}),
                encoding="utf-8",
            )
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            manifest = {
                "task_name": "click_bell",
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "episode_dir": "expert/episode_0",
                            "policy_name": "expert",
                            "seed": 7,
                            "tool_results": [],
                        },
                        {
                            "episode_dir": "act/episode_0",
                            "policy_name": "ACT",
                            "seed": 7,
                            "tool_results": [
                                {
                                    "tool": "official_check_success",
                                    "value": False,
                                    "evidence_steps": [],
                                }
                            ],
                        },
                    ]
                },
            }
            captured = {}

            def fake_vqa(**kwargs):
                captured.update(kwargs)
                return {
                    "schema_version": 1,
                    "observation": {},
                    "evidence_conflict": False,
                    "artifacts": {},
                }

            with patch(
                "mea.execution_vqa.runtime.run_execution_vqa",
                side_effect=fake_vqa,
            ):
                result = run_round_execution_vqa(
                    repo_root=repo_root,
                    child_manifest=manifest,
                    child_dir=child_dir,
                    tool_evaluation=None,
                    execution_dir=execution_dir,
                    provider=object(),
                    model="vision",
                    round_plan={
                        "route": "official",
                        "task_name": "click_bell",
                        "template_id": "task_execution.official_baseline",
                        "sub_aspect": "task_execution.official_baseline",
                        "execution": {"backend": "both"},
                        "tool_request": {
                            "task_name": "click_bell",
                            "metric": "official_check_success",
                        },
                    },
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(captured["video_path"], act_dir / "video.mp4")
            self.assertEqual(
                captured["numeric_tool_results"][0]["tool"],
                "official_check_success",
            )

    def test_official_route_uses_expert_video_and_numeric_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            episode_dir = child_dir / "evaluation/telemetry/expert/episode_0"
            episode_dir.mkdir(parents=True)
            (episode_dir / "video.mp4").write_bytes(b"video")
            (episode_dir / "episode.json").write_text(
                json.dumps(
                    {
                        "visual_capture": {"status": "completed"},
                        "artifacts": {"video": "video.mp4"},
                    }
                ),
                encoding="utf-8",
            )
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            manifest = {
                "task_name": "click_bell",
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "episode_dir": "expert/episode_0",
                            "policy_name": "expert",
                            "seed": 7,
                            "tool_results": [
                                {
                                    "tool": "official_check_success",
                                    "value": True,
                                    "evidence_steps": [12],
                                }
                            ],
                        }
                    ]
                },
            }
            round_plan = {
                "route": "official",
                "task_name": "click_bell",
                "template_id": "task_execution.official_baseline",
                "sub_aspect": "task_execution.official_baseline",
                "tool_request": {
                    "task_name": "click_bell",
                    "metric": "official_check_success",
                },
            }
            captured = {}

            def fake_vqa(**kwargs):
                captured.update(kwargs)
                return {
                    "schema_version": 1,
                    "observation": {},
                    "evidence_conflict": False,
                    "artifacts": {},
                }

            with patch(
                "mea.execution_vqa.runtime.run_execution_vqa",
                side_effect=fake_vqa,
            ):
                result = run_round_execution_vqa(
                    repo_root=repo_root,
                    child_manifest=manifest,
                    child_dir=child_dir,
                    tool_evaluation=None,
                    execution_dir=execution_dir,
                    provider=object(),
                    model="vision",
                    round_plan=round_plan,
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(captured["video_path"], episode_dir / "video.mp4")
            self.assertEqual(
                captured["numeric_tool_results"][0]["tool"],
                "official_check_success",
            )

    def test_official_episode_without_video_remains_auditable_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            episode_dir = child_dir / "evaluation/telemetry/expert/episode_0"
            episode_dir.mkdir(parents=True)
            (episode_dir / "episode.json").write_text(
                json.dumps(
                    {
                        "visual_capture": {
                            "profile_id": "event_keyframes_v1",
                            "status": "failed",
                        }
                    }
                ),
                encoding="utf-8",
            )
            # A residual file is not valid evidence when the episode contract
            # says visual capture failed.
            (episode_dir / "video.mp4").write_bytes(b"unapproved video")
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            manifest = {
                "task_name": "click_bell",
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "episode_dir": "expert/episode_0",
                            "policy_name": "expert",
                            "seed": 7,
                            "tool_results": [],
                        }
                    ]
                },
            }
            result = run_round_execution_vqa(
                repo_root=repo_root,
                child_manifest=manifest,
                child_dir=child_dir,
                tool_evaluation=None,
                execution_dir=execution_dir,
                provider=object(),
                model="vision",
                round_plan={
                    "route": "official",
                    "task_name": "click_bell",
                    "template_id": "task_execution.official_baseline",
                    "sub_aspect": "task_execution.official_baseline",
                    "tool_request": {
                        "task_name": "click_bell",
                        "metric": "official_check_success",
                    },
                },
            )
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["visual_capture"]["status"], "failed")
            self.assertTrue(
                (execution_dir / "execution_vqa_skipped.json").is_file()
            )

    def test_official_route_prefers_later_expert_episode_with_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run"
            missing = child_dir / "evaluation/telemetry/expert/episode_0"
            available = child_dir / "evaluation/telemetry/expert/episode_1"
            missing.mkdir(parents=True)
            available.mkdir(parents=True)
            (available / "video.mp4").write_bytes(b"video")
            (available / "episode.json").write_text(
                json.dumps(
                    {
                        "visual_capture": {"status": "completed"},
                        "artifacts": {"video": "video.mp4"},
                    }
                ),
                encoding="utf-8",
            )
            execution_dir = repo_root / "mea/evaluation_runs/e/execution/round_1"
            manifest = {
                "task_name": "click_bell",
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "episode_dir": "expert/episode_0",
                            "policy_name": "expert",
                            "seed": 7,
                            "tool_results": [],
                        },
                        {
                            "episode_dir": "expert/episode_1",
                            "policy_name": "expert",
                            "seed": 8,
                            "tool_results": [],
                        },
                    ]
                },
            }
            captured = {}

            def fake_vqa(**kwargs):
                captured.update(kwargs)
                return {
                    "schema_version": 1,
                    "observation": {},
                    "evidence_conflict": False,
                    "artifacts": {},
                }

            with patch(
                "mea.execution_vqa.runtime.run_execution_vqa",
                side_effect=fake_vqa,
            ):
                result = run_round_execution_vqa(
                    repo_root=repo_root,
                    child_manifest=manifest,
                    child_dir=child_dir,
                    tool_evaluation=None,
                    execution_dir=execution_dir,
                    provider=object(),
                    model="vision",
                    round_plan={
                        "route": "official",
                        "task_name": "click_bell",
                        "template_id": "task_execution.official_baseline",
                        "sub_aspect": "task_execution.official_baseline",
                        "tool_request": {
                            "task_name": "click_bell",
                            "metric": "official_check_success",
                        },
                    },
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(captured["video_path"], available / "video.mp4")


if __name__ == "__main__":
    unittest.main()
