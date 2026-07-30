import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.toolkit import aggregate_tool_executions
from mea.feedback.answer_scope import build_answer_scope
from scripts.manipeval_agent import (
    build_evidence_bundle,
    compact_aggregate_result,
    run_round_execution_vqa,
)


class AgentEvidenceIntegrationTests(unittest.TestCase):
    def test_taskgen_failure_does_not_count_requested_act_episode_as_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            evaluation_id = "eval_taskgen_failed_before_act"
            child_dir = repo_root / "mea/generated_tasks/failed_round"
            child_dir.mkdir(parents=True)
            round_plan = {
                "round_id": "round_1",
                "template_id": "performance.control",
                "sub_aspect": "performance.control",
                "task_instruction": "Run the control.",
                "route": "official",
                "execution": {
                    "backend": "act",
                    "seeds": [17],
                    "num_episodes": 1,
                },
            }
            round_summary = {
                "round_id": "round_1",
                "pipeline_passed": False,
                "observations": {
                    "execution_backend": "ACT",
                    "requested_seeds": [17],
                    "actual_seeds": [],
                    "scene_alignment": False,
                    "observed_color": None,
                    "expert_solvable": None,
                    "act_pipeline_status": False,
                    "policy_success": None,
                    "position_samples": [],
                    "position_metrics": {},
                    "aggregate": None,
                    "execution_vqa": {
                        "status": "failed",
                        "artifacts": {},
                    },
                },
            }
            evidence = build_evidence_bundle(
                repo_root,
                evaluation_id,
                "Does the policy pass the control?",
                {
                    "max_rounds": 1,
                    "planning_state": "stopped_after_round_1",
                    "round_decisions": [],
                    "requested_template_ids": ["performance.control"],
                    "requested_aspect_ids": ["performance.control"],
                },
                [
                    {
                        "round_plan": round_plan,
                        "child_manifest": {"run_id": "failed_round"},
                        "child_dir": child_dir,
                        "round_summary": round_summary,
                        "tool_evaluation": {"artifacts": {}},
                    }
                ],
            )

            observed_round = evidence["rounds"][0]
            self.assertEqual(observed_round["requested_seeds"], [17])
            self.assertEqual(observed_round["requested_num_episodes"], 1)
            self.assertEqual(observed_round["actual_seeds"], [])
            self.assertEqual(observed_round["seeds"], [])
            self.assertEqual(observed_round["num_episodes"], 0)
            self.assertEqual(evidence["requested_total_episodes"], 1)
            self.assertEqual(evidence["total_episodes"], 0)

            scope = build_answer_scope(evidence)
            self.assertEqual(scope["sample_count"], 0)
            self.assertEqual(scope["seeds"], [])
            self.assertEqual(scope["termination"], "pipeline_invalid")

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
                return {
                    "schema_version": 1,
                    "observation": {},
                    "evidence_conflict": False,
                    "artifacts": {},
                }

            with patch(
                "scripts.manipeval_agent.run_execution_vqa",
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
                "scripts.manipeval_agent.run_execution_vqa",
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
                "scripts.manipeval_agent.run_execution_vqa",
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
                "scripts.manipeval_agent.run_execution_vqa",
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
