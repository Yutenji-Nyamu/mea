import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.toolkit import aggregate_tool_executions
from scripts.manipeval_agent import (
    compact_aggregate_result,
    run_round_execution_vqa,
)


class AgentEvidenceIntegrationTests(unittest.TestCase):
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
            )
            self.assertEqual(result["status"], "failed")
            saved = json.loads(
                (execution_dir / "execution_vqa_error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("missing video.mp4", saved["reason"])


if __name__ == "__main__":
    unittest.main()
