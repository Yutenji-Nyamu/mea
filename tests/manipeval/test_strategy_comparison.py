import json
import tempfile
import unittest
from pathlib import Path

from mea.strategy_comparison import StrategyComparisonError, compare_fixed_dynamic


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(
    root: Path,
    evaluation_id: str,
    planning_policy: str,
    variants: list[str],
    *,
    candidates: list[str],
    seed: int = 7,
    checkpoint: str = "demo_clean",
) -> str:
    relative = f"mea/evaluation_runs/{evaluation_id}"
    evaluation = root / relative
    policy = {
        "name": "ACT",
        "checkpoint_setting": checkpoint,
        "expert_data_num": 50,
        "language_conditioned": False,
    }
    import hashlib

    digest = hashlib.sha256(
        json.dumps(
            candidates,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    rounds = []
    for index, variant in enumerate(variants, start=1):
        child_id = f"run_{evaluation_id}_{index}"
        episode_dir = f"act/episode_{index}"
        write_json(
            root / f"mea/generated_tasks/{child_id}/manifest.json",
            {
                "run_id": child_id,
                "status": "completed",
                "trusted_tool_evaluation": {
                    "episodes": [
                        {
                            "policy_name": "ACT",
                            "episode_dir": episode_dir,
                        }
                    ]
                }
            },
        )
        write_json(
            root
            / f"mea/generated_tasks/{child_id}/evaluation/telemetry/{episode_dir}/episode.json",
            {
                "task_name": "click_bell",
                "policy_name": "ACT",
                "seed": seed,
                "success": index % 2 == 0,
                "policy_steps": 10,
                "physics_steps": 100,
                "wall_duration_seconds": 2.0,
                "error": None,
            },
        )
        rounds.append(
            {
                "variant_id": variant,
                "taskgen_run_id": child_id,
                "pipeline_passed": True,
            }
        )
    write_json(
        evaluation / "manifest.json",
        {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "lifecycle_status": "completed",
            "created_at": "2026-07-17T10:00:00+08:00",
            "execution_finished_at": "2026-07-17T10:01:00+08:00",
            "base_commit": "abc",
            "task_name": "click_bell",
            "task_profile": "fixed_suite"
            if planning_policy.startswith("fixed")
            else "adaptive_properties",
            "telemetry_profile": "balanced_v1",
            "planning_policy": planning_policy,
            "candidate_suite_sha256": digest,
            "user_request": "Evaluate click_bell object generalization.",
            "global_route_selection": {
                "route": "supported",
                "task_name": "click_bell",
            },
            "plan": {
                "policy": policy,
                "requested_template_ids": candidates,
            },
        },
    )
    write_json(
        evaluation / "summary/summary.json", {"status": "completed", "rounds": rounds}
    )
    return relative


class StrategyComparisonTests(unittest.TestCase):
    def test_n1_fixed_four_dynamic_two_is_honest_micro_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = ["left", "right", "base0", "base1"]
            fixed = make_run(
                root,
                "eval_fixed",
                "fixed_predeclared_v1",
                candidates,
                candidates=candidates,
            )
            dynamic = make_run(
                root,
                "eval_dynamic",
                "dynamic_evidence_v1",
                candidates[:2],
                candidates=candidates,
            )
            result = compare_fixed_dynamic(
                root,
                {
                    "schema_version": 1,
                    "fixed_evaluation_dir": fixed,
                    "dynamic_evaluation_dir": dynamic,
                },
            )
            self.assertEqual(result["rollout_savings"], 2)
            self.assertEqual(result["overlap_exact_success_agreement_rate"], 1.0)
            self.assertFalse(result["paper_table_eligible"])
            self.assertIsNone(result["table2_consistency"])

    def test_checkpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = ["left", "right"]
            fixed = make_run(
                root,
                "eval_fixed",
                "fixed_predeclared_v1",
                candidates,
                candidates=candidates,
            )
            dynamic = make_run(
                root,
                "eval_dynamic",
                "dynamic_evidence_v1",
                candidates,
                candidates=candidates,
                checkpoint="different",
            )
            with self.assertRaisesRegex(StrategyComparisonError, "identity mismatch"):
                compare_fixed_dynamic(
                    root,
                    {
                        "schema_version": 1,
                        "fixed_evaluation_dir": fixed,
                        "dynamic_evaluation_dir": dynamic,
                    },
                )

    def test_dynamic_variant_outside_frozen_suite_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = ["left", "right"]
            fixed = make_run(
                root,
                "eval_fixed",
                "fixed_predeclared_v1",
                candidates,
                candidates=candidates,
            )
            # Its own manifest is internally valid but no longer matches the
            # fixed candidate suite, so the pair is rejected before reporting.
            dynamic = make_run(
                root,
                "eval_dynamic",
                "dynamic_evidence_v1",
                ["left", "clutter"],
                candidates=["left", "clutter"],
            )
            with self.assertRaisesRegex(StrategyComparisonError, "identity mismatch"):
                compare_fixed_dynamic(
                    root,
                    {
                        "schema_version": 1,
                        "fixed_evaluation_dir": fixed,
                        "dynamic_evaluation_dir": dynamic,
                    },
                )

    def test_incomplete_fixed_suite_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = ["left", "right"]
            fixed = make_run(
                root,
                "eval_fixed",
                "fixed_predeclared_v1",
                ["left"],
                candidates=candidates,
            )
            dynamic = make_run(
                root,
                "eval_dynamic",
                "dynamic_evidence_v1",
                ["left"],
                candidates=candidates,
            )
            with self.assertRaisesRegex(StrategyComparisonError, "complete frozen"):
                compare_fixed_dynamic(
                    root,
                    {
                        "schema_version": 1,
                        "fixed_evaluation_dir": fixed,
                        "dynamic_evaluation_dir": dynamic,
                    },
                )

    def test_missing_success_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = ["left"]
            fixed = make_run(
                root,
                "eval_fixed",
                "fixed_predeclared_v1",
                candidates,
                candidates=candidates,
            )
            dynamic = make_run(
                root,
                "eval_dynamic",
                "dynamic_evidence_v1",
                candidates,
                candidates=candidates,
            )
            episode_path = (
                root
                / "mea/generated_tasks/run_eval_fixed_1/evaluation/telemetry/act/episode_1/episode.json"
            )
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode.pop("success")
            write_json(episode_path, episode)
            with self.assertRaisesRegex(StrategyComparisonError, "explicit boolean"):
                compare_fixed_dynamic(
                    root,
                    {
                        "schema_version": 1,
                        "fixed_evaluation_dir": fixed,
                        "dynamic_evaluation_dir": dynamic,
                    },
                )

            episode["success"] = False
            write_json(episode_path, episode)
            summary_path = root / fixed / "summary/summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["rounds"][0]["taskgen_run_id"] = "run_../../escape"
            write_json(summary_path, summary)
            with self.assertRaisesRegex(StrategyComparisonError, "invalid round"):
                compare_fixed_dynamic(
                    root,
                    {
                        "schema_version": 1,
                        "fixed_evaluation_dir": fixed,
                        "dynamic_evaluation_dir": dynamic,
                    },
                )

    def test_query_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = ["left"]
            fixed = make_run(
                root,
                "eval_fixed",
                "fixed_predeclared_v1",
                candidates,
                candidates=candidates,
            )
            dynamic = make_run(
                root,
                "eval_dynamic",
                "dynamic_evidence_v1",
                candidates,
                candidates=candidates,
            )
            manifest_path = root / dynamic / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["user_request"] = "A different evaluation query."
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(StrategyComparisonError, "identity mismatch"):
                compare_fixed_dynamic(
                    root,
                    {
                        "schema_version": 1,
                        "fixed_evaluation_dir": fixed,
                        "dynamic_evaluation_dir": dynamic,
                    },
                )


if __name__ == "__main__":
    unittest.main()
