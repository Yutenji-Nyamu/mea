from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mea.paper_claim_demo import (
    CODEGEN_ABLATION_PROTOCOL,
    EFFICIENCY_PROTOCOL,
    ERROR_DISTRIBUTION_PROTOCOL,
    PROPOSAL_PROMPT_ABLATION_PROTOCOL,
    PROXY_VALIDITY_PROTOCOL,
    RANKING_PROTOCOL,
    PaperClaimDemoError,
    evaluate_codegen_ablation,
    evaluate_error_distribution,
    evaluate_paper_claim_manifest,
    evaluate_policy_ranking,
    evaluate_proposal_prompt_ablation,
    evaluate_proxy_validity,
    evaluate_small_efficiency,
)


def _trial(
    trial_id: str,
    candidate: str,
    seed: int,
    *,
    success: bool | None = None,
    score: float | None = None,
) -> dict:
    row = {
        "trial_id": trial_id,
        "candidate_id": candidate,
        "seed": seed,
        "rollout_ref": f"runs/{trial_id}/episode.json",
        "episode_status": "completed",
        "outcome_metric": "official_check_success",
        "outcome_authority": "official_equivalent",
        "outcome_tool_sha256": "a" * 64,
    }
    if success is not None:
        row["success"] = success
    if score is not None:
        row["score"] = score
    return row


class SmallEfficiencyTest(unittest.TestCase):
    def manifest(self) -> dict:
        fixed_left = _trial("fixed_left", "left", 7, success=False)
        adaptive_left = _trial("adaptive_left", "left", 7, success=False)
        adaptive_left["rollout_ref"] = fixed_left["rollout_ref"]
        return {
            "schema_version": 1,
            "protocol": EFFICIENCY_PROTOCOL,
            "evidence_source": "live_act_rollout",
            "study_id": "click_bell_n1",
            "query": "Does the policy retain success across the bounded suite?",
            "claim_type": "universal_all_candidates",
            "comparison_design": "cached_prefix_counterfactual",
            "comparison_timing": "post_hoc_after_observed_outcomes",
            "cost_scope": "policy_episode_wall_only",
            "adaptive_stop_assessment_ref": (
                "runs/adaptive_run/query_sufficiency.json"
            ),
            "candidate_universe": ["left", "right"],
            "seeds": [7],
            "conclusion_rule": {
                "metric": "success_rate",
                "operator": ">=",
                "threshold": 1.0,
            },
            "fixed": {
                "arm": "fixed",
                "run_id": "fixed_run",
                "policy_id": "act",
                "checkpoint_id": "act-ckpt",
                "wall_seconds": 120.0,
                "arm_trace_ref": "runs/fixed_run/fixed_schedule.json",
                "trials": [
                    fixed_left,
                    _trial("fixed_right", "right", 7, success=True),
                ],
            },
            "adaptive": {
                "arm": "adaptive",
                "run_id": "adaptive_run",
                "policy_id": "act",
                "checkpoint_id": "act-ckpt",
                "wall_seconds": 55.0,
                "arm_trace_ref": "runs/adaptive_run/planner_trace.json",
                "trials": [adaptive_left],
            },
        }

    def test_real_small_pair_reports_joint_savings(self) -> None:
        result = evaluate_small_efficiency(self.manifest())
        self.assertTrue(result["conclusion_agreement"])
        self.assertIsNone(result["act_rollout_saving"])
        self.assertIsNone(result["act_rollout_saving_fraction"])
        self.assertEqual(result["counterfactual_avoidable_rollout_count"], 1)
        self.assertEqual(
            result["counterfactual_avoidable_rollout_fraction"],
            0.5,
        )
        self.assertEqual(
            result["claim_status"],
            "post_hoc_cached_counterfactual_protocol_demo",
        )
        self.assertFalse(result["measured_independent_arm_wall_speedup"])
        self.assertIsNone(result["wall_second_saving"])

    def test_rejects_synthetic_source(self) -> None:
        value = self.manifest()
        value["evidence_source"] = "synthetic_fixture"
        with self.assertRaisesRegex(PaperClaimDemoError, "synthetic"):
            evaluate_small_efficiency(value)

    def test_fixed_arm_must_be_dense(self) -> None:
        value = self.manifest()
        value["fixed"]["trials"].pop()
        with self.assertRaisesRegex(PaperClaimDemoError, "complete"):
            evaluate_small_efficiency(value)

    def test_sparse_declared_candidate_seed_suite_is_honest(self) -> None:
        value = self.manifest()
        value["seeds"] = [7, 8]
        value["candidate_seed_pairs"] = [
            {"candidate_id": "left", "seed": 7},
            {"candidate_id": "right", "seed": 8},
        ]
        value["fixed"]["trials"][1]["seed"] = 8
        result = evaluate_small_efficiency(value)
        self.assertTrue(result["conclusion_agreement"])
        self.assertEqual(len(result["candidate_seed_pairs"]), 2)

    def test_rejects_mixed_outcome_authority(self) -> None:
        value = self.manifest()
        value["fixed"]["trials"][1][
            "outcome_authority"
        ] = "compiled_success_spec_experimental_bounded"
        with self.assertRaisesRegex(PaperClaimDemoError, "outcome binding"):
            evaluate_small_efficiency(value)

    def test_universal_claim_requires_threshold_one(self) -> None:
        value = self.manifest()
        value["conclusion_rule"]["threshold"] = 0.5
        with self.assertRaisesRegex(PaperClaimDemoError, "requires"):
            evaluate_small_efficiency(value)


class PolicyRankingTest(unittest.TestCase):
    def test_two_policy_live_scores_are_explicitly_toy(self) -> None:
        value = {
            "schema_version": 1,
            "protocol": RANKING_PROTOCOL,
            "evidence_source": "live_policy_rollout",
            "study_id": "act_dp_toy",
            "candidate_universe": ["base"],
            "seeds": [3],
            "reference_source_ref": "paper_table9_subset.json",
            "reference_scores": {"act": 0.8, "dp": 0.6},
            "policies": [
                {
                    "policy_id": "act",
                    "checkpoint_id": "act-ckpt",
                    "run_id": "act-run",
                    "trials": [_trial("act_base", "base", 3, score=0.9)],
                },
                {
                    "policy_id": "dp",
                    "checkpoint_id": "dp-ckpt",
                    "run_id": "dp-run",
                    "trials": [_trial("dp_base", "base", 3, score=0.4)],
                },
            ],
        }
        result = evaluate_policy_ranking(value)
        self.assertTrue(result["two_policy_toy"])
        self.assertEqual(result["spearman_rho"], 1.0)
        self.assertEqual(result["claim_scope"], "two_policy_toy_ranking_not_table9")
        self.assertTrue(result["exact_order_agreement"])

    def test_tied_observed_ranking_has_no_spearman(self) -> None:
        value = {
            "schema_version": 1,
            "protocol": RANKING_PROTOCOL,
            "evidence_source": "live_policy_rollout",
            "study_id": "tie_toy",
            "candidate_universe": ["base"],
            "seeds": [3],
            "reference_source_ref": "paper.json",
            "reference_scores": {"act": 0.8, "dp": 0.6},
            "policies": [
                {
                    "policy_id": policy,
                    "checkpoint_id": f"{policy}-ckpt",
                    "run_id": f"{policy}-run",
                    "trials": [
                        _trial(f"{policy}_base", "base", 3, score=0.5)
                    ],
                }
                for policy in ("act", "dp")
            ],
        }
        result = evaluate_policy_ranking(value)
        self.assertIsNone(result["spearman_rho"])
        self.assertFalse(result["exact_order_agreement"])
        self.assertEqual(result["claim_status"], "toy_order_inconclusive_tie")


class ProxyValidityTest(unittest.TestCase):
    def manifest(self) -> dict:
        vqa_items = []
        for condition_index, condition in enumerate(
            ("clean", "scene_clutter", "background_texture", "lighting")
        ):
            vqa_items.extend(
                [
                    {
                        "item_id": f"{condition_index}_positive",
                        "condition": condition,
                        "reference_observed": True,
                        "positive_score": 0.9,
                        "evidence_ref": f"clips/{condition}/positive.mp4",
                    },
                    {
                        "item_id": f"{condition_index}_negative",
                        "condition": condition,
                        "reference_observed": False,
                        "positive_score": 0.1,
                        "evidence_ref": f"clips/{condition}/negative.mp4",
                    },
                ]
            )
        return {
            "schema_version": 1,
            "protocol": PROXY_VALIDITY_PROTOCOL,
            "evidence_source": "development_agent_proxy",
            "study_id": "proxy_v1",
            "plan": {
                "reference_session_id": "proxy_reference_session",
                "prediction_session_id": "proxy_prediction_session",
                "items": [
                    {
                        "item_id": "query_1",
                        "paper_category": "generalization_object",
                        "reference_aspects": ["position", "appearance"],
                        "predicted_aspects": ["position"],
                    },
                    {
                        "item_id": "query_2",
                        "paper_category": "generalization_scene",
                        "reference_aspects": ["lighting"],
                        "predicted_aspects": ["lighting"],
                    },
                    {
                        "item_id": "query_3",
                        "paper_category": "performance",
                        "reference_aspects": ["motion_smoothness"],
                        "predicted_aspects": ["motion_smoothness"],
                    },
                    {
                        "item_id": "query_4",
                        "paper_category": "safety",
                        "reference_aspects": ["unintended_contact"],
                        "predicted_aspects": ["unintended_contact"],
                    },
                    {
                        "item_id": "query_5",
                        "paper_category": "language_or_multitask",
                        "reference_aspects": ["paraphrase_consistency"],
                        "predicted_aspects": ["paraphrase_consistency"],
                    },
                ],
            },
            "vqa": {
                "reference_session_id": "vqa_reference_session",
                "prediction_session_id": "vqa_prediction_session",
                "threshold": 0.5,
                "items": vqa_items,
            },
        }

    def test_proxy_plan_and_all_four_vqa_conditions(self) -> None:
        result = evaluate_proxy_validity(self.manifest())
        self.assertAlmostEqual(result["plan"]["micro_f1"], 10 / 11)
        self.assertEqual(
            set(result["plan"]["categories"]),
            {
                "generalization_object",
                "generalization_scene",
                "performance",
                "safety",
                "language_or_multitask",
            },
        )
        self.assertEqual(result["vqa"]["accuracy"], 1.0)
        self.assertEqual(result["vqa"]["auroc"], 1.0)
        self.assertEqual(
            set(result["vqa"]["conditions"]),
            {"clean", "scene_clutter", "background_texture", "lighting"},
        )
        self.assertIn("not_human", result["claim_scope"])

    def test_each_condition_needs_positive_and_negative_for_auc(self) -> None:
        value = self.manifest()
        value["vqa"]["items"] = [
            row
            for row in value["vqa"]["items"]
            if row["item_id"] != "0_negative"
        ]
        with self.assertRaisesRegex(PaperClaimDemoError, "AUROC"):
            evaluate_proxy_validity(value)


class CodegenAblationTest(unittest.TestCase):
    def test_real_attempt_matrix_statistics(self) -> None:
        attempts = []
        for component, conditions in (
            (
                "taskgen",
                (
                    "complete",
                    "base",
                    "no_rag",
                    "no_visual_self_check",
                    "no_readme_agent",
                ),
            ),
            ("toolgen", ("complete", "no_rag")),
        ):
            for condition in conditions:
                attempts.append(
                    {
                        "attempt_id": f"{component}_{condition}",
                        "component": component,
                        "condition": condition,
                        "input_id": f"{component}_unseen_1",
                        "provider_attempt_ref": (
                            f"provider/{component}/{condition}/attempt.json"
                        ),
                        "artifact_ref": (
                            f"provider/{component}/{condition}/candidate.py"
                        ),
                        "syntax_valid": True,
                        "downstream_valid": condition != "base",
                        "accepted": condition != "base",
                        "failure_stage": (
                            None if condition != "base" else "downstream_gate"
                        ),
                    }
                )
        value = {
            "schema_version": 1,
            "protocol": CODEGEN_ABLATION_PROTOCOL,
            "evidence_source": "live_provider_codegen",
            "study_id": "table3_minimal",
            "attempts": attempts,
        }
        result = evaluate_codegen_ablation(value)
        self.assertTrue(result["table3_minimum_condition_coverage"])
        self.assertEqual(result["attempt_count"], 7)
        self.assertEqual(
            result["conditions"]["taskgen"]["base"]["acceptance_rate"], 0.0
        )
        self.assertEqual(result["failure_stage_counts"], {"downstream_gate": 1})

    def test_accepted_attempt_must_pass_real_gates(self) -> None:
        value = {
            "schema_version": 1,
            "protocol": CODEGEN_ABLATION_PROTOCOL,
            "evidence_source": "live_provider_codegen",
            "study_id": "bad_attempt",
            "attempts": [
                {
                    "attempt_id": "attempt_1",
                    "component": "taskgen",
                    "condition": "complete",
                    "input_id": "input_1",
                    "provider_attempt_ref": "provider/attempt.json",
                    "artifact_ref": "provider/candidate.py",
                    "syntax_valid": True,
                    "downstream_valid": False,
                    "accepted": True,
                    "failure_stage": None,
                }
            ],
        }
        with self.assertRaisesRegex(PaperClaimDemoError, "validation gates"):
            evaluate_codegen_ablation(value)

    def test_proposal_prompt_ablation_cannot_claim_codegen(self) -> None:
        attempts = []
        for component, conditions in (
            ("taskgen", (
                "complete",
                "base",
                "no_rag",
                "no_visual_self_check",
                "no_readme_agent",
            )),
            ("toolgen", ("complete", "no_rag")),
        ):
            for condition in conditions:
                attempts.append(
                    {
                        "attempt_id": f"{component}_{condition}",
                        "component": component,
                        "condition": condition,
                        "input_id": f"{component}_unseen_1",
                        "provider_attempt_ref": (
                            f"provider/{component}/{condition}/attempt.json"
                        ),
                        "artifact_ref": (
                            f"provider/{component}/{condition}/candidate.json"
                        ),
                        "artifact_kind": "structured_proposal_json",
                        "syntax_valid": True,
                        "downstream_valid": True,
                        "accepted": True,
                        "failure_stage": None,
                    }
                )
        value = {
            "schema_version": 1,
            "protocol": PROPOSAL_PROMPT_ABLATION_PROTOCOL,
            "evidence_source": "live_provider_structured_proposal",
            "study_id": "proposal_prompt_minimal",
            "attempts": attempts,
        }
        result = evaluate_proposal_prompt_ablation(value)
        self.assertTrue(result["prompt_condition_matrix_complete"])
        self.assertFalse(result["task_or_tool_code_generated_and_executed"])
        self.assertEqual(
            result["claim_status"],
            "proposal_prompt_matrix_complete_no_codegen_evidence",
        )
        self.assertNotIn(
            "table3_minimum_condition_coverage",
            result,
        )


class ErrorDistributionTest(unittest.TestCase):
    def test_unified_stage_distribution_uses_all_operations(self) -> None:
        operations = []
        for index, stage in enumerate(
            ("plan", "taskgen", "toolgen", "simulator", "other")
        ):
            operations.append(
                {
                    "operation_id": f"operation_{index}",
                    "run_id": "run_1",
                    "stage": stage,
                    "operation_ref": f"logs/{stage}.json",
                    "status": "error" if stage == "simulator" else "success",
                    "error_code": "simulator_timeout" if stage == "simulator" else None,
                }
            )
        value = {
            "schema_version": 1,
            "protocol": ERROR_DISTRIBUTION_PROTOCOL,
            "evidence_source": "live_runtime_operation_log",
            "study_id": "error_pilot",
            "operations": operations,
        }
        result = evaluate_error_distribution(value)
        self.assertEqual(result["operation_count"], 5)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["error_rate"], 0.2)
        self.assertEqual(
            result["stages"]["simulator"]["share_of_all_errors"], 1.0
        )
        self.assertEqual(result["stage_coverage"]["missing"], [])

    def test_success_cannot_carry_error_code(self) -> None:
        value = {
            "schema_version": 1,
            "protocol": ERROR_DISTRIBUTION_PROTOCOL,
            "evidence_source": "live_runtime_operation_log",
            "study_id": "bad_error_log",
            "operations": [
                {
                    "operation_id": "operation_1",
                    "run_id": "run_1",
                    "stage": "plan",
                    "operation_ref": "logs/plan.json",
                    "status": "success",
                    "error_code": "not_really_an_error",
                }
            ],
        }
        with self.assertRaisesRegex(PaperClaimDemoError, "cannot have"):
            evaluate_error_distribution(value)


class DispatchAndCliTest(unittest.TestCase):
    def test_dispatches_explicit_protocol(self) -> None:
        manifest = SmallEfficiencyTest().manifest()
        self.assertEqual(
            evaluate_paper_claim_manifest(manifest)["claim_status"],
            "post_hoc_cached_counterfactual_protocol_demo",
        )

    def test_cli_requires_input_and_writes_result(self) -> None:
        manifest = SmallEfficiencyTest().manifest()
        repo_root = Path(__file__).resolve().parents[2]
        script = (
            repo_root
            / "experiments/paper/manipeval_paper_claim_demo.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.json"
            input_path.write_text(json.dumps(manifest), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                result["claim_status"],
                "post_hoc_cached_counterfactual_protocol_demo",
            )


if __name__ == "__main__":
    unittest.main()
