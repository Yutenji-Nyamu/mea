import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from mea.live_paper_protocols import (
    EFFICIENCY_PROTOCOL,
    RANKING_PROTOCOL,
    TABLE3_CONDITIONS,
    LivePaperProtocolError,
    build_click_bell_efficiency_preregistration,
    build_ranking_preregistration,
    build_table3_codegen_preregistration,
    evaluate_click_bell_efficiency,
    evaluate_exact_seed_ranking,
    evaluate_table3_codegen,
    materialize_click_bell_efficiency_preregistration,
    materialize_ranking_preregistration,
    materialize_table3_codegen_preregistration,
    validate_click_bell_efficiency_preregistration,
    validate_proxy_gold_manifest,
    validate_table3_codegen_preregistration,
)
from mea.prospective_error_ledger import (
    ProspectiveOperationLedger,
    initialize_ledger,
)
from mea.taskgen.prototype import (
    TaskGenError,
    validate_taskgen_ablation_switches,
)


def checkpoint(name):
    return {"checkpoint_id": name, "artifact_sha256": name[0] * 64}


def write_bytes(root, ref, payload):
    path = root / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_json(root, ref, value):
    return write_bytes(
        root,
        ref,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def bound_attempt(root, prereg, arm_name, arm_run_id, candidate, index, success):
    command = next(
        row
        for row in prereg["execution_schedule"][arm_name]
        if row["candidate_id"] == candidate
    )
    seed = prereg["seed"]
    seed_results = {
        "schema_version": 1,
        "protocol": "exact_seed_paired_v1",
        "requested_seeds": [seed],
        "requested_count": 1,
        "evaluated_count": 1,
        "seed_measurements": [{"seed": seed, "policy_success": success}],
    }
    seed_sha = write_json(
        root, command["expected_seed_results_ref"], seed_results
    )
    telemetry_sha = write_json(
        root,
        command["expected_telemetry_episode_ref"],
        {"schema_version": 1, "seed": seed, "success": success},
    )
    variant = next(
        row["variant_binding"]
        for row in prereg["candidate_universe"]
        if row["candidate_id"] == candidate
    )
    minute = index + 1
    attempt_id = f"{arm_name}_{candidate}_{index}"
    receipt = {
        "schema_version": 1,
        "protocol": "click_bell_bound_live_rollout_receipt_v1",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "arm": arm_name,
        "arm_run_id": arm_run_id,
        "attempt_id": attempt_id,
        "candidate_id": candidate,
        "variant_id": variant["variant_id"],
        "variant_manifest_sha256": variant["variant_manifest_sha256"],
        "command_sha256": command["command_sha256"],
        "checkpoint_sha256": prereg["checkpoint"]["artifact_sha256"],
        "seed": seed,
        "evidence_source": "live_policy_rollout",
        "started_at_utc": f"2026-07-24T00:{minute:02d}:00Z",
        "ended_at_utc": f"2026-07-24T00:{minute:02d}:10Z",
        "wall_seconds": 10.0,
        "status": "completed",
        "success": success,
        "seed_results_ref": command["expected_seed_results_ref"],
        "seed_results_sha256": seed_sha,
        "telemetry_episode_ref": command["expected_telemetry_episode_ref"],
        "telemetry_episode_sha256": telemetry_sha,
    }
    receipt_sha = write_json(root, command["receipt_ref"], receipt)
    return {
        "attempt_id": attempt_id,
        "candidate_id": candidate,
        "receipt_ref": command["receipt_ref"],
        "receipt_sha256": receipt_sha,
    }


def arm(prereg, name, attempts, stop_reason):
    return {
        "schema_version": 1,
        "protocol": f"{EFFICIENCY_PROTOCOL}_arm",
        "arm": name,
        "arm_run_id": f"{name}_independent_run",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "stop_reason": stop_reason,
        "attempts": attempts,
    }


class ClickBellEfficiencyTests(unittest.TestCase):
    def prereg(self, root, mode="toy_5to7act"):
        prereg = build_click_bell_efficiency_preregistration(
            study_id="click_efficiency_test",
            mode=mode,
            checkpoint=checkpoint("act"),
            seed=17,
            created_at_utc="2026-07-24T00:00:00Z",
            artifact_root_ref="artifacts/click_efficiency",
        )
        materialize_click_bell_efficiency_preregistration(root, prereg)
        return prereg

    def test_prereg_materializes_four_single_axis_variants_and_exact_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = self.prereg(root)
            self.assertEqual(len(prereg["candidate_universe"]), 4)
            for candidate in prereg["candidate_universe"]:
                binding = candidate["variant_binding"]
                self.assertEqual(binding["task_module"], "mea.tasks.click_bell")
                overlay = json.loads((root / binding["overlay_ref"]).read_text())
                self.assertTrue(overlay["mea"]["enabled"])
                self.assertEqual(
                    overlay["mea"]["bell"],
                    candidate["variant_hint"]["bell"],
                )
                self.assertEqual(binding["axis_id"], candidate["axis_id"])
            command = prereg["execution_schedule"]["fixed"][0]
            command_value = json.loads((root / command["command_ref"]).read_text())
            self.assertEqual(command_value["argv"][8], "1")
            self.assertEqual(command_value["argv"][9], "mea.tasks.click_bell")
            validate_click_bell_efficiency_preregistration(
                prereg, repo_root=root, require_materialized=True
            )
            (root / command["command_ref"]).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(
                LivePaperProtocolError, "command spec hash mismatch"
            ):
                validate_click_bell_efficiency_preregistration(
                    prereg, repo_root=root, require_materialized=True
                )

    def test_independent_toy_uses_bound_receipts_and_measured_wall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = self.prereg(root)
            fixed = [
                bound_attempt(
                    root,
                    prereg,
                    "fixed",
                    "fixed_independent_run",
                    candidate,
                    index,
                    success,
                )
                for index, (candidate, success) in enumerate(
                    (
                        ("object_position.left_fixed", False),
                        ("object_position.right_fixed", True),
                        ("object_instance.base0", False),
                        ("object_instance.base1", False),
                    )
                )
            ]
            adaptive = [
                bound_attempt(
                    root,
                    prereg,
                    "adaptive",
                    "adaptive_independent_run",
                    candidate,
                    index + 4,
                    success,
                )
                for index, (candidate, success) in enumerate(
                    (
                        ("object_position.right_fixed", True),
                        ("object_position.left_fixed", False),
                    )
                )
            ]
            result = evaluate_click_bell_efficiency(
                prereg,
                arm(prereg, "fixed", fixed, "fixed_suite_complete"),
                arm(prereg, "adaptive", adaptive, "query_sufficient"),
                repo_root=root,
            )
            self.assertEqual(
                result["resource_measurement"]["act_episode_start_saving"], 2
            )
            self.assertEqual(
                result["resource_measurement"]["measured_wall_second_saving"],
                20.0,
            )
            self.assertTrue(result["original_query_conclusion_agrees"])
            self.assertTrue(result["toy_efficiency_evidence_passed"])
            self.assertFalse(result["cached_prefix_used"])

    def test_missing_tampered_or_cached_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = self.prereg(root, "smoke_3act")
            fixed = [
                bound_attempt(
                    root,
                    prereg,
                    "fixed",
                    "fixed_independent_run",
                    candidate,
                    index,
                    success,
                )
                for index, (candidate, success) in enumerate(
                    (
                        ("object_position.left_fixed", False),
                        ("object_position.right_fixed", True),
                    )
                )
            ]
            adaptive = [
                bound_attempt(
                    root,
                    prereg,
                    "adaptive",
                    "adaptive_independent_run",
                    "object_position.left_fixed",
                    2,
                    False,
                )
            ]
            receipt_path = root / adaptive[0]["receipt_ref"]
            receipt = json.loads(receipt_path.read_text())
            receipt["evidence_source"] = "cached_artifact"
            adaptive[0]["receipt_sha256"] = write_json(
                root, adaptive[0]["receipt_ref"], receipt
            )
            with self.assertRaisesRegex(
                LivePaperProtocolError, "receipt identity mismatch"
            ):
                evaluate_click_bell_efficiency(
                    prereg,
                    arm(prereg, "fixed", fixed, "fixed_suite_complete"),
                    arm(prereg, "adaptive", adaptive, "query_sufficient"),
                    repo_root=root,
                )


class ExactSeedRankingTests(unittest.TestCase):
    def prereg(self, root):
        prereg = build_ranking_preregistration(
            study_id="act_dp3_n3",
            act_checkpoint=checkpoint("act"),
            dp3_checkpoint=checkpoint("dp3"),
            seeds=[101, 102, 103],
            created_at_utc="2026-07-24T00:00:00Z",
            reference_source_ref="official_leaderboard_snapshot.json",
            reference_scores={"act": 0.56, "dp3": 0.72},
            artifact_root_ref="artifacts/ranking",
        )
        materialize_ranking_preregistration(root, prereg)
        return prereg

    def runs(self, prereg, act_scores, dp3_scores):
        policies = []
        for policy_id, scores in (("act", act_scores), ("dp3", dp3_scores)):
            trials = []
            for index, (seed, score) in enumerate(zip(prereg["seeds"], scores)):
                trials.append(
                    {
                        "trial_id": f"{policy_id}_{seed}",
                        "seed": seed,
                        "evidence_source": "live_policy_rollout",
                        "rollout_ref": f"runs/{policy_id}/{seed}.json",
                        "status": "completed",
                        "score": score,
                        "started_at_utc": f"2026-07-24T01:0{index}:00Z",
                        "ended_at_utc": f"2026-07-24T01:0{index}:10Z",
                        "wall_seconds": 10.0,
                    }
                )
            policies.append(
                {
                    "policy_id": policy_id,
                    "checkpoint": prereg["policies"][policy_id],
                    "run_id": f"{policy_id}_exact_seed_run",
                    "trials": trials,
                }
            )
        return {
            "schema_version": 1,
            "protocol": f"{RANKING_PROTOCOL}_runs",
            "preregistration_sha256": prereg["preregistration_sha256"],
            "policies": policies,
        }

    def test_commands_use_exact_n1_act_and_direct_dp3_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = self.prereg(root)
            self.assertEqual(
                prereg["execution_entrypoints"]["dp3"], "script/eval_policy.py"
            )
            self.assertEqual(
                sum(len(rows) for rows in prereg["execution_schedule"].values()),
                6,
            )
            act = json.loads(
                (
                    root
                    / prereg["execution_schedule"]["act"][0]["command_ref"]
                ).read_text()
            )
            dp3 = json.loads(
                (
                    root
                    / prereg["execution_schedule"]["dp3"][0]["command_ref"]
                ).read_text()
            )
            self.assertIn("policy/ACT/eval_mea.sh", act["argv"])
            self.assertIn("--num_episodes", dp3["argv"])
            self.assertEqual(
                dp3["python_environment"],
                "/root/autodl-tmp/conda/envs/RoboTwin-DP3/bin/python",
            )
            self.assertIn("--seed_manifest", dp3["argv"])
            self.assertIn("--seed_results_path", dp3["argv"])
            self.assertIn("--output_dir", dp3["argv"])

    def test_three_seed_tie_leaves_spearman_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = self.prereg(root)
            result = evaluate_exact_seed_ranking(
                prereg,
                self.runs(prereg, [0, 0, 0], [0, 0, 0]),
                repo_root=root,
            )
            self.assertIsNone(result["spearman_rho"])
            self.assertEqual(result["claim_status"], "toy_order_inconclusive_tie")
            self.assertEqual(result["exact_total_policy_rollouts"], 6)
            self.assertFalse(result["paper_table9_eligible"])


class Table3AndProxyTests(unittest.TestCase):
    def table3_runs(self, root, prereg):
        cells = []
        for frozen in prereg["cells"]:
            expected = frozen["expected_stage_receipts"]
            task_sha = write_bytes(
                root, expected["codegen"]["artifact_ref"], b"def load_actors(self):\n    pass\n"
            )
            static_sha = write_json(
                root, expected["compile"]["receipt_ref"], {"passed": True}
            )
            scene_sha = write_json(
                root,
                expected["render"]["receipt_ref"],
                {"setup_success": True, "render_success": True},
            )
            oracle_sha = write_json(
                root,
                expected["oracle"]["receipt_ref"],
                {"positive_fixture_count": 1, "negative_fixture_count": 2},
            )
            stages = {
                "codegen": {
                    "generated_by_provider": True,
                    "artifact_ref": expected["codegen"]["artifact_ref"],
                    "artifact_sha256": task_sha,
                },
                "compile": {
                    "passed": True,
                    "receipt_ref": expected["compile"]["receipt_ref"],
                    "receipt_sha256": static_sha,
                },
                "render": {
                    "passed": True,
                    "receipt_ref": expected["render"]["receipt_ref"],
                    "receipt_sha256": scene_sha,
                },
                "simulator": {
                    "passed": True,
                    "receipt_ref": expected["simulator"]["receipt_ref"],
                    "receipt_sha256": scene_sha,
                },
                "oracle": {
                    "passed": True,
                    "receipt_ref": expected["oracle"]["receipt_ref"],
                    "receipt_sha256": oracle_sha,
                    "positive_fixture_count": 1,
                    "negative_fixture_count": 2,
                },
            }
            cells.append(
                {
                    "cell_id": frozen["cell_id"],
                    "proposal_id": frozen["proposal_id"],
                    "condition": frozen["condition"],
                    "stages": stages,
                }
            )
        return {
            "schema_version": 1,
            "protocol": "table3_real_codegen_ablation_v1_runs",
            "preregistration_sha256": prereg["preregistration_sha256"],
            "cells": cells,
        }

    def test_table3_materializes_25_executable_real_taskgen_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = build_table3_codegen_preregistration(
                study_id="table3_test",
                created_at_utc="2026-07-24T00:00:00Z",
                artifact_root_ref="artifacts/table3",
                text_model="frozen-text-model",
                vision_model="frozen-vision-model",
            )
            materialize_table3_codegen_preregistration(root, prereg)
            self.assertEqual(len(prereg["cells"]), 25)
            self.assertEqual(
                {cell["condition"] for cell in prereg["cells"]},
                set(TABLE3_CONDITIONS),
            )
            for cell in prereg["cells"]:
                runner = json.loads((root / cell["runner_ref"]).read_text())
                self.assertIn(
                    "--taskgen-ablation-json", runner["argv"]
                )
                self.assertIn("--accept-task-only", runner["argv"])
                self.assertNotIn("--run-act", runner["argv"])
                self.assertEqual(runner["module_switches"], cell["module_switches"])
            runs = self.table3_runs(root, prereg)
            result = evaluate_table3_codegen(
                prereg, runs, repo_root=root
            )
            self.assertEqual(result["provider_generation_count"], 25)
            self.assertTrue(
                all(value == 1.0 for value in result["success_rates"].values())
            )
            self.assertEqual(result["act_rollouts_started"], 0)

            runner_path = root / prereg["cells"][0]["runner_ref"]
            runner_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(
                LivePaperProtocolError, "runner hash mismatch"
            ):
                validate_table3_codegen_preregistration(
                    prereg, repo_root=root, require_materialized=True
                )

    def test_ablation_switch_schema_is_exact(self):
        self.assertEqual(
            validate_taskgen_ablation_switches(
                {
                    "rag": False,
                    "visual_self_check": True,
                    "readme_agent": True,
                }
            )["rag"],
            False,
        )
        with self.assertRaises(TaskGenError):
            validate_taskgen_ablation_switches({"rag": False})

    def test_checked_proxy_manifest_stays_non_human_and_partial(self):
        root = Path(__file__).resolve().parents[2]
        path = (
            root
            / "configs"
            / "manipeval_paper_evidence"
            / "plan_vqa_development_proxy_v1.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result = validate_proxy_gold_manifest(root, manifest)
        self.assertEqual(result["query_count"], 20)
        self.assertEqual(result["clip_slot_count"], 8)
        self.assertEqual(result["materialized_clip_count"], 2)
        self.assertFalse(result["ready_for_proxy_smoke"])


class ProspectiveLedgerTests(unittest.TestCase):
    def test_context_manager_freezes_start_denominator_and_error_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ledger"
            initialize_ledger(root, study_id="prospective_test")
            ledger = ProspectiveOperationLedger(root)
            with ledger.operation(
                operation_id="plan_001",
                run_id="run_001",
                category="plan_agent",
            ):
                pass
            with self.assertRaises(RuntimeError):
                with ledger.operation(
                    operation_id="taskgen_001",
                    run_id="run_001",
                    category="taskgen",
                ):
                    raise RuntimeError("fixture")
            summary = ledger.summarize()
            self.assertEqual(summary["denominator_operation_starts"], 2)
            self.assertEqual(summary["numerator_terminal_errors"], 1)
            self.assertEqual(summary["prospective_error_rate"], 0.5)
            self.assertEqual(summary["categories"]["taskgen"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
