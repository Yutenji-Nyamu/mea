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
    validate_proxy_gold_manifest,
)
from mea.prospective_error_ledger import (
    ProspectiveOperationLedger,
    initialize_ledger,
)


def checkpoint(name):
    return {"checkpoint_id": name, "artifact_sha256": name[0] * 64}


def attempt(candidate, index, success, *, seed=17, source="live_policy_rollout"):
    minute = index + 1
    return {
        "attempt_id": f"attempt_{candidate}_{index}",
        "candidate_id": candidate,
        "seed": seed,
        "evidence_source": source,
        "rollout_ref": f"runs/{candidate}_{index}/episode.json",
        "started_at_utc": f"2026-07-24T00:{minute:02d}:00Z",
        "ended_at_utc": f"2026-07-24T00:{minute:02d}:10Z",
        "wall_seconds": 10.0,
        "status": "completed",
        "success": success,
    }


def arm(prereg, name, attempts, stop_reason, wall):
    return {
        "schema_version": 1,
        "protocol": f"{EFFICIENCY_PROTOCOL}_arm",
        "arm": name,
        "arm_run_id": f"{name}_independent_run",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "started_at_utc": "2026-07-24T00:01:00Z",
        "ended_at_utc": "2026-07-24T00:10:00Z",
        "wall_seconds": wall,
        "stop_reason": stop_reason,
        "attempts": attempts,
    }


class ClickBellEfficiencyTests(unittest.TestCase):
    def prereg(self, mode="toy_5to7act"):
        return build_click_bell_efficiency_preregistration(
            study_id="click_efficiency_test",
            mode=mode,
            checkpoint=checkpoint("act"),
            seed=17,
            created_at_utc="2026-07-24T00:00:00Z",
        )

    def test_independent_toy_uses_real_starts_and_wall(self):
        prereg = self.prereg()
        fixed_attempts = [
            attempt("left_base0", 0, False),
            attempt("right_base0", 1, True),
            attempt("left_base1", 2, False),
            attempt("right_base1", 3, True),
        ]
        adaptive_attempts = [
            attempt("right_base0", 4, True),
            attempt("left_base0", 5, False),
        ]
        for row in adaptive_attempts:
            row["attempt_id"] = "adaptive_" + row["attempt_id"]
            row["rollout_ref"] = "adaptive/" + row["rollout_ref"]
        result = evaluate_click_bell_efficiency(
            prereg,
            arm(prereg, "fixed", fixed_attempts, "fixed_suite_complete", 44.0),
            arm(prereg, "adaptive", adaptive_attempts, "query_sufficient", 23.0),
        )
        self.assertEqual(result["resource_measurement"]["act_episode_start_saving"], 2)
        self.assertEqual(result["resource_measurement"]["measured_wall_second_saving"], 21.0)
        self.assertTrue(result["original_query_conclusion_agrees"])
        self.assertTrue(result["toy_efficiency_evidence_passed"])
        self.assertFalse(result["cached_prefix_used"])
        self.assertFalse(result["paper_tables_1_2_eligible"])

    def test_three_act_smoke_never_becomes_efficiency_claim(self):
        prereg = self.prereg("smoke_3act")
        fixed = [
            attempt("left_base0", 0, False),
            attempt("right_base0", 1, True),
        ]
        adaptive = [attempt("left_base0", 2, False)]
        adaptive[0]["attempt_id"] = "adaptive_attempt"
        adaptive[0]["rollout_ref"] = "adaptive/episode.json"
        result = evaluate_click_bell_efficiency(
            prereg,
            arm(prereg, "fixed", fixed, "fixed_suite_complete", 20.0),
            arm(prereg, "adaptive", adaptive, "query_sufficient", 9.0),
        )
        self.assertEqual(
            result["claim_scope"], "three_act_mechanism_smoke_not_dense_reference"
        )
        self.assertFalse(result["toy_efficiency_evidence_passed"])

    def test_cached_or_shared_receipts_fail_closed(self):
        prereg = self.prereg("smoke_3act")
        fixed = [
            attempt("left_base0", 0, False),
            attempt("right_base0", 1, True),
        ]
        adaptive = [attempt("left_base0", 2, False, source="cached_artifact")]
        with self.assertRaisesRegex(LivePaperProtocolError, "never cached"):
            evaluate_click_bell_efficiency(
                prereg,
                arm(prereg, "fixed", fixed, "fixed_suite_complete", 20.0),
                arm(prereg, "adaptive", adaptive, "query_sufficient", 9.0),
            )


class ExactSeedRankingTests(unittest.TestCase):
    def prereg(self):
        return build_ranking_preregistration(
            study_id="act_dp3_n3",
            act_checkpoint=checkpoint("act"),
            dp3_checkpoint=checkpoint("dp3"),
            seeds=[101, 102, 103],
            created_at_utc="2026-07-24T00:00:00Z",
            reference_source_ref="official_leaderboard_snapshot.json",
            reference_scores={"act": 0.56, "dp3": 0.72},
        )

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

    def test_three_seed_tie_leaves_spearman_null(self):
        prereg = self.prereg()
        result = evaluate_exact_seed_ranking(
            prereg, self.runs(prereg, [0, 0, 0], [0, 0, 0])
        )
        self.assertIsNone(result["spearman_rho"])
        self.assertEqual(result["claim_status"], "toy_order_inconclusive_tie")
        self.assertEqual(result["exact_total_policy_rollouts"], 6)
        self.assertFalse(result["paper_table9_eligible"])

    def test_missing_seed_fails_closed(self):
        prereg = self.prereg()
        runs = self.runs(prereg, [0, 0, 0], [0, 1, 1])
        runs["policies"][0]["trials"].pop()
        with self.assertRaisesRegex(LivePaperProtocolError, ">= 3|exactly three"):
            evaluate_exact_seed_ranking(prereg, runs)


class Table3AndProxyTests(unittest.TestCase):
    def test_table3_requires_real_downstream_receipts_for_all_25_cells(self):
        prereg = build_table3_codegen_preregistration(
            study_id="table3_test", created_at_utc="2026-07-24T00:00:00Z"
        )
        cells = []
        for frozen in prereg["cells"]:
            stages = {
                "codegen": {
                    "generated_by_provider": True,
                    "artifact_ref": f"artifacts/{frozen['cell_id']}/task.py",
                    "artifact_sha256": "a" * 64,
                }
            }
            for stage in ("compile", "render", "simulator", "oracle"):
                stages[stage] = {
                    "passed": True,
                    "receipt_ref": f"receipts/{frozen['cell_id']}/{stage}.json",
                    "receipt_sha256": "b" * 64,
                }
            stages["oracle"].update(
                {"positive_fixture_count": 1, "negative_fixture_count": 2}
            )
            cells.append({**frozen, "stages": stages})
        runs = {
            "schema_version": 1,
            "protocol": "table3_real_codegen_ablation_v1_runs",
            "preregistration_sha256": prereg["preregistration_sha256"],
            "cells": cells,
        }
        result = evaluate_table3_codegen(prereg, runs)
        self.assertEqual(result["provider_generation_count"], 25)
        self.assertEqual(set(result["success_rates"]), set(TABLE3_CONDITIONS))
        self.assertTrue(all(value == 1.0 for value in result["success_rates"].values()))
        self.assertEqual(result["act_rollouts_started"], 0)
        proposal_only = deepcopy(runs)
        proposal_only["cells"][0]["stages"]["codegen"]["generated_by_provider"] = False
        with self.assertRaisesRegex(LivePaperProtocolError, "proposal-only"):
            evaluate_table3_codegen(prereg, proposal_only)

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
        self.assertFalse(result["paper_plan_validity_eligible"])


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
            self.assertFalse(summary["paper_fig6_eligible"])


if __name__ == "__main__":
    unittest.main()
