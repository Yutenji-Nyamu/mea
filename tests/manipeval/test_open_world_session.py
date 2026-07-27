import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from mea.planner.catalog import build_act_catalog
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.planner.open_world_session import (
    OpenWorldPlanSession,
    OpenWorldSessionError,
)
from mea.planner.query_contract import build_query_sufficiency_contract


def _ready_official_only_catalog(root: Path) -> dict:
    task_name = "adjust_bottle"
    schema = root / "mea/toolkit/schemas" / f"{task_name}.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(
        json.dumps(
            {
                "task_name": task_name,
                "task_family": "object_manipulation",
            }
        ),
        encoding="utf-8",
    )
    checkpoint = (
        root
        / "policy/ACT/act_ckpt"
        / f"act-{task_name}"
        / "demo_clean-50"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
    (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
    return build_act_catalog(root)


def _control_round(session: OpenWorldPlanSession) -> dict:
    return {
        "round_id": "round_1",
        "task_name": session.target["task_name"],
        "template_id": session.target["control_template_id"],
        "execution": {
            "checkpoint_id": session.target["checkpoint"]["checkpoint_id"],
            "num_episodes": 1,
            "seeds": [1001],
        },
    }


def _candidate() -> dict:
    return build_experiment_candidate(
        source_query="Where does bottle stability first fail?",
        base_task="adjust_bottle",
        semantic_concern="motion.post_release_wobble",
        scene_need="reuse the official scene and vary release orientation",
        checker_need="check stable final pose after the policy releases",
        tool_need="measure post-release angular velocity",
    )


def _initial_plan(
    session: OpenWorldPlanSession,
    contract: dict,
) -> dict:
    return {
        "schema_version": 1,
        "task_name": session.target["task_name"],
        "policy": deepcopy(session.target["policy"]),
        "checkpoint": deepcopy(session.target["checkpoint"]),
        # The legacy official planner reports its catalog cap.  The open-world
        # session canonically expands it to its independently frozen budget.
        "max_rounds": session.target["catalog_max_rounds"],
        "evaluation_goal": "find a post-release stability weakness",
        "rounds": [_control_round(session)],
        "round_decisions": [],
        "planning_state": "awaiting_round_1_observation",
        "query_contract": contract,
    }


class OpenWorldPlanSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = _ready_official_only_catalog(self.root)
        self.contract = build_query_sufficiency_contract(
            "Is there some condition that exposes a failure?",
            candidate_universe=[],
            round_budget=2,
            claim_type="existential",
            candidate_universe_closed=False,
            existential_witness_outcome="fail",
        )
        self.session = OpenWorldPlanSession.from_catalog(
            self.catalog,
            "adjust_bottle",
            max_rounds=3,
            query_contract=self.contract,
        )
        self.plan = _initial_plan(self.session, self.contract)

    def tearDown(self):
        self.temp.cleanup()

    def test_official_only_catalog_does_not_limit_runtime_round_budget(self):
        self.assertEqual(self.session.target["catalog_max_rounds"], 1)
        self.assertEqual(self.session.target["max_rounds"], 3)
        normalized = self.session.normalize_plan(self.plan)
        self.assertEqual(normalized["max_rounds"], 3)
        self.assertEqual(
            normalized["checkpoint_id"],
            "act-adjust_bottle/demo_clean-50",
        )
        self.assertEqual(normalized["requested_candidate_ids"], [])

    def test_control_round_is_frozen_after_first_normalization(self):
        self.session.normalize_plan(self.plan)
        changed = deepcopy(self.plan)
        changed["rounds"][0]["execution"]["seeds"] = [9009]
        with self.assertRaisesRegex(
            OpenWorldSessionError, "rewrite the frozen official control"
        ):
            self.session.normalize_plan(changed)

    def test_runtime_candidate_needs_no_catalog_aspect_or_template(self):
        candidate = _candidate()
        step = {
            "schema_version": 2,
            "action": "propose",
            "candidate_id": candidate["candidate_id"],
            "execution_mode": "reuse_or_generate",
            "experiment_candidate": candidate,
            "rationale": "The control passed; inspect post-release wobble.",
            "answered_query": False,
        }
        generated_round = {
            "round_id": "round_2",
            "task_name": "adjust_bottle",
            "candidate_id": candidate["candidate_id"],
            "experiment_candidate": candidate,
            "task_module": "mea.generated.adjust_bottle_post_release_wobble",
            "execution": {
                "checkpoint_id": self.session.target["checkpoint"][
                    "checkpoint_id"
                ],
                "num_episodes": 1,
                "seeds": [1002],
            },
        }
        updated, decision, options = self.session.apply_plan_step(
            self.plan,
            [{"round_id": "round_1"}],
            step,
            materialized_round=generated_round,
            source="provider_claim_first_open_query",
        )

        self.assertNotIn("template_id", updated["rounds"][1])
        self.assertNotIn("aspect_id", updated["rounds"][1])
        self.assertEqual(
            updated["requested_candidate_ids"], [candidate["candidate_id"]]
        )
        self.assertIn(
            candidate["candidate_id"],
            updated["query_contract"]["candidate_universe"],
        )
        self.assertEqual(decision["transition"], "switch_concern")
        self.assertEqual(
            decision["decision_reason"], "provider_authored_open_world_step"
        )
        self.assertEqual(options["session_kind"], "open_world_claim_first")

    def test_runtime_candidate_cannot_change_task_or_checkpoint(self):
        candidate = {
            **_candidate(),
            "base_task": "grab_roller",
        }
        with self.assertRaisesRegex(
            OpenWorldSessionError, "cannot switch the base task"
        ):
            self.session.apply_plan_step(
                self.plan,
                [{"round_id": "round_1"}],
                {
                    "schema_version": 2,
                    "action": "propose",
                    "candidate_id": candidate["candidate_id"],
                    "experiment_candidate": candidate,
                    "rationale": "invalid task switch",
                    "answered_query": False,
                },
                materialized_round={
                    "round_id": "round_2",
                    "experiment_candidate": candidate,
                },
            )

        valid_candidate = _candidate()
        with self.assertRaisesRegex(
            OpenWorldSessionError, "cannot change bound checkpoint_id"
        ):
            self.session.apply_plan_step(
                self.plan,
                [{"round_id": "round_1"}],
                {
                    "schema_version": 2,
                    "action": "propose",
                    "candidate_id": valid_candidate["candidate_id"],
                    "experiment_candidate": valid_candidate,
                    "rationale": "invalid checkpoint switch",
                    "answered_query": False,
                },
                materialized_round={
                    "round_id": "round_2",
                    "experiment_candidate": valid_candidate,
                    "execution": {"checkpoint_id": "other/checkpoint"},
                },
            )

    def test_evidence_sufficient_stop_uses_dynamic_query_contract(self):
        candidate = _candidate()
        current = deepcopy(self.plan)
        current["rounds"].append(
            {
                "round_id": "round_2",
                "candidate_id": candidate["candidate_id"],
                "experiment_candidate": candidate,
            }
        )
        current["query_contract"] = {
            **self.contract,
            "candidate_universe": [candidate["candidate_id"]],
            "required_coverage": {
                **self.contract["required_coverage"],
                "candidate_ids": [candidate["candidate_id"]],
                "minimum_evaluated": 1,
            },
        }
        updated, decision, _ = self.session.apply_plan_step(
            current,
            [
                {"round_id": "round_1"},
                {
                    "round_id": "round_2",
                    "candidate_evidence": {
                        "candidate_id": candidate["candidate_id"],
                        "outcome": "fail",
                        "score": 0.0,
                        "diagnosis": "post-release wobble observed",
                    },
                },
            ],
            {
                "schema_version": 2,
                "action": "stop",
                "rationale": "A failing witness answers the existential Query.",
                "answered_query": True,
            },
        )
        self.assertEqual(updated["planning_state"], "stopped_after_round_2")
        self.assertTrue(decision["query_assessment"]["evidence_sufficient"])
        self.assertEqual(
            decision["query_assessment"]["stop_reason"],
            "evidence_sufficient",
        )

    def test_open_universal_cannot_claim_answer_before_domain_closes(self):
        universal = build_query_sufficiency_contract(
            "Do all generated variants pass?",
            candidate_universe=[],
            round_budget=2,
            claim_type="universal",
            candidate_universe_closed=False,
        )
        session = OpenWorldPlanSession.from_catalog(
            self.catalog,
            "adjust_bottle",
            max_rounds=3,
            query_contract=universal,
        )
        plan = _initial_plan(session, universal)
        with self.assertRaisesRegex(
            OpenWorldSessionError,
            "requires sufficient QueryContract evidence",
        ):
            session.apply_plan_step(
                plan,
                [{"round_id": "round_1"}],
                {
                    "schema_version": 2,
                    "action": "stop",
                    "rationale": "No failures yet.",
                    "answered_query": True,
                },
            )

    def test_snapshot_keeps_clean_control_candidate_dataflow(self):
        normalized = self.session.normalize_plan(self.plan)
        snapshot = self.session.snapshot(
            "Where does bottle stability first fail?",
            normalized,
            [{"round_id": "round_1"}],
        )
        self.assertEqual(
            snapshot["session_kind"],
            "open_world_single_task_adaptive_evaluation",
        )
        self.assertEqual(
            snapshot["control_round"]["template_id"],
            "task_execution.official_baseline",
        )
        self.assertEqual(snapshot["experiment_candidates"], [])
        self.assertTrue(
            snapshot["query_assessment"]["candidate_discovery_required"]
        )


if __name__ == "__main__":
    unittest.main()
