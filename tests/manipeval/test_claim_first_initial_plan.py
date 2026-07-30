import json
import tempfile
import unittest
from pathlib import Path

from mea.planner.claim_first_initial import (
    PlanAgentInitialPlanBuilder,
    PlanAgentInitialPlanError,
    ClaimFirstInitialPlanBuilder,
    ClaimFirstInitialPlanError,
)
from mea.planner.query_contract import build_query_sufficiency_contract


def target(task_name: str) -> dict:
    return {
        "schema_version": 1,
        "binding_mode": "single_task_single_checkpoint",
        "task_name": task_name,
        "task_family": "contact",
        "task_profile": "official",
        "planner_kind": "legacy_kind_is_not_consumed",
        "policy": {
            "policy_name": "ACT",
            "checkpoint_setting": "demo_clean",
            "expert_data_num": 50,
            "language_conditioned": False,
        },
        "checkpoint": {
            "checkpoint_id": f"act-{task_name}/demo_clean-50",
            "checkpoint_dir": f"policy/ACT/act_ckpt/act-{task_name}/demo_clean-50",
        },
        "max_rounds": 1,
        "aspects": [
            {
                "aspect_id": "task_execution.official_baseline",
                "description": "Unchanged official task.",
                "template_ids": ["task_execution.official_baseline"],
            }
        ],
    }


class ClaimFirstInitialPlanTests(unittest.TestCase):
    def test_plan_agent_builder_is_canonical_with_legacy_aliases(self):
        self.assertIs(
            ClaimFirstInitialPlanBuilder,
            PlanAgentInitialPlanBuilder,
        )
        self.assertIs(
            ClaimFirstInitialPlanError,
            PlanAgentInitialPlanError,
        )

    def test_builds_neutral_control_without_task_specific_planner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = PlanAgentInitialPlanBuilder(
                root,
                target=target("click_bell"),
                max_rounds=2,
                start_seed=41,
                num_episodes=2,
            ).plan(
                "Where does this policy first expose a weakness?",
                evaluation_id="eval_direct_control",
                control_required=True,
            )

            self.assertEqual(
                manifest["planner"]["kind"],
                "plan_agent_direct_initial_v1",
            )
            self.assertFalse(
                manifest["planner"]["task_specific_planner_used"]
            )
            first_round = manifest["plan"]["rounds"][0]
            self.assertEqual(
                first_round["template_id"],
                "task_execution.official_baseline",
            )
            self.assertEqual(first_round["execution"]["seeds"], [41, 42])
            self.assertEqual(first_round["execution"]["backend"], "act")
            self.assertEqual(
                first_round["tool_request"]["metric"],
                "official_check_success",
            )
            written = json.loads(
                (
                    root
                    / "mea/evaluation_runs/eval_direct_control/manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(written, manifest)

    def test_no_control_plan_exposes_binding_for_runtime_candidate(self):
        contract = build_query_sufficiency_contract(
            "Did the policy jerk before contact?",
            candidate_universe=["candidate_precontact_jerk"],
            round_budget=1,
            claim_type="existential",
            control_requirement="not_required",
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = ClaimFirstInitialPlanBuilder(
                temporary,
                target=target("adjust_bottle"),
                max_rounds=1,
                start_seed=17,
            ).plan(
                "Did the policy jerk before contact?",
                evaluation_id="eval_direct_no_control",
                control_required=False,
                query_contract=contract,
            )

            self.assertEqual(manifest["plan"]["rounds"], [])
            self.assertEqual(
                manifest["plan"]["planning_state"],
                "awaiting_initial_query_candidate_materialization",
            )
            self.assertEqual(
                manifest["initial_execution_binding"]["seeds"], [17]
            )
            self.assertEqual(
                manifest["plan"]["query_contract"]["control_requirement"],
                "not_required",
            )
            self.assertEqual(
                manifest["planner"]["proposal_source"],
                "runtime_plan_agent_proposal_pending",
            )

    def test_contract_control_requirement_must_match(self):
        contract = build_query_sufficiency_contract(
            "Does the translated task still succeed?",
            candidate_universe=["candidate_translation"],
            round_budget=1,
            control_requirement="required",
        )
        with tempfile.TemporaryDirectory() as temporary:
            builder = ClaimFirstInitialPlanBuilder(
                temporary,
                target=target("grab_roller"),
                max_rounds=2,
                start_seed=3,
            )
            with self.assertRaisesRegex(
                ClaimFirstInitialPlanError,
                "conflicts with QueryContract",
            ):
                builder.plan(
                    "Does the translated task still succeed?",
                    evaluation_id="eval_control_conflict",
                    control_required=False,
                    query_contract=contract,
                )

    def test_duplicate_evaluation_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            builder = ClaimFirstInitialPlanBuilder(
                temporary,
                target=target("place_phone_stand"),
                max_rounds=2,
                start_seed=5,
            )
            builder.plan(
                "Evaluate a bounded position change.",
                evaluation_id="eval_duplicate",
                control_required=True,
            )
            with self.assertRaisesRegex(
                ClaimFirstInitialPlanError,
                "already exists",
            ):
                builder.plan(
                    "Evaluate a bounded position change.",
                    evaluation_id="eval_duplicate",
                    control_required=True,
                )

    def test_control_plan_reserves_a_candidate_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            builder = ClaimFirstInitialPlanBuilder(
                temporary,
                target=target("beat_block_hammer"),
                max_rounds=1,
                start_seed=5,
            )
            with self.assertRaisesRegex(
                ClaimFirstInitialPlanError,
                "needs one candidate round",
            ):
                builder.plan(
                    "Where does this policy first expose a weakness?",
                    evaluation_id="eval_missing_candidate_budget",
                    control_required=True,
                )


if __name__ == "__main__":
    unittest.main()
