from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mea.libero.benchmark import LiberoContractError
from mea.libero.retrieval import (
    BDDLTaskIndex,
    BDDLTaskRecord,
    PolicyTaskCompatibility,
    authorize_controlled_change,
    pending_controlled_change,
    smolvla_policy_compatibility,
)


def _task(task_id: int, language: str) -> BDDLTaskRecord:
    return BDDLTaskRecord(
        suite="libero_object",
        task_id=task_id,
        problem_name=f"problem_{task_id}",
        language=language,
        bddl_path=f"/fixture/task_{task_id}.bddl",
        init_state_path=f"/fixture/task_{task_id}.pruned_init",
        objects=(f"object_{task_id}", "basket_1"),
        goal_predicates=(
            ("in", f"object_{task_id}", "basket_1_contain_region"),
        ),
    )


def _profile(*task_ids: int) -> PolicyTaskCompatibility:
    return PolicyTaskCompatibility(
        policy_name="fixture",
        checkpoint="/checkpoint",
        declared_scope="single_task" if len(task_ids) == 1 else "multi_task",
        authorized_task_ids={"libero_object": tuple(task_ids)},
        authorization_source="artifact_manifest",
        artifact_evidence={"fixture": True},
    )


class LiberoOpenRetrievalTests(unittest.TestCase):
    def test_similarity_never_overrides_single_task_policy_compatibility(self):
        supported = _task(0, "pick alphabet soup and place it in basket")
        tempting_but_unsupported = _task(
            1,
            "test texture robustness under lighting variation",
        )
        result = BDDLTaskIndex(
            [supported, tempting_but_unsupported]
        ).retrieve_nearest(
            "texture robustness under lighting variation",
            compatibility=_profile(0),
        )
        self.assertEqual(result.selected.task_id, 0)
        self.assertEqual(result.authorized_candidate_count, 1)
        self.assertEqual(result.not_authorized_candidate_count, 1)

    def test_explicit_multi_task_profile_may_retrieve_nearest_supported_task(self):
        result = BDDLTaskIndex(
            [
                _task(0, "pick alphabet soup and place it in basket"),
                _task(1, "test texture robustness under lighting variation"),
            ]
        ).retrieve_nearest(
            "texture robustness under lighting variation",
            compatibility=_profile(0, 1),
        )
        self.assertEqual(result.selected.task_id, 1)
        self.assertEqual(result.authorized_candidate_count, 2)

    def test_plan_only_contract_is_pending_not_false_authorization(self):
        retrieval = BDDLTaskIndex([_task(0, "pick object")]).retrieve_nearest(
            "where does this policy fail",
            compatibility=_profile(0),
        )
        contract = pending_controlled_change(retrieval)
        self.assertEqual(contract.status, "pending")
        self.assertFalse(contract.authorized)
        with self.assertRaisesRegex(LiberoContractError, "not authorized"):
            contract.require_authorized()

    def test_open_concern_can_be_unsupported_without_catalog_remap(self):
        retrieval = BDDLTaskIndex([_task(0, "pick object")]).retrieve_nearest(
            "pre-contact jerk",
            compatibility=_profile(0),
        )
        contract = authorize_controlled_change(
            retrieval,
            {
                "proposal": {
                    "sub_aspect": "precontact_jerk",
                    "requested_perturbation": {
                        "controlled_changes": ["trajectory smoothness"]
                    },
                }
            },
        )
        self.assertEqual(contract.status, "unsupported")
        self.assertIn("cannot express", contract.reason)

    def test_goal_identity_change_is_authorized_for_retrieved_source_only(self):
        retrieval = BDDLTaskIndex([_task(0, "pick object")]).retrieve_nearest(
            "object identity robustness",
            compatibility=_profile(0),
        )
        contract = authorize_controlled_change(
            retrieval,
            {
                "proposal": {
                    "requested_perturbation": {
                        "controlled_changes": ["goal object identity"]
                    }
                }
            },
        )
        self.assertTrue(contract.authorized)
        self.assertEqual(contract.source_task_id, retrieval.selected.task_id)

    def test_retrieval_fails_closed_when_profile_supports_no_indexed_task(self):
        with self.assertRaisesRegex(LiberoContractError, "no BDDL candidate"):
            BDDLTaskIndex([_task(0, "pick object")]).retrieve_nearest(
                "pick object",
                compatibility=_profile(9),
            )

    def test_unknown_checkpoint_scope_binding_is_not_training_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "README.md").write_text(
                "---\ndatasets: unknown\n---\n", encoding="utf-8"
            )
            (checkpoint / "config.json").write_text(
                json.dumps({"repo_id": "None"}), encoding="utf-8"
            )
            task = _task(0, "pick object")
            profile = smolvla_policy_compatibility(
                checkpoint=checkpoint,
                explicit_task_binding=task,
            )
        self.assertEqual(profile.declared_scope, "unknown")
        self.assertEqual(
            profile.authorization_source, "explicit_run_binding"
        )
        self.assertEqual(profile.verdict(task), "explicit_run_binding_only")
        self.assertEqual(
            profile.artifact_evidence["model_card_datasets"], "unknown"
        )


if __name__ == "__main__":
    unittest.main()
