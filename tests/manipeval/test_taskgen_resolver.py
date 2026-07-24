import unittest
from pathlib import Path
from unittest.mock import patch

from mea.capability_adapter import resolve_capability_contract
from mea.proposals import task_proposal_from_contract
from mea.taskgen.resolver import TaskResolutionError, resolve_task_proposal
from mea.taskgen.reviewed_registry import ReviewedTaskRegistryError
from scripts.manipeval_taskgen import reviewed_task_lookup_with_fallback


class TaskGenResolverTests(unittest.TestCase):
    @staticmethod
    def proposal(task_name: str, template_id: str) -> tuple[dict, dict]:
        contract = resolve_capability_contract(task_name, template_id)
        proposal = task_proposal_from_contract(
            contract, intent=f"evaluate {template_id}"
        )
        return contract, proposal

    def test_official_and_bounded_overlay_win_before_reviewed_lookup(self):
        def unexpected_lookup(_query):
            raise AssertionError("built-in routes must not consult reviewed artifacts")

        official, official_proposal = self.proposal(
            "beat_block_hammer", "safety.hammer_left_camera_contact.official"
        )
        official_decision = resolve_task_proposal(
            official_proposal, official, find_reviewed=unexpected_lookup
        )
        self.assertEqual(official_decision["requested_route"], "official")
        self.assertEqual(official_decision["resolved_route"], "official")
        self.assertEqual(official_decision["materialization"], "official_reuse")
        self.assertFalse(official_decision["provider_required"])
        self.assertFalse(official_decision["reviewed_lookup_attempted"])

        overlay, overlay_proposal = self.proposal(
            "click_bell", "object_position.left_fixed"
        )
        overlay_decision = resolve_task_proposal(
            overlay_proposal, overlay, find_reviewed=unexpected_lookup
        )
        self.assertEqual(overlay_decision["requested_route"], "reuse")
        self.assertEqual(overlay_decision["resolved_route"], "reuse")
        self.assertEqual(overlay_decision["materialization"], "bounded_overlay")
        self.assertFalse(overlay_decision["provider_required"])
        self.assertFalse(overlay_decision["reviewed_lookup_attempted"])

    def test_exact_approved_generated_artifact_reuses_without_provider(self):
        contract, proposal = self.proposal(
            "beat_block_hammer", "object_appearance.color_blue"
        )

        def exact_lookup(query):
            return {
                "schema_version": 1,
                "registration_id": "reviewed_bbh_blue",
                "artifact_id": "task_artifact_bbh_blue_v1",
                "status": "approved",
                "semantic_key": query["semantic_key"],
                "semantic_key_sha256": query["semantic_key_sha256"],
            }

        decision = resolve_task_proposal(
            proposal, contract, find_reviewed=exact_lookup
        )
        self.assertEqual(decision["requested_route"], "force_codegen")
        self.assertEqual(decision["resolved_route"], "reviewed_generated_reuse")
        self.assertEqual(
            decision["materialization"], "reviewed_generated_artifact"
        )
        self.assertFalse(decision["provider_required"])
        self.assertTrue(decision["reviewed_lookup_attempted"])
        self.assertEqual(
            decision["reviewed_registration"]["registration_id"],
            "reviewed_bbh_blue",
        )

    def test_query_rewording_does_not_change_executable_semantic_key(self):
        contract, first = self.proposal(
            "beat_block_hammer", "object_scale.bounded_1_2"
        )
        second = dict(first)
        second["proposal_id"] = "scale-query-rewrite"
        second["intent"] = "Use different wording for the same bounded scale."
        first_decision = resolve_task_proposal(first, contract)
        second_decision = resolve_task_proposal(second, contract)
        self.assertEqual(
            first_decision["semantic_key_sha256"],
            second_decision["semantic_key_sha256"],
        )
        self.assertFalse(first_decision["reviewed_lookup_attempted"])
        self.assertFalse(second_decision["reviewed_lookup_attempted"])
        self.assertIn("not configured", first_decision["reason"])

    def test_missing_reviewed_match_falls_back_to_generation(self):
        contract, proposal = self.proposal(
            "beat_block_hammer", "object_scale.bounded_1_2"
        )
        decision = resolve_task_proposal(
            proposal, contract, find_reviewed=lambda _query: None
        )
        self.assertEqual(decision["requested_route"], "force_codegen")
        self.assertEqual(decision["resolved_route"], "force_codegen")
        self.assertEqual(decision["materialization"], "generate")
        self.assertTrue(decision["provider_required"])
        self.assertTrue(decision["reviewed_lookup_attempted"])

    def test_invalid_registry_is_audited_and_falls_back_to_generation(self):
        with patch(
            "scripts.manipeval_taskgen.find_reviewed_task",
            side_effect=ReviewedTaskRegistryError("invalid layout"),
        ):
            match, issue = reviewed_task_lookup_with_fallback(
                Path("/registry"),
                {"schema_version": 1},
                repo_root=Path("/repo"),
            )

        self.assertIsNone(match)
        self.assertEqual(
            issue["status"],
            "invalid_registry_fallback_to_generation",
        )
        self.assertEqual(issue["error_type"], "ReviewedTaskRegistryError")
        self.assertIn("invalid layout", issue["message"])

    def test_non_exact_or_unapproved_reviewed_match_fails_closed(self):
        contract, proposal = self.proposal(
            "beat_block_hammer", "object_appearance.color_blue"
        )

        def corrupt_lookup(query):
            return {
                "schema_version": 1,
                "registration_id": "reviewed_wrong",
                "artifact_id": "wrong_artifact",
                "status": "approved",
                "semantic_key": query["semantic_key"],
                "semantic_key_sha256": "0" * 64,
            }

        with self.assertRaisesRegex(TaskResolutionError, "hash does not match"):
            resolve_task_proposal(proposal, contract, find_reviewed=corrupt_lookup)

    def test_proposal_cannot_cross_the_contract_capability(self):
        appearance, proposal = self.proposal(
            "beat_block_hammer", "object_appearance.color_blue"
        )
        scale = resolve_capability_contract(
            "beat_block_hammer", "object_scale.bounded_1_2"
        )
        self.assertNotEqual(
            appearance["taskgen"]["capability_id"],
            scale["taskgen"]["capability_id"],
        )
        with self.assertRaisesRegex(TaskResolutionError, "capability"):
            resolve_task_proposal(proposal, scale)


if __name__ == "__main__":
    unittest.main()
