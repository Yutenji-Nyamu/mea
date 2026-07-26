import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.paper.manipeval_click_bell_open_taskgen import (
    OpenClickBellGate0Error,
    proposal_from_open_resolution,
    run_gate0,
)


QUERY = (
    "Can this policy reliably click the intended object when another similar "
    "interactive object is nearby, and where would it first fail?"
)


def _resolution(
    *,
    decision: str = "retrieve_and_adapt",
    selected_task: str = "click_bell",
    requested_variation: str = "add one similar bell as a physical distractor",
) -> dict:
    return {
        "schema_version": 1,
        "decision": decision,
        "reason_code": "nearest_official_base",
        "free_concern": {
            "schema_version": 1,
            "source_query": QUERY,
            "sub_aspect": "target selectivity around a similar interactive object",
            "hypothesis": "the policy may click the nearby distractor",
            "task_intent": "press the intended bell",
            "requested_variation": requested_variation,
            "measurement_need": "contacts with the intended and distractor bells",
        },
        "policy_scope": {},
        "selected_base_task": {
            "task_name": selected_task,
            "score": 0.5,
        },
        "ranked_candidates": [],
        "resolution_contract": {},
    }


class ClickBellOpenTaskGenTests(unittest.TestCase):
    def test_adapts_only_after_open_resolution_selects_click_bell(self) -> None:
        proposal, concern = proposal_from_open_resolution(_resolution())

        self.assertEqual(proposal["task_name"], "click_bell")
        self.assertEqual(proposal["query"], QUERY)
        self.assertEqual(concern["task_intent"], "press the intended bell")
        self.assertIn("target selectivity", proposal["intent"])

    def test_rejects_policy_task_mismatch_without_provider_call(self) -> None:
        with self.assertRaisesRegex(
            OpenClickBellGate0Error,
            "did not authorize nearest-task adaptation",
        ):
            proposal_from_open_resolution(
                _resolution(decision="unsupported", selected_task="scan_object")
            )

    def test_rejects_non_click_bell_resolution(self) -> None:
        with self.assertRaisesRegex(
            OpenClickBellGate0Error,
            "did not select click_bell",
        ):
            proposal_from_open_resolution(_resolution(selected_task="scan_object"))

    def test_rejects_unrelated_concern(self) -> None:
        resolution = _resolution(
            requested_variation="change the table illumination",
        )
        resolution["free_concern"].update(
            {
                "sub_aspect": "illumination robustness",
                "hypothesis": "the policy may fail under dim lighting",
                "requested_variation": "change the table illumination",
                "measurement_need": "success under two light levels",
            }
        )
        with self.assertRaisesRegex(
            OpenClickBellGate0Error,
            "outside the available ClickBell distractor dialect",
        ):
            proposal_from_open_resolution(resolution)

    def test_frozen_replay_is_unwrapped_before_any_provider_call(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "deterministic_repair_replay.json"
            source.write_text(
                __import__("json").dumps(
                    {
                        "schema_version": 1,
                        "provider_calls": 0,
                        "repaired_resolution": _resolution(
                            decision="unsupported",
                            selected_task="scan_object",
                        ),
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                repo_root=root,
                open_resolution_json=source,
                run_id="must_not_start",
                text_model="unused",
                base_url=None,
                seed=0,
            )
            with patch(
                "experiments.paper.manipeval_click_bell_open_taskgen."
                "OpenAICompatibleProvider"
            ) as provider:
                with self.assertRaisesRegex(
                    OpenClickBellGate0Error,
                    "did not authorize nearest-task adaptation",
                ):
                    run_gate0(args)
            provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
