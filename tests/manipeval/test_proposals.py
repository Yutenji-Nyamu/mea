import unittest
from copy import deepcopy

from mea.proposals import ProposalError, validate_task_proposal
from mea.taskgen.success_spec import (
    default_bbh_success_spec_v2,
    development_bbh_success_spec_v2,
    experimental_bbh_success_spec_v2,
)


def _proposal(*, schema_version=1, success_spec=None):
    value = {
        "schema_version": schema_version,
        "proposal_id": "object_appearance.experimental_success",
        "task_name": "beat_block_hammer",
        "aspect_id": "object_appearance.color",
        "intent": "test a blue block with a bounded experimental success threshold",
        "capability_id": "object_appearance.color",
        "reuse_first": True,
        "changes": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.0, 0.2, 1.0],
            }
        },
        "preserve_success_semantics": schema_version == 1,
    }
    if schema_version == 2:
        value["success_spec"] = (
            success_spec
            if success_spec is not None
            else experimental_bbh_success_spec_v2()
        )
    return value


class TaskProposalSuccessSpecTests(unittest.TestCase):
    def test_v1_contract_remains_unchanged(self):
        proposal = _proposal()

        self.assertEqual(validate_task_proposal(proposal), proposal)

    def test_v2_accepts_only_non_preserving_bounded_act_success_spec(self):
        proposal = _proposal(schema_version=2)

        accepted = validate_task_proposal(proposal)

        self.assertFalse(accepted["preserve_success_semantics"])
        self.assertEqual(
            accepted["success_spec"], experimental_bbh_success_spec_v2()
        )

    def test_v2_rejects_other_success_envelopes_and_task_scope(self):
        cases = []
        preserved = _proposal(schema_version=2)
        preserved["preserve_success_semantics"] = True
        cases.append(("preserved", preserved, "must be false"))

        official = _proposal(
            schema_version=2, success_spec=default_bbh_success_spec_v2()
        )
        cases.append(("official", official, "experimental bounded"))

        development = _proposal(
            schema_version=2, success_spec=development_bbh_success_spec_v2()
        )
        cases.append(("development", development, "experimental bounded"))

        official_semantics_in_experimental_envelope = _proposal(
            schema_version=2,
            success_spec=experimental_bbh_success_spec_v2(
                thresholds_m=(0.02, 0.02)
            ),
        )
        cases.append(
            (
                "official_semantics_in_experimental_envelope",
                official_semantics_in_experimental_envelope,
                "must differ from official",
            )
        )

        other_task = _proposal(schema_version=2)
        other_task["task_name"] = "click_bell"
        cases.append(("task", other_task, "only supports beat_block_hammer"))

        for name, proposal, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                ProposalError, message
            ):
                validate_task_proposal(proposal)

    def test_v2_rejects_any_actor_axis_and_comparison_changes(self):
        mutations = {
            "logic": lambda spec: spec.update(logic="any"),
            "actor": lambda spec: spec["predicates"][1].update(
                actors=["hammer", "table"]
            ),
            "axis": lambda spec: spec["predicates"][0].update(axes=[0, 2]),
            "comparison": lambda spec: spec["predicates"][0].update(
                comparison="lte"
            ),
        }
        for name, mutate in mutations.items():
            proposal = _proposal(schema_version=2)
            spec = deepcopy(proposal["success_spec"])
            mutate(spec)
            proposal["success_spec"] = spec
            with self.subTest(name=name), self.assertRaisesRegex(
                ProposalError, "invalid TaskProposal SuccessSpec"
            ):
                validate_task_proposal(proposal)


if __name__ == "__main__":
    unittest.main()
