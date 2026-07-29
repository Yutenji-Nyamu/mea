import unittest
from copy import deepcopy

from mea.planner.policy_task_binding import (
    PolicyTaskBindingError,
    build_policy_task_binding,
    validate_policy_task_binding,
)


def _binding() -> dict:
    return build_policy_task_binding(
        task_name="adjust_bottle",
        task_family="object_manipulation",
        policy={
            "name": "ACT",
            "checkpoint_setting": "demo_clean",
            "expert_data_num": 50,
            "language_conditioned": False,
        },
        checkpoint={
            "policy_name": "ACT",
            "checkpoint_setting": "demo_clean",
            "expert_data_num": 50,
            "checkpoint_id": "act-adjust_bottle/demo_clean-50",
            "ready": True,
        },
    )


class PolicyTaskBindingTests(unittest.TestCase):
    def test_binding_contains_execution_authority_not_planner_inventory(self):
        binding = _binding()

        self.assertEqual(binding["task_module"], "envs.adjust_bottle")
        self.assertEqual(
            binding["hooks"]["official_success"]["method"],
            "check_success",
        )
        self.assertEqual(
            binding["hooks"]["rollout"]["entrypoint"],
            "policy/ACT/eval_mea.sh",
        )
        self.assertTrue(
            {
                "planner_kind",
                "task_profile",
                "max_rounds",
                "aspects",
                "templates",
            }.isdisjoint(binding)
        )

    def test_hook_cannot_switch_execution_module(self):
        changed = deepcopy(_binding())
        changed["hooks"]["render"]["module"] = "envs.other_task"

        with self.assertRaisesRegex(
            PolicyTaskBindingError,
            "render hook differs",
        ):
            validate_policy_task_binding(changed)


if __name__ == "__main__":
    unittest.main()
