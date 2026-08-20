import unittest

from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    build_experiment_candidate,
    validate_experiment_candidate,
)


class ExperimentCandidateTests(unittest.TestCase):
    def test_runtime_concern_materializes_without_template_id(self):
        candidate = build_experiment_candidate(
            source_query="Where does reflective appearance first fail?",
            base_task="beat_block_hammer",
            semantic_concern=(
                "novel.reflective_surface_confusion: "
                "A reflective target causes a pre-contact miss."
            ),
            scene_need="make only the target surface reflective",
            checker_need=(
                "Generate an experimental check_success predicate that decides "
                "whether the reflective target causes a pre-contact miss."
            ),
            tool_need="target contact and pre-contact trajectory",
        )

        self.assertEqual(candidate["base_task"], "beat_block_hammer")
        self.assertTrue(candidate["candidate_id"].startswith("dynamic."))
        self.assertNotIn("template_id", candidate)
        self.assertIn(
            "pre-contact miss", candidate["checker_need"]["description"]
        )
        self.assertEqual(candidate["schema_version"], 2)

    def test_tool_only_candidate_does_not_invent_scene_or_checker_work(self):
        candidate = build_experiment_candidate(
            source_query="Does the TCP jerk before contact?",
            base_task="beat_block_hammer",
            semantic_concern="observability.precontact_jerk",
            rule_tool_need={
                "kind": "measure",
                "description": "Measure peak jerk before first target contact.",
                "reuse_first": True,
            },
        )

        self.assertIsNone(candidate["scene_need"])
        self.assertIsNone(candidate["checker_need"])
        self.assertEqual(candidate["rule_tool_need"]["kind"], "measure")

    def test_default_id_names_the_concern_instead_of_hashing_the_payload(self):
        scene_candidate = build_experiment_candidate(
            source_query="Where does target pose fail?",
            base_task="alpha_task",
            semantic_concern="target.pose",
            scene_need="move only the target",
        )
        checker_candidate = build_experiment_candidate(
            source_query="Where does target pose fail?",
            base_task="alpha_task",
            semantic_concern="target.pose",
            checker_need="check the target pose",
        )

        self.assertEqual(
            scene_candidate["candidate_id"],
            "dynamic.alpha_task.target.pose",
        )
        self.assertEqual(
            checker_candidate["candidate_id"],
            scene_candidate["candidate_id"],
        )

    def test_rule_and_vqa_tool_needs_remain_independent(self):
        candidate = build_experiment_candidate(
            source_query="Does motion look unstable and exceed the jerk limit?",
            base_task="adjust_bottle",
            semantic_concern="motion.post_release_instability",
            rule_tool_need={
                "kind": "measure",
                "description": "Measure post-release angular jerk.",
                "reuse_first": True,
            },
            vqa_tool_need={
                "kind": "vqa",
                "description": "Check whether the bottle visibly wobbles.",
                "reuse_first": True,
            },
        )

        self.assertIsNone(candidate["scene_need"])
        self.assertIsNone(candidate["checker_need"])
        self.assertEqual(candidate["rule_tool_need"]["kind"], "measure")
        self.assertEqual(candidate["vqa_tool_need"]["kind"], "vqa")

    def test_unchanged_official_candidate_may_request_no_artifacts(self):
        candidate = build_experiment_candidate(
            source_query="Retry the unchanged official task.",
            base_task="beat_block_hammer",
            semantic_concern="task_execution.official_retry",
        )

        self.assertEqual(
            candidate["candidate_id"],
            "dynamic.beat_block_hammer.task_execution.official_retry",
        )
        for field in (
            "scene_need",
            "checker_need",
            "rule_tool_need",
            "vqa_tool_need",
            "tool_need",
        ):
            self.assertIsNone(candidate[field])

    def test_legacy_candidate_is_read_as_typed_needs(self):
        candidate = validate_experiment_candidate(
            {
                "schema_version": 1,
                "candidate_id": "legacy.candidate",
                "source_query": "Where does it fail?",
                "base_task": "adjust_bottle",
                "semantic_concern": "post-release stability",
                "scene_need": "Reuse the official scene.",
                "checker_need": "Check stable final pose.",
                "tool_need": "Measure angular velocity.",
            }
        )

        self.assertEqual(candidate["schema_version"], 2)
        self.assertEqual(candidate["scene_need"]["kind"], "adapt")
        self.assertEqual(candidate["checker_need"]["kind"], "generate")
        self.assertEqual(candidate["rule_tool_need"]["kind"], "measure")

    def test_direct_builder_rejects_a_predeclared_template(self):
        candidate = build_experiment_candidate(
            source_query="Does the bottle wobble after placement?",
            base_task="adjust_bottle",
            semantic_concern="post-placement stability",
            scene_need="reuse the official scene",
            checker_need="check final pose and sustained stability",
            rule_tool_need="measure post-release angular velocity",
        )
        invalid = {**candidate, "template_id": "predeclared.variant"}
        with self.assertRaisesRegex(
            ExperimentCandidateError, "fields must be exactly"
        ):
            validate_experiment_candidate(invalid)


if __name__ == "__main__":
    unittest.main()
