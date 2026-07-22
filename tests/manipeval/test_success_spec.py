import ast
import unittest
from pathlib import Path

import numpy as np

from mea.taskgen import (
    SuccessSpecError,
    SuccessSpecRepairError,
    compile_success_spec,
    default_bbh_success_spec,
    repair_success_spec,
    validate_compiled_success_method,
    validate_success_spec,
)


class _Pose:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=float)


class _Actor:
    def __init__(self, name, points):
        self._name = name
        self._points = points

    def get_functional_point(self, point_id, mode):
        if mode != "pose":
            raise AssertionError(mode)
        return _Pose(self._points[point_id])

    def get_name(self):
        return self._name


class _FixtureTask:
    def __init__(self, hammer_xy, block_xy, contact):
        self.hammer = _Actor("hammer", {0: [*hammer_xy, 0.8]})
        self.block = _Actor("box", {1: [*block_xy, 0.8]})
        self._contact = contact

    def check_actors_contact(self, left, right):
        if (left, right) != ("hammer", "box"):
            raise AssertionError((left, right))
        return self._contact


def _load_official_check_success():
    root = Path(__file__).resolve().parents[2]
    source = (root / "envs/beat_block_hammer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    task_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "beat_block_hammer"
    )
    method = next(
        node
        for node in task_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_success"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, "<official beat_block_hammer.check_success>", "exec"), namespace)
    return namespace["check_success"]


class SuccessSpecTests(unittest.TestCase):
    def test_accepts_valid_candidate_without_spending_repair_budget(self):
        candidate = default_bbh_success_spec()

        accepted, report = repair_success_spec(candidate)

        self.assertEqual(accepted, validate_success_spec(candidate))
        self.assertFalse(report["repaired"])
        self.assertEqual(report["final_source"], "candidate")
        self.assertEqual(
            report["attempts"],
            [
                {
                    "attempt_index": 0,
                    "source": "candidate",
                    "valid": True,
                    "diagnosis": None,
                }
            ],
        )

    def test_diagnoses_invalid_candidate_and_repairs_once_with_trusted_default(self):
        candidate = default_bbh_success_spec()
        candidate["predicates"][0]["thresholds_m"] = [0.03, 0.02]

        repaired, report = repair_success_spec(candidate, max_repairs=1)
        source, validation = compile_success_spec(repaired)

        self.assertEqual(repaired, default_bbh_success_spec())
        self.assertTrue(report["repaired"])
        self.assertEqual(report["final_source"], "trusted_default")
        self.assertEqual(len(report["attempts"]), 2)
        self.assertFalse(report["attempts"][0]["valid"])
        self.assertIn(
            "official thresholds",
            report["attempts"][0]["diagnosis"]["message"],
        )
        self.assertTrue(report["attempts"][1]["valid"])
        self.assertIn("def check_success(self):", source)
        self.assertTrue(validation["valid"])

    def test_zero_repair_budget_fails_closed_with_diagnosis_report(self):
        candidate = default_bbh_success_spec()
        candidate["task_name"] = "untrusted_task"

        with self.assertRaises(SuccessSpecRepairError) as raised:
            repair_success_spec(candidate, max_repairs=0)

        report = raised.exception.report
        self.assertFalse(report["repaired"])
        self.assertIsNone(report["final_source"])
        self.assertEqual(len(report["attempts"]), 1)
        self.assertFalse(report["attempts"][0]["valid"])
        self.assertEqual(report["attempts"][0]["source"], "candidate")
        self.assertIn(
            "only supports beat_block_hammer",
            report["attempts"][0]["diagnosis"]["message"],
        )

    def test_repair_budget_is_bounded_to_one(self):
        with self.assertRaisesRegex(SuccessSpecError, "must be 0 or 1"):
            repair_success_spec(default_bbh_success_spec(), max_repairs=2)

    def test_compiles_closed_two_predicate_contract(self):
        spec = validate_success_spec(default_bbh_success_spec())
        source, validation = compile_success_spec(spec)

        self.assertIn("def check_success(self):", source)
        self.assertIn("check_actors_contact", source)
        self.assertEqual(
            validation["predicates"],
            ["planar_axis_distance", "physical_contact"],
        )
        self.assertFalse(validation["arbitrary_code_accepted"])

    def test_rejects_threshold_actor_and_code_expansion(self):
        changed_threshold = default_bbh_success_spec()
        changed_threshold["predicates"][0]["thresholds_m"] = [0.03, 0.02]
        with self.assertRaisesRegex(SuccessSpecError, "official thresholds"):
            validate_success_spec(changed_threshold)

        changed_actor = default_bbh_success_spec()
        changed_actor["predicates"][1]["actors"] = ["hammer", "table"]
        with self.assertRaisesRegex(SuccessSpecError, "physical_contact.actors"):
            validate_success_spec(changed_actor)

        boolean_point = default_bbh_success_spec()
        boolean_point["predicates"][0]["left"]["functional_point_id"] = False
        with self.assertRaisesRegex(SuccessSpecError, "must be an integer"):
            validate_success_spec(boolean_point)

        boolean_schema = default_bbh_success_spec()
        boolean_schema["schema_version"] = True
        with self.assertRaisesRegex(SuccessSpecError, "must be an integer"):
            validate_success_spec(boolean_schema)

        spec = default_bbh_success_spec()
        source, _ = compile_success_spec(spec)
        with self.assertRaisesRegex(SuccessSpecError, "one function"):
            validate_compiled_success_method(
                source + "\ndef unwanted(self):\n    return open('/tmp/x')\n",
                spec,
            )

    def test_generated_method_matches_current_official_semantics(self):
        source, _ = compile_success_spec(default_bbh_success_spec())
        namespace = {"np": np}
        exec(compile(source, "<generated check_success>", "exec"), namespace)
        generated = namespace["check_success"]
        official = _load_official_check_success()
        fixtures = (
            ((0.0, 0.0), (0.019, -0.019), True, True),
            ((0.0, 0.0), (0.020, 0.0), True, False),
            ((0.0, 0.0), (0.0, 0.021), True, False),
            ((0.0, 0.0), (0.001, 0.001), False, False),
        )
        for hammer_xy, block_xy, contact, expected in fixtures:
            with self.subTest(
                hammer_xy=hammer_xy, block_xy=block_xy, contact=contact
            ):
                task = _FixtureTask(hammer_xy, block_xy, contact)
                self.assertEqual(bool(generated(task)), expected)
                self.assertEqual(bool(generated(task)), bool(official(task)))


if __name__ == "__main__":
    unittest.main()
