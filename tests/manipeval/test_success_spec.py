import ast
import unittest
from pathlib import Path

import numpy as np

from mea.taskgen import (
    SUCCESS_SPEC_V2_DEVELOPMENT_ENVELOPE,
    SUCCESS_SPEC_V2_OFFICIAL_ENVELOPE,
    SuccessSpecError,
    SuccessSpecRepairError,
    compile_success_spec,
    default_bbh_success_spec,
    default_bbh_success_spec_v2,
    development_bbh_success_spec_v2,
    repair_success_spec,
    success_spec_validation_report,
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

    def test_v2_official_envelope_remains_official_equivalent_and_act_eligible(self):
        spec = validate_success_spec(default_bbh_success_spec_v2())
        source, validation = compile_success_spec(spec)

        self.assertEqual(spec["schema_version"], 2)
        self.assertEqual(
            spec["envelope_id"], SUCCESS_SPEC_V2_OFFICIAL_ENVELOPE
        )
        self.assertEqual(validation["compiler"], "restricted_success_spec_v2")
        self.assertTrue(validation["official_equivalent"])
        self.assertTrue(validation["act_eligible"])
        self.assertFalse(validation["development_fixture"])
        self.assertFalse(validation["development_fixture_compilation"])
        self.assertIn(" and self.check_actors_contact(", source)

    def test_v2_validation_report_is_structured_and_envelope_bounded(self):
        spec = development_bbh_success_spec_v2()

        report = success_spec_validation_report(spec)

        self.assertEqual(
            report["envelope_id"], SUCCESS_SPEC_V2_DEVELOPMENT_ENVELOPE
        )
        self.assertEqual(report["logic"], "any")
        self.assertEqual(
            report["predicates"],
            ["planar_axis_distance", "physical_contact"],
        )
        self.assertFalse(report["official_equivalent"])
        self.assertFalse(report["act_eligible"])
        self.assertTrue(report["development_fixture"])
        self.assertEqual(
            set(report["checks"]),
            {
                "closed_schema",
                "trusted_envelope",
                "bounded_predicates",
                "trusted_actor_bindings",
                "bounded_thresholds",
                "official_equivalence_required_for_act",
            },
        )
        self.assertTrue(all(report["checks"].values()))

    def test_v2_any_is_development_only_and_never_implicitly_act_eligible(self):
        spec = development_bbh_success_spec_v2(logic="any")

        with self.assertRaisesRegex(SuccessSpecError, "not ACT eligible"):
            compile_success_spec(spec)

        source, validation = compile_success_spec(
            spec, allow_development_fixture=True
        )
        namespace = {"np": np}
        exec(compile(source, "<development SuccessSpec v2>", "exec"), namespace)
        generated = namespace["check_success"]

        self.assertTrue(validation["development_fixture_compilation"])
        self.assertFalse(validation["act_eligible"])
        self.assertIn(" or self.check_actors_contact(", source)
        self.assertTrue(generated(_FixtureTask((0.0, 0.0), (0.04, 0.0), True)))
        self.assertTrue(generated(_FixtureTask((0.0, 0.0), (0.01, 0.01), False)))
        self.assertFalse(generated(_FixtureTask((0.0, 0.0), (0.04, 0.0), False)))

    def test_v2_official_envelope_rejects_non_equivalent_logic_and_threshold(self):
        changed_logic = default_bbh_success_spec_v2()
        changed_logic["logic"] = "any"
        with self.assertRaisesRegex(SuccessSpecError, "must use logic 'all'"):
            validate_success_spec(changed_logic)

        changed_threshold = default_bbh_success_spec_v2()
        changed_threshold["predicates"][0]["thresholds_m"] = [0.03, 0.02]
        with self.assertRaisesRegex(SuccessSpecError, "official thresholds"):
            validate_success_spec(changed_threshold)

    def test_v2_development_envelope_still_rejects_unbounded_fields(self):
        oversized_threshold = development_bbh_success_spec_v2(
            thresholds_m=(0.051, 0.02)
        )
        with self.assertRaisesRegex(SuccessSpecError, r"\(0.0, 0.05\]"):
            validate_success_spec(oversized_threshold)

        changed_actor = development_bbh_success_spec_v2()
        changed_actor["predicates"][1]["actors"] = ["hammer", "table"]
        with self.assertRaisesRegex(SuccessSpecError, "physical_contact.actors"):
            validate_success_spec(changed_actor)

        injected_field = development_bbh_success_spec_v2()
        injected_field["python"] = "open('/tmp/x')"
        with self.assertRaisesRegex(SuccessSpecError, "fields must be exactly"):
            validate_success_spec(injected_field)

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
        official = _load_official_check_success()
        fixtures = (
            ((0.0, 0.0), (0.019, -0.019), True, True),
            ((0.0, 0.0), (0.020, 0.0), True, False),
            ((0.0, 0.0), (0.0, 0.021), True, False),
            ((0.0, 0.0), (0.001, 0.001), False, False),
        )
        for spec in (default_bbh_success_spec(), default_bbh_success_spec_v2()):
            source, _ = compile_success_spec(spec)
            namespace = {"np": np}
            exec(compile(source, "<generated check_success>", "exec"), namespace)
            generated = namespace["check_success"]
            for hammer_xy, block_xy, contact, expected in fixtures:
                with self.subTest(
                    schema_version=spec["schema_version"],
                    hammer_xy=hammer_xy,
                    block_xy=block_xy,
                    contact=contact,
                ):
                    task = _FixtureTask(hammer_xy, block_xy, contact)
                    self.assertEqual(bool(generated(task)), expected)
                    self.assertEqual(bool(generated(task)), bool(official(task)))


if __name__ == "__main__":
    unittest.main()
