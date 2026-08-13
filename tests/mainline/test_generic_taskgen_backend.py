from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from mea.planner.experiment_candidate import build_experiment_candidate
from mea.planner.semantic_coverage import build_evaluation_intent
from mea.taskgen.attempts import (
    CandidateUnexecutableError,
    task_generation_recovery_action,
)
from mea.taskgen.generic_backend import (
    GenericRoboTwinTaskAdapter,
    GenericRoboTwinTaskGenBackend,
    GenericTaskGenError,
    GenericTaskGenHooks,
    _GENERIC_READ_ONLY_METHOD_CALLS,
    _candidate_requires_official_core_conjunct,
    _semantic_field_access_guide,
    build_generic_task_subclass_module,
    discover_generic_robotwin_task_identity,
    generic_task_semantic_key,
    load_generic_robotwin_task_adapter,
    validate_generic_task_methods,
)
from mea.taskgen.provider_scene_checker import (
    run_provider_codegen,
    validate_method_ast,
)
from mea.taskgen.probe import robot_tcp_xyz_summary
from mea.taskgen.probe_runtime import _write_json as _write_probe_json
from mea.taskgen.runtime import (
    _checker_fixture_failure_diagnosis,
    _generated_checker_execution_failure,
    build_preservation_report,
    record_generic_taskgen_generation_failure,
)


class ProbeRuntimeContractTests(unittest.TestCase):
    def test_probe_json_writer_emits_json_whitespace_not_literal_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "probe.json"
            _write_probe_json(output, {"passed": True})

            text = output.read_text(encoding="utf-8")
            self.assertEqual(json.loads(text), {"passed": True})
            self.assertTrue(text.endswith("\n"))
            self.assertFalse(text.endswith("\\n"))


class _Provider:
    def __init__(self, responses: list[dict[str, str]]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []
        self.review_calls = 0
        self.review_prompts: list[str] = []
        self.last_metadata: dict[str, Any] = {}

    def text(self, prompt: str, **_kwargs: Any) -> str:
        if "TaskGen's separate checker semantic-review pass" in prompt:
            self.review_calls += 1
            self.review_prompts.append(prompt)
            self.last_metadata = {"review_call": self.review_calls}
            return json.dumps(
                {
                    "schema_version": 1,
                    "status": "approved",
                    "checks": {
                        "implements_every_checker_requirement": True,
                        "preserves_quantifiers_and_temporal_relations": True,
                        "uses_direct_current_simulator_observables": True,
                        "does_not_substitute_correlated_proxy": True,
                    },
                    "reason": "The fixture checker directly implements its Proposal.",
                }
            )
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        self.last_metadata = {"call": self.calls}
        return json.dumps(response)


class ProviderSceneCheckerRepairTests(unittest.TestCase):
    def test_preservation_failure_is_typed_and_keeps_probe_runtime(self):
        responses = [
            {
                "load_actors": (
                    "def load_actors(self):\n"
                    '    self.target = "generated"\n'
                ),
                "check_success": (
                    "def check_success(self):\n"
                    "    return True\n"
                ),
            }
        ]

        def validate(_methods: Mapping[str, str]) -> dict[str, Any]:
            raise GenericTaskGenError(
                "generated task violated a checked preservation condition: "
                "sampled y position",
                runtime={"simulator_probes": 2, "expert_probes": 1},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(GenericTaskGenError):
                run_provider_codegen(
                    attempt_root=root / "attempts",
                    proposal={"task_name": "fixture"},
                    prompt="generate methods",
                    provider=_Provider(responses),
                    model="fixture-model",
                    validate=validate,
                    error_type=GenericTaskGenError,
                    max_regenerations=0,
                )
            summary = json.loads(
                (root / "attempts/task_generation_attempt_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        failure = summary["attempts"][0]["failure"]
        self.assertEqual(failure["stage"], "preservation_validation")
        self.assertEqual(failure["failure_kind"], "failed")
        self.assertEqual(summary["runtime"]["simulator_probes"], 2)
        self.assertEqual(summary["runtime"]["expert_probes"], 1)
        self.assertEqual(
            task_generation_recovery_action(
                failure["stage"], failure["failure_kind"]
            ),
            "regenerate_candidate",
        )

    def test_unchanged_checker_repair_stops_before_revalidation(self):
        scene = (
            "def load_actors(self):\n"
            '    self.target = "stable_scene"\n'
        )
        checker = (
            "def check_success(self):\n"
            "    return False\n"
        )
        provider = _Provider(
            [
                {"load_actors": scene, "check_success": checker},
                {"load_actors": scene, "check_success": checker},
            ]
        )
        validations = 0

        def validate(_methods: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal validations
            validations += 1
            raise GenericTaskGenError(
                "generated checker failed live negative/positive fixtures"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            attempt_root = Path(temp_dir) / "attempts"
            with self.assertRaises(GenericTaskGenError):
                run_provider_codegen(
                    attempt_root=attempt_root,
                    proposal={"candidate_id": "dynamic.synthetic"},
                    prompt="Generate a method pair.",
                    provider=provider,
                    model="fixture-model",
                    validate=validate,
                    error_type=GenericTaskGenError,
                    max_regenerations=1,
                )
            terminal = json.loads(
                (attempt_root / "attempt_02/attempt_result.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(validations, 1)
        self.assertIn(
            "repeated the failing checker unchanged",
            terminal["failure"]["message"],
        )

    def test_live_checker_failure_preserves_simulator_validated_scene(self):
        first_scene = (
            "def load_actors(self):\n"
            '    self.target = "stable_scene"\n'
        )
        provider = _Provider(
            [
                {
                    "load_actors": first_scene,
                    "check_success": (
                        "def check_success(self):\n"
                        "    return False\n"
                    ),
                },
                {
                    "load_actors": (
                        "def load_actors(self):\n"
                        '    self.target = "changed_scene"\n'
                    ),
                    "check_success": (
                        "def check_success(self):\n"
                        "    return True\n"
                    ),
                },
            ]
        )
        validations = 0

        def validate(methods: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal validations
            validations += 1
            if validations == 1:
                raise GenericTaskGenError(
                    "generated checker failed live execution: "
                    '{"reason": "generated_checker_execution_error"}'
                )
            self.assertEqual(methods["load_actors"], first_scene)
            return {"valid": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_provider_codegen(
                attempt_root=Path(temp_dir) / "attempts",
                proposal={"candidate_id": "dynamic.synthetic"},
                prompt="Generate a method pair.",
                provider=provider,
                model="fixture-model",
                validate=validate,
                error_type=GenericTaskGenError,
                max_regenerations=1,
            )
            repair_scope = json.loads(
                (
                    Path(temp_dir)
                    / "attempts/attempt_02/repair_scope.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(result["methods"]["load_actors"], first_scene)
        self.assertIn("Copy that method exactly", provider.prompts[1])
        self.assertIn("PREVIOUS METHOD PAIR", provider.prompts[1])
        self.assertEqual(repair_scope["scope"], "checker_only")
        self.assertTrue(repair_scope["provider_scene_output_ignored"])

    def test_expert_failure_regenerates_scene_without_relaxing_checker(self):
        provider = _Provider(
            [
                {
                    "load_actors": (
                        "def load_actors(self):\n"
                        '    self.obstacle = "blocking"\n'
                    ),
                    "check_success": (
                        "def check_success(self):\n"
                        "    return self.target_height > 0.8\n"
                    ),
                },
                {
                    "load_actors": (
                        "def load_actors(self):\n"
                        '    self.obstacle = "clear"\n'
                    ),
                    "check_success": (
                        "def check_success(self):\n"
                        "    return self.target_height > 0.8\n"
                    ),
                },
            ]
        )
        validations = 0

        def validate(methods: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal validations
            validations += 1
            if validations == 1:
                raise GenericTaskGenError(
                    "generated scene/expert failed official terminal-state "
                    'authority: {"reason": "expert_execution_error"}'
                )
            return {"valid": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_provider_codegen(
                attempt_root=Path(temp_dir) / "attempts",
                proposal={"candidate_id": "dynamic.synthetic"},
                prompt="Generate a method pair.",
                provider=provider,
                model="fixture-model",
                validate=validate,
                error_type=GenericTaskGenError,
                max_regenerations=1,
            )

        self.assertEqual(
            result["methods"]["load_actors"],
            provider.responses[1]["load_actors"],
        )
        self.assertIn(
            "scene/expert-solvability failure", provider.prompts[1]
        )
        self.assertIn(
            "Preserve every official task predicate", provider.prompts[1]
        )
        self.assertNotIn("Copy that method exactly", provider.prompts[1])

        baseline_provider = _Provider([provider.responses[0]])

        def reject_official_baseline(
            _methods: Mapping[str, Any],
        ) -> dict[str, Any]:
            raise GenericTaskGenError(
                "official same-seed expert baseline is unavailable",
                runtime={"simulator_probes": 2, "expert_probes": 6},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                CandidateUnexecutableError, "recovery failed after 1 attempt"
            ) as caught:
                run_provider_codegen(
                    attempt_root=Path(temp_dir) / "attempts",
                    proposal={"candidate_id": "dynamic.synthetic"},
                    prompt="Generate a method pair.",
                    provider=baseline_provider,
                    model="fixture-model",
                    validate=reject_official_baseline,
                    error_type=GenericTaskGenError,
                    max_regenerations=1,
                )
        self.assertEqual(baseline_provider.calls, 1)
        summary = caught.exception.__cause__.summary
        self.assertEqual(summary["runtime"]["simulator_probes"], 2)
        self.assertEqual(summary["runtime"]["expert_probes"], 6)
        checker_error = _generated_checker_execution_failure(
            {
                "error": {
                    "type": "AttributeError",
                    "message": "Robot has no get_links",
                    "traceback": (
                        "generated_checker_success = "
                        "bool(task.check_success())"
                    ),
                }
            }
        )
        self.assertEqual(
            checker_error["repair_scope"],
            "checker_only_after_expert_action",
        )

    def test_expert_failure_exhaustion_is_typed_candidate_rejection(self):
        response = {
            "load_actors": (
                "def load_actors(self):\n"
                "    self.target_pose = None\n"
            ),
            "check_success": (
                "def check_success(self):\n"
                "    return False\n"
            ),
        }
        provider = _Provider([response, response])

        def reject(_methods: Mapping[str, Any]) -> dict[str, Any]:
            raise GenericTaskGenError(
                "generated scene/expert failed official terminal-state "
                'authority: {"reason":"expert_execution_error",'
                '"expert_error":{"message":"target_pose cannot be None"}}',
                runtime={"simulator_probes": 2, "expert_probes": 4},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(CandidateUnexecutableError) as raised:
                run_provider_codegen(
                    attempt_root=Path(temp_dir) / "attempts",
                    proposal={"candidate_id": "dynamic.unexecutable"},
                    prompt="Generate a method pair.",
                    provider=provider,
                    model="fixture-model",
                    validate=reject,
                    error_type=GenericTaskGenError,
                    max_regenerations=1,
                )

        summary = raised.exception.summary
        self.assertEqual(summary["attempt_count"], 2)
        self.assertEqual(summary["regenerations_used"], 1)
        self.assertEqual(summary["runtime"]["act_rollouts_started"], 0)
        self.assertTrue(
            all(
                item["failure"]["stage"] == "expert_gate"
                and item["failure"]["failure_kind"]
                == "candidate_unexecutable"
                for item in summary["attempts"]
            )
        )
        self.assertIn(
            "scene/expert-solvability failure", provider.prompts[1]
        )

def _write_cold_task_repo(root: Path) -> None:
    source = root / "envs/cold_unseen_task.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
class cold_unseen_task:
    def load_actors(self):
        self.target = "official"

    def check_success(self):
        return self.target == "official"
""".lstrip(),
        encoding="utf-8",
    )
    docs = root / "description/cold_unseen_task.md"
    docs.parent.mkdir(parents=True)
    docs.write_text(
        "The target string is the observable scene state.\n",
        encoding="utf-8",
    )
    asset = root / "assets/cold_target.asset"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"cold-target")
    readme = root / "mea/taskgen/README.Agent.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        "Generate bounded RoboTwin task methods.\n",
        encoding="utf-8",
    )


def _write_discoverable_task_repo(root: Path, task_name: str) -> None:
    source = root / "envs" / f"{task_name}.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        f"""
class {task_name}:
    def load_actors(self):
        self.target = create_actor(modelname="900_novel_target")

    def check_success(self):
        return self.target.is_ready()
""".lstrip(),
        encoding="utf-8",
    )
    schema = root / "mea/toolkit/schemas" / f"{task_name}.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": task_name,
                "task_family": "runtime_discovered",
                "trusted_tool_profile": "generic_success",
                "physics_timestep_seconds": 0.004,
                "action_dimension": 14,
                "probe_task_attributes": ["target"],
                "tracked_actors": [
                    {
                        "id": "target",
                        "task_attribute": "target",
                        "scene_name": "900_novel_target",
                        "functional_points": [],
                        "contact_points": [],
                    }
                ],
                "contact_focus_actor_ids": ["target"],
                "semantic_fields": [
                    {
                        "name": "target_position",
                        "source": "actor_position",
                        "actor_id": "target",
                    }
                ],
                "semantic_roles": {
                    "manipulated_object_position": "target_position"
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    instruction = (
        root / "description/task_instruction" / f"{task_name}.json"
    )
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_text(
        json.dumps({"full_description": "operate the novel target"}) + "\n",
        encoding="utf-8",
    )
    asset = (
        root
        / "description/objects_description/900_novel_target/base0.json"
    )
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(
        json.dumps({"description": "novel target asset"}) + "\n",
        encoding="utf-8",
    )
    readme = root / "mea/taskgen/README.Agent.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "Generate bounded RoboTwin task methods.\n",
        encoding="utf-8",
    )


def _candidate() -> dict[str, Any]:
    return build_experiment_candidate(
        candidate_id="dynamic.cold_unseen_task.contact_stability",
        source_query="Does this policy remain stable before target contact?",
        base_task="cold_unseen_task",
        semantic_concern="contact stability under a shifted target",
        scene_need="Move the existing target while preserving its identity.",
        checker_need="Require target completion without pre-contact failure.",
        tool_need="Measure pre-contact motion stability.",
    )


def _validate_methods(
    methods: Mapping[str, str],
    _candidate_value: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = {
        name: validate_method_ast(
            source,
            name,
            safe_direct_calls=set(),
            safe_module_calls=set(),
            safe_method_calls=set(),
            allowed_private_attributes=set(),
            error_type=GenericTaskGenError,
        )
        for name, source in methods.items()
    }
    namespace: dict[str, Any] = {}
    exec(compile(methods["load_actors"], "<scene-fixture>", "exec"), namespace)
    exec(compile(methods["check_success"], "<checker-fixture>", "exec"), namespace)
    task = SimpleNamespace()
    namespace["load_actors"](task)
    positive = bool(namespace["check_success"](task))
    task.target = ""
    negative = bool(namespace["check_success"](task))
    fixtures = [
        {
            "fixture": "generated_scene_succeeds",
            "observed": positive,
            "expected": True,
            "passed": positive is True,
        },
        {
            "fixture": "missing_target_fails",
            "observed": negative,
            "expected": False,
            "passed": negative is False,
        },
    ]
    return {
        "valid": all(item["passed"] for item in fixtures),
        "policy": "cold_task_shared_safe_ast_fixture_v1",
        "scene_ast_nodes": sum(
            1 for _ in ast.walk(parsed["load_actors"])
        ),
        "success_ast_nodes": sum(
            1 for _ in ast.walk(parsed["check_success"])
        ),
        "checker_fixtures": fixtures,
    }


def _build_module(
    methods: Mapping[str, str],
    _candidate_value: Mapping[str, Any],
) -> str:
    scene = "\n".join(
        "    " + line for line in methods["load_actors"].splitlines()
    )
    checker = "\n".join(
        "    " + line for line in methods["check_success"].splitlines()
    )
    return (
        "class cold_unseen_task:\n"
        + scene
        + "\n\n"
        + checker
        + "\n"
    )


def _preflight(
    attempt_dir: Path,
    module_source: str,
    _candidate_value: Mapping[str, Any],
) -> dict[str, Any]:
    render_passed = "render_bad" not in module_source
    (attempt_dir / "render_probe.json").write_text(
        json.dumps({"passed": render_passed}) + "\n",
        encoding="utf-8",
    )
    return {
        "render_passed": render_passed,
        "expert_passed": render_passed,
        "scene_change_passed": render_passed,
        "simulator_probes": 1,
        "expert_probes": 1,
    }


def _adapter() -> GenericRoboTwinTaskAdapter:
    return GenericRoboTwinTaskAdapter(
        task_name="cold_unseen_task",
        official_source="envs/cold_unseen_task.py",
        official_class="cold_unseen_task",
        task_schema={
            "tracked_actors": ["target"],
            "signals": ["target_position", "success"],
        },
        documentation_paths=("description/cold_unseen_task.md",),
        asset_paths=("assets/cold_target.asset",),
        hooks=GenericTaskGenHooks(
            validate_methods=_validate_methods,
            build_module=_build_module,
            preflight_candidate=_preflight,
            resolve_metric=lambda candidate: candidate["tool_need"],
            resolve_checker_contract=lambda candidate: {
                "semantic_concern": candidate["semantic_concern"],
                "experimental_success": True,
            },
            prompt_constraints=(
                "Use only self.target from the retrieved official task."
            ),
        ),
    )


class GenericTaskGenBackendTests(unittest.TestCase):
    def test_semantic_field_access_guide_exposes_exact_read_only_apis(
        self,
    ) -> None:
        guide = _semantic_field_access_guide(
            {
                "task_schema": {
                    "tracked_actors": [
                        {
                            "id": "roller",
                            "task_attribute": "roller",
                        }
                    ],
                    "semantic_fields": [
                        {
                            "name": "roller_left_contact_position",
                            "source": "actor_contact_position",
                            "actor_id": "roller",
                            "point_id": 0,
                        },
                        {
                            "name": "left_tcp_position",
                            "source": "robot_tcp_position",
                            "side": "left",
                        },
                    ],
                }
            }
        )

        self.assertIn(
            'self.roller.get_contact_point(0, "pose").p',
            guide,
        )
        self.assertIn("self.robot.get_left_tcp_pose()[:3]", guide)
        self.assertIn("Do not invent", guide)
        self.assertIn("Semantic field names describe evidence", guide)
        self.assertIn(
            "get_contact_point",
            _GENERIC_READ_ONLY_METHOD_CALLS,
        )
        self.assertIn(
            "get_left_tcp_pose",
            _GENERIC_READ_ONLY_METHOD_CALLS,
        )

    def _cold_safe_ast_allows_conventional_discard_loop_target(self) -> None:
        tree = validate_method_ast(
            "def load_actors(self):\n"
            "    for _ in []:\n"
            "        pass\n",
            "load_actors",
            safe_direct_calls=set(),
            safe_module_calls=set(),
            safe_method_calls=set(),
            allowed_private_attributes=set(),
            error_type=GenericTaskGenError,
        )

        self.assertIsInstance(tree, ast.Module)

    def test_infeasible_uniform_scale_preservation_fails_before_lookup(
        self,
    ) -> None:
        query = "Does uniform target scaling expose a policy weakness?"
        requested_change = (
            "Uniformly reduce the target size by 20% while preserving "
            "the contact-point world position and object center position."
        )
        intent = build_evaluation_intent(
            source_query=query,
            original_concern="uniform target scale robustness",
            hypothesis="Uniform target scaling may reduce task success.",
            requested_change=requested_change,
            preserved_conditions=(
                "contact-point world position",
                "object center position",
            ),
            required_observation="Observe task success after uniform scaling.",
        )
        candidate = build_experiment_candidate(
            source_query=query,
            base_task="cold_unseen_task",
            semantic_concern=(
                "Uniform target scaling may reduce task success."
            ),
            scene_need=requested_change,
            rule_tool_need="Observe task success after uniform scaling.",
            evaluation_intent=intent,
        )
        provider = _Provider([])
        lookup_calls = 0

        def find_exact(_query: Mapping[str, Any]) -> Mapping[str, Any]:
            nonlocal lookup_calls
            lookup_calls += 1
            raise AssertionError("infeasible candidate must not be looked up")

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                GenericTaskGenError,
                r"origin-centered uniform scale backend cannot guarantee "
                r"both.*Planner must revise",
            ):
                GenericRoboTwinTaskGenBackend(
                    Path(temp_dir),
                    provider,
                    model="fixture-model",
                    find_exact=find_exact,
                ).materialize(
                    candidate,
                    _adapter(),
                    run_id="run_infeasible_uniform_scale",
                )

        self.assertEqual(lookup_calls, 0)
        self.assertEqual(provider.calls, 0)

    def test_tool_only_candidate_bypasses_generic_taskgen(self) -> None:
        candidate = build_experiment_candidate(
            source_query="Measure pre-contact jerk.",
            base_task="cold_unseen_task",
            semantic_concern="motion.precontact_jerk",
            rule_tool_need="Measure peak jerk before first contact.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                GenericTaskGenError, "must bypass TaskGen"
            ):
                GenericRoboTwinTaskGenBackend(
                    Path(temp_dir),
                    _Provider([]),
                    model="fixture-model",
                ).materialize(
                    candidate,
                    _adapter(),
                    run_id="run_tool_only_must_bypass",
                )

    def test_runtime_only_change_fails_before_provider_generation(self) -> None:
        query = "Does lower gripper precision expose a weakness?"
        intent = build_evaluation_intent(
            source_query=query,
            original_concern="task_execution.gripper_precision",
            hypothesis="Reduced gripper precision may miss the target.",
            requested_change="Reduce the precision of the gripper.",
            preserved_conditions=("task identity", "policy checkpoint"),
            required_observation="Observe which object is lifted.",
        )
        candidate = build_experiment_candidate(
            source_query=query,
            base_task="cold_unseen_task",
            semantic_concern="task_execution.gripper_precision",
            scene_need="Reduce the precision of the gripper.",
            checker_need="Require only the target to be lifted.",
            rule_tool_need="Measure target and non-target lift.",
            evaluation_intent=intent,
        )
        provider = _Provider([])

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                GenericTaskGenError,
                "TaskGen cannot implement the requested runtime intervention",
            ):
                GenericRoboTwinTaskGenBackend(
                    Path(temp_dir),
                    provider,
                    model="fixture-model",
                ).materialize(
                    candidate,
                    _adapter(),
                    run_id="run_runtime_change_must_fail",
                )

        self.assertEqual(provider.calls, 0)

    def test_preservation_report_separates_available_authorities(self) -> None:
        report = build_preservation_report(
            [
                "task identity and policy checkpoint",
                "target color",
                "official success semantics",
                "target mass",
            ],
            scene_generated=True,
            checker_generated=True,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
        )

        checks = {
            item["condition"]: item for item in report["checks"]
        }
        self.assertTrue(
            checks["task identity and policy checkpoint"]["verified"]
        )
        self.assertEqual(
            checks["task identity and policy checkpoint"]["kind"],
            "frozen_runtime_binding",
        )
        self.assertTrue(checks["target color"]["verified"])
        self.assertEqual(checks["target color"]["kind"], "visual")
        self.assertIsNone(
            checks["official success semantics"]["verified"]
        )
        self.assertIsNone(checks["target mass"]["verified"])
        self.assertIsNone(report["verified"])
        self.assertEqual(report["status"], "partially_unverified")

    def test_preservation_report_marks_checked_visual_drift_failed(self) -> None:
        report = build_preservation_report(
            ["target color and policy checkpoint"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={
                "passed": False,
                "unexpected_changes": ["target color changed"],
            },
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["checks"][0]["kind"],
            "visual+frozen_runtime_binding",
        )

    def test_exact_center_uses_same_seed_simulator_state(self) -> None:
        report = build_preservation_report(
            ["exact target center position"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 7,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.1, -0.2, 0.3],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {},
                    }
                ],
            },
            generated_setup={
                "seed": 7,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.1, -0.2, 0.3],
                        "quaternion": [0.0, 1.0, 0.0, 0.0],
                        "contact_points": {},
                    },
                    {
                        "id": "new_distractor",
                        "position": [0.2, -0.2, 0.3],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {},
                    },
                ],
            },
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_simulator_state:tracked_actors.position",
        )

    def test_contact_point_mismatch_fails_simulator_state_check(self) -> None:
        report = build_preservation_report(
            ["target contact point world position"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 11,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.0, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {
                            "0": {
                                "position": [0.0, 0.0, 0.1],
                                "raw": [0.0, 0.0, 0.1],
                            }
                        },
                    }
                ],
            },
            generated_setup={
                "seed": 11,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.0, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {
                            "0": {
                                "position": [0.0, 0.0, 0.12],
                                "raw": [0.0, 0.0, 0.12],
                            }
                        },
                    }
                ],
            },
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_simulator_state:"
            "tracked_actors.contact_point_world_positions",
        )

    def test_batch39_axis_contact_and_model_preservation_passes(self) -> None:
        official_actor = {
            "id": "roller",
            "position": [-0.14362009, -0.06900436, 0.74150085],
            "quaternion": [0.9999999, 0.0, 0.0, 0.0002],
            "contact_points": {
                "0": {"position": [-0.2171124, -0.1161990, 0.7585314]},
                "1": {"position": [-0.0699741, -0.0218261, 0.7585396]},
            },
            "collision_geometry": [
                {
                    "geometry_type": "create_actor_asset",
                    "modelname": "066_roller",
                    "model_id": 0,
                    "collision_asset": "base0.glb",
                    "scale": [1.0, 1.0, 1.0],
                }
            ],
        }
        generated_actor = {
            "id": "roller",
            "position": [0.15001997, -0.06900253, 0.74150062],
            "quaternion": [0.9999998, 0.0001, 0.0, 0.0001],
            "contact_points": {
                "0": {"position": [0.0765210, -0.1161838, 0.7585394]},
                "1": {"position": [0.2236660, -0.0218243, 0.7585378]},
            },
            "collision_geometry": [
                {
                    "geometry_type": "create_actor_asset",
                    "modelname": "066_roller",
                    "model_id": 0,
                    "collision_asset": "base0.glb",
                    "scale": [1.0, 1.0, 1.0],
                }
            ],
        }
        official_setup = {
            "seed": 1000,
            "tracked_actors": [official_actor],
        }
        generated_setup = {
            "seed": 1000,
            "tracked_actors": [generated_actor],
        }
        report = build_preservation_report(
            [
                "roller model identity, sampled y position, and orientation",
                "declared roller contact points 0 and 1",
            ],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup=official_setup,
            generated_setup=generated_setup,
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")

        v4_wording = build_preservation_report(
            [
                "roller model instance: 0",
                "sampled roller y coordinate",
                "roller orientation",
                "official roller actor and declared contact references",
                "official lift goal and official check_success predicate",
            ],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup=official_setup,
            generated_setup=generated_setup,
        )

        self.assertTrue(v4_wording["verified"])
        self.assertEqual(v4_wording["status"], "verified")
        round_2_wording = build_preservation_report(
            [
                "official grab_roller task identity",
                "SmolVLA shared_official policy checkpoint",
                "roller declared contact reference point 0",
                "roller declared contact reference point 1",
                "sampled roller y coordinate",
                "sampled roller model instance",
                "sampled roller orientation",
                "official lift success predicate",
            ],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup=official_setup,
            generated_setup=generated_setup,
        )

        self.assertTrue(round_2_wording["verified"])
        self.assertEqual(round_2_wording["status"], "verified")
        self.assertIn("model_identity", report["checks"][0]["authority"])
        self.assertIn("position_y", report["checks"][0]["authority"])
        self.assertIn("quaternion", report["checks"][0]["authority"])
        self.assertIn(
            "contact_point_references",
            report["checks"][1]["authority"],
        )

    def test_explicit_y_preservation_rejects_y_change_only(self) -> None:
        common = {
            "id": "roller",
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "contact_points": {},
        }
        report = build_preservation_report(
            ["sampled y position"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 1000,
                "tracked_actors": [
                    {**common, "position": [-0.14, -0.069, 0.74]}
                ],
            },
            generated_setup={
                "seed": 1000,
                "tracked_actors": [
                    {**common, "position": [0.15, -0.05, 0.74]}
                ],
            },
        )

        self.assertFalse(report["verified"])
        self.assertIn("position_y", report["checks"][0]["authority"])

    def test_contact_reference_and_model_identity_changes_fail(self) -> None:
        base = {
            "id": "roller",
            "position": [0.0, 0.0, 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "contact_points": {"0": {"position": [0.1, 0.0, 0.0]}},
            "collision_geometry": [
                {"modelname": "066_roller", "model_id": 0}
            ],
        }
        changed = {
            **base,
            "contact_points": {},
            "collision_geometry": [
                {"modelname": "066_roller", "model_id": 1}
            ],
        }
        report = build_preservation_report(
            ["roller contact points", "roller model identity"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={"seed": 1000, "tracked_actors": [base]},
            generated_setup={"seed": 1000, "tracked_actors": [changed]},
        )

        self.assertFalse(report["verified"])
        self.assertFalse(report["checks"][0]["verified"])
        self.assertFalse(report["checks"][1]["verified"])

    def test_compound_contact_and_center_condition_checks_both(self) -> None:
        report = build_preservation_report(
            ["contact point and object center position"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 13,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.0, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {
                            "0": {"position": [0.0, 0.0, 0.1]}
                        },
                    }
                ],
            },
            generated_setup={
                "seed": 13,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.01, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {
                            "0": {"position": [0.0, 0.0, 0.1]}
                        },
                    }
                ],
            },
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_simulator_state:"
            "tracked_actors.contact_point_references+position",
        )

    def _cold_compound_position_and_orientation_checks_both(self) -> None:
        report = build_preservation_report(
            ["target position and orientation"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 17,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.0, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {},
                    }
                ],
            },
            generated_setup={
                "seed": 17,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.01, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {},
                    }
                ],
            },
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_simulator_state:"
            "tracked_actors.position+quaternion",
        )

    def test_same_seed_pose_probe_jitter_is_not_a_semantic_change(self) -> None:
        report = build_preservation_report(
            ["target position and orientation"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 18,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.1, -0.2, 0.3],
                        "quaternion": [0.0, 0.0, 0.7071068, 0.7071068],
                        "contact_points": {},
                    }
                ],
            },
            generated_setup={
                "seed": 18,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.1000005, -0.2000004, 0.3000001],
                        "quaternion": [
                            0.0001,
                            -0.0001,
                            0.7072068,
                            0.7070068,
                        ],
                        "contact_points": {},
                    }
                ],
            },
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")

    def test_contact_target_position_and_orientation_checks_all(self) -> None:
        report = build_preservation_report(
            ["contact point, target position, and orientation"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 19,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.0, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {
                            "0": {"position": [0.0, 0.0, 0.1]}
                        },
                    }
                ],
            },
            generated_setup={
                "seed": 19,
                "tracked_actors": [
                    {
                        "id": "target",
                        "position": [0.01, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "contact_points": {
                            "0": {"position": [0.0, 0.0, 0.1]}
                        },
                    }
                ],
            },
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_simulator_state:"
            "tracked_actors.contact_point_references+position+quaternion",
        )

    def test_geometry_without_simulator_or_ast_authority_is_partial(
        self,
    ) -> None:
        report = build_preservation_report(
            ["target geometry and shape"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={"seed": 3, "tracked_actors": []},
            generated_setup={"seed": 3, "tracked_actors": []},
        )

        self.assertIsNone(report["verified"])
        self.assertEqual(report["status"], "partially_unverified")
        self.assertEqual(report["checks"][0]["kind"], "geometry")
        self.assertEqual(
            report["checks"][0]["authority"],
            "no_comparable_simulator_collision_geometry",
        )

    def test_same_seed_collision_geometry_verifies_shape_preservation(
        self,
    ) -> None:
        geometry = [
            {
                "geometry_type": "BoxGeometry",
                "half_lengths": [0.03, 0.03, 0.02],
                "local_pose": {
                    "position": [0.0, 0.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        ]
        common_actor = {
            "id": "target",
            "position": [0.1, -0.2, 0.3],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "contact_points": {},
            "collision_geometry": geometry,
        }
        report = build_preservation_report(
            ["target shape and size"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={"seed": 23, "tracked_actors": [common_actor]},
            generated_setup={
                "seed": 23,
                "tracked_actors": [
                    dict(common_actor),
                    {
                        "id": "new_distractor",
                        "collision_geometry": [
                            {
                                "geometry_type": "BoxGeometry",
                                "half_lengths": [0.02, 0.02, 0.02],
                            }
                        ],
                    },
                ],
            },
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_simulator_state:"
            "tracked_actors.collision_geometry",
        )

    def test_changed_collision_geometry_fails_shape_preservation(
        self,
    ) -> None:
        official_actor = {
            "id": "target",
            "collision_geometry": [
                {
                    "geometry_type": "BoxGeometry",
                    "half_lengths": [0.03, 0.03, 0.02],
                }
            ],
        }
        generated_actor = {
            "id": "target",
            "collision_geometry": [
                {
                    "geometry_type": "BoxGeometry",
                    "half_lengths": [0.04, 0.03, 0.02],
                }
            ],
        }
        report = build_preservation_report(
            ["target geometry"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 29,
                "tracked_actors": [official_actor],
            },
            generated_setup={
                "seed": 29,
                "tracked_actors": [generated_actor],
            },
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "failed")

    def _cold_chinese_goal_and_contact_geometry_preservation_is_typed(
        self,
    ) -> None:
        actor = {
            "id": "bell",
            "collision_geometry": [
                {
                    "geometry_type": "create_actor_asset",
                    "modelname": "050_bell",
                    "model_id": 0,
                    "convex": True,
                    "is_static": True,
                    "scale": [0.05, 0.05, 0.05],
                }
            ],
        }
        report = build_preservation_report(
            ["任务目标与接触几何语义"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={"seed": 31, "tracked_actors": [actor]},
            generated_setup={"seed": 31, "tracked_actors": [dict(actor)]},
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(
            report["checks"][0]["kind"],
            "checker_semantics+geometry",
        )

    def _cold_height_preservation_ignores_requested_xy_offset(self) -> None:
        common = {
            "id": "bell",
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "contact_points": {},
            "collision_geometry": [],
        }
        report = build_preservation_report(
            ["height"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 37,
                "tracked_actors": [
                    {**common, "position": [0.0, 0.0, 0.76]}
                ],
            },
            generated_setup={
                "seed": 37,
                "tracked_actors": [
                    {**common, "position": [0.1, 0.0, 0.76]}
                ],
            },
        )

        self.assertTrue(report["verified"])
        self.assertIn(
            "position_z",
            report["checks"][0]["authority"],
        )

    def test_vertical_axis_position_preservation_compares_only_z(self) -> None:
        common = {
            "id": "bell",
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "contact_points": {},
            "collision_geometry": [],
        }
        report = build_preservation_report(
            ["center position along the vertical axis"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
            official_setup={
                "seed": 41,
                "tracked_actors": [
                    {**common, "position": [0.0, 0.0, 0.76]}
                ],
            },
            generated_setup={
                "seed": 41,
                "tracked_actors": [
                    {**common, "position": [0.05, 0.0, 0.76]}
                ],
            },
        )

        self.assertTrue(report["verified"])
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_simulator_state:tracked_actors.position_z",
        )

    def _cold_legacy_visual_color_preservation_still_passes(self) -> None:
        report = build_preservation_report(
            ["target color"],
            scene_generated=True,
            checker_generated=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["checks"][0]["kind"], "visual")
        self.assertEqual(
            report["checks"][0]["authority"],
            "same_seed_visual_diagnosis",
        )

    def test_exact_official_task_methods_verify_preservation(self) -> None:
        report = build_preservation_report(
            ["target mass", "goal semantics"],
            scene_generated=False,
            checker_generated=False,
            visual_self_check_enabled=False,
            visual={},
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")
        self.assertTrue(
            all(
                item["kind"] == "exact_task_method_reuse"
                for item in report["checks"]
            )
        )

    def test_generated_checker_can_preserve_official_core_conjunct(self) -> None:
        report = build_preservation_report(
            ["official core predicate as a required conjunct"],
            scene_generated=True,
            checker_generated=True,
            checker_references_official_core=True,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(
            report["checks"][0],
            {
                "condition": (
                    "official core predicate as a required conjunct"
                ),
                "kind": "official_core_conjunct",
                "verified": True,
                "authority": (
                    "generated_checker_direct_official_core_reference"
                ),
            },
        )

    def test_generated_checker_cannot_claim_official_core_without_reference(
        self,
    ) -> None:
        report = build_preservation_report(
            ["official core predicate as a required conjunct"],
            scene_generated=True,
            checker_generated=True,
            checker_references_official_core=False,
            visual_self_check_enabled=True,
            visual={"passed": True, "unexpected_changes": []},
        )

        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "failed")

    def test_generation_failure_still_writes_child_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "run_generic_failure_fixture"
            attempt_dir = (
                root / "mea/generated_task_attempts" / run_id
            )
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "task_generation_attempt_summary.json").write_text(
                json.dumps({"status": "failed", "attempt_count": 2}) + "\n",
                encoding="utf-8",
            )

            manifest = record_generic_taskgen_generation_failure(
                root,
                run_id=run_id,
                user_request="Evaluate a shifted target.",
                experiment_candidate=_candidate(),
                model="fixture-model",
                telemetry_profile="balanced_v1",
                error=GenericTaskGenError("repair exhausted"),
            )

            run_dir = root / "mea/generated_tasks" / run_id
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue(
                (
                    run_dir
                    / "validation/task_generation_attempt_summary.json"
                ).is_file()
            )
            self.assertEqual(
                manifest["failure"]["stage"],
                "provider_scene_checker_generation",
            )
            self.assertEqual(
                manifest["proposal_path"],
                "generation/proposal.json",
            )
            self.assertEqual(
                manifest["proposal"]["candidate_id"],
                _candidate()["candidate_id"],
            )
            self.assertTrue(
                (run_dir / "generation/proposal.json").is_file()
            )
            self.assertFalse(
                (
                    run_dir
                    / "generation/experiment_candidate.json"
                ).exists()
            )

    def test_loader_discovers_unknown_task_without_a_core_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_name = "runtime_novel_task"
            _write_discoverable_task_repo(root, task_name)
            identity = discover_generic_robotwin_task_identity(
                root,
                task_name,
            )
            self.assertEqual(identity["task_name"], task_name)
            self.assertEqual(
                identity["official_source"],
                f"envs/{task_name}.py",
            )
            self.assertEqual(identity["task_schema"]["task_name"], task_name)

            def fixtures(
                methods: Mapping[str, str],
                _candidate_value: Mapping[str, Any],
            ) -> list[Mapping[str, Any]]:
                namespace: dict[str, Any] = {}
                exec(methods["check_success"], namespace)
                positive = SimpleNamespace(target=SimpleNamespace())
                positive.target.is_ready = lambda: True
                negative = SimpleNamespace(target=SimpleNamespace())
                negative.target.is_ready = lambda: False
                checker = namespace["check_success"]
                return [
                    {
                        "fixture": "ready",
                        "passed": checker(positive) is True,
                    },
                    {
                        "fixture": "not_ready",
                        "passed": checker(negative) is False,
                    },
                ]

            adapter = load_generic_robotwin_task_adapter(
                root,
                task_name,
                checker_fixtures=fixtures,
                preflight_candidate=lambda _path, _source, _candidate_value: {
                    "render_passed": True,
                    "expert_passed": True,
                    "scene_change_passed": True,
                },
                resolve_metric=lambda _candidate_value: (
                    "runtime_novel_completion"
                ),
                resolve_checker_contract=lambda candidate_value: {
                    "semantic_concern": candidate_value[
                        "semantic_concern"
                    ]
                },
            )

            self.assertEqual(adapter.task_name, task_name)
            self.assertEqual(
                adapter.official_source,
                f"envs/{task_name}.py",
            )
            self.assertNotIn(
                f"description/task_instruction/{task_name}.json",
                adapter.documentation_paths,
            )
            self.assertIn(
                (
                    "description/objects_description/"
                    "900_novel_target/base0.json"
                ),
                adapter.asset_paths,
            )
            self.assertNotIn(
                "template",
                json.dumps(
                    {
                        "task": adapter.task_name,
                        "docs": adapter.documentation_paths,
                        "assets": adapter.asset_paths,
                    }
                ),
            )

            candidate = build_experiment_candidate(
                source_query="Can it operate an unseen shifted target?",
                base_task=task_name,
                semantic_concern="target shift response",
                scene_need="Move the target to a new valid pose.",
                checker_need="Require the generated target to be ready.",
                tool_need="Measure target completion.",
            )
            provider = _Provider(
                [
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            "    self.target = create_actor("
                            'modelname="901_novel_target")\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            "    return self.target.is_ready() and "
                            "self.target is not None\n"
                        ),
                    }
                ]
            )
            result = GenericRoboTwinTaskGenBackend(
                root, provider, model="fixture-model"
            ).materialize(
                candidate,
                adapter,
                run_id="run_runtime_novel_task",
            )
            self.assertEqual(result["status"], "generated")
            module = (
                Path(result["run_dir"]) / "task.py"
            ).read_text(encoding="utf-8")
            self.assertIn(
                f"from envs.{task_name} import *",
                module,
            )
            self.assertIn(
                f"class {task_name}("
                f"_official_task_module.{task_name}):",
                module,
            )

    def test_official_only_tasks_share_the_generic_cold_start_contract(
        self,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        def fixtures(
            _methods: Mapping[str, str],
            _candidate_value: Mapping[str, Any],
        ) -> list[Mapping[str, Any]]:
            return [{"fixture": "live_hook_owned", "passed": True}]

        for task_name in (
            "adjust_bottle",
            "grab_roller",
            "place_phone_stand",
        ):
            with self.subTest(task_name=task_name):
                adapter = load_generic_robotwin_task_adapter(
                    repo_root,
                    task_name,
                    checker_fixtures=fixtures,
                    preflight_candidate=(
                        lambda _path, _source, _candidate_value: {
                            "render_passed": True,
                            "expert_passed": True,
                            "scene_change_passed": True,
                        }
                    ),
                    resolve_metric=lambda _candidate_value: (
                        "query_derived_metric"
                    ),
                    resolve_checker_contract=lambda candidate_value: {
                        "semantic_concern": candidate_value[
                            "semantic_concern"
                        ]
                    },
                )
                candidate = build_experiment_candidate(
                    source_query=(
                        "Where does object-pose robustness first fail?"
                    ),
                    base_task=task_name,
                    semantic_concern=(
                        "cold-start object-pose robustness"
                    ),
                    scene_need=(
                        "Adapt one existing object pose inside the official "
                        "workspace."
                    ),
                    checker_need=(
                        "Check completion under only that adapted scene."
                    ),
                    tool_need=(
                        "Measure target-relative clearance and completion."
                    ),
                )
                semantic_key = generic_task_semantic_key(
                    candidate,
                    adapter,
                    repo_root=repo_root,
                )

                self.assertEqual(adapter.task_name, task_name)
                self.assertEqual(
                    adapter.official_source,
                    f"envs/{task_name}.py",
                )
                self.assertEqual(
                    adapter.task_schema["task_name"],
                    task_name,
                )
                # Language-only task_instruction JSON is not implementation
                # authority. A short task guide is optional and is added only
                # after a source-backed Agent/runtime failure.
                if task_name == "grab_roller":
                    self.assertIn(
                        "mea/knowledge/tasks/grab_roller.md",
                        adapter.documentation_paths,
                    )
                else:
                    self.assertEqual(adapter.documentation_paths, ())
                self.assertTrue(adapter.asset_paths)
                self.assertEqual(
                    semantic_key["base_task"],
                    task_name,
                )
                self.assertIsNotNone(semantic_key["scene_need"])
                self.assertIsNotNone(semantic_key["checker_need"])
                self.assertIsNotNone(candidate["tool_need"])
                serialized = json.dumps(
                    semantic_key,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                self.assertNotIn("template_id", serialized)
                self.assertNotIn("aspect_id", serialized)

    def test_loader_requires_real_preflight_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_discoverable_task_repo(root, "runtime_novel_task")
            with self.assertRaisesRegex(
                GenericTaskGenError, "preflight_candidate"
            ):
                load_generic_robotwin_task_adapter(
                    root,
                    "runtime_novel_task",
                    checker_fixtures=lambda _methods, _candidate: [],
                    preflight_candidate=None,  # type: ignore[arg-type]
                    resolve_metric=lambda _candidate: "metric",
                    resolve_checker_contract=lambda _candidate: {},
                )

    def test_common_ast_and_subclass_builder_are_data_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_name = "runtime_novel_task"
            _write_discoverable_task_repo(root, task_name)
            methods = {
                "load_actors": (
                    "def load_actors(self):\n"
                    "    self.target = create_actor("
                    'modelname="901_novel_target")\n'
                ),
                "check_success": (
                    "def check_success(self):\n"
                    "    return self.target.is_ready() and "
                    "self.target is not None\n"
                ),
            }
            report = validate_generic_task_methods(
                methods,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
            )
            self.assertTrue(report["valid"])
            self.assertTrue(
                report["policy"].startswith(
                    "generic_official_api_ast_v1:"
                )
            )
            module = build_generic_task_subclass_module(
                methods,
                official_module=f"envs.{task_name}",
                official_class=task_name,
            )
            compile(module, "<runtime-novel-task>", "exec")
            self.assertIn("def mea_official_check_success(self):", module)
            self.assertIn(
                (
                    f"return _official_task_module.{task_name}."
                    "check_success(self)"
                ),
                module,
            )
            scene_only = build_generic_task_subclass_module(
                methods,
                official_module=f"envs.{task_name}",
                official_class=task_name,
                emit_overrides={
                    "load_actors": True,
                    "check_success": False,
                },
            )
            scene_class = next(
                node
                for node in ast.parse(scene_only).body
                if isinstance(node, ast.ClassDef)
            )
            self.assertEqual(
                {
                    node.name
                    for node in scene_class.body
                    if isinstance(node, ast.FunctionDef)
                },
                {"load_actors", "mea_official_check_success"},
            )
            checker_only = build_generic_task_subclass_module(
                methods,
                official_module=f"envs.{task_name}",
                official_class=task_name,
                emit_overrides={
                    "load_actors": False,
                    "check_success": True,
                },
            )
            checker_class = next(
                node
                for node in ast.parse(checker_only).body
                if isinstance(node, ast.ClassDef)
            )
            self.assertEqual(
                {
                    node.name
                    for node in checker_class.body
                    if isinstance(node, ast.FunctionDef)
                },
                {"check_success", "mea_official_check_success"},
            )
            with self.assertRaisesRegex(
                GenericTaskGenError,
                "emit_overrides",
            ):
                build_generic_task_subclass_module(
                    methods,
                    official_module=f"envs.{task_name}",
                    official_class=task_name,
                    emit_overrides={"load_actors": True},
                )
            with self.assertRaisesRegex(
                GenericTaskGenError, "forbidden AST node Import"
            ):
                validate_generic_task_methods(
                    {
                        **methods,
                        "check_success": (
                            "def check_success(self):\n"
                            "    import os\n"
                            "    return True\n"
                        ),
                    },
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                )
            numpy_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    return bool(np.asarray(["
                    "self.target.is_ready()])[0])\n"
                ),
            }
            report = validate_generic_task_methods(
                numpy_checker,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
            )
            self.assertTrue(report["valid"])
            contact_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    robot_links = self.robot.get_links()\n"
                    "    return not any(\n"
                    "        contact.actor0 in robot_links\n"
                    "        for contact in self.scene.get_contacts()\n"
                    "    )\n"
                ),
            }
            report = validate_generic_task_methods(
                contact_checker,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
            )
            self.assertTrue(report["valid"])
            conjunct_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    return self.mea_official_check_success() and "
                    "self.target is not None\n"
                ),
            }
            report = validate_generic_task_methods(
                conjunct_checker,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
                require_official_core_conjunct=True,
            )
            self.assertTrue(report["official_core_directly_called"])
            self.assertTrue(
                report["official_core_enforced_as_conjunct"]
            )
            aliased_conjunct_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    official_success = "
                    "self.mea_official_check_success()\n"
                    "    return official_success and "
                    "self.target is not None\n"
                ),
            }
            report = validate_generic_task_methods(
                aliased_conjunct_checker,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
                require_official_core_conjunct=True,
            )
            self.assertTrue(report["official_core_directly_called"])
            self.assertTrue(
                report["official_core_enforced_as_conjunct"]
            )
            bool_wrapped_alias_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    official_success = "
                    "self.mea_official_check_success()\n"
                    "    terminal_distance = 0.02\n"
                    "    return bool(official_success and "
                    "terminal_distance <= 0.03)\n"
                ),
            }
            report = validate_generic_task_methods(
                bool_wrapped_alias_checker,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
                require_official_core_conjunct=True,
            )
            self.assertTrue(
                report["official_core_enforced_as_conjunct"]
            )
            shadowed_bool_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    def bool(_value):\n"
                    "        return True\n"
                    "    official_success = "
                    "self.mea_official_check_success()\n"
                    "    return bool(official_success and "
                    "self.target is not None)\n"
                ),
            }
            with self.assertRaisesRegex(
                GenericTaskGenError,
                "required boolean conjunct",
            ):
                validate_generic_task_methods(
                    shadowed_bool_checker,
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                    require_official_core_conjunct=True,
                )
            overwritten_alias_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    official_success = "
                    "self.mea_official_check_success()\n"
                    "    official_success = True\n"
                    "    return official_success and "
                    "self.target is not None\n"
                ),
            }
            with self.assertRaisesRegex(
                GenericTaskGenError,
                "required boolean conjunct",
            ):
                validate_generic_task_methods(
                    overwritten_alias_checker,
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                    require_official_core_conjunct=True,
                )
            guarded_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    if not self.mea_official_check_success():\n"
                    "        return False\n"
                    "    return self.target is not None\n"
                ),
            }
            report = validate_generic_task_methods(
                guarded_checker,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
                require_official_core_conjunct=True,
            )
            self.assertTrue(
                report["official_core_enforced_as_conjunct"]
            )
            bypassable_guard_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    if not self.mea_official_check_success():\n"
                    "        if self.target is not None:\n"
                    "            return True\n"
                    "        return False\n"
                    "    return True\n"
                ),
            }
            with self.assertRaisesRegex(
                GenericTaskGenError,
                "required boolean conjunct",
            ):
                validate_generic_task_methods(
                    bypassable_guard_checker,
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                    require_official_core_conjunct=True,
                )
            discarded_call_checker = {
                **methods,
                "check_success": (
                    "def check_success(self):\n"
                    "    self.mea_official_check_success()\n"
                    "    return self.target is not None\n"
                ),
            }
            with self.assertRaisesRegex(
                GenericTaskGenError,
                "required boolean conjunct",
            ):
                validate_generic_task_methods(
                    discarded_call_checker,
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                    require_official_core_conjunct=True,
                )
            with self.assertRaisesRegex(
                GenericTaskGenError,
                "required boolean conjunct",
            ):
                validate_generic_task_methods(
                    methods,
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                    require_official_core_conjunct=True,
                )

    def test_canonical_official_core_phrase_requires_conjunct(self) -> None:
        self.assertTrue(
            _candidate_requires_official_core_conjunct(
                {
                    "checker_need": {
                        "kind": "generate",
                        "description": "Check one extra contact condition.",
                        "reuse_first": True,
                    },
                    "evaluation_intent": {
                        "preserved_conditions": [
                            "official core predicate as a required conjunct"
                        ]
                    },
                }
            )
        )

    def _cold_pose_property_item_assignment_is_rejected(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        methods = {
            "load_actors": (
                "def load_actors(self):\n"
                "    rand_pos = rand_pose(\n"
                "        xlim=[-0.25, 0.25],\n"
                "        ylim=[-0.2, 0.0],\n"
                "        qpos=[0.5, 0.5, 0.5, 0.5],\n"
                "    )\n"
                "    rand_pos.p[0] += 0.08\n"
                "    self.bell = create_actor(\n"
                "        scene=self,\n"
                "        pose=rand_pos,\n"
                '        modelname="050_bell",\n'
                "        convex=True,\n"
                "        model_id=0,\n"
                "        is_static=True,\n"
                "    )\n"
            ),
            "check_success": (
                "def check_success(self):\n"
                "    return bool(self.stage_success_tag)\n"
            ),
        }
        with self.assertRaisesRegex(
            GenericTaskGenError,
            r"mutates Pose\.p.*construct a new sapien\.Pose",
        ):
            validate_generic_task_methods(
                methods,
                official_source=repo_root / "envs/click_bell.py",
                official_class="click_bell",
            )

    def test_literal_scale_multiplier_matches_requested_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_name = "runtime_novel_task"
            _write_discoverable_task_repo(root, task_name)
            methods = {
                "load_actors": (
                    "def load_actors(self):\n"
                    "    self.target = create_actor(\n"
                    '        modelname="900_novel_target",\n'
                    "        scale_multiplier=0.5,\n"
                    "    )\n"
                ),
                "check_success": (
                    "def check_success(self):\n"
                    "    return self.target.is_ready() and "
                    "self.target is not None\n"
                ),
            }
            with self.assertRaisesRegex(
                GenericTaskGenError,
                r"expected 1\.5.*observed \[0\.5\]",
            ):
                validate_generic_task_methods(
                    methods,
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                    scene_need="Increase the target diameter by 50%.",
                )
            report = validate_generic_task_methods(
                methods,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
                scene_need={
                    "description": "Reduce the target size to 50%."
                },
            )
            self.assertTrue(report["valid"])

    def _cold_scale_gate_defers_nonliteral_or_irrelevant_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_name = "runtime_novel_task"
            _write_discoverable_task_repo(root, task_name)
            checker = (
                "def check_success(self):\n"
                "    return self.target.is_ready() and "
                "self.target is not None\n"
            )
            for scene_need, scale_expression in (
                ("Move the target laterally.", "0.5"),
                ("Increase the target size by 50%.", "self.scale_factor"),
            ):
                with self.subTest(scene_need=scene_need):
                    report = validate_generic_task_methods(
                        {
                            "load_actors": (
                                "def load_actors(self):\n"
                                "    self.target = create_actor(\n"
                                '        modelname="900_novel_target",\n'
                                "        scale_multiplier="
                                f"{scale_expression},\n"
                                "    )\n"
                            ),
                            "check_success": checker,
                        },
                        official_source=root / f"envs/{task_name}.py",
                        official_class=task_name,
                        scene_need=scene_need,
                    )
                    self.assertTrue(report["valid"])

    def test_semantic_reuse_ignores_query_wording_and_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            first = _candidate()
            second = {
                **first,
                "candidate_id": "dynamic.cold_unseen_task.rephrased",
                "source_query": (
                    "Rephrased: is motion stable before contact?"
                ),
                "tool_need": "Measure the same scene with a different metric.",
            }
            self.assertEqual(
                generic_task_semantic_key(
                    first, _adapter(), repo_root=root
                ),
                generic_task_semantic_key(
                    second, _adapter(), repo_root=root
                ),
            )

    def test_exact_semantic_reuse_skips_provider_and_has_no_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = _Provider([])
            observed_queries: list[dict[str, Any]] = []

            def find_exact(query: Mapping[str, Any]) -> Mapping[str, Any]:
                observed_queries.append(dict(query))
                return {
                    "schema_version": 1,
                    "status": "approved",
                    "artifact_id": "task_artifact_cold_exact",
                    "semantic_key": query["semantic_key"],
                    "semantic_key_sha256": query["semantic_key_sha256"],
                }

            backend = GenericRoboTwinTaskGenBackend(
                root,
                provider,
                model="fixture-model",
                find_exact=find_exact,
            )
            result = backend.materialize(
                _candidate(), _adapter(), run_id="run_cold_exact"
            )

            self.assertEqual(result["status"], "reused")
            self.assertEqual(result["provider_call_count"], 0)
            self.assertEqual(provider.calls, 0)
            self.assertEqual(len(observed_queries), 1)
            serialized = json.dumps(observed_queries[0], sort_keys=True)
            self.assertNotIn("template", serialized)
            self.assertNotIn("aspect_id", serialized)
            self.assertFalse(
                (root / "mea/generated_tasks/run_cold_exact").exists()
            )

    def _cold_ablation_condition_never_reuses_complete_task_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = _Provider(
                [
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "generated"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target == "generated"\n'
                        ),
                    }
                ]
            )
            lookup_calls = 0

            def find_exact(_query: Mapping[str, Any]) -> Mapping[str, Any]:
                nonlocal lookup_calls
                lookup_calls += 1
                raise AssertionError("ablation must not reuse an artifact")

            result = GenericRoboTwinTaskGenBackend(
                root,
                provider,
                model="fixture-model",
                find_exact=find_exact,
            ).materialize(
                _candidate(),
                _adapter(),
                run_id="run_cold_ablation",
                ablation_switches={
                    "rag": False,
                    "visual_self_check": True,
                    "readme_agent": True,
                },
            )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(lookup_calls, 0)
            self.assertEqual(provider.calls, 1)

    def test_unseen_task_generates_after_one_render_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = _Provider(
                [
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "render_bad"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target != ""\n'
                        ),
                    },
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "generated"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target != ""\n'
                        ),
                    },
                ]
            )
            backend = GenericRoboTwinTaskGenBackend(
                root, provider, model="fixture-model"
            )

            result = backend.materialize(
                _candidate(),
                _adapter(),
                run_id="run_cold_generated",
                max_regenerations=1,
            )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(result["provider_call_count"], 4)
            self.assertEqual(result["local_regeneration_count"], 1)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(provider.review_calls, 2)
            self.assertTrue(result["validation"]["preflight"]["render_passed"])
            self.assertTrue(result["validation"]["preflight"]["expert_passed"])
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "task.py").is_file())
            self.assertTrue(
                (run_dir / "generic_taskgen_resolution.json").is_file()
            )
            self.assertEqual(
                result["candidate_manifest"]["task_name"],
                "cold_unseen_task",
            )
            self.assertTrue(
                result["candidate_manifest"]["codegen_provenance"][
                    "generated_by_model"
                ]
            )
            attempt_summary = json.loads(
                (
                    root
                    / "mea/generated_task_attempts/run_cold_generated"
                    / "task_generation_attempt_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(attempt_summary["attempt_count"], 2)
            self.assertEqual(
                attempt_summary["attempts"][0]["recovery_action"],
                "regenerate_candidate",
            )
            self.assertEqual(attempt_summary["max_regenerations"], 1)
            prompt = provider.prompts[0]
            self.assertIn("cold_unseen_task", prompt)
            self.assertIn("TASK TELEMETRY/EXECUTION SCHEMA", prompt)
            self.assertIn("description/cold_unseen_task.md", prompt)
            self.assertIn("assets/cold_target.asset", prompt)
            self.assertNotIn("template_id", prompt)
            self.assertNotIn("aspect_id", prompt)
            self.assertIn("README.AGENT CONTEXT", prompt)

    def test_semantic_review_rejects_proxy_and_repairs_only_checker(
        self,
    ) -> None:
        class ReviewSequenceProvider:
            def __init__(self) -> None:
                self.method_calls = 0
                self.review_calls = 0
                self.method_prompts: list[str] = []
                self.review_prompts: list[str] = []
                self.last_metadata: dict[str, Any] = {}
                self.methods = [
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "generated"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target != ""\n'
                        ),
                    },
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "changed_scene"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target == "generated"\n'
                        ),
                    },
                ]
                self.reviews = [
                    {
                        "schema_version": 1,
                        "status": "rejected",
                        "checks": {
                            "implements_every_checker_requirement": False,
                            "preserves_quantifiers_and_temporal_relations": False,
                            "uses_direct_current_simulator_observables": True,
                            "does_not_substitute_correlated_proxy": False,
                        },
                        "reason": (
                            "A correlated state proxy does not implement the "
                            "frozen checker relation."
                        ),
                    },
                    {
                        "schema_version": 1,
                        "status": "approved",
                        "checks": {
                            "implements_every_checker_requirement": True,
                            "preserves_quantifiers_and_temporal_relations": True,
                            "uses_direct_current_simulator_observables": True,
                            "does_not_substitute_correlated_proxy": True,
                        },
                        "reason": "The repaired checker implements the relation.",
                    },
                ]

            def text(self, prompt: str, **_kwargs: Any) -> str:
                if "TaskGen's separate checker semantic-review pass" in prompt:
                    value = self.reviews[self.review_calls]
                    self.review_calls += 1
                    self.review_prompts.append(prompt)
                    self.last_metadata = {
                        "review_call": self.review_calls
                    }
                    return json.dumps(value)
                value = self.methods[self.method_calls]
                self.method_calls += 1
                self.method_prompts.append(prompt)
                self.last_metadata = {"method_call": self.method_calls}
                return json.dumps(value)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = ReviewSequenceProvider()
            result = GenericRoboTwinTaskGenBackend(
                root,
                provider,
                model="fixture-model",
            ).materialize(
                _candidate(),
                _adapter(),
                run_id="run_semantic_checker_repair",
                max_regenerations=1,
            )

            task_source = (
                Path(result["run_dir"]) / "task.py"
            ).read_text(encoding="utf-8")
            review = result["validation"]["checker_semantic_review"]

        self.assertEqual(provider.method_calls, 2)
        self.assertEqual(provider.review_calls, 2)
        self.assertEqual(result["provider_call_count"], 4)
        self.assertEqual(result["local_regeneration_count"], 1)
        self.assertIn('self.target = "generated"', task_source)
        self.assertNotIn("changed_scene", task_source)
        self.assertIn("correlated proxy", provider.method_prompts[1])
        self.assertIn(
            "SUPPORTED ROBOTWIN CHECKER API",
            provider.review_prompts[1],
        )
        self.assertIn(
            "joint tuple's `[0].child_link`",
            provider.review_prompts[1],
        )
        self.assertIn(
            "runtime-bound to the supplied",
            provider.review_prompts[1],
        )
        self.assertIn(
            "get_left_tcp_pose()[:3]",
            provider.review_prompts[1],
        )
        self.assertIn(
            'get_contact_point(i, "pose").p',
            provider.review_prompts[1],
        )
        self.assertIn(
            "exact TaskContext expression ending in",
            provider.review_prompts[1],
        )
        self.assertEqual(review["status"], "approved")
        self.assertEqual(review["authority"], "development_agent_proxy")

    def _cold_unexpected_preflight_failure_is_terminal_and_counted(self) -> None:
        def failing_preflight(
            _attempt_dir: Path,
            _module_source: str,
            _candidate_value: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            raise ValueError("fixture preflight failure")

        base = _adapter()
        adapter = GenericRoboTwinTaskAdapter(
            task_name=base.task_name,
            official_source=base.official_source,
            official_class=base.official_class,
            task_schema=base.task_schema,
            documentation_paths=base.documentation_paths,
            asset_paths=base.asset_paths,
            hooks=GenericTaskGenHooks(
                validate_methods=base.hooks.validate_methods,
                build_module=base.hooks.build_module,
                preflight_candidate=failing_preflight,
                resolve_metric=base.hooks.resolve_metric,
                resolve_checker_contract=(
                    base.hooks.resolve_checker_contract
                ),
                prompt_constraints=base.hooks.prompt_constraints,
            ),
        )
        response = {
            "load_actors": (
                "def load_actors(self):\n"
                '    self.target = "generated"\n'
            ),
            "check_success": (
                "def check_success(self):\n"
                '    return self.target == "generated"\n'
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            with self.assertRaises(GenericTaskGenError):
                GenericRoboTwinTaskGenBackend(
                    root,
                    _Provider([response, response]),
                    model="fixture-model",
                ).materialize(
                    _candidate(),
                    adapter,
                    run_id="run_preflight_exception_count",
                )
            summary = json.loads(
                (
                    root
                    / "mea/generated_task_attempts/"
                    "run_preflight_exception_count/"
                    "task_generation_attempt_summary.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(summary["attempt_count"], 1)
        self.assertEqual(summary["runtime"]["provider_calls"], 2)
        self.assertEqual(
            summary["attempts"][0]["failure"]["stage"],
            "task_generation",
        )
        self.assertEqual(
            summary["attempts"][0]["failure"]["failure_kind"],
            "unclassified_exception",
        )

    def _cold_unavailable_review_is_terminal_not_a_checker_repair(self) -> None:
        class UnavailableReviewProvider(_Provider):
            def text(self, prompt: str, **kwargs: Any) -> str:
                if "TaskGen's separate checker semantic-review pass" in prompt:
                    self.review_calls += 1
                    return "not review JSON"
                return super().text(prompt, **kwargs)

        response = {
            "load_actors": (
                "def load_actors(self):\n"
                '    self.target = "generated"\n'
            ),
            "check_success": (
                "def check_success(self):\n"
                '    return self.target == "generated"\n'
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = UnavailableReviewProvider([response])
            with self.assertRaises(GenericTaskGenError):
                GenericRoboTwinTaskGenBackend(
                    root,
                    provider,
                    model="fixture-model",
                ).materialize(
                    _candidate(),
                    _adapter(),
                    run_id="run_semantic_review_unavailable",
                )
            summary = json.loads(
                (
                    root
                    / "mea/generated_task_attempts/"
                    "run_semantic_review_unavailable/"
                    "task_generation_attempt_summary.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(summary["attempt_count"], 1)
        self.assertEqual(summary["runtime"]["provider_calls"], 2)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.review_calls, 1)

    def _cold_checker_repair_diagnosis_includes_terminal_xyz_state(self) -> None:
        initial_tcp = robot_tcp_xyz_summary(
            SimpleNamespace(
                robot=SimpleNamespace(
                    get_left_tcp_pose=lambda: [0.01, -0.02, 0.78, 1.0],
                )
            )
        )
        terminal_tcp = robot_tcp_xyz_summary(
            SimpleNamespace(
                robot=SimpleNamespace(
                    get_left_tcp_pose=lambda: [0.11, -0.02, 0.74, 1.0],
                    get_right_tcp_pose=lambda: [0.03, 0.04, 0.75, 1.0],
                )
            )
        )
        diagnosis = _checker_fixture_failure_diagnosis(
            [
                {
                    "fixture_id": "official_expert_terminal_positive",
                    "expected": True,
                    "observed": False,
                    "passed": False,
                }
            ],
            setup={
                "tracked_actors": [
                    {"id": "target", "position": [0.1, -0.2, 0.74]}
                ],
                "initial_robot_tcp_xyz_m": initial_tcp,
            },
            expert={
                "expert_terminal_tracked_actors": [
                    {"id": "target", "position": [0.12, -0.2, 0.85]}
                ],
                "expert_terminal_robot_tcp_xyz_m": terminal_tcp,
            },
        )

        self.assertIn('"initial_actor_xyz_m"', diagnosis)
        self.assertIn('"expert_terminal_actor_xyz_m"', diagnosis)
        self.assertIn('"target": [0.12, -0.2, 0.85]', diagnosis)
        self.assertIn('"initial_robot_tcp_xyz_m"', diagnosis)
        self.assertIn('"expert_terminal_robot_tcp_xyz_m"', diagnosis)
        self.assertIn('"right": [0.03, 0.04, 0.75]', diagnosis)

    def test_partial_generation_reuses_unrequested_official_method(
        self,
    ) -> None:
        cases = (
            (
                "scene_only",
                "Tag the official target as shifted.",
                None,
                {
                    "load_actors": (
                        "def load_actors(self):\n"
                        "    self.target = create_actor("
                        'modelname="901_novel_target")\n'
                    ),
                    "check_success": "IGNORED_PROVIDER_CHECKER",
                },
                "check_success",
            ),
            (
                "checker_only",
                None,
                "Require the official target to be ready.",
                {
                    "load_actors": "IGNORED_PROVIDER_SCENE",
                    "check_success": (
                        "def check_success(self):\n"
                        "    return self.target.is_ready() and "
                        "self.target is not None\n"
                    ),
                },
                "load_actors",
            ),
        )
        for label, scene_need, checker_need, response, reused in cases:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    task_name = "runtime_novel_task"
                    _write_discoverable_task_repo(root, task_name)
                    observed: dict[str, Any] = {}

                    def fixtures(
                        methods: Mapping[str, str],
                        _candidate_value: Mapping[str, Any],
                    ) -> list[Mapping[str, Any]]:
                        observed["fixture_methods"] = dict(methods)
                        return [{"fixture": "semantic_hook", "passed": True}]

                    def preflight(
                        _path: Path,
                        module_source: str,
                        _candidate_value: Mapping[str, Any],
                    ) -> Mapping[str, Any]:
                        observed["module_source"] = module_source
                        return {
                            "render_passed": True,
                            "expert_passed": True,
                            # Legacy hooks report literal change here. A
                            # checker-only candidate instead proves
                            # preservation by exact official method reuse.
                            "scene_change_passed": (
                                scene_need is not None
                            ),
                        }

                    adapter = load_generic_robotwin_task_adapter(
                        root,
                        task_name,
                        checker_fixtures=fixtures,
                        preflight_candidate=preflight,
                        resolve_metric=lambda _candidate_value: (
                            "runtime_novel_completion"
                        ),
                        resolve_checker_contract=lambda _candidate_value: {
                            "metric": "hook_override",
                            "authority": "hook_override",
                            "official_success": "hook_override",
                        },
                    )
                    candidate = build_experiment_candidate(
                        source_query="Evaluate one independently typed need.",
                        base_task=task_name,
                        semantic_concern=label,
                        scene_need=scene_need,
                        checker_need=checker_need,
                    )
                    provider = _Provider([response])
                    result = GenericRoboTwinTaskGenBackend(
                        root,
                        provider,
                        model="fixture-model",
                    ).materialize(
                        candidate,
                        adapter,
                        run_id=f"run_partial_{label}",
                    )

                    generated = (
                        "check_success"
                        if reused == "load_actors"
                        else "load_actors"
                    )
                    provenance = result["validation"]["method_provenance"]
                    self.assertEqual(provenance[reused], "official_reused")
                    self.assertEqual(
                        provenance[generated], "provider_generated"
                    )
                    self.assertEqual(
                        result["validation"]["official_reused_methods"],
                        [reused],
                    )
                    manifest_provenance = result["candidate_manifest"][
                        "codegen_provenance"
                    ]
                    self.assertEqual(
                        manifest_provenance["method_provenance"],
                        provenance,
                    )
                    self.assertEqual(
                        manifest_provenance["official_reused_methods"],
                        [reused],
                    )
                    self.assertEqual(
                        result["candidate_manifest"]["checker_contract"][
                            "official_success"
                        ],
                        reused == "check_success",
                    )
                    self.assertEqual(
                        result["candidate_manifest"]["checker_contract"][
                            "authority"
                        ],
                        (
                            "official_task_method_reused"
                            if reused == "check_success"
                            else "llm_generated_python_ast_validated"
                        ),
                    )
                    self.assertEqual(
                        result["candidate_manifest"]["checker_contract"][
                            "metric"
                        ],
                        "runtime_novel_completion",
                    )
                    module_source = str(observed["module_source"])
                    self.assertNotIn("IGNORED_PROVIDER_", module_source)
                    generated_class = next(
                        node
                        for node in ast.parse(module_source).body
                        if isinstance(node, ast.ClassDef)
                    )
                    direct_methods = {
                        node.name
                        for node in generated_class.body
                        if isinstance(node, ast.FunctionDef)
                    }
                    self.assertIn(generated, direct_methods)
                    self.assertNotIn(reused, direct_methods)
                    fixture_methods = observed["fixture_methods"]
                    self.assertNotIn(
                        "IGNORED_PROVIDER_", json.dumps(fixture_methods)
                    )
                    self.assertEqual(
                        result["validation"]["scene_alignment"][
                            "expected_state"
                        ],
                        "changed" if scene_need is not None else "preserved",
                    )
                    self.assertIn(
                        "runtime injects the exact official method",
                        provider.prompts[0],
                    )

    def test_direct_method_validation_does_not_ignore_unrequested_text(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_name = "runtime_novel_task"
            _write_discoverable_task_repo(root, task_name)
            with self.assertRaises(GenericTaskGenError):
                validate_generic_task_methods(
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            "    self.target = create_actor("
                            'modelname="900_novel_target")\n'
                            '    self.scene_variant = "shifted"\n'
                        ),
                        "check_success": "IGNORED_PROVIDER_CHECKER",
                    },
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                    required_method_changes={
                        "load_actors": True,
                        "check_success": False,
                    },
                )

    def test_candidate_must_bind_the_adapter_base_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            candidate = _candidate()
            candidate["base_task"] = "another_task"
            backend = GenericRoboTwinTaskGenBackend(
                root, _Provider([]), model="fixture-model"
            )
            with self.assertRaisesRegex(
                GenericTaskGenError, "base_task differs"
            ):
                backend.materialize(
                    candidate, _adapter(), run_id="run_wrong_task"
                )


if __name__ == "__main__":
    unittest.main()
