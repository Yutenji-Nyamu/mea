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
from mea.taskgen.generic_backend import (
    GenericRoboTwinTaskAdapter,
    GenericRoboTwinTaskGenBackend,
    GenericTaskGenError,
    GenericTaskGenHooks,
    build_generic_task_subclass_module,
    discover_generic_robotwin_task_identity,
    generic_task_semantic_key,
    load_generic_robotwin_task_adapter,
    validate_generic_task_methods,
)
from mea.taskgen.provider_scene_checker import validate_method_ast
from scripts.manipeval_taskgen import (
    build_preservation_report,
    record_generic_taskgen_generation_failure,
)


class _Provider:
    def __init__(self, responses: list[dict[str, str]]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []
        self.last_metadata: dict[str, Any] = {}

    def text(self, prompt: str, **_kwargs: Any) -> str:
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        self.last_metadata = {"call": self.calls}
        return json.dumps(response)


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
    def test_safe_ast_allows_conventional_discard_loop_target(self) -> None:
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
            "same_seed_simulator_state:tracked_actors.contact_points",
        )

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
            "tracked_actors.contact_points+position",
        )

    def test_compound_position_and_orientation_checks_both(self) -> None:
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
            "tracked_actors.contact_points+position+quaternion",
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

    def test_chinese_goal_and_contact_geometry_preservation_is_typed(
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

    def test_height_preservation_ignores_requested_xy_offset(self) -> None:
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

    def test_legacy_visual_color_preservation_still_passes(self) -> None:
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
            self.assertIn(
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
                self.assertTrue(adapter.documentation_paths)
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

    def test_pose_property_item_assignment_is_rejected(self) -> None:
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

    def test_scale_gate_defers_nonliteral_or_irrelevant_changes(self) -> None:
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

    def test_ablation_condition_never_reuses_complete_task_artifact(self) -> None:
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
            self.assertEqual(result["provider_call_count"], 2)
            self.assertEqual(result["local_regeneration_count"], 1)
            self.assertEqual(provider.calls, 2)
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
            prompt = provider.prompts[0]
            self.assertIn("cold_unseen_task", prompt)
            self.assertIn("TASK TELEMETRY/EXECUTION SCHEMA", prompt)
            self.assertIn("description/cold_unseen_task.md", prompt)
            self.assertIn("assets/cold_target.asset", prompt)
            self.assertNotIn("template_id", prompt)
            self.assertNotIn("aspect_id", prompt)
            self.assertIn("increasing size by 50% uses 1.5", prompt)
            self.assertIn("reducing size by 50% (or to 50%) uses 0.5", prompt)
            self.assertIn(
                "tracked automatically even when their pose or instance is "
                "replaced",
                prompt,
            )
            self.assertIn(
                "Assign it only when adding an entirely new actor",
                prompt,
            )

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
                        "runtime ignores that text",
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
