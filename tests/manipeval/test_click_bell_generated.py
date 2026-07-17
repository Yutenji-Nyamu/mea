import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mea.planner import (
    ClickBellAdaptivePlanAgent,
    ClickBellPositionPlanAgent,
    PlanAgentError,
)
from mea.taskgen import (
    ClickBellTaskGenError,
    compile_click_bell_overlay,
    create_click_bell_variant_run,
    validate_click_bell_variant_hint,
    validate_click_bell_vision_observation,
)
from mea.taskgen.probe import task_attribute_summary
from scripts.manipeval_agent import build_taskgen_command
from scripts.manipeval_taskgen import (
    validate_click_bell_scene_contract,
    validate_click_bell_scene_position,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_repo(root: Path) -> None:
    schema_dir = root / "mea/toolkit/schemas"
    schema_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
        schema_dir / "click_bell.json",
    )
    for relative in (
        "envs/click_bell.py",
        "policy/ACT/eval.sh",
        "script/eval_policy.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# protected fixture\n", encoding="utf-8")


class AdaptiveProvider:
    last_metadata = {"model": "fake-click-bell-planner"}

    def __init__(self, *, transition: str, next_aspect_id: str | None):
        self.transition = transition
        self.next_aspect_id = next_aspect_id
        self.prompts = []

    def text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if "REAL OBSERVATION HISTORY" not in prompt:
            return json.dumps(
                {
                    "schema_version": 1,
                    "task_name": "click_bell",
                    "evaluation_goal": "evaluate_position_and_instance",
                    "requested_aspect_ids": [
                        "object_position",
                        "object_instance",
                    ],
                    "first_aspect_id": "object_position",
                }
            )
        action = "stop" if self.transition == "stop" else "continue"
        return json.dumps(
            {
                "schema_version": 1,
                "action": action,
                "transition": self.transition,
                "observation_summary": "read real policy, aggregate, and VQA evidence",
                "decision_reason": "bounded evidence-conditioned choice",
                "next_aspect_id": self.next_aspect_id,
            }
        )


class SequenceProvider:
    last_metadata = {"model": "fake-sequence-planner"}

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def adaptive_observation(
    *,
    policy_success: float | None,
    pipeline_passed: bool = True,
    evidence_conflict: bool = False,
    aggregate_complete: bool = True,
):
    quality = (
        {"valid": 1, "missing": 0, "invalid": 0}
        if aggregate_complete
        else {"valid": 0, "missing": 1, "invalid": 0}
    )
    return {
        "round_id": "round_1",
        "pipeline_passed": pipeline_passed,
        "observations": {
            "policy_success": policy_success,
            "planned_tool": {
                "route_decision": {"metric": "official_check_success"},
                "episodes": [
                    {
                        "role": "policy_under_evaluation",
                        "value": policy_success,
                        "details": {},
                    }
                ],
            },
            "aggregate": {
                "status": "passed",
                "input_issues": [] if aggregate_complete else ["missing row"],
                "metrics": [
                    {
                        "metric": "official_check_success",
                        "cohorts": [
                            {
                                "role": "policy_under_evaluation",
                                "summary": {"quality": quality},
                            }
                        ],
                    }
                ],
            },
            "execution_vqa": {"evidence_conflict": evidence_conflict},
        },
    }


class ClickBellGeneratedTests(unittest.TestCase):
    def test_variant_hint_is_strict_and_compiles_overlay(self):
        hint = {"bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}}
        self.assertEqual(validate_click_bell_variant_hint(hint), hint)
        overlay = compile_click_bell_overlay(hint)
        self.assertEqual(overlay["mea"]["bell"]["xy"], [-0.2, -0.08])
        with self.assertRaises(ClickBellTaskGenError):
            validate_click_bell_variant_hint(
                {"bell": {"position_mode": "fixed", "xy": [0.01, -0.08]}}
            )

    def test_instance_variant_is_strict_and_compiles_overlay(self):
        hint = {
            "bell": {
                "position_mode": "official_random",
                "instance_mode": "fixed",
                "bell_id": 0,
            }
        }
        self.assertEqual(validate_click_bell_variant_hint(hint), hint)
        self.assertEqual(
            compile_click_bell_overlay(hint)["mea"]["bell"]["bell_id"], 0
        )
        for invalid in (True, -1, 2):
            with self.subTest(bell_id=invalid), self.assertRaises(
                ClickBellTaskGenError
            ):
                validate_click_bell_variant_hint(
                    {
                        "bell": {
                            "position_mode": "official_random",
                            "instance_mode": "fixed",
                            "bell_id": invalid,
                        }
                    }
                )
        with self.assertRaises(ClickBellTaskGenError):
            validate_click_bell_variant_hint(
                {
                    "bell": {
                        "position_mode": "official_random",
                        "instance_mode": "fixed",
                        "bell_id": 0,
                        "extra": "forbidden",
                    }
                }
            )

    def test_bounded_run_records_overlay_not_codegen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            manifest = create_click_bell_variant_run(
                root,
                "evaluate bell position",
                variant_hint={
                    "bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}
                },
                run_id="run_click_bell_position_test",
            )
            self.assertEqual(manifest["mode"], "reuse")
            self.assertEqual(manifest["generation_kind"], "bounded_variant_overlay")
            self.assertEqual(manifest["task_module"], "mea.tasks.click_bell")
            self.assertFalse(
                manifest["static_validation"]["code_generation"]["performed"]
            )

    def test_instance_run_records_controlled_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            manifest = create_click_bell_variant_run(
                root,
                "evaluate bell instance",
                variant_hint={
                    "bell": {
                        "position_mode": "official_random",
                        "instance_mode": "fixed",
                        "bell_id": 1,
                    }
                },
                run_id="run_click_bell_instance_test",
            )
            spec = json.loads(
                (
                    root
                    / "mea/generated_tasks/run_click_bell_instance_test/variant_spec.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(spec["controlled_axis"], "object_instance")
            self.assertEqual(spec["changes"]["bell"]["bell_id"], 1)

    def test_planner_continues_left_to_right_with_same_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            planner = ClickBellPositionPlanAgent(
                root, start_seed=10, num_episodes=1, max_rounds=2
            )
            manifest = planner.plan(
                "evaluate bell position", evaluation_id="eval_bell_position_test"
            )
            plan = manifest["plan"]
            first = plan["rounds"][0]
            self.assertEqual(first["template_id"], "object_position.left_fixed")
            updated, decision = planner.decide_next_round(
                evaluation_id="eval_bell_position_test",
                user_request="evaluate bell position",
                current_plan=plan,
                observation_history=[{"pipeline_passed": True}],
            )
            self.assertEqual(decision["action"], "continue")
            second = updated["rounds"][1]
            self.assertEqual(second["template_id"], "object_position.right_fixed")
            self.assertEqual(first["execution"]["seeds"], second["execution"]["seeds"])
            stopped, final = planner.decide_next_round(
                evaluation_id="eval_bell_position_test",
                user_request="evaluate bell position",
                current_plan=updated,
                observation_history=[
                    {"pipeline_passed": True},
                    {"pipeline_passed": True},
                ],
            )
            self.assertEqual(final["action"], "stop")
            self.assertEqual(stopped["planning_state"], "stopped_after_round_2")

    def test_adaptive_planner_counterfactual_evidence_changes_direction(self):
        cases = (
            (
                0.0,
                "drill_down",
                "object_position",
                "object_position.right_fixed",
            ),
            (
                1.0,
                "switch_aspect",
                "object_instance",
                "object_instance.base0",
            ),
        )
        for index, (
            policy_success,
            transition,
            next_aspect,
            next_template,
        ) in enumerate(cases):
            with self.subTest(policy_success=policy_success), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                make_repo(root)
                provider = AdaptiveProvider(
                    transition=transition, next_aspect_id=next_aspect
                )
                planner = ClickBellAdaptivePlanAgent(
                    root,
                    provider,
                    model="fake",
                    start_seed=10,
                    num_episodes=1,
                    max_rounds=3,
                )
                evaluation_id = f"eval_bell_adaptive_{index}"
                manifest = planner.plan(
                    "evaluate bell properties", evaluation_id=evaluation_id
                )
                plan = manifest["plan"]
                self.assertEqual(
                    plan["rounds"][0]["template_id"],
                    "object_position.left_fixed",
                )
                updated, decision = planner.decide_next_round(
                    evaluation_id=evaluation_id,
                    user_request="evaluate bell properties",
                    current_plan=plan,
                    observation_history=[
                        adaptive_observation(policy_success=policy_success)
                    ],
                )
                self.assertEqual(decision["transition"], transition)
                self.assertEqual(decision["next_template_id"], next_template)
                self.assertEqual(updated["rounds"][1]["template_id"], next_template)
                decision_prompt = provider.prompts[-1]
                self.assertIn(f'"policy_success": {policy_success}', decision_prompt)
                self.assertIn('"aggregate"', decision_prompt)
                self.assertIn('"execution_vqa"', decision_prompt)

    def test_adaptive_planner_rejects_decisions_against_required_evidence(self):
        cases = (
            (
                "success_cannot_drill",
                adaptive_observation(policy_success=1.0),
                "drill_down",
                "object_position",
            ),
            (
                "failure_cannot_switch",
                adaptive_observation(policy_success=0.0),
                "switch_aspect",
                "object_instance",
            ),
            (
                "conflict_cannot_switch",
                adaptive_observation(
                    policy_success=1.0, evidence_conflict=True
                ),
                "switch_aspect",
                "object_instance",
            ),
        )
        for name, observation, transition, next_aspect in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                make_repo(root)
                provider = AdaptiveProvider(
                    transition=transition, next_aspect_id=next_aspect
                )
                planner = ClickBellAdaptivePlanAgent(
                    root, provider, model="fake", max_rounds=3
                )
                evaluation_id = f"eval_bell_reject_{name}"
                manifest = planner.plan(
                    "evaluate bell properties", evaluation_id=evaluation_id
                )
                with self.assertRaises(PlanAgentError):
                    planner.decide_next_round(
                        evaluation_id=evaluation_id,
                        user_request="evaluate bell properties",
                        current_plan=manifest["plan"],
                        observation_history=[observation],
                    )
                failed = json.loads(
                    (root / f"mea/evaluation_runs/{evaluation_id}/manifest.json")
                    .read_text(encoding="utf-8")
                )
                self.assertEqual(
                    failed["status"], "decision_failed_after_round_1"
                )

    def test_adaptive_planner_retries_malformed_proposal_and_decision(self):
        proposal = json.dumps(
            {
                "schema_version": 1,
                "task_name": "click_bell",
                "evaluation_goal": "evaluate_position_and_instance",
                "requested_aspect_ids": [
                    "object_position",
                    "object_instance",
                ],
                "first_aspect_id": "object_position",
            }
        )
        decision = json.dumps(
            {
                "schema_version": 1,
                "action": "continue",
                "transition": "switch_aspect",
                "observation_summary": "position sentinel passed",
                "decision_reason": "cover the requested instance axis",
                "next_aspect_id": "object_instance",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            provider = SequenceProvider(["not JSON", proposal])
            planner = ClickBellAdaptivePlanAgent(
                root, provider, model="fake", max_rounds=3
            )
            manifest = planner.plan(
                "evaluate bell properties", evaluation_id="eval_bell_retry_plan"
            )
            self.assertEqual(len(provider.prompts), 2)
            self.assertEqual(
                len(manifest["planner"]["round_1_validation_errors"]), 1
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            provider = SequenceProvider([proposal, "not JSON", decision])
            planner = ClickBellAdaptivePlanAgent(
                root, provider, model="fake", max_rounds=3
            )
            manifest = planner.plan(
                "evaluate bell properties",
                evaluation_id="eval_bell_retry_decision",
            )
            _, resolved = planner.decide_next_round(
                evaluation_id="eval_bell_retry_decision",
                user_request="evaluate bell properties",
                current_plan=manifest["plan"],
                observation_history=[adaptive_observation(policy_success=1.0)],
            )
            self.assertEqual(resolved["transition"], "switch_aspect")
            saved = json.loads(
                (root / "mea/evaluation_runs/eval_bell_retry_decision/manifest.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                len(
                    saved["planner"][
                        "decision_after_round_1_validation_errors"
                    ]
                ),
                1,
            )

    def test_adaptive_planner_pipeline_failure_can_only_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            provider = AdaptiveProvider(transition="stop", next_aspect_id=None)
            planner = ClickBellAdaptivePlanAgent(
                root, provider, model="fake", max_rounds=3
            )
            manifest = planner.plan(
                "evaluate bell properties", evaluation_id="eval_bell_adaptive_stop"
            )
            updated, decision = planner.decide_next_round(
                evaluation_id="eval_bell_adaptive_stop",
                user_request="evaluate bell properties",
                current_plan=manifest["plan"],
                observation_history=[
                    adaptive_observation(
                        policy_success=0.0, pipeline_passed=False
                    )
                ],
            )
            self.assertEqual(decision["action"], "stop")
            self.assertEqual(
                decision["evidence_assessment"]["state"], "pipeline_failure"
            )
            self.assertEqual(updated["planning_state"], "stopped_after_round_1")

    def test_agent_command_carries_trusted_variant_json(self):
        round_plan = {
            "round_id": "round_1",
            "task_instruction": "evaluate",
            "task_name": "click_bell",
            "task_module": "mea.tasks.click_bell",
            "route": "reuse",
            "variant_hint": {
                "bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}
            },
            "execution": {"backend": "act", "seeds": [7], "num_episodes": 1},
        }
        command, _ = build_taskgen_command(
            Path("/repo"),
            "eval_click_generated",
            round_plan,
            text_model="text",
            vision_model="vision",
            base_url=None,
            gpu=0,
            max_reflections=0,
        )
        index = command.index("--variant-hint-json")
        self.assertEqual(json.loads(command[index + 1]), round_plan["variant_hint"])
        self.assertIn("--vision-check", command)
        self.assertIn("--run-act", command)

    def test_scene_xy_is_numeric_authority_and_vqa_is_plausibility_only(self):
        spec = {
            "task_name": "click_bell",
            "controlled_axis": "object_position",
            "changes": {
                "bell": {
                    "position_mode": "fixed",
                    "xy": [-0.2, -0.08],
                }
            },
        }
        scene = {
            "tracked_actors": [
                {"id": "bell", "position": [-0.2, -0.08, 0.741]}
            ]
        }
        self.assertTrue(validate_click_bell_scene_position(scene, spec)["passed"])
        vision = validate_click_bell_vision_observation(
            {
                "aligned": True,
                "target_actor": "bell",
                "bell_visible": True,
                "unexpected_changes": [],
                "diagnosis": "ok",
                "suggestions": [],
                "confidence": 0.9,
            }
        )
        self.assertTrue(vision["passed"])
        self.assertEqual(vision["position_authority"], "simulator_tracked_actor_xy")

    def test_scene_contract_rejects_malformed_or_mislabeled_spec(self):
        scene = {
            "tracked_actors": [
                {"id": "bell", "position": [-0.2, -0.08, 0.741]}
            ]
        }
        invalid_specs = (
            {"task_name": "click_bell", "changes": {"bell": {}}},
            {
                "task_name": "click_bell",
                "controlled_axis": "object_instance",
                "changes": {
                    "bell": {
                        "position_mode": "fixed",
                        "xy": [-0.2, -0.08],
                    }
                },
            },
        )
        for spec in invalid_specs:
            with self.subTest(spec=spec), self.assertRaises(
                ClickBellTaskGenError
            ):
                validate_click_bell_scene_contract(scene, spec)

    def test_scene_contract_infers_axis_for_legacy_strict_position_spec(self):
        spec = {
            "task_name": "click_bell",
            "changes": {
                "bell": {
                    "position_mode": "fixed",
                    "xy": [-0.2, -0.08],
                }
            },
        }
        scene = {
            "tracked_actors": [
                {"id": "bell", "position": [-0.2, -0.08, 0.741]}
            ]
        }
        result = validate_click_bell_scene_contract(scene, spec)
        self.assertTrue(result["passed"])
        self.assertEqual(result["controlled_axis"], "object_position")

    def test_scene_instance_id_is_simulator_authority(self):
        spec = {
            "task_name": "click_bell",
            "controlled_axis": "object_instance",
            "changes": {
                "bell": {
                    "position_mode": "official_random",
                    "instance_mode": "fixed",
                    "bell_id": 1,
                }
            },
        }
        scene = {
            "tracked_actors": [
                {"id": "bell", "position": [-0.2, -0.08, 0.741]}
            ],
            "task_attributes": {"bell_id": 1},
        }
        result = validate_click_bell_scene_contract(scene, spec)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["instance"]["authority"],
            "simulator_task_attribute:bell_id",
        )
        mismatch = validate_click_bell_scene_contract(
            {**scene, "task_attributes": {"bell_id": 0}}, spec
        )
        self.assertFalse(mismatch["passed"])

    def test_probe_task_attribute_summary_normalizes_scalar(self):
        class Scalar:
            def item(self):
                return 1

        summary = task_attribute_summary(
            SimpleNamespace(bell_id=Scalar()),
            {"probe_task_attributes": ["bell_id"]},
        )
        self.assertEqual(summary, {"bell_id": 1})

    def test_click_bell_vqa_rejects_string_booleans_and_wrong_actor(self):
        base = {
            "aligned": True,
            "target_actor": "bell",
            "bell_visible": True,
            "unexpected_changes": [],
            "diagnosis": "ok",
            "suggestions": [],
            "confidence": 0.9,
        }
        for update in (
            {"aligned": "false"},
            {"bell_visible": "false"},
            {"target_actor": "block"},
        ):
            with self.subTest(update=update), self.assertRaises(Exception):
                validate_click_bell_vision_observation({**base, **update})

    def test_failed_first_round_preserves_unexecuted_right_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            planner = ClickBellPositionPlanAgent(root, max_rounds=2)
            manifest = planner.plan("evaluate", evaluation_id="eval_bell_failed")
            _, decision = planner.decide_next_round(
                evaluation_id="eval_bell_failed",
                user_request="evaluate",
                current_plan=manifest["plan"],
                observation_history=[{"pipeline_passed": False}],
            )
            self.assertEqual(decision["action"], "stop")
            self.assertEqual(
                decision["remaining_template_ids_before_decision"],
                ["object_position.right_fixed"],
            )


if __name__ == "__main__":
    unittest.main()
