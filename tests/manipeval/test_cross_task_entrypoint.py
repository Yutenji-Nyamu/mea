import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mea.agent_acceptance import build_compact_flagship_acceptance
from mea.agent_evidence import build_evidence_bundle
from mea.planner import (
    BoundTaskPlanSession,
    PlanAgentQueryInterpreter,
    OfficialTaskPlanAgent,
    build_act_catalog,
    discover_robotwin_task_inventory,
    resolve_open_task,
)
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.taskgen import create_official_task_run
from mea.taskgen.round_materialization import build_taskgen_command
from mea.execution_vqa.runtime import run_round_execution_vqa
from mea.round_summary import (
    normalize_outcome_semantics,
    summarize_round,
    taskgen_ast_gate_passed,
)
from scripts.manipeval_agent import (
    bind_ready_task_after_query_interpretation,
    build_bound_plan_agent_handoff,
    build_pending_task_binding_policy_card,
    concern_candidate_domain_is_executable,
    finish_unsupported_global_route,
    finish_unsupported_open_task_resolution,
    persist_query_contract,
)
from scripts.manipeval_taskgen import (
    _checker_fixture_failure_diagnosis,
    _expert_terminal_authority_failure,
    _tracked_actor_heights,
    run_act,
    run_official_expert_episodes,
    run_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_schema_repo(root: Path) -> None:
    (root / "envs").mkdir(parents=True)
    (root / "envs/click_bell.py").write_text(
        "class click_bell:\n"
        "    def load_actors(self):\n"
        "        return None\n"
        "\n"
        "    def check_success(self):\n"
        "        return False\n",
        encoding="utf-8",
    )
    schema_dir = root / "mea/toolkit/schemas"
    schema_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
        schema_dir / "click_bell.json",
    )


def official_round(execution_backend: str | None = None) -> dict:
    execution = {"seeds": [7], "num_episodes": 1}
    if execution_backend is not None:
        execution["backend"] = execution_backend
    return {
        "round_id": "round_1",
        "template_id": "task_execution.official_baseline",
        "sub_aspect": "task_execution.official_baseline",
        "task_instruction": "evaluate click_bell",
        "task_name": "click_bell",
        "task_module": "envs.click_bell",
        "route": "official",
        "execution": execution,
        "tool_request": {
            "schema_version": 1,
            "task_name": "click_bell",
            "metric": "official_check_success",
            "question": "Did the bell task succeed?",
        },
    }


class CrossTaskEntrypointTests(unittest.TestCase):
    class FrozenConcernProvider:
        last_metadata = {"provider": "fixture"}

        def __init__(self, response: dict) -> None:
            self.response = response
            self.calls = 0

        def text(self, _prompt: str, **_kwargs) -> str:
            self.calls += 1
            return json.dumps(self.response, ensure_ascii=False)

    def test_ast_gate_accepts_valid_provider_scene_checker_codegen(self):
        self.assertTrue(
            taskgen_ast_gate_passed(
                {
                    "provider_scene_checker": {
                        "valid": True,
                        "ast_policy": (
                            "bbh_distractor_safe_ast_semantic_fixtures_v2"
                        ),
                        "model_written_python": True,
                        "restricted_success_spec_compiler_used": False,
                    }
                }
            )
        )
        self.assertTrue(
            taskgen_ast_gate_passed(
                {"load_actors_ast": {"valid": True}}
            )
        )
        self.assertFalse(
            taskgen_ast_gate_passed(
                {
                    "provider_scene_checker": {
                        "valid": True,
                        "ast_policy": "fixture",
                        "model_written_python": True,
                        "restricted_success_spec_compiler_used": True,
                    }
                }
            )
        )

    def test_outcome_semantics_separates_extension_from_conflict(self):
        official = normalize_outcome_semantics(
            {
                "outcome_authority": "official_check_success",
                "episodes": [],
            },
            {"success_official_equivalent": True},
        )
        self.assertEqual(official["status"], "official_only")
        self.assertFalse(official["evidence_conflict"])

        trusted = {
            "outcome_authority": "llm_generated_python_ast_validated",
            "episodes": [
                {
                    "seed": 100405,
                    "result": {
                        "details": {
                            "generated_checker_success": True,
                            "official_success": False,
                            "official_core_predicate_satisfied": True,
                        }
                    },
                }
            ]
        }
        extension = normalize_outcome_semantics(
            trusted,
            {"success_official_equivalent": False},
        )
        self.assertEqual(extension["status"], "expected_semantic_extension")
        self.assertFalse(extension["evidence_conflict"])

        trusted["episodes"][0]["result"]["details"][
            "official_core_predicate_satisfied"
        ] = False
        conflict = normalize_outcome_semantics(
            trusted,
            {"success_official_equivalent": False},
        )
        self.assertEqual(conflict["status"], "conflict")
        self.assertTrue(conflict["evidence_conflict"])

    def test_official_outcome_defaults_missing_summary_to_equivalent(self):
        for label, authority, episodes in (
            ("no_episode_details", "official_check_success", []),
            (
                "scene_only_official_reuse",
                "official_check_success_reused",
                [],
            ),
            (
                "official_result",
                "official_check_success",
                [
                    {
                        "seed": 100405,
                        "result": {
                            "details": {
                                "official_success": True,
                            }
                        },
                    }
                ],
            ),
        ):
            with self.subTest(label=label):
                semantics = normalize_outcome_semantics(
                    {
                        "outcome_authority": authority,
                        "episodes": episodes,
                    },
                    {},
                )
                self.assertEqual(semantics["status"], "official_only")
                self.assertTrue(semantics["official_equivalent"])
                self.assertFalse(semantics["evidence_conflict"])
                if semantics["episodes"]:
                    self.assertEqual(
                        semantics["episodes"][0]["status"],
                        "official_only",
                    )
                    self.assertTrue(
                        semantics["episodes"][0][
                            "official_equivalent"
                        ]
                    )

    def test_runtime_query_contract_artifact_tracks_open_discovery(self):
        contract = {
            "schema_version": 3,
            "claim_type": "diagnostic",
            "candidate_universe": ["catalog_candidate", "dynamic_candidate"],
            "required_coverage": {
                "candidate_ids": [
                    "catalog_candidate",
                    "dynamic_candidate",
                ],
                "minimum_evaluated": 1,
                "minimum_per_group": None,
            },
            "round_budget": 2,
            "comparison_groups": None,
            "candidate_universe_closed": False,
            "existential_witness_outcome": None,
            "control_requirement": "required",
        }
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)
            plan = {"query_contract": {"candidate_universe_closed": True}}

            persisted = persist_query_contract(
                evaluation,
                plan,
                contract,
            )

            artifact = json.loads(
                (
                    evaluation
                    / "plan/query_sufficiency_contract.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(artifact, persisted)
            self.assertEqual(plan["query_contract"], persisted)
            self.assertFalse(artifact["candidate_universe_closed"])
            self.assertIn(
                "dynamic_candidate",
                artifact["candidate_universe"],
            )

    def test_compact_flagship_acceptance_requires_online_sufficient_reuse(self):
        module_sha256 = "a" * 64
        round_runs = [
            {
                "round_summary": {
                    "route": "official",
                    "observations": {
                        "execution_backend": "ACT",
                        "actual_seeds": [100405],
                        "outcome_semantics": {"status": "official_only"},
                    }
                },
            },
            {
                "round_summary": {
                    "route": "provider_scene_checker_codegen",
                    "observations": {
                        "execution_backend": "ACT",
                        "actual_seeds": [100405],
                        "outcome_semantics": {
                            "status": "expected_semantic_extension"
                        },
                    }
                },
                "tool_evaluation": {
                    "route": "bound_child_trusted_checker",
                    "route_decision": {
                        "provider_called": False,
                        "exact_match": True,
                        "metric": "generated_check_success",
                    },
                    "source": {
                        "authority": "llm_generated_python_ast_validated"
                    },
                    "validation": {
                        "status": "passed",
                        "exact_metric_match": True,
                    },
                    "episodes": [
                        {
                            "role": "policy_under_evaluation",
                            "result": {
                                "tool": "generated_check_success",
                                "value": True,
                                "details": {
                                    "authority": (
                                        "llm_generated_python_ast_validated"
                                    ),
                                    "module_sha256": module_sha256,
                                },
                            },
                        }
                    ],
                },
            },
        ]
        acceptance = build_compact_flagship_acceptance(
            round_runs,
            global_route_result={
                "global_router_provider_calls": 0,
                "provider_called": False,
                "route_source": "runtime_task_checkpoint_binding",
            },
            claim_first_runtime_state={
                "assessment": {
                    "stop_reason": "evidence_sufficient",
                    "evidence_sufficient": True,
                    "observed_candidate_ids": [
                        "robustness.distractor_avoidance.lookalike_bell"
                    ],
                },
                "query_contract": {
                    "candidate_universe": [
                        "robustness.distractor_avoidance.lookalike_bell"
                    ],
                }
            },
            claim_first_query_answer={
                "answered": True,
                "stop_reason": "evidence_sufficient",
                "answer_scope": "bounded_experimental_query_semantics",
            },
            free_concern_bundle={
                "source": "provider_plan_agent_query_interpretation",
                "provider": {
                    "called": True,
                    "attempt_count": 1,
                    "errors": [],
                },
            },
            open_task_resolution={"decision": "retrieve_and_adapt"},
            concern_candidate_resolution={
                "decision": "bind_single_aspect",
                "resolution": "unique_query_supported_concern",
                "candidate_aspect_ids": [
                    "robustness.distractor_avoidance"
                ],
                "selected_template_ids": [
                    "robustness.distractor_avoidance.lookalike_bell"
                ],
                "concern_created_before_catalog": True,
                "catalog_was_model_visible": False,
            },
            history_disabled=True,
        )

        self.assertTrue(acceptance["accepted"])
        self.assertEqual(acceptance["act_rollouts"], 2)
        self.assertTrue(acceptance["history_replay_disabled"])
        self.assertTrue(acceptance["online_query_interpretation"])
        self.assertTrue(acceptance["same_bundle_bound_checker_reuse"])
        self.assertFalse(
            acceptance["cross_query_registry_reuse_established"]
        )

        no_act = json.loads(json.dumps(round_runs))
        for item in no_act:
            item["round_summary"]["observations"]["actual_seeds"] = []
        rejected = build_compact_flagship_acceptance(
            no_act,
            global_route_result={
                "global_router_provider_calls": 0,
                "provider_called": False,
                "route_source": "runtime_task_checkpoint_binding",
            },
            claim_first_runtime_state={
                "assessment": {
                    "stop_reason": "evidence_sufficient",
                    "evidence_sufficient": True,
                    "observed_candidate_ids": [
                        "robustness.distractor_avoidance.lookalike_bell"
                    ],
                },
                "query_contract": {
                    "candidate_universe": [
                        "robustness.distractor_avoidance.lookalike_bell"
                    ],
                }
            },
            claim_first_query_answer={
                "answered": True,
                "answer_scope": "bounded_experimental_query_semantics",
            },
            free_concern_bundle={
                "source": "provider_plan_agent_query_interpretation",
                "provider": {
                    "called": True,
                    "attempt_count": 1,
                    "errors": [],
                },
            },
            open_task_resolution={"decision": "retrieve_and_adapt"},
            concern_candidate_resolution={
                "decision": "bind_single_aspect",
                "resolution": "unique_query_supported_concern",
                "candidate_aspect_ids": [
                    "robustness.distractor_avoidance"
                ],
                "selected_template_ids": [
                    "robustness.distractor_avoidance.lookalike_bell"
                ],
                "concern_created_before_catalog": True,
                "catalog_was_model_visible": False,
            },
            history_disabled=True,
        )
        self.assertFalse(rejected["accepted"])

    def test_compact_flagship_accepts_broad_runtime_candidate_discovery(self):
        candidate_id = "dynamic.click_bell.lateral_translation"
        module_sha256 = "b" * 64
        acceptance = build_compact_flagship_acceptance(
            [
                {
                    "round_summary": {
                        "route": "official",
                        "observations": {
                            "execution_backend": "ACT",
                            "actual_seeds": [100405],
                            "outcome_semantics": {
                                "status": "official_only"
                            },
                        },
                    },
                },
                {
                    "round_summary": {
                        "route": "generic_provider_scene_checker_codegen",
                        "semantic_need_execution": {
                            "candidate_id": candidate_id
                        },
                        "observations": {
                            "execution_backend": "ACT",
                            "actual_seeds": [100405],
                            "outcome_semantics": {
                                "status": "expected_semantic_extension"
                            },
                        },
                    },
                    "tool_evaluation": {
                        "route": "bound_child_trusted_checker",
                        "route_decision": {
                            "provider_called": False,
                            "exact_match": True,
                            "metric": "generated_check_success",
                        },
                        "source": {
                            "authority": (
                                "llm_generated_python_ast_validated"
                            )
                        },
                        "validation": {
                            "status": "passed",
                            "exact_metric_match": True,
                        },
                        "episodes": [
                            {
                                "role": "policy_under_evaluation",
                                "result": {
                                    "tool": "generated_check_success",
                                    "value": False,
                                    "details": {
                                        "authority": (
                                            "llm_generated_python_ast_validated"
                                        ),
                                        "module_sha256": module_sha256,
                                    },
                                },
                            }
                        ],
                    },
                },
            ],
            global_route_result={
                "global_router_provider_calls": 0,
                "provider_called": False,
                "route_source": "runtime_task_checkpoint_binding",
            },
            claim_first_runtime_state={
                "assessment": {
                    "stop_reason": "evidence_sufficient",
                    "evidence_sufficient": True,
                    "observed_candidate_ids": [candidate_id],
                },
                "query_contract": {
                    "candidate_universe": [
                        "object_position.left_fixed",
                        candidate_id,
                    ],
                },
            },
            claim_first_query_answer={
                "answered": True,
                "stop_reason": "evidence_sufficient",
                "answer_scope": "bounded_experimental_query_semantics",
            },
            free_concern_bundle={
                "source": "provider_plan_agent_query_interpretation",
                "provider": {
                    "called": True,
                    "attempt_count": 1,
                    "errors": [],
                },
            },
            open_task_resolution={"decision": "retrieve_and_adapt"},
            concern_candidate_resolution={
                "decision": "catalog_external",
                "resolution": "open_world_candidate_discovery_required",
                "candidate_aspect_ids": None,
                "selected_template_ids": [],
                "concern_created_before_catalog": True,
                "catalog_was_model_visible": False,
            },
            history_disabled=True,
        )

        self.assertTrue(acceptance["accepted"])
        self.assertEqual(
            acceptance["candidate_binding_mode"],
            "online_runtime_discovery",
        )
        self.assertTrue(acceptance["singleton_query_candidate"])

    def test_compact_flagship_accepts_query_authorized_no_control_sequence(
        self,
    ):
        candidate_ids = [
            "dynamic.grab_roller.clearance",
            "dynamic.grab_roller.clearance.refined",
        ]
        round_runs = []
        for candidate_id in candidate_ids:
            round_runs.append(
                {
                    "round_summary": {
                        "route": "generic_provider_scene_checker_codegen",
                        "semantic_need_execution": {
                            "candidate_id": candidate_id
                        },
                        "observations": {
                            "execution_backend": "ACT",
                            "actual_seeds": [100401],
                            "outcome_semantics": {
                                "status": "expected_semantic_extension"
                            },
                            "implementation_trace": {
                                "candidate_id": candidate_id,
                                "relationship": "direct",
                                "coverage_status": "complete",
                            },
                        },
                    }
                }
            )

        acceptance = build_compact_flagship_acceptance(
            round_runs,
            global_route_result={
                "global_router_provider_calls": 0,
                "provider_called": False,
                "route_source": "runtime_task_checkpoint_binding",
            },
            claim_first_runtime_state={
                "assessment": {
                    "stop_reason": "evidence_sufficient",
                    "evidence_sufficient": True,
                    "observed_candidate_ids": candidate_ids,
                    "decisive_candidate_ids": [candidate_ids[-1]],
                },
                "query_contract": {
                    "candidate_universe": candidate_ids,
                    "control_requirement": "not_required",
                },
            },
            claim_first_query_answer={
                "answered": True,
                "stop_reason": "evidence_sufficient",
                "answer_scope": "bounded_experimental_query_semantics",
            },
            free_concern_bundle={
                "source": "provider_plan_agent_query_interpretation",
                "provider": {
                    "called": True,
                    "attempt_count": 1,
                    "errors": [],
                },
            },
            open_task_resolution={"decision": "retrieve_and_adapt"},
            concern_candidate_resolution={
                "decision": "catalog_external",
                "resolution": "open_world_candidate_discovery_required",
                "candidate_aspect_ids": None,
                "selected_template_ids": [],
                "concern_created_before_catalog": True,
                "catalog_was_model_visible": False,
            },
            history_disabled=True,
        )

        self.assertTrue(acceptance["accepted"])
        self.assertEqual(acceptance["policy_rollouts"], 2)
        self.assertEqual(
            acceptance["control_requirement"], "not_required"
        )

    def test_compact_flagship_accepts_direct_typed_official_candidate(self):
        candidate_id = "dynamic.click_bell.target_scale"
        refined_candidate_id = "dynamic.click_bell.target_scale.smaller"
        acceptance = build_compact_flagship_acceptance(
            [
                {
                    "round_summary": {
                        "route": "official",
                        "observations": {
                            "execution_backend": "ACT",
                            "actual_seeds": [100000],
                            "outcome_semantics": {
                                "status": "official_only"
                            },
                        },
                    },
                },
                {
                    "round_summary": {
                        "route": "generic_provider_scene_checker_codegen",
                        "semantic_need_execution": {
                            "candidate_id": candidate_id,
                        },
                        "observations": {
                            "execution_backend": "ACT",
                            "actual_seeds": [100000],
                            "outcome_semantics": {
                                "status": "official_only"
                            },
                            "implementation_trace": {
                                "candidate_id": candidate_id,
                                "relationship": "direct",
                                "coverage_status": "complete",
                            },
                        },
                    },
                },
                {
                    "round_summary": {
                        "route": "generic_provider_scene_checker_codegen",
                        "semantic_need_execution": {
                            "candidate_id": refined_candidate_id,
                        },
                        "observations": {
                            "execution_backend": "ACT",
                            "actual_seeds": [100000],
                            "outcome_semantics": {
                                "status": "official_only"
                            },
                            "implementation_trace": {
                                "candidate_id": refined_candidate_id,
                                "relationship": "direct",
                                "coverage_status": "complete",
                            },
                        },
                    },
                },
            ],
            global_route_result={
                "global_router_provider_calls": 0,
                "provider_called": False,
                "route_source": "runtime_task_checkpoint_binding",
            },
            claim_first_runtime_state={
                "assessment": {
                    "stop_reason": "evidence_sufficient",
                    "evidence_sufficient": True,
                    "observed_candidate_ids": [
                        candidate_id,
                        refined_candidate_id,
                    ],
                    "decisive_candidate_ids": [refined_candidate_id],
                },
                "query_contract": {
                    "candidate_universe": [
                        candidate_id,
                        refined_candidate_id,
                    ],
                },
            },
            claim_first_query_answer={
                "answered": True,
                "stop_reason": "evidence_sufficient",
                "answer_scope": "official_equivalent",
            },
            free_concern_bundle={
                "source": "provider_plan_agent_query_interpretation",
                "provider": {
                    "called": True,
                    "attempt_count": 1,
                    "errors": [],
                },
            },
            open_task_resolution={"decision": "retrieve_and_adapt"},
            concern_candidate_resolution={
                "decision": "catalog_external",
                "resolution": "open_world_candidate_discovery_required",
                "candidate_aspect_ids": None,
                "selected_template_ids": [],
                "concern_created_before_catalog": True,
                "catalog_was_model_visible": False,
            },
            history_disabled=True,
        )

        self.assertTrue(acceptance["accepted"])
        self.assertTrue(acceptance["typed_execution_complete"])
        self.assertTrue(acceptance["candidate_execution_accepted"])
        self.assertFalse(acceptance["same_bundle_bound_checker_reuse"])
        self.assertFalse(acceptance["singleton_query_candidate"])
        self.assertTrue(acceptance["query_candidates_bound"])
        self.assertEqual(acceptance["act_rollouts"], 3)

    def test_auto_route_rejects_task_module_override_before_provider_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment.pop("UIUI_API_KEY", None)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_agent.py"),
                    "--repo-root",
                    temporary,
                    "--request",
                    "evaluate a bell",
                    "--auto-route",
                    "--task-module",
                    "envs.click_bell",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("do not pass --task-module", process.stderr)

    def test_unsupported_global_route_rejects_path_escape_evaluation_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "evaluation_id"):
                finish_unsupported_global_route(
                    root,
                    evaluation_id="../escape",
                    user_request="unsupported query",
                    catalog={},
                    route_result={},
                    router=object(),
                    history_retrieval={},
                )
            self.assertFalse((root / "mea/escape").exists())

    def test_unsupported_open_task_resolution_writes_query_first_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            concern_bundle = {
                "schema_version": 1,
                "source": "provider_plan_agent_query_interpretation",
                "concern": {"sub_aspect": "novel.concern"},
                "provider": {"called": True},
            }
            resolution = {
                "schema_version": 1,
                "decision": "unsupported",
                "reason_code": "policy_task_mismatch",
            }
            manifest = finish_unsupported_open_task_resolution(
                root,
                evaluation_id="eval_open_task_mismatch",
                user_request="open a laptop",
                catalog={"tasks": []},
                concern_bundle=concern_bundle,
                task_inventory=[],
                task_resolution=resolution,
                concern_agent=SimpleNamespace(
                    last_prompt="catalog-free prompt",
                    last_responses=['{"schema_version": 1}'],
                ),
            )
            run_dir = root / "mea/evaluation_runs/eval_open_task_mismatch"
            self.assertEqual(manifest["status"], "unsupported")
            self.assertEqual(manifest["route"]["reason_code"], "policy_task_mismatch")
            self.assertTrue(
                (run_dir / "plan/query_interpretation.json").is_file()
            )
            self.assertTrue(
                (run_dir / "plan/open_task_resolution.json").is_file()
            )
            self.assertTrue(
                (run_dir / "plan/robotwin_task_inventory.json").is_file()
            )
            self.assertFalse(any((run_dir / "execution").iterdir()))

    def test_bound_query_first_uses_one_concern_call_and_no_global_router(self):
        query = "这个ACT策略在目标附近有相似物体时是否仍能可靠点击正确目标？"
        frozen_concern = {
            "schema_version": 1,
            "source_query": query,
            "sub_aspect": (
                "Reliability of target selection when visually similar "
                "objects are nearby."
            ),
            "hypothesis": (
                "The ACT policy reliably clicks the correct target even when "
                "visually similar objects are placed near the target."
            ),
            "task_intent": (
                "Click the correct target object based on its predefined "
                "identity and location."
            ),
            "requested_variation": (
                "Place visually similar objects near the target to test "
                "potential confusion."
            ),
            "measurement_need": (
                "Observe whether the policy consistently clicks the correct "
                "target object without errors or confusion."
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_schema_repo(root)
            instruction_dir = root / "description/task_instruction"
            instruction_dir.mkdir(parents=True)
            (instruction_dir / "click_bell.json").write_text(
                json.dumps(
                    {"full_description": "click the bell's top center"},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            checkpoint_dir = (
                root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            catalog = build_act_catalog(root)
            policy_card = BoundTaskPlanSession.from_catalog(
                catalog, "click_bell"
            ).planning_context(root)["policy_card"]
            provider = self.FrozenConcernProvider(frozen_concern)
            concern_bundle = PlanAgentQueryInterpreter(
                provider,
                model="fixture",
                max_attempts=1,
            ).propose(query, policy_card=policy_card)
            inventory = discover_robotwin_task_inventory(
                root,
                capability_catalog=catalog,
            )
            resolution = resolve_open_task(
                concern_bundle["concern"],
                policy_card=policy_card,
                inventory=inventory,
                can_generate_new_task=False,
            )
            route_result, routed = build_bound_plan_agent_handoff(
                catalog,
                task_name="click_bell",
                user_request=query,
            )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(resolution["decision"], "retrieve_and_adapt")
        self.assertEqual(
            resolution["selected_base_task"]["task_name"], "click_bell"
        )
        self.assertFalse(route_result["provider_called"])
        self.assertEqual(route_result["global_router_provider_calls"], 0)
        self.assertEqual(route_result["attempt_count"], 0)
        self.assertEqual(routed["task_name"], "click_bell")
        self.assertIsNone(routed["proposal"])
        self.assertTrue(route_result["selection"]["binding_only"])
        self.assertEqual(
            route_result["selection"]["requested_aspect_ids"],
            [],
        )
        self.assertIsNone(route_result["selection"]["first_aspect_id"])
        self.assertIn(
            query,
            route_result["selection"]["evaluation_goal"],
        )

    def test_unbound_query_discovers_concern_before_checkpoint_binding(self):
        query = "Where does this policy first expose a task weakness?"
        frozen_concern = {
            "schema_version": 1,
            "source_query": query,
            "sub_aspect": "Object pose robustness.",
            "hypothesis": "The policy fails when the bottle starts rotated.",
            "task_intent": "Adjust a bottle into the required upright pose.",
            "requested_variation": "Rotate the initial bottle pose.",
            "measurement_need": "Measure final bottle orientation and success.",
        }
        provider = self.FrozenConcernProvider(frozen_concern)
        concern_agent = PlanAgentQueryInterpreter(
            provider,
            model="fixture",
            max_attempts=1,
        )
        bundle = concern_agent.propose(
            query,
            policy_card=build_pending_task_binding_policy_card(),
        )
        inventory = [
            {
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "description": "adjust a bottle into an upright target pose",
                "execution_status": "official_base_only",
                "capability_aspects": [],
            },
            {
                "schema_version": 1,
                "task_name": "click_bell",
                "description": "click the top center of a bell",
                "execution_status": "capability_registered",
                "capability_aspects": ["object_position"],
            },
        ]
        binding = bind_ready_task_after_query_interpretation(
            bundle["concern"],
            inventory=inventory,
            ready_task_names=["adjust_bottle", "click_bell"],
            default_task_name="click_bell",
        )

        self.assertEqual(binding["selected_task_name"], "adjust_bottle")
        self.assertFalse(binding["fallback_used"])
        self.assertFalse(binding["catalog_visible_to_concern_model"])
        self.assertNotIn("adjust_bottle", concern_agent.last_prompt)
        self.assertNotIn("click_bell", concern_agent.last_prompt)

    def test_task_underspecified_unbound_query_does_not_bind_default(self):
        concern = {
            "schema_version": 1,
            "source_query": "Where does this policy first expose a weakness?",
            "sub_aspect": "Sensitivity to object scale.",
            "hypothesis": "A larger target exposes a failure.",
            "task_intent": (
                "Perform the trained manipulation action and satisfy its goal."
            ),
            "requested_variation": "Increase target scale by twenty percent.",
            "measurement_need": "Compare success with the unchanged control.",
        }
        inventory = [
            {
                "schema_version": 1,
                "task_name": "beat_block_hammer",
                "description": "beat a block with a hammer",
                "execution_status": "capability_registered",
                "capability_aspects": ["object_scale"],
            },
            {
                "schema_version": 1,
                "task_name": "click_bell",
                "description": "click the top center of a bell",
                "execution_status": "capability_registered",
                "capability_aspects": ["object_position"],
            },
        ]
        binding = bind_ready_task_after_query_interpretation(
            concern,
            inventory=inventory,
            ready_task_names=["beat_block_hammer", "click_bell"],
            default_task_name="beat_block_hammer",
        )

        self.assertFalse(binding["fallback_used"])
        self.assertEqual(binding["binding_status"], "ambiguous")
        self.assertEqual(
            binding["reason_code"],
            "task_underspecified_no_checkpoint_binding",
        )
        self.assertIsNone(binding["selected_task_name"])

    def test_broad_domain_does_not_require_preselected_template(self):
        broad = {
            "decision": "discover_candidates",
            "candidate_aspect_ids": ["object_scale", "object_position"],
            "selected_template_ids": [],
        }
        self.assertTrue(
            concern_candidate_domain_is_executable(
                broad,
                candidate_budget=1,
            )
        )
        self.assertFalse(
            concern_candidate_domain_is_executable(
                broad,
                candidate_budget=0,
            )
        )

    def test_explicit_legacy_official_plan_only_does_not_require_provider_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_dir = root / "mea/toolkit/schemas"
            schema_dir.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
                schema_dir / "click_bell.json",
            )
            environment = dict(os.environ)
            environment.pop("UIUI_API_KEY", None)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_agent.py"),
                    "--repo-root",
                    str(root),
                    "--request",
                    "evaluate click_bell",
                    "--task-name",
                    "click_bell",
                    "--evaluation-id",
                    "eval_click_bell_no_key",
                    "--open-query-planner",
                    "catalog_step_v1",
                    "--plan-only",
                    "--no-history",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            plan = json.loads(process.stdout)
            self.assertEqual(plan["task_name"], "click_bell")

    def test_bound_claim_first_plan_only_is_providerless_control_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_dir = root / "envs"
            env_dir.mkdir(parents=True)
            (env_dir / "click_bell.py").write_text(
                "class click_bell:\n"
                "    def load_actors(self):\n"
                "        return None\n"
                "\n"
                "    def check_success(self):\n"
                "        return False\n",
                encoding="utf-8",
            )
            schema_dir = root / "mea/toolkit/schemas"
            schema_dir.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
                schema_dir / "click_bell.json",
            )
            checkpoint_dir = (
                root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            environment = dict(os.environ)
            environment.pop("UIUI_API_KEY", None)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_agent.py"),
                    "--repo-root",
                    str(root),
                    "--request",
                    "Where does this policy first expose a weakness?",
                    "--open-query-planner",
                    "plan_agent_v1",
                    "--bound-task-name",
                    "click_bell",
                    "--generated-rounds",
                    "2",
                    "--evaluation-id",
                    "eval_claim_first_bound_plan_only",
                    "--plan-only",
                    "--no-history",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            plan = json.loads(process.stdout)
            self.assertEqual(plan["task_name"], "click_bell")
            self.assertEqual(
                plan["rounds"][0]["template_id"],
                "task_execution.official_baseline",
            )
            manifest = json.loads(
                (
                    root
                    / "mea/evaluation_runs/eval_claim_first_bound_plan_only/"
                    "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["planner"]["public_planner"],
                "PlanAgent",
            )
            self.assertEqual(
                manifest["planner"]["kind"],
                "plan_agent_direct_initial_v1",
            )
            self.assertFalse(
                manifest["planner"]["task_specific_planner_used"]
            )
            self.assertFalse(manifest["planner"]["provider_called"])

    def test_bound_claim_first_accepts_runtime_task_outside_catalog(self):
        task_name = "novel_runtime_task"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_dir = root / "envs"
            env_dir.mkdir(parents=True)
            (env_dir / f"{task_name}.py").write_text(
                f"class {task_name}:\n"
                "    def load_actors(self):\n"
                "        return None\n"
                "\n"
                "    def check_success(self):\n"
                "        return False\n",
                encoding="utf-8",
            )
            schema_dir = root / "mea/toolkit/schemas"
            schema_dir.mkdir(parents=True)
            schema = json.loads(
                (
                    REPO_ROOT / "mea/toolkit/schemas/adjust_bottle.json"
                ).read_text(encoding="utf-8")
            )
            schema["task_name"] = task_name
            (schema_dir / f"{task_name}.json").write_text(
                json.dumps(schema),
                encoding="utf-8",
            )
            checkpoint_dir = (
                root
                / "policy/ACT/act_ckpt"
                / f"act-{task_name}/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            environment = dict(os.environ)
            environment.pop("UIUI_API_KEY", None)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_agent.py"),
                    "--repo-root",
                    str(root),
                    "--request",
                    "Where does this policy first expose a weakness?",
                    "--bound-task-name",
                    task_name,
                    "--evaluation-id",
                    "eval_runtime_task_outside_catalog",
                    "--plan-only",
                    "--no-history",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            plan = json.loads(process.stdout)
            self.assertEqual(plan["task_name"], task_name)
            self.assertEqual(
                plan["rounds"][0]["template_id"],
                "task_execution.official_baseline",
            )

    def test_official_task_run_records_no_codegen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_schema_repo(root)
            manifest = create_official_task_run(
                root,
                "evaluate the official bell task",
                task_name="click_bell",
                run_id="run_click_bell_test",
                telemetry_profile="balanced_v1",
            )
            run_dir = root / "mea/generated_tasks/run_click_bell_test"
            self.assertEqual(manifest["mode"], "official")
            self.assertEqual(manifest["task_module"], "envs.click_bell")
            self.assertFalse(manifest["provider"]["called"])
            self.assertFalse(
                manifest["static_validation"]["code_generation"]["performed"]
            )
            self.assertEqual(
                (run_dir / "overlay.yml").read_text(encoding="utf-8"), "{}\n"
            )
            self.assertTrue(
                (run_dir / "generation/official_source.json").is_file()
            )
            bundle = json.loads(
                (run_dir / "generation/task_artifact_bundle.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(bundle["scene_method"]["origin"], "official_reuse")
            self.assertEqual(bundle["success_method"]["origin"], "official_reuse")

    def test_official_planner_materializes_one_expert_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_schema_repo(root)
            planner = OfficialTaskPlanAgent(
                root,
                task_name="click_bell",
                start_seed=10,
                num_episodes=2,
                telemetry_profile="legacy_v1",
            )
            manifest = planner.plan(
                "evaluate click_bell",
                evaluation_id="eval_click_bell_test",
            )
            plan = manifest["plan"]
            round_plan = plan["rounds"][0]
            self.assertEqual(plan["policy"]["name"], "expert")
            self.assertEqual(round_plan["execution"]["seeds"], [10, 11])
            self.assertEqual(round_plan["route"], "official")
            self.assertEqual(round_plan["telemetry_profile"], "legacy_v1")
            updated, decision = planner.decide_next_round(
                evaluation_id="eval_click_bell_test",
                user_request="evaluate click_bell",
                current_plan=plan,
                observation_history=[
                    {"round_id": "round_1", "pipeline_passed": True}
                ],
            )
            self.assertEqual(decision["action"], "stop")
            self.assertEqual(updated["planning_state"], "stopped_after_round_1")

    def test_official_planner_materializes_requested_execution_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_schema_repo(root)
            for backend in ("act", "both"):
                planner = OfficialTaskPlanAgent(
                    root,
                    task_name="click_bell",
                    start_seed=20,
                    num_episodes=2,
                    execution_backend=backend,
                )
                manifest = planner.plan(
                    "evaluate click_bell",
                    evaluation_id=f"eval_click_bell_{backend}",
                )
                plan = manifest["plan"]
                round_plan = plan["rounds"][0]
                self.assertEqual(round_plan["route"], "official")
                self.assertEqual(round_plan["execution"]["backend"], backend)
                self.assertEqual(round_plan["execution"]["seeds"], [20, 21])
                self.assertIn("act", round_plan["execution"]["gates"])
                self.assertEqual(plan["policy"]["name"], "ACT")
                if backend == "both":
                    self.assertIn("expert", round_plan["execution"]["gates"])

    def test_official_planner_accepts_validated_control_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_dir = root / "mea/toolkit/schemas"
            schema_dir.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "mea/toolkit/schemas/adjust_bottle.json",
                schema_dir / "adjust_bottle.json",
            )
            planner = OfficialTaskPlanAgent(
                root,
                task_name="adjust_bottle",
                execution_backend="act",
            )
            proposal = {
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "evaluation_goal": (
                    "establish_clean_control_before_claim_first_attribution"
                ),
                "requested_aspect_ids": [
                    "task_execution.official_baseline"
                ],
                "first_aspect_id": "task_execution.official_baseline",
            }
            manifest = planner.plan(
                "evaluate adjust_bottle",
                evaluation_id="eval_adjust_bottle_control",
                validated_proposal=proposal,
            )

            self.assertEqual(
                manifest["plan"]["evaluation_goal"],
                proposal["evaluation_goal"],
            )
            self.assertEqual(
                manifest["planner"]["proposal_source"],
                "validated_route_or_control",
            )
            self.assertEqual(
                manifest["plan"]["rounds"][0]["template_id"],
                "task_execution.official_baseline",
            )

    def test_official_command_uses_expert_probe_without_act_or_codegen_vqa(self):
        command, _ = build_taskgen_command(
            Path("/repo"),
            "eval_click",
            official_round(),
            text_model="text",
            vision_model="vision",
            base_url=None,
            gpu=0,
            max_reflections=2,
            telemetry_profile="legacy_v1",
        )
        self.assertIn("official", command)
        self.assertIn("envs.click_bell", command)
        self.assertIn("legacy_v1", command)
        self.assertIn("--expert", command)
        self.assertNotIn("--run-act", command)
        self.assertNotIn("--vision-check", command)

    def test_official_command_flags_follow_execution_backend(self):
        expected = {
            "expert": {"--expert"},
            "act": {"--run-act"},
            "both": {"--expert", "--run-act"},
        }
        for backend, expected_flags in expected.items():
            with self.subTest(backend=backend):
                command, _ = build_taskgen_command(
                    Path("/repo"),
                    f"eval_click_{backend}",
                    official_round(backend),
                    text_model="text",
                    vision_model="vision",
                    base_url=None,
                    gpu=0,
                    max_reflections=2,
                )
                actual_flags = {
                    flag
                    for flag in ("--expert", "--run-act")
                    if flag in command
                }
                self.assertEqual(actual_flags, expected_flags)
                self.assertNotIn("--vision-check", command)

    def test_click_bell_vqa_query_is_saved_even_when_execution_vqa_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "mea/generated_tasks/run_click"
            execution = root / "mea/evaluation_runs/e/execution/round_1"
            child.mkdir(parents=True)
            result = run_round_execution_vqa(
                repo_root=root,
                child_manifest={"task_name": "click_bell"},
                child_dir=child,
                tool_evaluation=None,
                execution_dir=execution,
                provider=object(),
                model="vision",
                round_plan=official_round(),
            )
            query = json.loads(
                (execution / "execution_vqa_query.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(query["phenomenon_ids"], ["bell_visibly_pressed"])

    def test_open_vqa_need_materializes_run_local_question(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "mea/generated_tasks/run_click"
            execution = root / "mea/evaluation_runs/e/execution/round_1"
            child.mkdir(parents=True)
            episode = child / "evaluation/telemetry/act/episode_0"
            episode.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
                episode / "schema.json",
            )
            round_plan = official_round()
            round_plan["semantic_need_execution"] = {
                "vqa_tool": {
                    "requested": True,
                    "description": "the distractor remains untouched",
                }
            }
            round_plan["proposal"] = build_experiment_candidate(
                source_query="Does the policy avoid touching a distractor?",
                base_task="click_bell",
                semantic_concern="target discrimination near a distractor",
                vqa_tool_need="Determine whether the distractor remains untouched.",
            )
            provider = self.FrozenConcernProvider(
                {
                    "schema_version": 1,
                    "question_spec": {
                        "id": "run_local.distractor_remains_untouched",
                        "question_type": "visible_unintended_contact",
                        "target_role": "distractor",
                        "question": (
                            "Does the evidence show that the distractor "
                            "remains untouched throughout the rollout?"
                        ),
                        "visual_scope": "rollout_change",
                        "numeric_authority": "no_numeric_oracle",
                    },
                }
            )
            result = run_round_execution_vqa(
                repo_root=root,
                child_manifest={"task_name": "click_bell"},
                child_dir=child,
                tool_evaluation=None,
                execution_dir=execution,
                provider=provider,
                model="vision",
                round_plan=round_plan,
            )
            query = json.loads(
                (execution / "execution_vqa_query.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(len(query["phenomenon_ids"]), 1)
            self.assertEqual(
                query["phenomenon_ids"][0],
                "run_local.distractor_remains_untouched",
            )
            self.assertEqual(provider.calls, 1)
            self.assertIn(
                "the distractor remains untouched",
                query["questions"][0]["question"],
            )

    def test_official_summary_uses_its_declared_gates(self):
        round_plan = official_round()
        child_manifest = {
            "run_id": "run_click",
            "status": "completed_without_act",
            "scene_validation": {
                "render_success": True,
                "rule_check": {"passed": True},
                "expert": {"passed": True},
                "expert_batch": {"passed": True},
            },
            "trusted_tool_evaluation": {"episode_count": 1, "episodes": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_dir = root / "mea/generated_tasks/run_click"
            (child_dir / "evaluation").mkdir(parents=True)
            summary = summarize_round(
                round_plan,
                child_manifest,
                child_dir,
                {"status": "passed", "episodes": []},
                {"status": "passed", "metrics": []},
                {"status": "skipped", "evidence_conflict": False},
                0,
            )
            evidence = build_evidence_bundle(
                root,
                "eval_click",
                "evaluate click_bell",
                {
                    "max_rounds": 1,
                    "requested_template_ids": [round_plan["template_id"]],
                    "planning_state": "stopped_after_round_1",
                    "round_decisions": [],
                },
                [
                    {
                        "round_plan": round_plan,
                        "child_manifest": child_manifest,
                        "child_dir": child_dir,
                        "round_summary": summary,
                        "tool_evaluation": {"status": "passed"},
                    }
                ],
            )
            self.assertTrue(summary["pipeline_passed"])
            self.assertEqual(
                summary["observations"]["execution_backend"], "expert"
            )
            self.assertIsNone(summary["observations"]["act_pipeline_status"])
            self.assertIsNone(summary["observations"]["policy_success"])
            self.assertEqual(
                summary["observations"]["scene_clutter"],
                {
                    "expected": False,
                    "counts": [],
                    "all_matched": None,
                    "authority": None,
                },
            )
            self.assertEqual(
                evidence["observations"]["execution_backends"], ["expert"]
            )
            self.assertIsNone(evidence["observations"]["act_pipeline_status"])

    def test_official_act_policy_failure_is_not_pipeline_failure(self):
        round_plan = official_round("act")
        child_manifest = {
            "run_id": "run_click_act",
            "status": "completed",
            "scene_validation": {
                "render_success": True,
                "rule_check": {"passed": True},
            },
            "act_evaluation": {"passed": True, "actual_seeds": [8]},
            "trusted_tool_evaluation": {
                "episode_count": 1,
                "episodes": [],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            child_dir = Path(temporary) / "mea/generated_tasks/run_click_act"
            evaluation_dir = child_dir / "evaluation"
            evaluation_dir.mkdir(parents=True)
            (evaluation_dir / "_result.txt").write_text(
                "0.0\n", encoding="utf-8"
            )
            summary = summarize_round(
                round_plan,
                child_manifest,
                child_dir,
                {"status": "passed", "episodes": []},
                {"status": "passed", "metrics": []},
                {"status": "passed", "evidence_conflict": False},
                0,
            )

        self.assertTrue(summary["pipeline_passed"])
        self.assertEqual(summary["observations"]["execution_backend"], "ACT")
        self.assertTrue(summary["observations"]["act_pipeline_status"])
        self.assertEqual(summary["observations"]["policy_success"], 0.0)
        self.assertIsNone(summary["observations"]["expert_solvable"])
        self.assertEqual(summary["observations"]["requested_seeds"], [7])
        self.assertEqual(summary["observations"]["actual_seeds"], [8])

    def test_official_episode_index_is_forwarded_to_recorder_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click"
            scenes = [
                {
                    "setup_success": True,
                    "render_success": True,
                    "rule_check": {"passed": True},
                    "expert": {"passed": True},
                    "image": f"image-{index}",
                    "telemetry": {
                        "episode_dir": f"episode-{index}",
                        "metadata": {
                            "artifacts": {"video": "video.mp4"},
                            "visual_capture": {
                                "profile_id": "event_keyframes_v1",
                                "status": "completed",
                            },
                        },
                    },
                }
                for index in range(2)
            ]
            with patch(
                "scripts.manipeval_taskgen.run_probe",
                side_effect=scenes,
            ) as probe:
                result = run_official_expert_episodes(
                    root,
                    run_dir,
                    {"task_name": "click_bell"},
                    start_seed=10,
                    num_episodes=2,
                    telemetry_profile="balanced_v1",
                )
            self.assertTrue(result["expert_batch"]["passed"])
            self.assertEqual(
                [call.kwargs["episode_index"] for call in probe.call_args_list],
                [0, 1],
            )
            self.assertEqual(
                [
                    call.kwargs["visual_capture_profile_id"]
                    for call in probe.call_args_list
                ],
                ["event_keyframes_v1", "event_keyframes_v1"],
            )
            self.assertEqual(
                result["expert_batch"]["episodes"][0]["video"],
                str(Path("episode-0") / "video.mp4"),
            )
            self.assertEqual(result["expert_batch"]["rejected_seed_count"], 0)

    def test_official_expert_skips_unstable_seed_with_audit_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_adjust"

            def accepted_scene(index):
                return {
                    "returncode": 0,
                    "setup_success": True,
                    "render_success": True,
                    "rule_check": {"passed": True},
                    "expert": {"passed": True},
                    "image": f"image-{index}",
                    "telemetry": {
                        "episode_dir": f"episode-{index}",
                        "metadata": {
                            "artifacts": {"video": "video.mp4"},
                            "visual_capture": {"status": "completed"},
                        },
                    },
                }

            scenes = [
                {
                    "returncode": 1,
                    "error": {
                        "type": "UnStableError",
                        "message": "bottle unstable",
                    },
                },
                accepted_scene(0),
                accepted_scene(1),
            ]
            with patch(
                "scripts.manipeval_taskgen.run_probe", side_effect=scenes
            ) as probe:
                result = run_official_expert_episodes(
                    root,
                    run_dir,
                    {"task_name": "adjust_bottle"},
                    start_seed=100,
                    num_episodes=2,
                    telemetry_profile="balanced_v1",
                    max_seed_candidates=3,
                )

            self.assertEqual(
                [call.kwargs["seed"] for call in probe.call_args_list],
                [100, 101, 102],
            )
            self.assertEqual(
                [call.kwargs["episode_index"] for call in probe.call_args_list],
                [0, 0, 1],
            )
            self.assertTrue(
                all(
                    call.kwargs["raise_on_failure"] is False
                    for call in probe.call_args_list
                )
            )
            self.assertEqual(result["expert_batch"]["episode_count"], 2)
            self.assertEqual(result["expert_batch"]["candidate_count"], 3)
            self.assertEqual(result["expert_batch"]["rejected_seed_count"], 1)
            self.assertEqual(
                result["expert_batch"]["rejected_seeds"][0]["seed"], 100
            )

    def test_official_expert_skips_unsolvable_seed_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click"
            accepted = {
                "returncode": 0,
                "setup_success": True,
                "render_success": True,
                "rule_check": {"passed": True},
                "expert": {"passed": True},
                "telemetry": {"episode_dir": "expert/accepted", "metadata": {}},
            }
            with patch(
                "scripts.manipeval_taskgen.run_probe",
                side_effect=[
                    {"returncode": 2, "expert": {"passed": False}},
                    accepted,
                ],
            ) as probe:
                result = run_official_expert_episodes(
                    root,
                    run_dir,
                    {"task_name": "click_bell"},
                    start_seed=7,
                    num_episodes=1,
                    telemetry_profile="balanced_v1",
                    max_seed_candidates=2,
                )
            self.assertEqual(result["expert_batch"]["episodes"][0]["seed"], 8)
            self.assertEqual(
                result["expert_batch"]["rejected_seeds"][0]["reason"],
                "expert_unsolvable",
            )
            self.assertTrue(
                all(
                    call.kwargs["max_expert_attempts"] == 1
                    for call in probe.call_args_list
                )
            )

    def test_probe_command_forwards_visual_capture_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click"
            (run_dir / "validation").mkdir(parents=True)
            manifest = {
                "task_name": "click_bell",
                "task_module": "envs.click_bell",
            }
            with patch(
                "scripts.manipeval_taskgen.run_command", return_value=0
            ) as invoked:
                run_probe(
                    root,
                    run_dir,
                    manifest,
                    seed=7,
                    expert=True,
                    visual_capture_profile_id="event_keyframes_v1",
                )
                visual_command = invoked.call_args.args[0]
                run_probe(
                    root,
                    run_dir,
                    manifest,
                    seed=8,
                    expert=False,
                )
                default_command = invoked.call_args.args[0]
            flag_index = visual_command.index("--visual-capture-profile")
            self.assertEqual(
                visual_command[flag_index + 1], "event_keyframes_v1"
            )
            self.assertNotIn("--visual-capture-profile", default_command)

    def test_probe_command_forwards_execution_receipt_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_receipt_probe"
            (run_dir / "validation").mkdir(parents=True)
            receipt = run_dir / "expert_receipt.json"
            manifest = {
                "task_name": "beat_block_hammer",
                "task_module": "envs.beat_block_hammer",
            }
            with patch(
                "scripts.manipeval_taskgen.run_command", return_value=0
            ) as invoked:
                run_probe(
                    root,
                    run_dir,
                    manifest,
                    seed=7,
                    expert=True,
                    execution_receipt=receipt,
                )
            command = invoked.call_args.args[0]
            flag_index = command.index("--execution-receipt")
            self.assertEqual(command[flag_index + 1], str(receipt))

    def test_probe_failure_surfaces_simulator_validation_diagnosis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_failed_probe"
            (run_dir / "validation").mkdir(parents=True)
            manifest = {
                "task_name": "click_bell",
                "task_module": "generated.click_bell",
            }

            def fail_with_diagnosis(command, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "error": {
                                "type": "RecorderError",
                                "message": "runtime actor identity mismatch",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return 1

            with patch(
                "scripts.manipeval_taskgen.run_command",
                side_effect=fail_with_diagnosis,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "RecorderError: runtime actor identity mismatch",
                ):
                    run_probe(
                        root,
                        run_dir,
                        manifest,
                        seed=7,
                        expert=False,
                    )

    def test_checker_repair_uses_simulator_heights_not_only_returncode(self):
        setup = {
            "tracked_actors": [
                {"id": "target", "position": [0.0, 0.0, 0.741]},
                {"id": "distractor", "position": [0.1, 0.0, 0.742]},
            ]
        }
        expert = {
            "tracked_actors": [
                {"id": "target", "position": [0.0, 0.0, 0.741]},
                {"id": "distractor", "position": [0.1, 0.0, 0.742]},
            ],
            "expert_terminal_tracked_actors": [
                {"id": "target", "position": [0.0, 0.0, 0.831]},
                {"id": "distractor", "position": [0.1, 0.0, 0.743]},
            ]
        }
        fixtures = [
            {
                "fixture_id": "simulator_initial_negative",
                "expected": False,
                "observed": False,
                "passed": True,
            },
            {
                "fixture_id": "official_expert_terminal_positive",
                "expected": True,
                "observed": False,
                "passed": False,
            },
        ]

        self.assertEqual(
            _tracked_actor_heights(setup),
            {"target": 0.741, "distractor": 0.742},
        )
        diagnosis = _checker_fixture_failure_diagnosis(
            fixtures,
            setup=setup,
            expert=expert,
            success_contract={
                "target_actor_id": "target",
                "minimum_height_m": 0.8,
            },
        )
        self.assertIn("official_expert_terminal_positive", diagnosis)
        self.assertIn('"target": 0.831', diagnosis)
        self.assertIn('"distractor": 0.743', diagnosis)
        self.assertIn('"minimum_height_m": 0.8', diagnosis)
        self.assertNotIn("repair_rule", diagnosis)

    def test_expert_terminal_authority_separates_scene_failure_from_checker_repair(
        self,
    ):
        terminal_state = {
            "tracked_actors": [
                {"id": "roller", "position": [0.0, 0.0, 0.741]},
            ],
            "expert_terminal_tracked_actors": [
                {"id": "roller", "position": [0.0, 0.0, 0.781]},
            ],
            "expert": {
                "plan_success": True,
                "check_success": False,
                "official_core_predicate_satisfied": False,
            },
        }
        failure = _expert_terminal_authority_failure(terminal_state)
        self.assertEqual(
            failure["reason"],
            "official_success_false_after_expert_plan",
        )
        self.assertEqual(
            failure["expert_terminal_actor_z_m"],
            {"roller": 0.781},
        )
        self.assertEqual(
            failure["repair_scope"],
            "scene_or_expert_plan_not_checker_only",
        )

        terminal_state["expert"]["official_core_predicate_satisfied"] = True
        self.assertIsNone(_expert_terminal_authority_failure(terminal_state))

        execution_error = _expert_terminal_authority_failure(
            {
                "expert": {"attempts_used": 3},
                "error": {
                    "type": "AssertionError",
                    "message": "target_pose cannot be None",
                },
            }
        )
        self.assertEqual(execution_error["reason"], "expert_execution_error")
        self.assertEqual(
            execution_error["repair_scope"],
            "scene_or_expert_plan_not_checker_only",
        )

    def test_act_wrapper_receives_telemetry_profile_as_twelfth_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_act_profile"
            (run_dir / "evaluation").mkdir(parents=True)
            checkpoint_dir = (
                root
                / "policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            with patch(
                "scripts.manipeval_taskgen.run_command", return_value=0
            ) as invoked:
                with self.assertRaises(RuntimeError):
                    run_act(
                        root,
                        run_dir,
                        {
                            "task_name": "beat_block_hammer",
                            "task_module": "mea.tasks.beat_block_hammer",
                        },
                        seed=7,
                        gpu=0,
                        num_episodes=1,
                        telemetry_profile="legacy_v1",
                    )
            command = invoked.call_args.args[0]
            self.assertEqual(command[-1], "legacy_v1")

    def test_act_wrapper_places_execution_receipt_in_sixteenth_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_act_receipt"
            (run_dir / "evaluation").mkdir(parents=True)
            checkpoint_dir = (
                root
                / "policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            receipt = run_dir / "act_execution_receipt.json"
            with patch(
                "scripts.manipeval_taskgen.run_command", return_value=0
            ) as invoked:
                with self.assertRaises(RuntimeError):
                    run_act(
                        root,
                        run_dir,
                        {
                            "task_name": "beat_block_hammer",
                            "task_module": "mea.tasks.beat_block_hammer",
                        },
                        seed=7,
                        gpu=0,
                        num_episodes=1,
                        telemetry_profile="legacy_v1",
                        execution_receipt=receipt,
                    )
            command = invoked.call_args.args[0]
            shell_args = command[4:]
            self.assertEqual(shell_args[15], str(receipt))
            self.assertEqual(shell_args[12:15], ["", "", ""])

    def test_act_wrapper_uses_click_bell_checkpoint_and_eval_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click_act"
            (run_dir / "evaluation").mkdir(parents=True)
            checkpoint_dir = (
                root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            eval_root = root / "eval_result/click_bell/ACT/demo_clean/demo_clean"
            telemetry_episode = (
                run_dir / "evaluation/telemetry/act/episode_000_seed_7"
            )

            def fake_run(command, *, cwd, log_path):
                self.assertEqual(cwd, root)
                output = eval_root / "mock_eval"
                output.mkdir(parents=True)
                (output / "episode0.mp4").write_bytes(b"video")
                (output / "_result.txt").write_text(
                    "1.0\n", encoding="utf-8"
                )
                telemetry_episode.mkdir(parents=True)
                (telemetry_episode / "episode.json").write_text(
                    json.dumps({"seed": 7}), encoding="utf-8"
                )
                return 0

            with patch(
                "scripts.manipeval_taskgen.run_command",
                side_effect=fake_run,
            ) as invoked:
                result = run_act(
                    root,
                    run_dir,
                    {
                        "task_name": "click_bell",
                        "task_module": "envs.click_bell",
                    },
                    seed=7,
                    gpu=0,
                    num_episodes=1,
                    telemetry_profile="legacy_v1",
                )

            command = invoked.call_args.args[0]
            self.assertEqual(
                command[4:10],
                ["click_bell", "demo_clean", "demo_clean", "50", "0", "0"],
            )
            self.assertEqual(command[11], "envs.click_bell")
            self.assertEqual(command[-1], "legacy_v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["task_name"], "click_bell")
            self.assertEqual(result["actual_seeds"], [7])
            self.assertTrue(result["checkpoint"]["preflight_passed"])
            metadata = json.loads(
                (telemetry_episode / "episode.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["artifacts"]["video"], "video.mp4")

    def test_act_checkpoint_preflight_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click_act"
            (run_dir / "evaluation").mkdir(parents=True)
            with patch("scripts.manipeval_taskgen.run_command") as invoked:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "ACT checkpoint preflight failed for click_bell",
                ):
                    run_act(
                        root,
                        run_dir,
                        {
                            "task_name": "click_bell",
                            "task_module": "envs.click_bell",
                        },
                        seed=7,
                        gpu=0,
                        num_episodes=1,
                    )
            invoked.assert_not_called()

    def test_act_video_association_uses_numeric_episode_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click_act"
            (run_dir / "evaluation").mkdir(parents=True)
            checkpoint_dir = (
                root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            eval_root = root / "eval_result/click_bell/ACT/demo_clean/demo_clean"

            def fake_run(command, *, cwd, log_path):
                output = eval_root / "mock_eval"
                output.mkdir(parents=True)
                (output / "episode2.mp4").write_bytes(b"video-two")
                (output / "episode10.mp4").write_bytes(b"video-ten")
                (output / "_result.txt").write_text("0.5\n", encoding="utf-8")
                for index, seed in ((2, 22), (10, 110)):
                    episode = (
                        run_dir
                        / "evaluation/telemetry/act"
                        / f"episode_{index:03d}_seed_{seed}"
                    )
                    episode.mkdir(parents=True)
                    (episode / "episode.json").write_text(
                        json.dumps({"seed": seed}), encoding="utf-8"
                    )
                return 0

            with patch(
                "scripts.manipeval_taskgen.run_command",
                side_effect=fake_run,
            ):
                result = run_act(
                    root,
                    run_dir,
                    {
                        "task_name": "click_bell",
                        "task_module": "envs.click_bell",
                    },
                    seed=7,
                    gpu=0,
                    num_episodes=2,
                )
            self.assertTrue(result["episode_index_alignment"]["passed"])
            self.assertEqual(result["actual_seeds"], [22, 110])
            self.assertEqual(
                (
                    run_dir
                    / "evaluation/telemetry/act/episode_002_seed_22/video.mp4"
                ).read_bytes(),
                b"video-two",
            )
            self.assertEqual(
                (
                    run_dir
                    / "evaluation/telemetry/act/episode_010_seed_110/video.mp4"
                ).read_bytes(),
                b"video-ten",
            )


if __name__ == "__main__":
    unittest.main()
