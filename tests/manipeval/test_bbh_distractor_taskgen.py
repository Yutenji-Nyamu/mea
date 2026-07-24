"""Tests for the bounded Fig. 3/4 target/distractor candidate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mea.taskgen.bbh_distractor import (
    BBHDistractorTaskGenError,
    bbh_distractor_rollout_execution,
    default_bbh_distractor_proposal,
    materialize_bbh_distractor_candidate,
    reference_bbh_distractor_methods,
    run_bbh_distractor_checker_fixtures,
    validate_bbh_distractor_methods,
    validate_bbh_distractor_proposal,
)
from mea.capability_adapter import resolve_capability_contract
from mea.proposals import task_proposal_from_contract
from mea.taskgen.production_acceptance import (
    record_production_task_acceptance,
    require_production_task_acceptance,
    require_task_artifact_act_runtime_eligible,
)
from scripts.manipeval_taskgen import (
    create_bbh_distractor_taskgen_run,
    evaluate_run_telemetry,
    prepare_planner_capability_binding,
    run_visual_self_reflection,
    validate_planner_capability_binding,
)
from mea.toolgen.query_induced import (
    QueryInducedToolError,
    query_induced_result_to_tool_execution,
)
from mea.toolkit.aggregate import aggregate_tool_executions


class _Provider:
    def __init__(self, response: dict[str, str]) -> None:
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []
        self.last_metadata = {"id": "fixture-codegen-call"}

    def text(self, prompt: str, **_kwargs: object) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return json.dumps(self.response)


class _FencedProvider(_Provider):
    def text(self, prompt: str, **kwargs: object) -> str:
        value = super().text(prompt, **kwargs)
        return f"```json\n{value}\n```"


class _SequenceProvider:
    def __init__(self, responses: list[dict[str, str]]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.last_metadata: dict[str, object] = {}

    def text(self, _prompt: str, **_kwargs: object) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        self.last_metadata = {"attempt": self.calls}
        return json.dumps(response)


class _VisualRepairProvider:
    def __init__(self, methods: dict[str, str]) -> None:
        self.methods = methods
        self.text_calls = 0
        self.vision_calls = 0
        self.text_prompts: list[str] = []
        self.vision_prompts: list[str] = []
        self.last_metadata: dict[str, object] = {}

    def text(self, prompt: str, **_kwargs: object) -> str:
        self.text_calls += 1
        self.text_prompts.append(prompt)
        self.last_metadata = {
            "stage": "text",
            "call": self.text_calls,
        }
        return json.dumps(self.methods)

    def vision(
        self,
        prompt: str,
        _image_path: Path,
        **_kwargs: object,
    ) -> str:
        self.vision_calls += 1
        self.vision_prompts.append(prompt)
        self.last_metadata = {
            "stage": "vision",
            "call": self.vision_calls,
        }
        return json.dumps(
            {
                "aligned": self.vision_calls > 1,
                "target_actor": "block",
                "target_visible": True,
                "lookalike_distractor_visible": self.vision_calls > 1,
                "scene_physically_plausible": True,
                "unexpected_changes": (
                    [] if self.vision_calls > 1 else ["distractor not visible"]
                ),
                "diagnosis": (
                    "Both intended blocks are visible."
                    if self.vision_calls > 1
                    else "Only the target block is visible."
                ),
                "suggestions": (
                    [] if self.vision_calls > 1 else ["Regenerate both methods."]
                ),
                "confidence": 0.9,
            }
        )


def _episode(
    root: Path,
    *,
    task_module: str,
    success: bool = True,
) -> Path:
    episode = root / "episode_seed_17"
    episode.mkdir(parents=True)
    count = 3
    vector = np.zeros((count, 3), dtype=float)
    trace = {
        "physics_step": np.arange(count, dtype=np.int64),
        "policy_step": np.arange(count, dtype=np.int64),
        "simulation_time_seconds": np.arange(count, dtype=float) * 0.02,
        "success": np.asarray([False, False, success], dtype=bool),
        "hammer_position": vector,
        "block_position": vector,
        "hammer_functional_position": vector,
        "block_functional_position": vector,
        "left_tcp_position": vector,
        "right_tcp_position": vector,
    }
    np.savez(episode / "semantic_trace.npz", **trace)
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "task_name": "beat_block_hammer",
                "task_module": task_module,
                "physics_steps": count - 1,
                "semantic_trace_rows": count,
                "success": success,
                "policy_name": "ACT",
                "seed": 17,
            }
        ),
        encoding="utf-8",
    )
    (episode / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": "beat_block_hammer",
                "signals": sorted(trace),
            }
        ),
        encoding="utf-8",
    )
    (episode / "states.csv").write_text(
        "policy_step\n0\n1\n2\n", encoding="utf-8"
    )
    events = (
        [
            {
                "type": "success_transition",
                "physics_step": 2,
                "policy_step": 2,
                "simulation_time_seconds": 0.04,
            }
        ]
        if success
        else []
    )
    (episode / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
    )
    return episode


def _jerk_proposal() -> dict[str, object]:
    return {
        "schema_version": 2,
        "metric_id": "precontact_tcp_jerk_peak",
        "finite_difference_order": 3,
        "window_seconds": 0.12,
        "reducer": "peak_l2",
        "time_normalization": "physical_seconds",
        "threshold": 50.0,
        "unit": "m_per_second_cubed",
        "null_semantics": (
            "null_if_no_target_contact_or_insufficient_precontact_samples"
        ),
        "rationale": "query asks about abrupt pre-contact motion",
    }


def _query_result(value: float | None, reason: str | None) -> dict[str, object]:
    return {
        "schema_version": 2,
        "query": "Is there abrupt motion before target contact?",
        "route": "provider_generate_validate_register",
        "proposal": _jerk_proposal(),
        "registration_id": "run_local_fixture",
        "live_telemetry": {
            "episode_dir": "/recorded/episode_seed_17",
            "task_name": "beat_block_hammer",
            "synthetic_fallback_used": False,
        },
        "tool_result": {
            "tool": "precontact_tcp_jerk_peak",
            "value": value,
            "unit": "m_per_second_cubed",
            "passed": value is not None and value <= 50.0,
            "null_semantics": (
                "null_if_no_target_contact_or_insufficient_precontact_samples"
            ),
            "null_reason": reason,
            "active_arm": "left",
            "signal": "left_tcp_position",
            "first_target_contact_trace_index": (
                9 if value is not None else None
            ),
            "evidence_steps": [9] if value is not None else [],
        },
    }


class BBHDistractorTaskGenTests(unittest.TestCase):
    def test_proposal_ast_policy_and_semantic_fixtures_are_fail_closed(self) -> None:
        proposal = default_bbh_distractor_proposal()
        validated = validate_bbh_distractor_proposal(proposal)
        methods = reference_bbh_distractor_methods(validated)
        report = validate_bbh_distractor_methods(methods, validated)
        self.assertTrue(report["valid"])
        self.assertTrue(report["model_written_python"])
        self.assertFalse(report["restricted_success_spec_compiler_used"])

        too_far = default_bbh_distractor_proposal()
        too_far["scene"]["distractor_offset_xy_m"] = [0.3, 0.0]
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError, "separation"
        ):
            validate_bbh_distractor_proposal(too_far)

        unsafe = dict(methods)
        unsafe["check_success"] = "import os\n" + unsafe["check_success"]
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError, "exactly one function"
        ):
            validate_bbh_distractor_methods(unsafe, validated)

        accepts_distractor = dict(methods)
        accepts_distractor["check_success"] = accepts_distractor[
            "check_success"
        ].replace("and not self._mea_distractor_contact_seen", "")
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError, "semantic fixtures failed"
        ):
            validate_bbh_distractor_methods(accepts_distractor, validated)

        moved_hammer = dict(methods)
        moved_hammer["load_actors"] = moved_hammer["load_actors"].replace(
            "sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105])",
            "sapien.Pose()",
        )
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError, "official hammer"
        ):
            validate_bbh_distractor_methods(moved_hammer, validated)

        malformed_pose = dict(methods)
        malformed_pose["load_actors"] = malformed_pose["load_actors"].replace(
            "sapien.Pose(block_pose.p + distractor_offset, block_pose.q)",
            "sapien.Pose([0.1], block_pose.q)",
        )
        with self.assertRaisesRegex(
            BBHDistractorTaskGenError,
            "semantic fixture raised",
        ):
            validate_bbh_distractor_methods(malformed_pose, validated)

    def test_structurally_different_semantically_equivalent_code_is_accepted(self) -> None:
        proposal = default_bbh_distractor_proposal()
        methods = reference_bbh_distractor_methods(proposal)
        methods["load_actors"] = methods["load_actors"].replace(
            "distractor_offset = np.array([\n        0.1,\n        0.0,\n        0.0,\n    ])",
            "distractor_offset = np.asarray((0.1, 0.0, 0.0))",
        ).replace(
            "half_size=(0.025, 0.025, 0.025)",
            "half_size=[0.025, 0.025, 0.025]",
        ).replace(
            "self._mea_target_contact_seen",
            "self.target_contact_latched",
        ).replace(
            "self._mea_distractor_contact_seen",
            "self.distractor_contact_latched",
        )
        methods["check_success"] = """
def check_success(self):
    hit_target = self.check_actors_contact(
        self.hammer.get_name(), self.block.get_name()
    )
    hit_other = self.check_actors_contact(
        self.hammer.get_name(), self.distractor.get_name()
    )
    if hit_target:
        self.target_contact_latched = True
    if hit_other:
        self.distractor_contact_latched = True
    tip_xy = self.hammer.get_functional_point(0, "pose").p[:2]
    target_xy = self.block.get_functional_point(1, "pose").p[:2]
    within_target = bool(
        np.all(np.abs(tip_xy - target_xy) < np.asarray((0.025, 0.025)))
    )
    return (
        within_target
        and bool(self.target_contact_latched)
        and not bool(self.distractor_contact_latched)
    )
"""
        report = validate_bbh_distractor_methods(methods, proposal)
        self.assertEqual(
            report["policy"],
            "bbh_distractor_safe_ast_semantic_fixtures_v2",
        )
        reference = reference_bbh_distractor_methods(proposal)
        self.assertNotEqual(
            report["success_sha256"],
            validate_bbh_distractor_methods(reference, proposal)[
                "success_sha256"
            ],
        )

    def test_checker_fixtures_include_latched_contacts_and_alignment(self) -> None:
        proposal = default_bbh_distractor_proposal()
        methods = reference_bbh_distractor_methods(proposal)
        fixtures = run_bbh_distractor_checker_fixtures(
            methods["check_success"], proposal
        )
        self.assertEqual(
            [item["fixture"] for item in fixtures],
            [
                "target_contact",
                "target_contact_latched",
                "distractor_contact_latched",
                "no_contact",
                "misaligned_target_contact",
                "z_offset_target_contact",
            ],
        )
        self.assertTrue(all(item["passed"] for item in fixtures))
        self.assertEqual(fixtures[1]["calls"], [True, True])
        self.assertEqual(fixtures[2]["calls"], [False, False])

    def test_model_provenance_rollout_binding_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = default_bbh_distractor_proposal()
            provider = _FencedProvider(
                reference_bbh_distractor_methods(proposal)
            )
            manifest = materialize_bbh_distractor_candidate(
                repo_root=root,
                run_id="run_fixture_distractor",
                proposal=proposal,
                provider=provider,
                model="fixture-model",
            )
            self.assertEqual(provider.calls, 1)
            self.assertIn(
                '"proposal_id": "bbh.lookalike_distractor.v1"',
                provider.prompts[0],
            )
            self.assertNotIn("def load_actors(self):", provider.prompts[0])
            self.assertNotIn("structural reference", provider.prompts[0])
            self.assertIn(
                "base task has no self.create_actor",
                provider.prompts[0],
            )
            self.assertIn(
                "Actors have no get_contacts method",
                provider.prompts[0],
            )
            self.assertIn(
                "immutable official hammer contract",
                provider.prompts[0],
            )
            self.assertIn("is_static=True", provider.prompts[0])
            provenance = manifest["codegen_provenance"]
            self.assertEqual(
                provenance["source_kind"], "provider_response_python"
            )
            self.assertTrue(provenance["generated_by_model"])
            self.assertFalse(
                provenance["restricted_success_spec_compiler_used"]
            )
            self.assertEqual(
                manifest["live_boundary"]["act_rollouts_completed"], 0
            )
            candidate = (
                root
                / "mea/generated_tasks/run_fixture_distractor"
            )
            episode = _episode(
                root,
                task_module=manifest["task_module"],
            )
            execution = bbh_distractor_rollout_execution(
                episode_dir=episode,
                candidate_dir=candidate,
            )
            aggregate = aggregate_tool_executions([execution])
            metric = aggregate["metrics"][0]
            self.assertEqual(
                metric["metric"],
                "bbh_target_without_distractor_success",
            )
            cohort = metric["cohorts"][0]
            self.assertEqual(
                cohort["summary"]["statistics"]["true_rate"]["value"],
                1.0,
            )
            details = execution["episodes"][0]["result"]["details"]
            self.assertFalse(details["official_success"])
            self.assertEqual(
                details["authority"],
                "llm_generated_python_ast_validated",
            )

            wrong = _episode(
                root / "wrong",
                task_module="envs.beat_block_hammer",
            )
            with self.assertRaisesRegex(
                BBHDistractorTaskGenError, "task_module differs"
            ):
                bbh_distractor_rollout_execution(
                    episode_dir=wrong,
                    candidate_dir=candidate,
                )

            metadata_path = episode / "episode.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["success"] = 1
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                BBHDistractorTaskGenError,
                "JSON boolean",
            ):
                bbh_distractor_rollout_execution(
                    episode_dir=episode,
                    candidate_dir=candidate,
                )

    def test_final_codegen_failure_preserves_attempt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = default_bbh_distractor_proposal()
            invalid = reference_bbh_distractor_methods(proposal)
            invalid["check_success"] = (
                "def check_success(self):\n"
                "    return bool(self.check_actors_contact("
                "self.hammer.get_name(), self.distractor.get_name()))\n"
            )
            provider = _SequenceProvider([invalid, invalid])
            with self.assertRaises(BBHDistractorTaskGenError):
                materialize_bbh_distractor_candidate(
                    repo_root=root,
                    run_id="run_failed_distractor",
                    proposal=proposal,
                    provider=provider,
                    model="fixture-model",
                    max_regenerations=1,
                )
            run_dir = root / "mea/generated_tasks/run_failed_distractor"
            failure = json.loads(
                (run_dir / "failure_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                failure["status"], "codegen_validation_failed"
            )
            self.assertEqual(failure["provider_call_count"], 2)
            self.assertEqual(failure["act_rollouts_completed"], 0)
            self.assertTrue((run_dir / "proposal_prompt.md").is_file())
            self.assertTrue(
                (
                    run_dir
                    / "provider_attempts/attempt_01_response.txt"
                ).is_file()
            )
            self.assertTrue(
                (
                    run_dir
                    / "provider_attempts/attempt_02_response.txt"
                ).is_file()
            )
            self.assertFalse((run_dir / "task.py").exists())

    def test_one_local_regeneration_and_standard_taskgen_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "envs/beat_block_hammer.py",
                "policy/ACT/eval.sh",
                "script/eval_policy.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture\n", encoding="utf-8")
            contract = resolve_capability_contract(
                "beat_block_hammer",
                "robustness.distractor_avoidance.lookalike",
            )
            public_proposal = task_proposal_from_contract(
                contract,
                intent="test target selection with a physical distractor",
            )
            _, spec = prepare_planner_capability_binding(
                contract,
                task_name="beat_block_hammer",
                mode="provider_scene_checker_codegen",
                variant_id=public_proposal["proposal_id"],
                task_proposal=public_proposal,
            )
            self.assertIsNotNone(spec)
            bounded = default_bbh_distractor_proposal()
            valid = reference_bbh_distractor_methods(bounded)
            invalid = dict(valid)
            invalid["check_success"] = (
                "def check_success(self):\n"
                "    return bool(self.check_actors_contact("
                "self.hammer.get_name(), self.distractor.get_name()))\n"
            )
            provider = _SequenceProvider([invalid, valid])
            manifest = create_bbh_distractor_taskgen_run(
                root,
                user_request="Can ACT avoid a look-alike distractor?",
                provider=provider,
                model="fixture-model",
                variant_spec=spec,
                task_proposal=public_proposal,
                run_id="run_standard_distractor",
            )
            self.assertEqual(provider.calls, 2)
            self.assertEqual(
                manifest["mode"], "provider_scene_checker_codegen"
            )
            self.assertEqual(
                manifest["provider"]["local_regeneration_count"], 1
            )
            run_dir = root / "mea/generated_tasks/run_standard_distractor"
            binding = validate_planner_capability_binding(
                contract,
                task_name="beat_block_hammer",
                mode="provider_scene_checker_codegen",
                variant_id=public_proposal["proposal_id"],
                run_dir=run_dir,
                task_proposal=public_proposal,
            )
            self.assertEqual(binding["status"], "passed")
            self.assertEqual(
                (run_dir / "overlay.yml").read_text(encoding="utf-8"),
                "{}\n",
            )
            bundle = json.loads(
                (
                    run_dir / "generation/task_artifact_bundle.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                bundle["scene_method"]["origin"],
                "provider_generated_code",
            )
            self.assertEqual(
                bundle["success_method"]["origin"],
                "provider_generated_python",
            )
            self.assertFalse(bundle["success_semantics"]["preserved"])
            self.assertTrue(
                bundle["success_semantics"]["generated_by_model"]
            )
            require_task_artifact_act_runtime_eligible(run_dir, manifest)
            acceptance = record_production_task_acceptance(
                run_dir,
                manifest,
                scene={
                    "setup_success": True,
                    "render_success": True,
                    "rule_check": {"passed": True},
                    "expert": {"passed": True},
                },
                position_samples={"passed": True},
                require_expert=True,
            )
            self.assertEqual(acceptance["status"], "accepted")
            require_production_task_acceptance(
                run_dir,
                manifest,
                for_act=True,
            )
            _episode(
                run_dir / "evaluation/telemetry/act",
                task_module=manifest["task_module"],
            )
            checker_execution = evaluate_run_telemetry(
                root,
                run_dir,
                manifest,
            )
            self.assertEqual(
                checker_execution["outcome_metric"],
                "bbh_target_without_distractor_success",
            )
            self.assertEqual(
                checker_execution["aggregate"]["metrics"][0]["metric"],
                "bbh_target_without_distractor_success",
            )

    def test_visual_failure_regenerates_both_methods_once_and_refreshes_bundle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "envs/beat_block_hammer.py",
                "policy/ACT/eval.sh",
                "script/eval_policy.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# protected fixture\n", encoding="utf-8")
            contract = resolve_capability_contract(
                "beat_block_hammer",
                "robustness.distractor_avoidance.lookalike",
            )
            public_proposal = task_proposal_from_contract(
                contract,
                intent="find whether ACT confuses a physical lookalike",
            )
            _, spec = prepare_planner_capability_binding(
                contract,
                task_name="beat_block_hammer",
                mode="provider_scene_checker_codegen",
                variant_id=public_proposal["proposal_id"],
                task_proposal=public_proposal,
            )
            self.assertIsNotNone(spec)
            methods = reference_bbh_distractor_methods(
                default_bbh_distractor_proposal()
            )
            provider = _VisualRepairProvider(methods)
            manifest = create_bbh_distractor_taskgen_run(
                root,
                user_request="Where does ACT confuse the target and distractor?",
                provider=provider,
                model="fixture-text",
                variant_spec=spec,
                task_proposal=public_proposal,
                run_id="run_visual_distractor",
            )
            run_dir = root / "mea/generated_tasks/run_visual_distractor"

            def fake_probe(
                _repo_root: Path,
                _run_dir: Path,
                _manifest: dict[str, object],
                **kwargs: object,
            ) -> dict[str, object]:
                scene_path = Path(kwargs["scene_json"])
                image_path = Path(kwargs["image"])
                scene_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                scene = {
                    "setup_success": True,
                    "render_success": True,
                    "rule_check": {"passed": True},
                    "returncode": 0,
                }
                scene_path.write_text(json.dumps(scene), encoding="utf-8")
                image_path.write_bytes(b"fixture-png")
                return scene

            with patch(
                "scripts.manipeval_taskgen.run_probe",
                side_effect=fake_probe,
            ):
                summary, _scene, vision = run_visual_self_reflection(
                    root,
                    run_dir,
                    manifest,
                    provider,
                    seed=17,
                    text_model="fixture-text",
                    vision_model="fixture-vision",
                    max_repairs=2,
                )

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["repairs_used"], 1)
            self.assertEqual(summary["repair_limit"], 1)
            self.assertEqual(provider.text_calls, 2)
            self.assertEqual(provider.vision_calls, 2)
            self.assertTrue(vision["target_visible"])
            self.assertTrue(vision["lookalike_distractor_visible"])
            self.assertIn("TASK PROPOSAL", provider.vision_prompts[0])
            self.assertIn(
                "lookalike_distractor_visible",
                provider.vision_prompts[0],
            )
            self.assertIn(
                "Do not infer exact actor identity",
                provider.vision_prompts[0],
            )
            repair_prompt = (
                run_dir / "reflection/attempt_00/repair_prompt.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Regenerate both complete methods", repair_prompt)
            self.assertIn("TASK PROPOSAL", repair_prompt)
            self.assertTrue(
                (run_dir / "reflection/attempt_00/render.png").is_file()
            )
            self.assertTrue(
                (run_dir / "reflection/attempt_01/render.png").is_file()
            )
            candidate = json.loads(
                (run_dir / "candidate_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                candidate["codegen_provenance"][
                    "visual_regeneration_count"
                ],
                1,
            )
            bundle = json.loads(
                (
                    run_dir / "generation/task_artifact_bundle.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                bundle["scene_method"]["source_sha256"],
                summary["final_scene_source_sha256"],
            )
            self.assertTrue(bundle["scene_method"]["symbol_declared"])
            self.assertTrue(bundle["success_method"]["symbol_declared"])

    def test_query_tool_bridge_preserves_numeric_and_null_evidence(self) -> None:
        numeric = query_induced_result_to_tool_execution(
            _query_result(12.5, None),
            seed=17,
        )
        missing = query_induced_result_to_tool_execution(
            _query_result(None, "no_target_contact_event"),
            seed=18,
        )
        aggregate = aggregate_tool_executions([numeric, missing])
        cohort = aggregate["metrics"][0]["cohorts"][0]
        self.assertEqual(cohort["summary"]["quality"]["valid"]["value"], 1)
        self.assertEqual(cohort["summary"]["quality"]["missing"]["value"], 1)
        self.assertEqual(
            cohort["summary"]["statistics"]["mean"]["value"], 12.5
        )
        self.assertEqual(
            missing["episodes"][0]["result"]["details"]["reason"],
            "no_target_contact_event",
        )
        self.assertFalse(
            missing["episodes"][0]["result"]["details"][
                "synthetic_fallback_used"
            ]
        )

        invalid = _query_result(None, "no_target_contact_event")
        invalid["live_telemetry"]["synthetic_fallback_used"] = True
        with self.assertRaisesRegex(
            QueryInducedToolError, "real recorded telemetry"
        ):
            query_induced_result_to_tool_execution(invalid)


if __name__ == "__main__":
    unittest.main()
