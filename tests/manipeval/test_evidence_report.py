import json
import re
import tempfile
import unittest
from pathlib import Path

from mea.feedback import EvidenceReportError, write_evidence_report


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


class EvidenceReportTests(unittest.TestCase):
    def test_publish_bundle_renders_real_code_images_video_and_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_id = "eval_fixture"
            evaluation = root / "mea/evaluation_runs" / evaluation_id
            rounds = []
            summaries = []
            child_ids = []
            for index, route in enumerate(("force_codegen", "reuse"), start=1):
                round_id = f"round_{index}"
                child_id = f"run_fixture_{round_id}"
                child_ids.append(child_id)
                child = root / "mea/generated_tasks" / child_id
                child.mkdir(parents=True)
                if route == "force_codegen":
                    (child / "task.py").write_text(
                        "class GeneratedTask:\n    pass\n", encoding="utf-8"
                    )
                    generation_kind = "python_codegen"
                else:
                    (child / "overlay.yml").write_text(
                        "mea:\n  enabled: true\n", encoding="utf-8"
                    )
                    generation_kind = "bounded_variant_overlay"
                (child / "variant_spec.json").write_text("{}", encoding="utf-8")
                (child / "evidence").mkdir()
                (child / "evidence/initial_head.png").write_bytes(
                    b"\x89PNG\r\n\x1a\nfixture"
                )
                (child / "evidence/scene_comparison.png").write_bytes(
                    b"\x89PNG\r\n\x1a\ncomparison"
                )
                for relative, value in {
                    "generation/code_prompt.md": "generate the bounded scene",
                    "generation/provider_response.txt": "```python\npass\n```\n",
                    "validation/vision_prompt.md": "inspect the first frame",
                    "validation/vision_response.txt": "scene is valid\n",
                    "debug.log": "not part of the public bundle",
                }.items():
                    _write_text(child / relative, value)
                for relative, value in {
                    "generation/experiment_candidate.json": {
                        "candidate_id": f"candidate_{index}"
                    },
                    "validation/static.json": {"passed": True},
                    "validation/checker_fixtures.json": {"passed": True},
                    "validation/implementation_trace.json": {
                        "implementation": "direct",
                        "complete": True,
                    },
                    "validation/vision.json": {"passed": True},
                }.items():
                    _write_json(child / relative, value)
                (child / "evaluation").mkdir()
                (child / "evaluation/episode0.mp4").write_bytes(b"fixture-video")
                _write_json(
                    child / "manifest.json",
                    {"run_id": child_id, "generation_kind": generation_kind},
                )
                task_proposal = {
                    "schema_version": 1,
                    "proposal_id": f"proposal_{index}",
                    "task_name": "beat_block_hammer",
                    "aspect_id": "object_appearance.color",
                    "intent": "test appearance",
                    "capability_id": "object_appearance.color",
                    "reuse_first": True,
                    "changes": {"block": {"color": [0, 1, 0]}},
                    "preserve_success_semantics": True,
                }
                tool_proposal = {
                    "schema_version": 1,
                    "proposal_id": f"proposal_{index}.tool",
                    "task_name": "beat_block_hammer",
                    "aspect_id": "object_appearance.color",
                    "evaluation_goal": "measure contact",
                    "metric": "hammer_block_contact_ever",
                    "question": "Did contact occur?",
                    "vqa_phenomenon_ids": ["block_color_blue"],
                    "reuse_first": True,
                }
                rounds.append(
                    {
                        "round_id": round_id,
                        "template_id": f"template_{index}",
                        "sub_aspect": "object_appearance.color",
                        "task_name": "beat_block_hammer",
                        "task_instruction": "test one appearance",
                        "route": route,
                        "execution": {"seeds": [100000], "num_episodes": 1},
                        "task_proposal": task_proposal,
                        "tool_proposal": tool_proposal,
                    }
                )
                summaries.append(
                    {
                        "round_id": round_id,
                        "taskgen_run_id": child_id,
                        "pipeline_passed": True,
                        "observations": {
                            "execution_backend": "ACT",
                            "policy_success": 1.0 if index == 2 else 0.0,
                        },
                    }
                )
                execution = evaluation / "execution" / round_id
                (execution / "execution_vqa").mkdir(parents=True)
                (execution / "execution_vqa/execution_montage.png").write_bytes(
                    b"\x89PNG\r\n\x1a\nmontage"
                )
                planned_tool = execution / "planned_tool"
                generated_tool = planned_tool / "generated"
                generated_tool.mkdir(parents=True)
                tool_source = generated_tool / "generated_tool.py"
                tool_source.write_text(
                    "def evaluate(x):\n    return x\n", encoding="utf-8"
                )
                tool_files = {
                    "tool_request": planned_tool / "tool_request.json",
                    "route_decision": planned_tool / "route_decision.json",
                    "registration": generated_tool / "registration.json",
                    "property_validation": (
                        generated_tool / "property_validation.json"
                    ),
                }
                for name, path in tool_files.items():
                    _write_json(path, {"artifact": name, "passed": True})
                tool_artifacts = {
                    name: path.relative_to(root).as_posix()
                    for name, path in tool_files.items()
                }
                if index == 2:
                    _write_json(
                        generated_tool / "manifest.json",
                        {"status": "passed", "successful_attempt": 0},
                    )
                    attempt = generated_tool / "attempts/attempt_0"
                    _write_text(attempt / "prompt.md", "generate a trajectory Tool")
                    _write_text(attempt / "response.txt", "```python\npass\n```\n")
                    _write_json(
                        attempt / "validation.json",
                        {"valid": True},
                    )
                    tool_artifacts["toolgen_manifest"] = (
                        generated_tool / "manifest.json"
                    ).relative_to(root).as_posix()
                _write_json(
                    execution / "planned_tool/tool_execution.json",
                    {
                        "route": "force_codegen" if index == 2 else "reuse",
                        "tool_request": {"metric": "hammer_block_contact_ever"},
                        "source": {
                            "artifact": tool_source.relative_to(root).as_posix()
                        },
                        "artifacts": tool_artifacts,
                        "episodes": [
                            {
                                "role": "policy_under_evaluation",
                                "policy_name": "ACT",
                                "seed": 100000,
                                "result": {"value": index == 2, "passed": index == 2},
                            },
                            {
                                "role": "expert_validation",
                                "policy_name": "expert",
                                "seed": 100000,
                                "result": {"value": True, "passed": True},
                            },
                        ],
                    },
                )
                _write_json(
                    execution / "execution_vqa/execution_vqa.json",
                    {
                        "status": "passed",
                        "query": {
                            "questions": [
                                {"id": "block_color_blue", "question": "Is it blue?"}
                            ]
                        },
                        "observation": {
                            "phenomena": [
                                {
                                    "id": "block_color_blue",
                                    "observed": False,
                                    "description": "not blue",
                                    "confidence": 0.9,
                                    "frame_ids": ["initial"],
                                }
                            ],
                            "numeric_consistency": "consistent",
                        },
                        "evidence_conflict": False,
                    },
                )
                _write_json(execution / "aggregate_result.json", {"status": "passed"})

            decisions = [
                {
                    "action": "continue",
                    "transition": "switch_aspect",
                    "decision_reason": "first-round evidence requested timing",
                },
                {"action": "stop", "decision_reason": "budget complete"},
            ]
            plan = {
                "evaluation_goal": "appearance robustness",
                "requested_aspect_ids": ["object_appearance.color"],
                "requested_template_ids": ["template_1", "template_2"],
                "max_rounds": 2,
                "rounds": rounds,
                "round_decisions": decisions,
                "planning_state": "stopped_after_round_2",
            }
            _write_json(evaluation / "plan/evaluation_plan.json", plan)
            for relative, value in {
                "request.json": {
                    "user_request": "How does ACT handle appearance variation?"
                },
                "plan/global_query_route.json": {
                    "selection": "beat_block_hammer"
                },
                "plan/free_concern.json": {
                    "sub_aspect": "object_appearance.color"
                },
                "plan/query_sufficiency_contract.json": {"claim_type": "some"},
            }.items():
                _write_json(evaluation / relative, value)
            _write_text(
                evaluation / "plan/free_concern_prompt.md", "find one concern"
            )
            _write_text(
                evaluation / "plan/free_concern_response_1.txt",
                "color robustness\n",
            )
            claim_runtime = evaluation / "plan/claim_first_runtime"
            for name, value in {
                "evidence_after_round_01.json": {
                    "assessment": {"should_stop": False}
                },
                "evidence_after_round_02.json": {
                    "assessment": {"should_stop": True}
                },
                "query_answer.json": {"answered": True, "answer": "mixed"},
            }.items():
                _write_json(claim_runtime / name, value)
            _write_json(
                evaluation / "plan/semantic_preservation_audit.json",
                {"accepted": True},
            )
            _write_text(evaluation / "debug.log", "not in public bundle")
            _write_json(
                evaluation / "plan/bound_task_session.json",
                {
                    "user_query": "How does ACT handle appearance variation?",
                    "target": {
                        "schema_version": 3,
                        "binding_mode": (
                            "single_task_single_checkpoint_open_world"
                        ),
                        "policy_task_binding": {
                            "task_name": "beat_block_hammer",
                            "policy": {"name": "ACT"},
                            "checkpoint": {
                                "checkpoint_id": "act-bbh/demo_clean-50"
                            },
                        },
                        "max_rounds": 2,
                    },
                    "selected_aspect_ids": ["object_appearance.color"],
                    "round_budget": 2,
                },
            )
            _write_json(
                evaluation / "manifest.json",
                {
                    "evaluation_id": evaluation_id,
                    "user_request": "How does ACT handle appearance variation?",
                    "task_name": "beat_block_hammer",
                    "child_run_ids": child_ids,
                    "plan": plan,
                },
            )
            _write_json(evaluation / "summary/summary.json", {"rounds": summaries})
            _write_json(
                evaluation / "feedback/feedback.json",
                {
                    "answer": "ACT was mixed in this tiny run.",
                    "findings": ["one failure and one success"],
                    "limitations": ["N=1"],
                    "recommended_next_step": "repeat with N=3",
                },
            )

            destination = root / f"docs/evidence_runs/{evaluation_id}/README.md"
            bundle = write_evidence_report(
                root,
                evaluation,
                destination=destination,
                publish=True,
            )
            report = destination.read_text(encoding="utf-8")
            self.assertIn("How does ACT handle appearance variation?", report)
            self.assertIn("act-bbh/demo_clean-50", report)
            self.assertIn("TaskProposal", report)
            self.assertIn("```python", report)
            self.assertIn("```yaml", report)
            self.assertIn("![round_1 initial scene]", report)
            self.assertIn("Open ACT video", report)
            self.assertIn("first-round evidence requested timing", report)
            self.assertNotIn("/root/", report)
            self.assertEqual(bundle["round_count"], 2)
            expected_artifacts = (
                "artifacts/query/request.json",
                "artifacts/plan/free_concern_prompt.md",
                "artifacts/plan/free_concern_response_1.txt",
                "artifacts/taskgen/round_1/generation/code_prompt.md",
                "artifacts/taskgen/round_1/validation/static.json",
                "artifacts/taskgen/round_1/evidence/scene_comparison.png",
                "artifacts/tool/round_2/tool_execution.json",
                "artifacts/tool/round_2/codegen_prompt.md",
                "artifacts/tool/round_2/codegen_response.txt",
                "artifacts/tool/round_2/codegen_validation.json",
                "artifacts/aggregate/round_1.json",
                "artifacts/answer/query_answer.json",
                "artifacts/answer/feedback.json",
                "artifacts/audit/semantic_preservation_audit.json",
            )
            for relative in expected_artifacts:
                self.assertTrue((destination.parent / relative).is_file(), relative)
            for relative in bundle["files"]:
                self.assertTrue((root / relative).is_file(), relative)
            self.assertFalse(
                (
                    destination.parent
                    / "artifacts/taskgen/round_1/debug.log"
                ).exists()
            )
            self.assertNotIn("debug.log", json.dumps(bundle))

            for link in re.findall(r"\]\(([^)]+)\)", report):
                self.assertTrue((destination.parent / link).resolve().is_file(), link)
            replaced = write_evidence_report(
                root, evaluation, destination=destination, publish=True
            )
            self.assertEqual(replaced["files"], bundle["files"])

    def test_legacy_round_is_labeled_as_projection_not_as_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "mea/evaluation_runs/eval_legacy"
            plan = {
                "evaluation_goal": "legacy compatibility",
                "requested_aspect_ids": ["object_position"],
                "requested_template_ids": ["object_position.left_fixed"],
                "max_rounds": 1,
                "rounds": [
                    {
                        "round_id": "round_1",
                        "template_id": "object_position.left_fixed",
                        "aspect_id": "object_position",
                        "task_name": "click_bell",
                        "task_instruction": "legacy intent",
                        "route": "reuse",
                        "execution": {"seeds": [1], "num_episodes": 1},
                    }
                ],
                "round_decisions": [],
                "planning_state": "awaiting_round_1_observation",
            }
            _write_json(evaluation / "plan/evaluation_plan.json", plan)
            _write_json(
                evaluation / "manifest.json",
                {
                    "evaluation_id": "eval_legacy",
                    "task_name": "click_bell",
                    "user_request": "legacy query",
                    "plan": plan,
                },
            )
            destination = root / "docs/evidence_runs/eval_legacy/README.md"
            write_evidence_report(
                root,
                evaluation,
                destination=destination,
                publish=True,
            )
            report = destination.read_text(encoding="utf-8")
            published_manifest = json.loads(
                (
                    destination.parent / "evidence_bundle_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                published_manifest["source_server_path"],
                str(evaluation.resolve()),
            )
            self.assertIn(
                "### Compatibility task projection (not used for planning)",
                report,
            )
            self.assertIn(
                "### Compatibility Tool projection (not used for planning)",
                report,
            )
            self.assertIn(
                '"proposal_status": "not_projected_in_compatibility_view"',
                report,
            )
            self.assertNotIn("### Plan -> TaskProposal", report)
            self.assertNotIn("### ToolProposal -> ToolGen / reuse", report)

    def test_publish_rejects_unsafe_ids_and_nonfresh_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "mea/evaluation_runs/eval_boundary"
            plan = {
                "rounds": [
                    {
                        "round_id": "../escape",
                        "execution": {"seeds": [1], "num_episodes": 1},
                    }
                ]
            }
            _write_json(evaluation / "plan/evaluation_plan.json", plan)
            _write_json(
                evaluation / "manifest.json",
                {"evaluation_id": "eval_boundary", "plan": plan},
            )
            destination = root / "docs/unsafe/README.md"
            with self.assertRaisesRegex(EvidenceReportError, "safe artifact id"):
                write_evidence_report(
                    root,
                    evaluation,
                    destination=destination,
                    publish=True,
                )
            self.assertFalse(destination.exists())

            dirty = root / "docs/dirty"
            dirty.mkdir(parents=True)
            (dirty / "untracked.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceReportError, "must be fresh"):
                write_evidence_report(
                    root,
                    evaluation,
                    destination=dirty / "README.md",
                    publish=True,
                )

    def test_publish_can_include_one_completed_round_reuse_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "mea/evaluation_runs/eval_reuse_audit"
            plan = {"rounds": [], "max_rounds": 0}
            _write_json(
                evaluation / "manifest.json",
                {
                    "evaluation_id": "eval_reuse_audit",
                    "user_request": "Is a cached Tool reused exactly?",
                    "plan": plan,
                },
            )
            _write_json(evaluation / "plan/evaluation_plan.json", plan)
            repair = evaluation / "repairs/reuse_check"
            _write_json(
                repair / "result.json",
                {
                    "status": "completed",
                    "repair_id": "reuse_check",
                    "act_rollouts_started": 0,
                    "first_query_route": "typed_metric_spec_compile",
                    "first_query_measurements": [0.4],
                    "exact_reuse_route": "run_local_reuse",
                    "exact_reuse_provider_called": False,
                    "aggregate_status": "passed",
                },
            )
            _write_json(
                repair / "repair_provenance.json",
                {"status": "completed"},
            )
            for relative, route in (
                (
                    "first_query/planned_tool/tool_execution.json",
                    "typed_metric_spec_compile",
                ),
                (
                    "second_query_exact_reuse/planned_tool/tool_execution.json",
                    "run_local_reuse",
                ),
            ):
                _write_json(repair / relative, {"route": route})

            destination = root / "docs/evidence/current/README.md"
            bundle = write_evidence_report(
                root,
                evaluation,
                destination=destination,
                publish=True,
                include_repair_id="reuse_check",
            )

            self.assertEqual(bundle["included_repair_id"], "reuse_check")
            self.assertIn(
                "not independent cross-evaluation reuse",
                destination.read_text(encoding="utf-8"),
            )
            self.assertTrue(
                (
                    destination.parent
                    / "artifacts/audit/completed_round_reuse/"
                    "exact_reuse_tool_execution.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
