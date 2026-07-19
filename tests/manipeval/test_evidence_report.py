import json
import re
import tempfile
import unittest
from pathlib import Path

from mea.feedback import write_evidence_report


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


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
                tool_source = root / "mea/tool_registry/generated_tool.py"
                tool_source.parent.mkdir(parents=True, exist_ok=True)
                tool_source.write_text("def evaluate(x):\n    return x\n", encoding="utf-8")
                _write_json(
                    execution / "planned_tool/tool_execution.json",
                    {
                        "route": "force_codegen" if index == 2 else "reuse",
                        "tool_request": {"metric": "hammer_block_contact_ever"},
                        "source": {"artifact": "mea/tool_registry/generated_tool.py"},
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
            _write_json(
                evaluation / "plan/bound_task_session.json",
                {
                    "user_query": "How does ACT handle appearance variation?",
                    "target": {
                        "binding_mode": "single_task_single_checkpoint",
                        "task_name": "beat_block_hammer",
                        "task_profile": "generated",
                        "policy": {"name": "ACT"},
                        "checkpoint": {"checkpoint_id": "act-bbh/demo_clean-50"},
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
            self.assertIn("TaskProposal", report)
            self.assertIn("```python", report)
            self.assertIn("```yaml", report)
            self.assertIn("![round_1 initial scene]", report)
            self.assertIn("Open ACT video", report)
            self.assertIn("first-round evidence requested timing", report)
            self.assertNotIn("/root/", report)
            self.assertEqual(bundle["round_count"], 2)

            for link in re.findall(r"\]\(([^)]+)\)", report):
                self.assertTrue((destination.parent / link).resolve().is_file(), link)

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
            self.assertIn("### Legacy plan intent", report)
            self.assertIn("### Legacy Tool request", report)
            self.assertIn('"proposal_status": "missing_legacy_projection"', report)
            self.assertNotIn("### Plan -> TaskProposal", report)
            self.assertNotIn("### ToolProposal -> ToolGen / reuse", report)


if __name__ == "__main__":
    unittest.main()
