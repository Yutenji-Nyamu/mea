from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from mea.execution_vqa.open_question import (
    OpenVQAQuestionAgent,
    load_run_local_vqa_questions,
    register_run_local_vqa_question,
)
from mea.execution_vqa.reviewed_generated_questions import (
    build_generated_vqa_question_review_template,
    install_reviewed_generated_vqa_question,
)
from mea.toolgen.artifact_context import (
    ToolArtifactContextError,
    build_tool_artifact_context,
)
from mea.toolgen.open_request import OpenToolRequestAgent


class _Provider:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.last_metadata = {"provider": "fixture"}

    def text(self, *_args, **_kwargs):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return json.dumps(payload)


def _proposal(
    task_name: str = "adjust_bottle",
    *,
    vqa: bool = True,
):
    return {
        "schema_version": 2,
        "candidate_id": "candidate.runtime.novel",
        "source_query": "Where does this policy first expose a weakness?",
        "base_task": task_name,
        "semantic_concern": "novel.visual.object_stability",
        "scene_need": None,
        "checker_need": None,
        "rule_tool_need": {
            "kind": "measure",
            "description": "Measure terminal object height.",
            "reuse_first": True,
        },
        "vqa_tool_need": (
            {
                "kind": "vqa",
                "description": (
                    "Check whether the manipulated object visibly wobbles."
                ),
                "reuse_first": True,
            }
            if vqa
            else None
        ),
        "tool_need": None,
    }


def _task_artifact():
    return {
        "scene_origin": "provider_generated_code",
        "success_origin": "provider_generated_checker",
        "success_semantics_preserved": True,
        "success_official_equivalent": False,
        "success_compiler_eligible": False,
        "success_act_eligible": True,
        "success_execution_scope": "provider_generated_checker",
        "success_outcome_label": "experimental_target_pose",
    }


def _question(*, numeric_authority: str = "no_numeric_oracle"):
    return {
        "schema_version": 1,
        "question_spec": {
            "id": "run_local.manipulated_object_wobble",
            "question_type": "visible_state_change",
            "target_role": "manipulated_object",
            "question": (
                "Does the manipulated object visibly wobble during the rollout?"
            ),
            "visual_scope": "rollout_change",
            "numeric_authority": numeric_authority,
        },
    }


class OpenToolArtifactTest(unittest.TestCase):
    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def test_shared_context_carries_method_inputs_and_validation_contract(self):
        context = build_tool_artifact_context(
            self.root,
            task_name="adjust_bottle",
            proposal=_proposal(),
            task_artifact_summary=_task_artifact(),
        )

        self.assertEqual(
            context["proposal"]["candidate_id"], "candidate.runtime.novel"
        )
        self.assertEqual(
            context["task_artifact"]["success_execution_scope"],
            "provider_generated_checker",
        )
        self.assertIn(
            "bottle_functional_position",
            {
                field["name"]
                for field in context["runtime_schema"]["semantic_fields"]
            },
        )
        self.assertEqual(
            context["oracle_broker"]["derived_observable"]["status"],
            "available",
        )
        self.assertEqual(
            context["oracle_broker"]["derived_observable"]["source"],
            "toolgen_semantic_review_runtime",
        )
        self.assertFalse(
            context["oracle_broker"]["derived_observable"][
                "validation_contract"
            ]["success_or_reward_authority"]
        )

    def test_shared_context_rejects_proposal_task_mismatch(self):
        with self.assertRaisesRegex(
            ToolArtifactContextError, "differs from the bound runtime task"
        ):
            build_tool_artifact_context(
                self.root,
                task_name="adjust_bottle",
                proposal=_proposal("grab_roller"),
            )

    def test_rule_tool_agent_consumes_proposal_and_task_artifact(self):
        provider = _Provider(
            {
                "schema_version": 2,
                "task_name": "adjust_bottle",
                "metric": "terminal_bottle_height",
                "question": "What was the terminal bottle height?",
                "metric_spec": {
                    "schema_version": 1,
                    "operation": "terminal_signal_component",
                    "signal": "bottle_functional_position",
                    "component": "z",
                    "absolute": False,
                    "unit": "m",
                    "null_semantics": "null_if_terminal_not_finite",
                },
            }
        )
        bundle = OpenToolRequestAgent(
            self.root, provider, model="fixture-model"
        ).propose(
            source_query=_proposal()["source_query"],
            semantic_concern=_proposal()["semantic_concern"],
            tool_need="Measure terminal bottle height.",
            task_name="adjust_bottle",
            proposal=_proposal(),
            task_artifact_summary=_task_artifact(),
        )

        context = bundle["context"]["artifact_context"]
        self.assertEqual(bundle["status"], "selected")
        self.assertEqual(
            context["proposal"]["candidate_id"], "candidate.runtime.novel"
        )
        self.assertFalse(
            context["task_artifact"]["success_official_equivalent"]
        )

    def test_derived_observable_without_broker_uses_toolgen_validation(self):
        provider = _Provider(
            {
                "schema_version": 2,
                "task_name": "adjust_bottle",
                "metric": "query_object_wobble",
                "question": "How much did the bottle wobble?",
                "metric_spec": {
                    "schema_version": 2,
                    "operation": "derived_observable",
                    "observable_id": "query_object_wobble",
                    "description": "Peak lateral oscillation of the bottle.",
                    "required_signals": ["bottle_position"],
                    "unit": "m",
                    "null_semantics": "null_if_no_finite_sample",
                },
            }
        )
        bundle = OpenToolRequestAgent(
            self.root, provider, model="fixture-model"
        ).propose(
            source_query=_proposal()["source_query"],
            semantic_concern=_proposal()["semantic_concern"],
            tool_need="Measure bottle wobble.",
            task_name="adjust_bottle",
            proposal=_proposal(),
            task_artifact_summary=_task_artifact(),
            allow_unsupported=True,
        )

        self.assertEqual(bundle["status"], "selected")
        self.assertEqual(
            bundle["tool_request"]["metric_spec"]["operation"],
            "derived_observable",
        )
        self.assertEqual(provider.calls, 1)

    def test_vqa_semantic_miss_generates_structured_question(self):
        context = build_tool_artifact_context(
            self.root,
            task_name="adjust_bottle",
            proposal=_proposal(),
            task_artifact_summary=_task_artifact(),
        )
        provider = _Provider(_question())
        bundle = OpenVQAQuestionAgent(
            provider, model="fixture-model"
        ).propose(artifact_context=context)

        self.assertEqual(bundle["status"], "generated")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(
            bundle["query"]["phenomenon_ids"],
            ["run_local.manipulated_object_wobble"],
        )
        self.assertEqual(
            bundle["question_spec"]["numeric_authority"],
            "no_numeric_oracle",
        )

    def test_vqa_allows_only_one_repair(self):
        context = build_tool_artifact_context(
            self.root,
            task_name="adjust_bottle",
            proposal=_proposal(),
            task_artifact_summary=_task_artifact(),
        )
        provider = _Provider(
            _question(
                numeric_authority=(
                    "simulator_pose_is_authoritative_when_available"
                )
            ),
            _question(),
        )
        agent = OpenVQAQuestionAgent(provider, model="fixture-model")
        bundle = agent.propose(artifact_context=context)

        self.assertEqual(bundle["status"], "generated")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(bundle["provider"]["attempt_count"], 2)
        self.assertEqual(len(bundle["provider"]["errors"]), 1)

    def test_catalog_hit_with_different_need_is_not_exact_reuse(self):
        proposal = _proposal("beat_block_hammer")
        proposal["semantic_concern"] = "object_appearance.color"
        context = build_tool_artifact_context(
            self.root,
            task_name="beat_block_hammer",
            proposal=proposal,
        )
        provider = _Provider(_question())
        bundle = OpenVQAQuestionAgent(
            provider, model="fixture-model"
        ).propose(artifact_context=context)

        self.assertEqual(bundle["status"], "generated")
        self.assertTrue(bundle["provider"]["called"])
        self.assertEqual(provider.calls, 1)

    def test_generated_vqa_question_registers_and_exactly_reuses(self):
        context = build_tool_artifact_context(
            self.root,
            task_name="adjust_bottle",
            proposal=_proposal(),
            task_artifact_summary=_task_artifact(),
        )
        first_provider = _Provider(_question())
        generated = OpenVQAQuestionAgent(
            first_provider,
            model="fixture-model",
        ).propose(artifact_context=context)

        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "vqa_registry"
            registration = register_run_local_vqa_question(
                registry,
                generated,
                artifact_path="evaluation/round_1/question_bundle.json",
            )
            reusable_context = build_tool_artifact_context(
                self.root,
                task_name="adjust_bottle",
                proposal=_proposal(),
                task_artifact_summary=_task_artifact(),
                reusable_vqa_questions=load_run_local_vqa_questions(
                    registry
                ),
            )
            second_provider = _Provider(_question())
            reused = OpenVQAQuestionAgent(
                second_provider,
                model="fixture-model",
            ).propose(artifact_context=reusable_context)
            reused_registration = register_run_local_vqa_question(
                registry,
                reused,
                artifact_path="evaluation/round_2/question_bundle.json",
            )

        self.assertEqual(generated["status"], "generated")
        self.assertEqual(registration["reuse_count"], 0)
        self.assertEqual(reused["status"], "reused")
        self.assertEqual(
            reused["source"],
            "evaluation_local_exact_vqa_need_reuse",
        )
        self.assertEqual(
            reused["question_spec"],
            generated["question_spec"],
        )
        self.assertEqual(second_provider.calls, 0)
        self.assertEqual(reused_registration["reuse_count"], 1)

    def test_reviewed_generated_vqa_question_reuses_in_new_evaluation(self):
        context = build_tool_artifact_context(
            self.root,
            task_name="adjust_bottle",
            proposal=_proposal(),
            task_artifact_summary=_task_artifact(),
        )
        generated = OpenVQAQuestionAgent(
            _Provider(_question()), model="fixture-model"
        ).propose(artifact_context=context)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_registry = root / "evaluation_a/vqa_registry"
            reviewed_registry = root / "reviewed_vqa_registry"
            source = register_run_local_vqa_question(
                source_registry,
                generated,
                artifact_path="evaluation_a/question_bundle.json",
            )
            review = build_generated_vqa_question_review_template(
                source_registry,
                source["semantic_key"],
            )
            review.update(
                {
                    "decision": "approved",
                    "reviewer": {
                        "id": "development-agent-test",
                        "kind": "development_agent_proxy",
                    },
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "notes": "Fixture-only explicit review.",
                }
            )
            review["checks"] = {
                key: True for key in review["checks"]
            }
            install_reviewed_generated_vqa_question(
                source_registry,
                source["semantic_key"],
                review,
                reviewed_registry,
            )

            never_called = _Provider(_question())
            reused = OpenVQAQuestionAgent(
                never_called, model="fixture-model"
            ).propose(
                artifact_context=context,
                reviewed_registry_dir=reviewed_registry,
            )

        self.assertEqual(reused["status"], "reused")
        self.assertEqual(
            reused["source"],
            "reviewed_persistent_exact_vqa_need_reuse",
        )
        self.assertEqual(reused["question_spec"], generated["question_spec"])
        self.assertEqual(never_called.calls, 0)
        self.assertTrue(
            reused["validation"]["current_rollout_vqa_execution_required"]
        )


if __name__ == "__main__":
    unittest.main()
