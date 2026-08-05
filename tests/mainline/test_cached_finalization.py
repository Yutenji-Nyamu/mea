import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mea.cached_finalization import finalize_cached_evaluation
from mea.plan_agent_application import PlanAgentApplication
from mea.plan_agent_decision_resume import (
    PlanAgentDecisionResumeError,
    _latest_runtime_capabilities,
    resume_plan_agent_decision,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FinalSummary:
    def __init__(self, _root, _provider, *, model):
        self.model = model

    def generate(self, _evidence, *, output_dir):
        feedback = {
            "answer": "bounded answer",
            "evaluation_scope": "one cached round",
            "findings": ["official success"],
            "limitations": ["N=1"],
            "recommended_next_step": "repeat with more seeds",
        }
        _write_json(output_dir / "answer.json", feedback)
        (output_dir / "answer.md").write_text("bounded answer\n", encoding="utf-8")
        return feedback


class CachedFinalizationTests(unittest.TestCase):
    def test_decision_resume_restores_latest_persisted_capabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)
            _write_json(
                evaluation
                / "plan/plan_agent_steps/after_round_03/runtime_capabilities.json",
                {"schema_version": 2, "source": "runtime_round_3"},
            )

            restored = _latest_runtime_capabilities(
                evaluation,
                completed_rounds=3,
            )

            self.assertEqual(restored["source"], "runtime_round_3")

            with self.assertRaises(PlanAgentDecisionResumeError):
                _latest_runtime_capabilities(
                    evaluation,
                    completed_rounds=2,
                )

    def test_decision_resume_rejects_completed_or_retried_boundary(self):
        for mode in ("proposal_persisted", "retry_used"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                evaluation_id = "eval_resume_guard"
                evaluation = root / "mea/evaluation_runs" / evaluation_id
                manifest = {
                    "evaluation_id": evaluation_id,
                    "status": "failed",
                    "lifecycle_status": "failed",
                    "failure_stage": "plan_agent_decision_after_round_1",
                    "completed_rounds": 1,
                    "failure": {"type": "ClaimFirstPlanError"},
                }
                if mode == "retry_used":
                    manifest["plan_agent_cached_retry"] = {
                        "after_round": 1,
                        "attempt_count": 1,
                    }
                _write_json(evaluation / "manifest.json", manifest)
                step_dir = (
                    evaluation
                    / "plan/plan_agent_steps/after_round_01"
                )
                _write_json(
                    step_dir / "runtime_capabilities.json",
                    {"schema_version": 2},
                )
                if mode == "proposal_persisted":
                    _write_json(
                        step_dir / "semantic_proposal_bundle.json",
                        {"schema_version": 1},
                    )

                with self.assertRaises(PlanAgentDecisionResumeError):
                    resume_plan_agent_decision(
                        root,
                        evaluation_id,
                        provider=object(),
                        models={"planner": "fixture", "feedback": "fixture"},
                    )

    def test_recovers_final_answer_without_mutating_method_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_id = "eval_cached_finalization"
            evaluation = root / "mea/evaluation_runs" / evaluation_id
            aggregate = evaluation / "summary/aggregate_result.json"
            summary = evaluation / "summary/summary.json"
            evidence = evaluation / "summary/evidence_bundle.json"
            _write_json(
                evaluation / "manifest.json",
                {
                    "evaluation_id": evaluation_id,
                    "status": "failed",
                    "lifecycle_status": "failed",
                    "failure_stage": "final_answer",
                    "failure": {"type": "ProviderError", "message": "503"},
                    "history_retrieval_status": "disabled",
                },
            )
            _write_json(aggregate, {"status": "passed", "source_count": 1})
            _write_json(
                summary,
                {
                    "evaluation_id": evaluation_id,
                    "status": "completed",
                    "rounds": [{"round_id": "round_1"}],
                },
            )
            _write_json(
                evidence,
                {
                    "evaluation_id": evaluation_id,
                    "rounds": [{"round_id": "round_1"}],
                },
            )
            before = {path: _sha256(path) for path in (aggregate, summary, evidence)}

            def _write_report(_root, _evaluation, *, destination):
                destination.write_text("compact evidence\n", encoding="utf-8")
                return {"report": "evidence_report.md"}

            with (
                patch(
                    "mea.cached_finalization.PlanAgentFinalSummary",
                    _FinalSummary,
                ),
                patch(
                    "mea.cached_finalization.render_evaluation_report",
                    return_value="evaluation report\n",
                ),
                patch(
                    "mea.cached_finalization.write_evidence_report",
                    side_effect=_write_report,
                ),
            ):
                result = finalize_cached_evaluation(
                    root,
                    evaluation_id,
                    provider=object(),
                    feedback_model="fixture-model",
                )

            self.assertEqual(result["rollouts_executed"], 0)
            self.assertEqual(
                before,
                {path: _sha256(path) for path in (aggregate, summary, evidence)},
            )
            manifest = json.loads(
                (evaluation / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["lifecycle_status"], "completed")
            self.assertIsNone(manifest["failure"])
            self.assertEqual(manifest["history_index"], {"status": "disabled"})
            self.assertTrue((evaluation / "answer/answer.json").is_file())
            self.assertTrue((evaluation / "evaluation_report.md").is_file())
            self.assertTrue((evaluation / "evidence_report.md").is_file())

    def test_plan_agent_decision_resume_persists_continue_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)
            _write_json(evaluation / "manifest.json", {"status": "failed"})
            app = PlanAgentApplication.__new__(PlanAgentApplication)
            app.evaluation_dir = evaluation
            app.evaluation_id = "eval_resume_continue"
            app.plan = {"rounds": [{"round_id": "round_1"}]}
            app._observe = Mock(
                return_value={"assessment": {"should_stop": False}}
            )
            next_round = {"round_id": "round_2", "candidate_id": "new"}
            app._decide_next_step = Mock(
                return_value=(
                    {
                        "rounds": [*app.plan["rounds"], next_round],
                        "planning_state": "awaiting_round_2_observation",
                    },
                    {"action": "continue", "next_round": next_round},
                    None,
                )
            )
            app._finalize = Mock()

            result = app.resume_decision(
                round_runs=[
                    {
                        "round_plan": app.plan["rounds"][0],
                        "round_summary": {},
                    }
                ]
            )

            self.assertEqual(result["decision_resume"]["action"], "continue")
            self.assertEqual(result["decision_resume"]["rollouts_executed"], 0)
            self.assertFalse(
                result["decision_resume"]["automatic_round_execution"]
            )
            app._finalize.assert_not_called()
            manifest = json.loads(
                (evaluation / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["lifecycle_status"],
                "awaiting_explicit_round_execution",
            )

    def test_plan_agent_decision_resume_finalizes_validated_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)
            _write_json(evaluation / "manifest.json", {"status": "failed"})
            app = PlanAgentApplication.__new__(PlanAgentApplication)
            app.evaluation_dir = evaluation
            app.evaluation_id = "eval_resume_stop"
            app.plan = {"rounds": [{"round_id": "round_1"}]}
            state = {
                "assessment": {
                    "should_stop": True,
                    "evidence_sufficient": True,
                }
            }
            app._observe = Mock(return_value=state)
            app._decide_next_step = Mock(
                return_value=(
                    {**app.plan, "planning_state": "stopped_after_round_1"},
                    {"action": "stop", "next_round": None},
                    {"answered": True},
                )
            )
            app._finalize = Mock(return_value={"evaluation_id": app.evaluation_id})

            result = app.resume_decision(
                round_runs=[
                    {
                        "round_plan": app.plan["rounds"][0],
                        "round_summary": {},
                    }
                ]
            )

            self.assertEqual(result["decision_resume"]["action"], "stop")
            self.assertEqual(result["decision_resume"]["rollouts_executed"], 0)
            app._finalize.assert_called_once()
            manifest = json.loads(
                (evaluation / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(manifest["failure"])
            self.assertIsNone(manifest["failure_stage"])

    def test_pending_round_executes_once_without_replaying_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)
            _write_json(evaluation / "manifest.json", {"status": "pending"})
            app = PlanAgentApplication.__new__(PlanAgentApplication)
            app.evaluation_dir = evaluation
            app.evaluation_id = "eval_pending_round"
            app.plan = {
                "rounds": [
                    {"round_id": "round_1"},
                    {"round_id": "round_2"},
                ]
            }
            executed = SimpleNamespace(
                child_manifest={"status": "completed"},
                child_dir=evaluation / "child",
                round_summary={"round_id": "round_2"},
                tool_evaluation={"status": "passed"},
                returncode=0,
            )
            app._execute_round_plan = Mock(return_value=executed)
            app.resume_decision = Mock(return_value={"status": "completed"})
            prior = [
                {
                    "round_plan": app.plan["rounds"][0],
                    "round_summary": {"round_id": "round_1"},
                }
            ]

            result = app.execute_pending_round(round_runs=prior)

            app._execute_round_plan.assert_called_once_with(app.plan["rounds"][1])
            app.resume_decision.assert_called_once()
            resumed_runs = app.resume_decision.call_args.kwargs["round_runs"]
            self.assertEqual(len(resumed_runs), 2)
            self.assertEqual(
                result["pending_round_continuation"]["prior_rounds_replayed"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
