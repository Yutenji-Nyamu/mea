"""Answer, evidence bundle, report, and history finalization."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping

from mea.agent_evidence import build_evidence_bundle, compact_aggregate_result
from mea.feedback import (
    PlanAgentFinalSummary,
    render_evaluation_report,
    write_evidence_report,
)
from mea.plan_artifacts import PLAN_AGENT_SESSION
from mea.planner.plan_agent_evidence import render_query_answer
from mea.round_evidence import aggregate_evaluation_results

from .plan_agent_persistence import _write_json, update_manifest


class PlanAgentFinalizationMixin:
    def _finalize(
        self,
        *,
        plan: dict[str, Any],
        round_runs: list[dict[str, Any]],
        runtime_state: Mapping[str, Any],
        query_answer: Mapping[str, Any] | None,
        executed_rounds: int,
    ) -> dict[str, Any]:
        policy_round_runs = [
            item
            for item in round_runs
            if not isinstance(
                item["round_summary"].get("observations", {}).get(
                    "planning_observation"
                ),
                Mapping,
            )
        ]
        planning_observations = [
            deepcopy(
                item["round_summary"]["observations"][
                    "planning_observation"
                ]
            )
            for item in round_runs
            if isinstance(
                item["round_summary"].get("observations", {}).get(
                    "planning_observation"
                ),
                Mapping,
            )
        ]
        evaluation_aggregate = aggregate_evaluation_results(
            round_runs,
            self.evaluation_dir / "summary/aggregate_result.json",
        )
        summary = {
            "schema_version": 2,
            "evaluation_id": self.evaluation_id,
            "status": (
                "completed"
                if policy_round_runs
                and all(
                    item["round_summary"]["pipeline_passed"]
                    for item in policy_round_runs
                )
                else "completed_with_planning_gap"
                if planning_observations and not policy_round_runs
                else "completed_with_pipeline_failure"
            ),
            "rounds": [item["round_summary"] for item in round_runs],
            "planning_observations": planning_observations,
            "policy_round_count": len(policy_round_runs),
            "aggregate": compact_aggregate_result(evaluation_aggregate),
        }
        evidence = build_evidence_bundle(
            self.repo_root,
            self.evaluation_id,
            self.user_request,
            plan,
            round_runs,
            evaluation_aggregate,
        )
        final_query_answer = (
            deepcopy(dict(query_answer))
            if query_answer is not None
            else None
        )
        if final_query_answer is None:
            externally_stopped_assessment = {
                **runtime_state["assessment"],
                "should_stop": True,
                "stop_reason": "external_hard_round_cap",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "An external execution cap stopped the run before the "
                    "the Plan Agent produced a supported answer."
                ),
            }
            final_query_answer = render_query_answer(
                self.user_request,
                externally_stopped_assessment,
                runtime_state["records"],
                baseline_valid=bool(runtime_state["control_passed"]),
            )
            _write_json(
                self.evaluation_dir
                / PLAN_AGENT_SESSION
                / "query_answer.json",
                final_query_answer,
            )
        evidence["plan_agent_session"] = {
            "schema_version": 1,
            "runtime_limits": runtime_state["runtime_limits"],
            "assessment": runtime_state["assessment"],
            "query_answer": final_query_answer,
            "records": runtime_state["records"],
            "artifacts": {
                "query_answer": (
                    f"mea/evaluation_runs/{self.evaluation_id}/plan/"
                    "plan_agent_session/query_answer.json"
                ),
                "latest_evidence": (
                    f"mea/evaluation_runs/{self.evaluation_id}/plan/"
                    "plan_agent_session/"
                    f"evidence_after_round_{executed_rounds:02d}.json"
                ),
            },
        }
        _write_json(self.evaluation_dir / "summary/summary.json", summary)
        _write_json(
            self.evaluation_dir / "summary/evidence_bundle.json",
            evidence,
        )
        update_manifest(
            self.evaluation_dir,
            status="generating_answer",
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            summary=summary,
        )
        feedback = PlanAgentFinalSummary(
            self.repo_root,
            self.provider,
            model=self.models["feedback"],
        ).generate(
            evidence,
            output_dir=self.evaluation_dir / "answer",
        )
        report_path = self.evaluation_dir / "evaluation_report.md"
        report_path.write_text(
            render_evaluation_report(evidence, feedback),
            encoding="utf-8",
        )
        update_manifest(
            self.evaluation_dir,
            status=summary["status"],
            lifecycle_status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            answer_path="answer/answer.json",
            report_path="evaluation_report.md",
            child_run_ids=[
                item["child_manifest"].get("run_id")
                for item in round_runs
            ],
            summary=summary,
            answer=feedback,
        )
        compact_evidence_report = write_evidence_report(
            self.repo_root,
            self.evaluation_dir,
            destination=self.evaluation_dir / "evidence_report.md",
        )
        update_manifest(
            self.evaluation_dir,
            evidence_report_path="evidence_report.md",
            evidence_report_bundle=compact_evidence_report,
        )
        history_index: dict[str, Any] = {
            "status": (
                "disabled" if self.history_disabled else "not_available"
            )
        }
        if self.history_database is not None:
            try:
                history_index = {
                    "status": "passed",
                    **self.history_database.index_evaluation_dir(
                        self.evaluation_dir
                    ),
                }
            except Exception as exc:
                history_index = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        update_manifest(self.evaluation_dir, history_index=history_index)
        retrieval = self.history_retrieval or {}
        return {
            "evaluation_id": self.evaluation_id,
            "child_run_ids": [
                item["child_manifest"].get("run_id")
                for item in round_runs
            ],
            "summary": summary,
            "answer": feedback,
            "history_retrieval": {
                "status": retrieval.get("status"),
                "selected_count": self.history_context_count,
            },
            "history_index": history_index,
            "report_path": str(report_path.relative_to(self.repo_root)),
            "evidence_report_path": str(
                (self.evaluation_dir / "evidence_report.md").relative_to(
                    self.repo_root
                )
            ),
        }



__all__ = ["PlanAgentFinalizationMixin"]
