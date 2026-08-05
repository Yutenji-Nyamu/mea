"""Final Answer, audit, and completed-round reuse evidence publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evidence_projection import (
    EvidenceReportError,
    _read_json,
    _safe_artifact_id,
    _semantic_aggregate,
)
from .evidence_publication import EvidencePublisher


FINAL_AUDIT_ARTIFACTS = (
    (
        "plan/semantic_preservation_audit.json",
        "audit/semantic_preservation_audit.json",
    ),
    (
        "audit/semantic_alignment_reaudit.json",
        "audit/semantic_alignment_reaudit.json",
    ),
    ("audit/final_audit.json", "audit/final_audit.json"),
    ("audit/protocol_audit.json", "audit/protocol_audit.json"),
)


@dataclass(frozen=True)
class FinalEvidenceProjection:
    final_payload: dict[str, Any]
    alignment_reaudit_summary: dict[str, Any] | None
    repair_result: dict[str, Any] | None
    reuse_summary: dict[str, Any] | None
    run_summary_path: Path


def publish_final_evidence(
    *,
    publisher: EvidencePublisher,
    evaluation: Path,
    rounds: list[dict[str, Any]],
    feedback: Mapping[str, Any],
    manifest: Mapping[str, Any],
    query: Any,
    target: Mapping[str, Any],
    plan: Mapping[str, Any],
    session: Mapping[str, Any],
    summary: Mapping[str, Any],
    compact_rounds: list[dict[str, Any]],
    include_repair_id: str | None,
) -> FinalEvidenceProjection:
    """Publish immutable final evidence and its compact machine projection."""

    publisher.publish_first(
        (
            evaluation / "answer/answer.json",
            evaluation / "feedback/feedback.json",
        ),
        publisher.semantic_dir / "answer/answer.json",
    )
    for source, destination in FINAL_AUDIT_ARTIFACTS:
        publisher.publish_copy(
            evaluation / source,
            publisher.semantic_dir / destination,
        )
    for round_plan in rounds:
        round_id = str(round_plan.get("round_id") or "")
        if round_id:
            publisher.publish_copy(
                evaluation / f"plan/decision_after_{round_id}.json",
                publisher.semantic_dir
                / "plan/decisions"
                / f"after_{round_id}.json",
            )

    repair_result = None
    acceptance_projection = None
    if include_repair_id is not None:
        repair_id = _safe_artifact_id(
            include_repair_id,
            label="include_repair_id",
        )
        repair_root = evaluation / "repairs" / repair_id
        repair_result = _read_json(repair_root / "result.json", required=True)
        if repair_result.get("status") != "completed":
            raise EvidenceReportError(
                f"requested repair is not completed: {repair_id}"
            )
        acceptance_projection_path = repair_root / "acceptance_projection.json"
        if acceptance_projection_path.is_file():
            acceptance_projection = _read_json(
                acceptance_projection_path,
                required=True,
            )
            if not (
                acceptance_projection.get("status") == "completed"
                and acceptance_projection.get("source_summary_path")
                == "summary/summary.json"
                and acceptance_projection.get("projection_source")
                == "current_code_post_run"
                and isinstance(
                    acceptance_projection.get("projection"), Mapping
                )
            ):
                raise EvidenceReportError(
                    "repair acceptance projection has invalid provenance"
                )
            publisher.publish_copy(
                acceptance_projection_path,
                publisher.semantic_dir
                / "audit/completed_round_reuse/acceptance_projection.json",
            )
        for source, destination in (
            ("result.json", "result.json"),
            ("repair_provenance.json", "repair_provenance.json"),
            (
                "first_query/planned_tool/tool_execution.json",
                "first_query_tool_execution.json",
            ),
            (
                "second_query_exact_reuse/planned_tool/tool_execution.json",
                "exact_reuse_tool_execution.json",
            ),
        ):
            publisher.publish_copy(
                repair_root / source,
                publisher.semantic_dir
                / "audit/completed_round_reuse"
                / destination,
            )
        typed_root = repair_root / "first_query/planned_tool/typed_metric_spec"
        publisher.publish_copy(
            typed_root / "generated_tool.py",
            publisher.semantic_dir
            / "audit/completed_round_reuse/provider_generated_tool.py",
            allowed_suffixes=frozenset({".py"}),
        )
        typed_execution = _read_json(typed_root / "execution.json")
        successful_attempt = (
            typed_execution.get("generation") or {}
        ).get("successful_attempt")
        if isinstance(successful_attempt, int) and successful_attempt >= 0:
            attempt = typed_root / "attempts" / f"attempt_{successful_attempt}"
            for source_name, destination_name in (
                ("prompt.md", "provider_codegen_prompt.md"),
                ("response.txt", "provider_codegen_response.txt"),
                ("validation.json", "provider_codegen_validation.json"),
            ):
                publisher.publish_copy(
                    attempt / source_name,
                    publisher.semantic_dir
                    / "audit/completed_round_reuse"
                    / destination_name,
                    allowed_suffixes=frozenset(
                        {Path(source_name).suffix.lower()}
                    ),
                )

    final_payload = {
        "answer": feedback.get("answer"),
        "findings": feedback.get("findings"),
        "recommended_next_step": feedback.get("recommended_next_step"),
        "limitations": feedback.get("limitations"),
    }
    alignment_reaudit = _read_json(
        evaluation / "audit/semantic_alignment_reaudit.json"
    )
    alignment_reaudit_summary = None
    if alignment_reaudit.get("status") == "completed":
        alignment_reaudit_summary = {
            "act_rollouts_started": alignment_reaudit.get(
                "act_rollouts_started"
            ),
            "mutates_source_evaluation": alignment_reaudit.get(
                "mutates_source_evaluation"
            ),
            "rounds": [
                {
                    "round_id": item.get("round_id"),
                    "original_relationship": (
                        item.get("original_alignment") or {}
                    ).get("relationship"),
                    "recomputed_relationship": (
                        item.get("recomputed_alignment") or {}
                    ).get("relationship"),
                    "recomputed_coverage": (
                        item.get("recomputed_execution_trace") or {}
                    ).get("coverage_status"),
                    "pending_intent_fields": (
                        item.get("recomputed_execution_trace") or {}
                    ).get("pending_intent_fields"),
                }
                for item in alignment_reaudit.get("rounds", [])
                if isinstance(item, Mapping)
            ],
            "conclusion": alignment_reaudit.get("conclusion"),
        }
    reuse_summary = None
    if repair_result is not None:
        reuse_summary = {
            "repair_id": repair_result.get("repair_id"),
            "act_rollouts_started": repair_result.get("act_rollouts_started"),
            "first_query_route": repair_result.get("first_query_route"),
            "first_query_measurements": repair_result.get(
                "first_query_measurements"
            ),
            "exact_reuse_route": repair_result.get("exact_reuse_route"),
            "exact_reuse_provider_called": repair_result.get(
                "exact_reuse_provider_called"
            ),
            "aggregate_status": repair_result.get("aggregate_status"),
        }
        if acceptance_projection is not None:
            projection = acceptance_projection["projection"]
            reuse_summary["acceptance_projection"] = {
                "status": acceptance_projection.get("status"),
                "source_summary_path": acceptance_projection.get(
                    "source_summary_path"
                ),
                "projection_source": acceptance_projection.get(
                    "projection_source"
                ),
                "artifact": (
                    "artifacts/audit/completed_round_reuse/"
                    "acceptance_projection.json"
                ),
                "accepted": projection.get("accepted"),
                "candidate_execution_accepted": projection.get(
                    "candidate_execution_accepted"
                ),
            }

    final_aggregate = _read_json(
        evaluation / "summary/aggregate_result.json"
    )
    semantic_aggregate = _semantic_aggregate(final_aggregate)
    publisher.publish_json(
        semantic_aggregate,
        publisher.semantic_dir / "aggregate/final.json",
    )
    run_summary_path = publisher.publish_json(
        {
            "schema_version": 1,
            "evaluation_id": manifest.get("evaluation_id"),
            "source_evaluation": str(
                evaluation.relative_to(publisher.root)
            ).replace("\\", "/"),
            "query": query,
            "target": target,
            "plan": {
                "evaluation_goal": plan.get("evaluation_goal"),
                "planning_state": plan.get("planning_state"),
                "round_budget": session.get("round_budget")
                or plan.get("max_rounds"),
                "source_flagship_acceptance": summary.get(
                    "flagship_acceptance"
                ),
                "current_acceptance_projection": (
                    acceptance_projection["projection"]
                    if acceptance_projection is not None
                    else summary.get("flagship_acceptance")
                ),
            },
            "rounds": compact_rounds,
            "final_aggregate": semantic_aggregate,
            "answer": final_payload,
            "post_run_semantic_alignment_reaudit": (
                alignment_reaudit_summary
            ),
            "completed_round_reuse": reuse_summary,
        },
        publisher.bundle_root / "run_summary.json",
    )
    return FinalEvidenceProjection(
        final_payload=final_payload,
        alignment_reaudit_summary=alignment_reaudit_summary,
        repair_result=repair_result,
        reuse_summary=reuse_summary,
        run_summary_path=run_summary_path,
    )


__all__ = ["FinalEvidenceProjection", "publish_final_evidence"]
