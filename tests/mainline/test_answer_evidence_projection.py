import json
from pathlib import Path

from mea.feedback.answer_evidence import project_answer_evidence
from mea.feedback.evidence_projection import _compact_decision
from mea.feedback.prototype import _feedback_prompt


def _evidence():
    return {
        "evaluation_id": "eval_projection",
        "user_request": "Where does this policy first become weak?",
        "total_episodes": 1,
        "rounds": [
            {
                "round_id": "round_1",
                "candidate_id": "dynamic.alpha",
                "sub_aspect": "object_pose",
                "route": "generic_provider_scene_checker_codegen",
                "seeds": [1000],
                "num_episodes": 1,
                "proposal": {"large": "must not enter the answer prompt"},
                "observations": {
                    "execution_backend": "SmolVLA",
                    "pipeline_passed": True,
                    "policy_success": 0.0,
                    "policy_outcome": {
                        "metric": "generated_check_success",
                        "authority": "llm_generated_python_ast_validated",
                        "value": False,
                        "official_equivalent": False,
                    },
                    "planning_observation": {
                        "kind": "candidate_unexecutable",
                        "failure_stage": "expert_fixture",
                        "diagnosis": "The requested relation was unsatisfiable.",
                        "policy_sample_count": 0,
                        "taskgen_result": {"large": "drop"},
                    },
                    "implementation_trace": {
                        "candidate_id": "dynamic.alpha",
                        "stage": "runtime",
                        "relationship": "direct",
                        "coverage_status": "complete",
                        "covered_intent_fields": ["requested_change"],
                        "pending_intent_fields": [],
                        "validation_evidence": {"large": "drop"},
                    },
                },
                "tool_evaluation": {
                    "route": "generated",
                    "source": {"source_code": "drop"},
                    "validation": {"oracle": "drop"},
                    "episodes": [
                        {
                            "policy_name": "SmolVLA",
                            "role": "policy_under_evaluation",
                            "seed": 1000,
                            "result": {
                                "tool": "terminal_distance",
                                "value": 0.12,
                                "unit": "m",
                                "passed": True,
                            },
                        }
                    ],
                },
                "execution_vqa": {
                    "status": "abstained",
                    "reason": "insufficient_visual_evidence",
                    "evidence_conflict": True,
                    "observation": {
                        "answer": None,
                        "confidence": 0.4,
                        "conflicts": ["two frozen calls disagreed"],
                        "raw_response": "drop",
                    },
                    "artifacts": {"montage": "drop.png"},
                },
                "aggregate": {"duplicated": "drop"},
                "artifacts": {"video": "drop.mp4"},
            }
        ],
        "observations": {
            "aggregate": {
                "status": "passed",
                "unique_episode_count": 1,
                "metrics": [{"metric": "terminal_distance"}],
            }
        },
        "plan": {
            "planning_state": "stopped_after_round_1",
            "round_decisions": [{"large": "drop"}],
        },
        "plan_agent_session": {
            "assessment": {
                "should_stop": True,
                "evidence_sufficient": True,
                "stop_reason": "evidence_sufficient",
                "claim_verdict": "supported",
                "limitations": ["N=1"],
            },
            "query_answer": {
                "answered": True,
                "answer": "A bounded weakness was observed.",
                "claim_verdict": "supported",
                "limitations": ["N=1"],
                "evidence_refs": [{"path": "drop"}],
            },
            "records": [{"large": "drop"}],
            "artifacts": {"latest_evidence": "drop"},
        },
        "history_retrieval": {
            "status": "completed",
            "matches": [{"large": "drop"}, {"large": "drop"}],
        },
        "artifacts": {"everything": "drop"},
    }


def test_answer_projection_keeps_decision_evidence_and_live_values():
    projected = project_answer_evidence(_evidence())

    round_evidence = projected["rounds"][0]
    assert round_evidence["candidate_id"] == "dynamic.alpha"
    assert round_evidence["execution"]["policy_success"] == 0.0
    assert round_evidence["planning_observation"]["failure_stage"] == (
        "expert_fixture"
    )
    assert round_evidence["implementation_trace"]["relationship"] == "direct"
    assert round_evidence["tool_measurements"][0]["metric"] == (
        "terminal_distance"
    )
    assert round_evidence["tool_measurements"][0]["value"] == 0.12
    assert round_evidence["execution_vqa"]["status"] == "abstained"
    assert round_evidence["execution_vqa"]["evidence_conflict"] is True
    assert projected["final_aggregate"]["unique_episode_count"] == 1
    assert projected["final_plan"]["stop_reason"] == "evidence_sufficient"
    assert projected["final_plan"]["query_answer"]["answer"] == (
        "A bounded weakness was observed."
    )
    assert projected["history_retrieval"]["match_count"] == 2


def test_answer_projection_excludes_transport_and_repeated_payloads():
    serialized = json.dumps(project_answer_evidence(_evidence()))

    for excluded in (
        "artifacts",
        "round_decisions",
        "records",
        "evidence_refs",
        "proposal",
        "source_code",
        "validation_evidence",
        "taskgen_result",
        '"aggregate": {"duplicated"',
    ):
        assert excluded not in serialized


def test_feedback_prompt_uses_projection_but_scope_still_uses_raw_evidence():
    prompt = _feedback_prompt(Path(__file__).resolve().parents[2], _evidence())

    assert "ANSWER EVIDENCE PROJECTION:" in prompt
    assert "EVIDENCE BUNDLE:" not in prompt
    assert "terminal_distance" in prompt
    assert "ANSWER SCOPE:" in prompt
    assert "round_decisions" not in prompt
    assert "drop.mp4" not in prompt


def test_compact_decision_reads_current_query_assessment():
    projected = _compact_decision(
        {
            "action": "stop",
            "answered_query": True,
            "query_assessment": {
                "evidence_sufficient": True,
                "claim_verdict": "supported",
                "stop_reason": "evidence_sufficient",
            },
        }
    )

    assert projected["evidence_sufficient"] is True
    assert projected["claim_verdict"] == "supported"
    assert projected["stop_reason"] == "evidence_sufficient"
