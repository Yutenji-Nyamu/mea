import json

from mea.feedback.answer_evidence import project_answer_evidence
from mea.feedback.evidence_projection import _compact_decision


def _evidence():
    return {
        "schema_version": 3,
        "evaluation_id": "eval_projection",
        "query": "Where does this policy first become weak?",
        "total_policy_episodes": 1,
        "plan": {
            "max_rounds": 2,
            "executed_rounds": 1,
            "planning_state": "stopped_after_round_1",
            "round_budget_remaining": 1,
        },
        "rounds": [
            {
                "schema_version": 1,
                "round_id": "round_1",
                "candidate_id": "dynamic.alpha",
                "pipeline": {"passed": True, "failure_stage": None},
                "planning_observation": None,
                "policy": {
                    "success_rate": 0.0,
                    "metric": "generated_check_success",
                    "authority": "llm_generated_python_ast_validated",
                    "official_equivalent": False,
                    "execution_scope": "provider_generated_checker",
                    "seeds": [1000],
                },
                "rule": {
                    "requested": True,
                    "status": "passed",
                    "metric": "terminal_distance",
                    "route": "generated",
                    "source": None,
                    "results": [
                        {
                            "policy_name": "SmolVLA",
                            "seed": 1000,
                            "role": "policy_under_evaluation",
                            "metric": "terminal_distance",
                            "value": 0.12,
                            "unit": "m",
                            "passed": True,
                            "evidence_steps": [42],
                            "details": {},
                        }
                    ],
                },
                "vqa": {
                    "required": True,
                    "status": "abstained",
                    "evidence_conflict": True,
                    "observation": {
                        "confidence": 0.4,
                        "conflicts": ["visual and numeric evidence disagree"],
                    },
                },
                "outcome_semantics": {
                    "status": "expected_semantic_extension",
                    "evidence_conflict": False,
                },
                "scene_change": None,
            }
        ],
        "plan_agent_session": {
            "assessment": {
                "should_stop": True,
                "evidence_sufficient": False,
                "stop_reason": "agent_stop",
                "claim_verdict": "inconclusive",
                "limitations": ["N=1"],
                "observed_candidate_ids": ["dynamic.alpha"],
                "untested_candidate_ids": [],
                "conflict_candidate_ids": ["dynamic.alpha"],
            },
            "query_answer": {
                "answered": False,
                "answer": "The bounded result remains inconclusive.",
                "claim_verdict": "inconclusive",
                "limitations": ["N=1"],
                "evidence_conflict": True,
                "evidence_refs": [{"path": "drop"}],
            },
            "records": [{"large": "drop"}],
        },
        "artifacts": {"aggregate": "drop.json"},
    }


def test_answer_projection_keeps_round_evidence_and_final_plan():
    evidence = _evidence()
    projected = project_answer_evidence(evidence)

    round_evidence = projected["rounds"][0]
    assert projected["rounds"] == evidence["rounds"]
    assert round_evidence["candidate_id"] == "dynamic.alpha"
    assert round_evidence["policy"]["success_rate"] == 0.0
    assert round_evidence["policy"]["seeds"] == [1000]
    assert round_evidence["rule"]["results"][0]["metric"] == (
        "terminal_distance"
    )
    assert round_evidence["vqa"]["status"] == "abstained"
    assert round_evidence["vqa"]["evidence_conflict"] is True
    assert projected["final_plan"]["stop_reason"] == "agent_stop"
    assert projected["final_plan"]["query_answer"]["answer"] == (
        "The bounded result remains inconclusive."
    )


def test_answer_projection_excludes_bundle_transport():
    serialized = json.dumps(project_answer_evidence(_evidence()))

    for excluded in (
        "artifacts",
        "records",
        "evidence_refs",
        "drop.json",
    ):
        assert excluded not in serialized


def test_compact_decision_reads_current_query_assessment():
    projected = _compact_decision(
        {
            "action": "stop",
            "answered_query": True,
            "query_assessment": {
                "evidence_sufficient": True,
                "claim_verdict": "supported",
                "stop_reason": "agent_stop",
            },
        }
    )

    assert projected["evidence_sufficient"] is True
    assert projected["claim_verdict"] == "supported"
    assert projected["stop_reason"] == "agent_stop"
