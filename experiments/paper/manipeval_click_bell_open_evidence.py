"""Package the frozen Batch25 ClickBell open-TaskGen run.

This is evidence assembly only. It never invokes a provider, simulator, expert,
or policy.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.feedback.answer_scope import build_answer_scope
from mea.taskgen.click_bell_distractor import (
    click_bell_distractor_rollout_execution,
)
from mea.toolkit.aggregate import aggregate_tool_executions


CUSTOM_ID = "run_20260726_batch25_click_bell_open_provider_gate0_v2"
CONTROL_ID = "run_20260726_batch25_click_bell_open_official_control_seed100405"
V1_ID = "run_20260726_batch25_click_bell_open_provider_gate0_v1"
RESULT_NAME = "batch25_click_bell_open_taskgen_v2"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _compact_checker_aggregate(
    aggregate: dict[str, Any],
    checker: dict[str, Any],
) -> dict[str, Any]:
    """Project the general Aggregate output to this bundle's one metric."""
    episodes = []
    for item in checker.get("episodes", []):
        result = item.get("result", {})
        value = result.get("value")
        if not isinstance(value, bool):
            raise ValueError("checker result must be boolean")
        episodes.append(
            {
                "seed": item.get("seed"),
                "policy_name": item.get("policy_name"),
                "value": value,
                "passed": result.get("passed") is True,
                "evidence_steps": result.get("evidence_steps", []),
            }
        )
    if aggregate.get("episode_result_count") != len(episodes):
        raise ValueError("Aggregate/checker episode counts disagree")
    true_count = sum(item["value"] for item in episodes)
    episode_count = len(episodes)
    return {
        "schema_version": 1,
        "representation": "compact_metric_summary",
        "status": aggregate.get("status"),
        "metric": checker["tool_spec"]["metric"],
        "value_kind": "boolean",
        "source_count": aggregate.get("source_count"),
        "episode_result_count": episode_count,
        "unique_episode_count": aggregate.get("unique_episode_count"),
        "valid_count": episode_count,
        "missing_count": 0,
        "invalid_count": 0,
        "true_count": true_count,
        "false_count": episode_count - true_count,
        "true_rate": true_count / episode_count if episode_count else None,
        "episodes": episodes,
    }


def main() -> None:
    generated = REPO_ROOT / "mea/generated_tasks"
    custom = generated / CUSTOM_ID
    control = generated / CONTROL_ID
    result = REPO_ROOT / "experiments/paper/results" / RESULT_NAME
    if result.exists():
        raise SystemExit(f"append-only result already exists: {result}")

    gate0 = _read(custom / "gate0_result.json")
    custom_act = _read(custom / "evaluation/act.json")
    control_manifest = _read(control / "manifest.json")
    if (
        gate0.get("status") != "taskgen_gate0_passed"
        or custom_act.get("passed") is not True
        or control_manifest.get("status") != "completed"
        or control_manifest["act_evaluation"].get("passed") is not True
    ):
        raise SystemExit("frozen runs do not satisfy the technical bundle gate")

    custom_episode = (
        custom
        / "evaluation/telemetry/act/episode_000_seed_100405"
    )
    control_episode = (
        control
        / "evaluation/telemetry/act/episode_000_seed_100405"
    )
    control_metadata = _read(control_episode / "episode.json")
    custom_metadata = _read(custom_episode / "episode.json")
    if (
        control_metadata.get("success") is not True
        or custom_metadata.get("success") is not True
    ):
        raise SystemExit("the frozen paired rollout is not the positive pair")

    checker = click_bell_distractor_rollout_execution(
        episode_dir=custom_episode,
        candidate_dir=custom,
    )
    aggregate = aggregate_tool_executions([checker])
    compact_aggregate = _compact_checker_aggregate(aggregate, checker)
    checker_result = checker["episodes"][0]["result"]
    evidence_packet = {
        "schema_version": 1,
        "query": gate0["query"],
        "total_episodes": 2,
        "seeds": [100405],
        "rounds": [
            {
                "template_id": "official_click_bell_control",
                "seed": 100405,
                "success": True,
                "authority": "official_check_success",
            },
            {
                "template_id": "similar_bell_offset_y_0.12",
                "seed": 100405,
                "success": True,
                "authority": "llm_generated_python_ast_validated",
            },
        ],
        "checker_aggregate_artifact": "checker_aggregate.json",
        "observations": {
            "pipeline_passed": True,
            "execution_vqa_conflict": False,
            "official_control_success": True,
            "generated_checker_success": True,
            "official_core_predicate_satisfied": True,
            "distractor_contact_latched": False,
            "distractor_latch_authority": (
                "logical_implication_of_validated_checker_success"
            ),
            "distractor_contact_event_recorded": False,
            "distractor_trace_coverage": (
                "not_registered_in_current_click_bell_task_schema"
            ),
        },
        "execution_vqa": {
            "status": "not_run",
            "provider_calls": 0,
            "evidence_conflict": False,
            "reason": (
                "A rollout VQA call would require another provider request; "
                "the validated generated checker is the outcome authority."
            ),
        },
        "query_sufficiency": {
            "observed_candidate_ids": [
                "official_click_bell_control",
                "similar_bell_offset_y_0.12",
            ],
            "untested_candidate_ids": [
                "other_similar_object_geometries",
                "other_distractor_offsets",
                "other_random_seeds",
            ],
            "conflict_candidate_ids": [],
            "evidence_sufficient": False,
            "should_stop": True,
            "stop_reason": "budget_exhausted",
            "claim_verdict": (
                "single_case_positive_but_reliability_inconclusive"
            ),
        },
    }
    answer_scope = build_answer_scope(evidence_packet)
    query_answer = {
        "schema_version": 1,
        "query": gate0["query"],
        "verdict": "bounded_positive_but_general_reliability_inconclusive",
        "answer": (
            "At seed 100405, ACT succeeded on both the unchanged ClickBell "
            "control and one provider-generated scene containing a second "
            "physical bell 0.12 m away. The generated checker accepted the "
            "target click and forbade any latched distractor contact. This "
            "supports one bounded positive case, not reliable generalization "
            "to similar objects."
        ),
        "official_control_success": True,
        "generated_checker_success": bool(checker_result["value"]),
        "official_core_predicate_satisfied": checker_result["details"][
            "official_core_predicate_satisfied"
        ],
        "distractor_contact_latched": checker_result["details"][
            "distractor_contact_latched"
        ],
        "distractor_latch_authority": checker_result["details"][
            "distractor_latch_authority"
        ],
        "answer_scope": answer_scope,
    }

    _write(result / "checker_execution.json", checker)
    _write(result / "checker_aggregate.json", compact_aggregate)
    _write(result / "evidence_packet.json", evidence_packet)
    _write(result / "answer_scope.json", answer_scope)
    _write(result / "query_answer.json", query_answer)
    _write(
        result / "technical_summary.json",
        {
            "schema_version": 1,
            "provider_codegen": {
                "v2_calls": gate0["candidate_manifest"][
                    "codegen_provenance"
                ]["provider_call_count"],
                "v2_local_regenerations": gate0["candidate_manifest"][
                    "codegen_provenance"
                ]["local_regeneration_count"],
                "fixture_pass_count": gate0["candidate_manifest"][
                    "checker_contract"
                ]["fixture_pass_count"],
                "fixture_count": gate0["candidate_manifest"][
                    "checker_contract"
                ]["fixture_count"],
            },
            "gate0": {
                "render_passed": gate0["render_passed"],
                "expert_passed": gate0["expert_passed"],
            },
            "act": {
                "rollout_count": 2,
                "retry_count": 0,
                "control_technical_passed": control_manifest[
                    "act_evaluation"
                ]["passed"],
                "custom_technical_passed": custom_act["passed"],
                "video_count": 2,
                "telemetry_episode_count": 2,
            },
        },
    )

    v1_root = REPO_ROOT / "mea/generated_task_attempts" / V1_ID
    v1_summary = _read(v1_root / "task_generation_attempt_summary.json")
    _write(
        result / "v1_rag_gap_failure.json",
        {
            "schema_version": 1,
            "status": "frozen_negative_codegen_result",
            "provider_calls": v1_summary["runtime"]["provider_calls"],
            "local_regenerations": v1_summary["regenerations_used"],
            "simulator_probes": 0,
            "act_rollouts": 0,
            "attempts": [
                {
                    "failure": item["failure"],
                    "provider_response": _read(
                        v1_root
                        / f"attempt_{item['attempt_index']:02d}"
                        / "provider_response.txt"
                    ),
                }
                for item in v1_summary["attempts"]
            ],
            "diagnosis": (
                "The v1 prompt used abstract RAG prose but did not retrieve "
                "the official ClickBell methods or exact public API."
            ),
        },
    )

    copies = {
        custom / "proposal_prompt.md": result / "taskgen_prompt.md",
        custom / "provider_response.txt": result / "provider_response.json",
        custom / "task.py": result / "task.py",
        custom / "checker_fixtures.json": result / "checker_fixtures.json",
        custom / "evidence/initial_head.png": result / "scene.png",
        custom / "evaluation/episode0.mp4": result / "custom_video.mp4",
        custom_episode / "episode.json": result / "custom_episode.json",
        custom_episode / "events.jsonl": result / "custom_events.jsonl",
        control / "evaluation/episode0.mp4": result / "control_video.mp4",
        control_episode / "episode.json": result / "control_episode.json",
        control_episode / "events.jsonl": result / "control_events.jsonl",
    }
    for source, target in copies.items():
        _copy(source, target)

    _write(
        result / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete_bounded_positive_case",
            "query_source": (
                "../batch25_bound_click_open_plan_v2/free_concern.json"
            ),
            "retrieval_source": (
                "../batch25_bound_click_open_plan_v2/"
                "deterministic_repair_replay.json"
            ),
            "raw_custom_run": f"mea/generated_tasks/{CUSTOM_ID}",
            "raw_control_run": f"mea/generated_tasks/{CONTROL_ID}",
            "raw_v1_negative": f"mea/generated_task_attempts/{V1_ID}",
            "artifacts": sorted(
                [
                    "answer_scope.json",
                    "checker_aggregate.json",
                    "checker_execution.json",
                    "checker_fixtures.json",
                    "control_episode.json",
                    "control_events.jsonl",
                    "control_video.mp4",
                    "custom_episode.json",
                    "custom_events.jsonl",
                    "custom_video.mp4",
                    "evidence_packet.json",
                    "provider_response.json",
                    "query_answer.json",
                    "scene.png",
                    "task.py",
                    "taskgen_prompt.md",
                    "technical_summary.json",
                    "v1_rag_gap_failure.json",
                ]
            ),
            "policy_performance_scope": (
                "one control and one custom episode at the same seed"
            ),
        },
    )
    (result / "README.md").write_text(
        "# Batch25 ClickBell open TaskGen\n\n"
        "The online v2 resolver originally failed its routing gate and did not "
        "select the policy-compatible `click_bell` base task. A subsequent "
        "deterministic replay made zero provider calls and selected `click_bell`; "
        "the standalone TaskGen/ACT scripts then retrieved the official methods, "
        "generated one physical second-bell scene plus replacement checker, and "
        "ran Gate0, official-control ACT, and custom ACT.\n\n"
        "This is an explicitly composed evidence bundle, not a unified Agent "
        "automatically completing a two-round chain from Query through planning, "
        "TaskGen, rollout, replanning, and answer.\n\n"
        "The bounded result is positive at one shared seed. It is not evidence "
        "that ACT is generally reliable around similar objects. The current "
        "rollout did not independently trace the distractor latch; latch=false "
        "is a logical implication of validated-checker success. See "
        "`query_answer.json` and `answer_scope.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps(_read(result / "manifest.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
