"""Replay ToolGen and ClaimFirst evidence over a completed ACT round.

This is a repair/provenance utility, not a production execution path.  It never
starts TaskGen, a simulator, or ACT, and it never rewrites the source
evaluation.  All generated artifacts and the run-local Tool registry live
under ``repairs/<repair_id>`` in the source evaluation directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.planner import ClaimFirstRuntimeController, build_claim_first_evidence_record
from mea.providers import OpenAICompatibleProvider
from mea.round_evidence import aggregate_round_results
from mea.toolgen import execute_tool_request
from scripts.manipeval_agent import (
    compact_aggregate_result,
    summarize_round,
)


class CompletedToolReplayError(RuntimeError):
    """Raised when a source run cannot be replayed without ambiguity."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CompletedToolReplayError(f"missing source artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompletedToolReplayError(f"source artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _tool_execution_sha256(execution: Mapping[str, Any]) -> str | None:
    """Return one internally consistent Tool code hash."""

    declared: list[Any] = []
    source = execution.get("source")
    if isinstance(source, Mapping) and source.get("tool_sha256") is not None:
        declared.append(source.get("tool_sha256"))
    episodes = execution.get("episodes")
    if isinstance(episodes, list):
        for episode in episodes:
            result = (
                episode.get("result")
                if isinstance(episode, Mapping)
                else None
            )
            if isinstance(result, Mapping) and result.get("tool_sha256") is not None:
                declared.append(result.get("tool_sha256"))
    if not declared or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in declared
    ):
        return None
    hashes = set(declared)
    return hashes.pop() if len(hashes) == 1 else None


def _route_matches_request(
    execution: Mapping[str, Any],
    request: Mapping[str, Any],
) -> bool:
    decision = execution.get("route_decision")
    return bool(
        isinstance(decision, Mapping)
        and decision.get("status") == "resolved"
        and decision.get("resolved_route") == execution.get("route")
        and decision.get("task_name") == request.get("task_name")
        and decision.get("metric") == request.get("metric")
    )


def _exact_reuse_kind(
    first: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> str | None:
    first_request = first.get("tool_request")
    replay_request = replay.get("tool_request")
    first_validation = first.get("validation")
    replay_validation = replay.get("validation")
    if not (
        first.get("status") == replay.get("status") == "passed"
        and isinstance(first_request, Mapping)
        and isinstance(replay_request, Mapping)
        and isinstance(first_validation, Mapping)
        and isinstance(replay_validation, Mapping)
        and dict(first_request) == dict(replay_request)
        and replay_validation.get("provider_called") is False
        and _route_matches_request(first, first_request)
        and _route_matches_request(replay, replay_request)
    ):
        return None
    first_source = first.get("source")
    replay_source = replay.get("source")
    if not isinstance(first_source, Mapping) or not isinstance(
        replay_source, Mapping
    ):
        return None
    first_tool = first_source.get("tool")
    replay_tool = replay_source.get("tool")
    first_hash = _tool_execution_sha256(first)
    replay_hash = _tool_execution_sha256(replay)

    if replay.get("route") == "run_local_reuse":
        first_registration = first_source.get("registration_id")
        replay_registration = replay_source.get("registration_id")
        if not (
            first.get("route")
            in {
                "force_codegen",
                "provider_python_codegen",
                "typed_metric_spec_compile",
            }
            and first_source.get("scope") == "run_local_generated"
            and replay_source.get("scope") == "run_local_registry"
            and isinstance(first_registration, str)
            and bool(first_registration.strip())
            and first_registration == replay_registration
            and isinstance(first_tool, str)
            and bool(first_tool.strip())
            and first_tool == replay_tool
            and first_hash is not None
            and first_hash == replay_hash
        ):
            return None
        return "run_local_registry"

    first_route = first.get("route_decision")
    replay_route = replay.get("route_decision")
    if not isinstance(first_route, Mapping) or not isinstance(
        replay_route, Mapping
    ):
        return None
    if not (
        first.get("route") == replay.get("route") == "reuse"
        and first_source.get("scope")
        == replay_source.get("scope")
        == "trusted_catalog"
        and isinstance(first_tool, str)
        and bool(first_tool.strip())
        and first_tool == replay_tool
        and first_hash is not None
        and first_hash == replay_hash
        and first_route.get("exact_match") is True
        and replay_route.get("exact_match") is True
        and first_validation.get("provider_called") is False
    ):
        return None
    return "trusted_catalog"


def _source_context(
    repo_root: Path,
    evaluation_id: str,
    round_id: str,
) -> dict[str, Any]:
    evaluation_dir = repo_root / "mea/evaluation_runs" / evaluation_id
    manifest_path = evaluation_dir / "manifest.json"
    plan_path = evaluation_dir / "plan/evaluation_plan.json"
    contract_path = evaluation_dir / "plan/query_sufficiency_contract.json"
    summary_path = evaluation_dir / "summary" / f"{round_id}.json"
    manifest = _read_json(manifest_path)
    plan = _read_json(plan_path)
    contract = _read_json(contract_path)
    summary_available = summary_path.is_file()
    child_run_path = evaluation_dir / "execution" / round_id / "child_run.json"
    if summary_available:
        summary = _read_json(summary_path)
        child_run_id = summary.get("taskgen_run_id")
    else:
        expected_failure_stage = f"{round_id}_execution"
        if (
            manifest.get("status") != "failed"
            or manifest.get("failure_stage") != expected_failure_stage
        ):
            raise CompletedToolReplayError(
                "missing round summary is replayable only when the parent failed "
                "during that exact completed round"
            )
        child_run = _read_json(child_run_path)
        if (
            child_run.get("returncode") != 0
            or child_run.get("status") != "completed"
        ):
            raise CompletedToolReplayError(
                "missing-summary replay requires a completed child_run.json"
            )
        child_run_id = child_run.get("run_id")
        summary = {
            "round_id": round_id,
            "taskgen_run_id": child_run_id,
            "taskgen_returncode": 0,
            "observations": {},
            "source": "synthesized_from_completed_child_run",
        }
    round_plans = plan.get("rounds")
    if not isinstance(round_plans, list):
        raise CompletedToolReplayError("evaluation plan has no rounds list")
    matches = [
        deepcopy(item)
        for item in round_plans
        if isinstance(item, Mapping) and item.get("round_id") == round_id
    ]
    if len(matches) != 1:
        raise CompletedToolReplayError(
            f"expected one source round {round_id!r}, found {len(matches)}"
        )
    if not isinstance(child_run_id, str) or not child_run_id.strip():
        raise CompletedToolReplayError(
            "round summary or child_run.json has no taskgen_run_id"
        )
    child_dir = repo_root / "mea/generated_tasks" / child_run_id
    child_manifest_path = child_dir / "manifest.json"
    child_manifest = _read_json(child_manifest_path)
    if child_manifest.get("status") not in {"completed", "completed_without_act"}:
        raise CompletedToolReplayError("source child run did not complete")
    telemetry = sorted(
        (child_dir / "evaluation/telemetry").glob("act/episode_*/episode.json")
    )
    if not telemetry:
        raise CompletedToolReplayError(
            "source child run has no completed ACT telemetry episode"
        )
    return {
        "evaluation_dir": evaluation_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "plan": plan,
        "plan_path": plan_path,
        "contract": contract,
        "contract_path": contract_path,
        "summary": summary,
        "summary_path": summary_path if summary_available else None,
        "summary_available": summary_available,
        "child_run_path": child_run_path if not summary_available else None,
        "round_plan": matches[0],
        "child_dir": child_dir,
        "child_manifest": child_manifest,
        "child_manifest_path": child_manifest_path,
        "telemetry_paths": telemetry,
    }


def _evolved_query_contract(
    plan: Mapping[str, Any],
    *,
    round_id: str,
    initial_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Recover the contract in force when a dynamically proposed round ran."""

    decisions = plan.get("round_decisions")
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, Mapping):
                continue
            next_round = decision.get("next_round")
            if not isinstance(next_round, Mapping):
                continue
            if next_round.get("round_id") != round_id:
                continue
            assessment = decision.get("query_assessment")
            if isinstance(assessment, Mapping) and isinstance(
                assessment.get("contract"), Mapping
            ):
                return (
                    deepcopy(dict(assessment["contract"])),
                    "round_decision.query_assessment.contract",
                )
        for decision in reversed(decisions):
            if not isinstance(decision, Mapping):
                continue
            for assessment_key in ("evidence_assessment", "query_assessment"):
                assessment = decision.get(assessment_key)
                if isinstance(assessment, Mapping) and isinstance(
                    assessment.get("contract"), Mapping
                ):
                    return (
                        deepcopy(dict(assessment["contract"])),
                        f"round_decision.{assessment_key}.contract",
                    )
    return deepcopy(dict(initial_contract)), "plan.query_sufficiency_contract"


def replay_completed_round_tool(
    repo_root: str | Path,
    *,
    evaluation_id: str,
    round_id: str,
    repair_id: str,
    tool_request_path: str | Path,
    execution_vqa_path: str | Path | None = None,
    provider: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Replay one query-induced Tool twice and recompute planner-facing evidence."""

    root = Path(repo_root).expanduser().resolve()
    context = _source_context(root, evaluation_id, round_id)
    repair_dir = context["evaluation_dir"] / "repairs" / repair_id
    if repair_dir.exists():
        raise CompletedToolReplayError(
            f"repair output already exists; choose a new repair_id: {repair_dir}"
        )
    request_path = Path(tool_request_path).expanduser().resolve()
    bundle = _read_json(request_path)
    request = bundle.get("tool_request", bundle)
    if not isinstance(request, dict):
        raise CompletedToolReplayError("Tool request bundle has no tool_request")
    replacement_vqa_path = (
        Path(execution_vqa_path).expanduser().resolve()
        if execution_vqa_path is not None
        else None
    )
    replacement_vqa = None
    if replacement_vqa_path is not None:
        replacement_vqa_bundle = _read_json(replacement_vqa_path)
        replacement_vqa = replacement_vqa_bundle.get(
            "result", replacement_vqa_bundle
        )
        if not isinstance(replacement_vqa, Mapping):
            raise CompletedToolReplayError(
                "replacement Execution VQA artifact has no result object"
            )
        replacement_vqa = deepcopy(dict(replacement_vqa))

    original_request_paths = [
        context["evaluation_dir"]
        / "execution"
        / round_id
        / "open_tool_request/tool_request_bundle.json",
        context["evaluation_dir"]
        / "execution"
        / round_id
        / "planned_tool/tool_request.json",
    ]
    original_request_path = next(
        (path for path in original_request_paths if path.is_file()),
        None,
    )
    source_paths = [
        context["manifest_path"],
        context["plan_path"],
        context["contract_path"],
        context["child_manifest_path"],
        *context["telemetry_paths"],
    ]
    if context["summary_path"] is not None:
        source_paths.append(context["summary_path"])
    if context["child_run_path"] is not None:
        source_paths.append(context["child_run_path"])
    if original_request_path is not None:
        source_paths.append(original_request_path)
    if replacement_vqa_path is not None:
        source_paths.append(replacement_vqa_path)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "repair_id": repair_id,
        "source_evaluation_id": evaluation_id,
        "source_round_id": round_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "completed_act_tool_aggregate_planner_replay",
        "act_rollouts_started": 0,
        "source_artifacts_immutable": True,
        "source_round_summary": (
            "recorded"
            if context["summary_available"]
            else "synthesized_from_completed_child_run"
        ),
        "source_artifacts": [
            {"path": _relative(path, root), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "original_tool_request": (
            {
                "status": "recorded",
                "path": _relative(original_request_path, root),
                "sha256": _sha256(original_request_path),
            }
            if original_request_path is not None
            else {"status": "not_found", "path": None, "sha256": None}
        ),
        "repaired_tool_request": {
            "path": _relative(request_path, root),
            "sha256": _sha256(request_path),
            "request": deepcopy(request),
        },
        "execution_vqa": (
            {
                "source": "append_only_replay",
                "path": _relative(replacement_vqa_path, root),
                "sha256": _sha256(replacement_vqa_path),
            }
            if replacement_vqa_path is not None
            else {
                "source": (
                    "original_round_summary"
                    if context["summary_available"]
                    else "not_available"
                ),
                "path": None,
                "sha256": None,
            }
        ),
        "status": "running",
    }
    _write_json(repair_dir / "repair_provenance.json", provenance)

    repaired_plan = deepcopy(context["round_plan"])
    repaired_plan["tool_request"] = deepcopy(request)
    repaired_plan["open_tool_request_deferred"] = False
    registry = repair_dir / "tool_registry"
    try:
        first = execute_tool_request(
            root,
            context["child_dir"],
            repair_dir / "first_query/planned_tool",
            request,
            provider=provider,
            model=model,
            run_local_registry_dir=registry,
        )
        replay = execute_tool_request(
            root,
            context["child_dir"],
            repair_dir / "second_query_exact_reuse/planned_tool",
            request,
            provider=provider,
            model=model,
            run_local_registry_dir=registry,
        )
        exact_reuse_kind = _exact_reuse_kind(first, replay)
        if exact_reuse_kind is None:
            raise CompletedToolReplayError(
                "second identical Query did not resolve to the same validated "
                "run-local or trusted-catalog Tool"
            )
        aggregate = aggregate_round_results(
            repaired_plan,
            context["child_manifest"],
            first,
            repair_dir / "aggregate_result.json",
        )
        source_vqa = replacement_vqa
        if source_vqa is None:
            source_vqa = (
                context["summary"].get("observations", {}).get("execution_vqa")
            )
        repaired_summary = summarize_round(
            repaired_plan,
            context["child_manifest"],
            context["child_dir"],
            first,
            aggregate,
            source_vqa if isinstance(source_vqa, dict) else None,
            int(context["summary"].get("taskgen_returncode") or 0),
        )
        repaired_summary["execution_artifact_dir"] = _relative(repair_dir, root)
        repaired_summary["evidence_artifact_paths"] = {
            "round_aggregate": _relative(
                repair_dir / "aggregate_result.json", root
            ),
            "tool_execution": str(
                first.get("artifacts", {}).get("tool_execution") or ""
            ),
        }
        _write_json(repair_dir / "repaired_round_summary.json", repaired_summary)
        evidence_record = build_claim_first_evidence_record(
            repaired_plan,
            repaired_summary,
        )
        _write_json(
            repair_dir / "claim_first_evidence_record.json",
            evidence_record,
        )

        all_plans = [
            repaired_plan
            if item.get("round_id") == round_id
            else deepcopy(item)
            for item in context["plan"]["rounds"]
        ]
        all_summaries = []
        for item in all_plans:
            item_round_id = str(item["round_id"])
            all_summaries.append(
                repaired_summary
                if item_round_id == round_id
                else _read_json(
                    context["evaluation_dir"]
                    / "summary"
                    / f"{item_round_id}.json"
                )
            )
        target = context["manifest"].get("evaluation_target")
        request_text = context["manifest"].get("user_request")
        if not isinstance(request_text, str) or not request_text.strip():
            request_artifact = context["evaluation_dir"] / "request.json"
            request_text = _read_json(request_artifact).get("user_request")
        if not isinstance(target, dict) or not isinstance(request_text, str):
            raise CompletedToolReplayError(
                "source manifest lacks evaluation_target or original request"
            )
        replay_contract, replay_contract_source = _evolved_query_contract(
            context["plan"],
            round_id=round_id,
            initial_contract=context["contract"],
        )
        controller = ClaimFirstRuntimeController(
            request_text,
            target,
            query_contract=replay_contract,
        )
        planner_replay = controller.observe(all_plans, all_summaries)
        planner_replay["query_contract_source"] = replay_contract_source
        _write_json(repair_dir / "claim_first_planner_replay.json", planner_replay)

        result = {
            "schema_version": 1,
            "status": "completed",
            "source_evaluation_id": evaluation_id,
            "source_round_id": round_id,
            "repair_id": repair_id,
            "act_rollouts_started": 0,
            "first_query_route": first.get("route"),
            "first_query_provider_called": (
                first.get("validation", {}).get("provider_called")
            ),
            "first_query_measurements": [
                item.get("result", {}).get("value")
                for item in first.get("episodes", [])
            ],
            "exact_reuse_route": replay.get("route"),
            "exact_reuse_kind": exact_reuse_kind,
            "exact_reuse_provider_called": (
                replay.get("validation", {}).get("provider_called")
            ),
            "aggregate_status": aggregate.get("status"),
            "aggregate": compact_aggregate_result(aggregate),
            "evidence_strength": evidence_record["evidence_packet"][
                "evidence_strength"
            ],
            "query_contract_source": replay_contract_source,
            "execution_vqa_source": provenance["execution_vqa"],
            "planner_assessment": planner_replay["assessment"],
            "artifacts": {
                "repair_provenance": _relative(
                    repair_dir / "repair_provenance.json", root
                ),
                "first_tool": first.get("artifacts", {}).get("tool_execution"),
                "exact_reuse_tool": replay.get("artifacts", {}).get(
                    "tool_execution"
                ),
                "aggregate": _relative(
                    repair_dir / "aggregate_result.json", root
                ),
                "evidence_record": _relative(
                    repair_dir / "claim_first_evidence_record.json", root
                ),
                "planner_replay": _relative(
                    repair_dir / "claim_first_planner_replay.json", root
                ),
            },
        }
        _write_json(repair_dir / "result.json", result)
        provenance.update(
            {
                "status": "completed",
                "completed_at": datetime.now().astimezone().isoformat(),
                "result_path": _relative(repair_dir / "result.json", root),
            }
        )
        _write_json(repair_dir / "repair_provenance.json", provenance)
        return result
    except Exception as exc:
        provenance.update(
            {
                "status": "failed",
                "completed_at": datetime.now().astimezone().isoformat(),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_json(repair_dir / "repair_provenance.json", provenance)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--repair-id", required=True)
    parser.add_argument(
        "--tool-request",
        type=Path,
        required=True,
        help=(
            "New legal Tool request JSON, or a bundle containing tool_request. "
            "It is recorded separately from the original failed request."
        ),
    )
    parser.add_argument(
        "--execution-vqa",
        type=Path,
        help=(
            "Optional append-only cached VQA replay manifest or result. When "
            "provided, its result replaces the source round VQA before "
            "Aggregate/EvidencePacket/Planner replay."
        ),
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--allow-provider-codegen",
        action="store_true",
        help=(
            "Allow a Tool request that needs codegen to use UIUI_API_KEY from "
            "the current process. Typed MetricSpec/reuse paths need no provider."
        ),
    )
    args = parser.parse_args()
    provider = None
    if args.allow_provider_codegen:
        if not args.model:
            parser.error("--model is required with --allow-provider-codegen")
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=args.model,
            vision_model=args.model,
        )
    elif os.getenv("UIUI_API_KEY") and args.model:
        parser.error(
            "credential is present but provider use was not explicitly selected "
            "for this repair command"
        )
    result = replay_completed_round_tool(
        args.repo_root,
        evaluation_id=args.evaluation_id,
        round_id=args.round_id,
        repair_id=args.repair_id,
        tool_request_path=args.tool_request,
        execution_vqa_path=args.execution_vqa,
        provider=provider,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
