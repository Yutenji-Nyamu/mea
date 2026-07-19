"""Plan and execute a bounded, evidence-driven multi-round MEA evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.execution_vqa import build_execution_vqa_query, run_execution_vqa
from mea.capability_adapter import (
    CapabilityAdapterError,
    build_contract_tool_request,
    taskgen_route,
    validate_capability_contract,
)
from mea.feedback import FeedbackAgent, render_evaluation_report
from mea.history import EvaluationHistoryDB
from mea.planner import (
    ClickBellAdaptivePlanAgent,
    ClickBellFixedSuitePlanAgent,
    ClickBellPositionPlanAgent,
    GlobalQueryRouter,
    OfficialTaskPlanAgent,
    PlanAgentPrototype,
    build_act_catalog,
    make_evaluation_id,
    route_to_planner_proposal,
)
from mea.providers import (
    OpenAICompatibleProvider,
    available_model_profiles,
    resolve_model_profile,
)
from mea.recovery import (
    BoundedRecoveryError,
    UnexpectedToolExecutionError,
    run_bounded_tool_recovery,
)
from mea.round_recovery import (
    StageFailure,
    WholeRoundRecoveryError,
    run_stage_aware_round_recovery,
)
from mea.toolgen import ToolOrchestrationError, execute_tool_request
from mea.toolkit import aggregate_tool_executions
from mea.strategy_plan import (
    StrategyPlanError,
    load_registered_execution,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def directory_tree_sha256(root: Path) -> str:
    """Hash immutable Tool inputs without depending on filesystem mtimes."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"telemetry directory does not exist: {resolved}")
    digest = hashlib.sha256()
    files = sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    if not files:
        raise RuntimeError(f"telemetry directory is empty: {resolved}")
    for path in files:
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def update_manifest(evaluation_dir: Path, **updates: Any) -> dict[str, Any]:
    path = evaluation_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def write_global_route_trace(
    evaluation_dir: Path,
    *,
    catalog: dict[str, Any],
    route_result: dict[str, Any],
    router: GlobalQueryRouter,
    history_retrieval: dict[str, Any],
) -> None:
    """Persist the bounded global route without leaking credentials."""

    write_json(evaluation_dir / "plan/global_act_catalog.json", catalog)
    write_json(
        evaluation_dir / "plan/global_query_route.json",
        {
            **route_result,
            "history_retrieval": history_retrieval,
        },
    )
    if router.last_prompt is not None:
        (evaluation_dir / "plan/global_query_prompt.md").write_text(
            router.last_prompt, encoding="utf-8"
        )
    for index, response in enumerate(router.last_responses, start=1):
        (evaluation_dir / f"plan/global_query_response_{index}.txt").write_text(
            response + "\n", encoding="utf-8"
        )


def finish_unsupported_global_route(
    repo_root: Path,
    *,
    evaluation_id: str | None,
    user_request: str,
    catalog: dict[str, Any],
    route_result: dict[str, Any],
    router: GlobalQueryRouter,
    history_retrieval: dict[str, Any],
) -> dict[str, Any]:
    """Create an auditable no-execution result for an unsupported query."""

    resolved_id = evaluation_id or make_evaluation_id()
    if not re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id):
        raise ValueError("evaluation_id must match eval_[A-Za-z0-9_]+")
    evaluation_dir = repo_root / "mea/evaluation_runs" / resolved_id
    if evaluation_dir.exists():
        raise RuntimeError(f"evaluation directory already exists: {evaluation_dir}")
    for child in ("plan", "execution", "summary"):
        (evaluation_dir / child).mkdir(parents=True, exist_ok=False)
    write_json(evaluation_dir / "request.json", {"user_request": user_request})
    write_global_route_trace(
        evaluation_dir,
        catalog=catalog,
        route_result=route_result,
        router=router,
        history_retrieval=history_retrieval,
    )
    manifest = {
        "schema_version": 1,
        "evaluation_id": resolved_id,
        "status": "unsupported",
        "lifecycle_status": "completed_without_execution",
        "created_at": datetime.now().astimezone().isoformat(),
        "execution_finished_at": datetime.now().astimezone().isoformat(),
        "user_request": user_request,
        "auto_route": True,
        "global_query_route_path": "plan/global_query_route.json",
        "global_act_catalog_path": "plan/global_act_catalog.json",
        "route": route_result["selection"],
        "limitations": ["query requires an aspect outside the trusted ACT catalog"],
    }
    write_json(evaluation_dir / "manifest.json", manifest)
    return manifest


def child_run_id(evaluation_id: str, round_id: str) -> str:
    return f"run_{evaluation_id.removeprefix('eval_')}_{round_id}"


def round_execution_backend(round_plan: dict[str, Any]) -> str:
    """Resolve policy execution independently from the TaskGen route."""

    raw = (round_plan.get("execution") or {}).get("backend")
    if raw is None:
        raw = "expert" if round_plan.get("route") == "official" else "act"
    backend = str(raw).casefold()
    if backend not in {"expert", "act", "both"}:
        raise ValueError(f"unsupported execution backend: {raw!r}")
    return backend


def validate_round_capability_contract(
    round_plan: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind every duplicated runtime field to one trusted adapter contract."""

    raw = round_plan.get("capability_contract")
    if raw is None:
        return None
    try:
        contract = validate_capability_contract(raw)
        expected_tool = build_contract_tool_request(contract)
    except (CapabilityAdapterError, ValueError) as exc:
        raise ValueError(f"invalid round capability contract: {exc}") from exc
    taskgen = contract["taskgen"]
    expected = {
        "task_name": contract["task_name"],
        "template_id": contract["template_id"],
        "capability_id": taskgen["capability_id"],
        "task_variant_id": taskgen["task_variant_id"],
        "sub_aspect": contract["aspect"]["aspect_id"],
        "route": taskgen_route(contract),
        "variant_hint": taskgen["changes"],
        "tool_request": expected_tool,
        "vqa_phenomenon_ids": contract["vqa"]["phenomenon_ids"],
        "required_gates": contract["required_gates"],
    }
    observed = {
        "task_name": str(round_plan.get("task_name") or "beat_block_hammer"),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "task_variant_id": round_plan.get("task_variant_id"),
        "sub_aspect": round_plan.get("sub_aspect"),
        "route": round_plan.get("route"),
        "variant_hint": round_plan.get("variant_hint") or {},
        "tool_request": round_plan.get("tool_request"),
        "vqa_phenomenon_ids": round_plan.get("vqa_phenomenon_ids"),
        "required_gates": (round_plan.get("execution") or {}).get("gates"),
    }
    mismatches = sorted(key for key in expected if observed[key] != expected[key])
    if mismatches:
        raise ValueError(
            "round fields differ from capability contract: " + ", ".join(mismatches)
        )
    return contract


def build_taskgen_command(
    repo_root: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    telemetry_profile: str = "balanced_v1",
    registration_identity: dict[str, Any] | None = None,
    run_id_suffix: str = "",
) -> tuple[list[str], str]:
    capability_contract = validate_round_capability_contract(round_plan)
    if run_id_suffix and re.fullmatch(r"_[A-Za-z0-9_]+", run_id_suffix) is None:
        raise ValueError("run_id_suffix must be empty or a safe underscore suffix")
    run_id = child_run_id(evaluation_id, round_plan["round_id"]) + run_id_suffix
    execution = round_plan["execution"]
    seed = execution["seeds"][0]
    task_name = (
        capability_contract["task_name"]
        if capability_contract is not None
        else str(round_plan.get("task_name") or "beat_block_hammer")
    )
    task_module = round_plan.get("task_module")
    route = (
        taskgen_route(capability_contract)
        if capability_contract is not None
        else str(round_plan["route"])
    )
    execution_backend = round_execution_backend(round_plan)
    command = [
        sys.executable,
        str(repo_root / "scripts/manipeval_taskgen.py"),
        "--repo-root",
        str(repo_root),
        "--request",
        round_plan["task_instruction"],
        "--run-id",
        run_id,
        "--task-name",
        task_name,
        "--mode",
        route,
        "--text-model",
        text_model,
        "--vision-model",
        vision_model,
        "--seed",
        str(seed),
        "--num-episodes",
        str(execution["num_episodes"]),
        "--gpu",
        str(gpu),
        "--telemetry-profile",
        telemetry_profile,
        "--probe",
        "--max-reflections",
        str(max_reflections),
    ]
    if task_module:
        command.extend(["--task-module", str(task_module)])
    if round_plan.get("variant_hint"):
        command.extend(
            [
                "--variant-hint-json",
                json.dumps(
                    round_plan["variant_hint"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    task_variant_id = round_plan.get("task_variant_id")
    if task_variant_id:
        command.extend(["--variant-id", str(task_variant_id)])
    elif (
        round_plan.get("template_id")
        and round_plan.get("capability_contract") is None
    ):
        # Compatibility for hand-authored legacy plans that predate the
        # capability adapter's template/task-variant identity split.
        command.extend(["--variant-id", str(round_plan["template_id"])])
    if round_plan.get("capability_contract") is not None:
        command.extend(
            [
                "--capability-contract-json",
                json.dumps(
                    round_plan["capability_contract"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if route == "official":
        if execution_backend in {"expert", "both"}:
            command.append("--expert")
        if execution_backend in {"act", "both"}:
            command.append("--run-act")
    else:
        # The bounded generated-task prototype keeps its original expert
        # solvability gate before the ACT policy rollout.
        command.extend(["--expert", "--vision-check", "--run-act"])
    if base_url:
        command.extend(["--base-url", base_url])
    if registration_identity is not None:
        command.extend(
            [
                "--registration-identity-json",
                json.dumps(
                    registration_identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    return command, run_id


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def read_policy_success(result_path: Path) -> float | None:
    if not result_path.is_file():
        return None
    for line in reversed(result_path.read_text(encoding="utf-8").splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


def compact_trusted_tools(child_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the numerical Toolkit evidence small enough for planner/feedback use."""

    evaluation = child_manifest.get("trusted_tool_evaluation") or {}
    episodes = []
    for episode in evaluation.get("episodes", []):
        episodes.append(
            {
                "episode_dir": episode.get("episode_dir"),
                "policy_name": episode.get("policy_name"),
                "seed": episode.get("seed"),
                "success": episode.get("success"),
                "results": [
                    {
                        "tool": result.get("tool"),
                        "value": result.get("value"),
                        "unit": result.get("unit"),
                        "passed": result.get("passed"),
                        "evidence_steps": result.get("evidence_steps", []),
                        "details": result.get("details", {}),
                    }
                    for result in episode.get("tool_results", [])
                ],
            }
        )
    return episodes


def compact_tool_evaluation(
    tool_evaluation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep planned Tool evidence compact while preserving ACT/expert roles."""

    if not tool_evaluation:
        return None
    return {
        "status": tool_evaluation.get("status"),
        "requested_route": tool_evaluation.get("requested_route"),
        "route": tool_evaluation.get("route"),
        "reference_tool": tool_evaluation.get("reference_tool"),
        "route_decision": tool_evaluation.get("route_decision", {}),
        "source": tool_evaluation.get("source", {}),
        "episodes": [
            {
                "policy_name": item.get("policy_name"),
                "seed": item.get("seed"),
                "role": item.get("role"),
                "value": item.get("result", {}).get("value"),
                "passed": item.get("result", {}).get("passed"),
                "evidence_steps": item.get("result", {}).get("evidence_steps", []),
                "details": item.get("result", {}).get("details", {}),
            }
            for item in tool_evaluation.get("episodes", [])
        ],
        "validation": tool_evaluation.get("validation", {}),
    }


def _aggregate_sources(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build one de-duplicated set of episode ToolResult sources."""

    context = {
        "round_id": round_plan["round_id"],
        "variant": round_plan.get("template_id")
        or round_plan.get("sub_aspect")
        or round_plan.get("route"),
    }
    sources: list[dict[str, Any]] = []
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    trusted_tools = {
        result.get("tool")
        for episode in trusted.get("episodes", [])
        for result in episode.get("tool_results", [])
        if result.get("tool")
    }
    if trusted.get("episodes"):
        sources.append(
            {
                **trusted,
                "context": {
                    **context,
                    "source_artifact": trusted.get("artifact"),
                },
            }
        )
    if tool_evaluation and tool_evaluation.get("episodes"):
        request = tool_evaluation.get("tool_request") or tool_evaluation.get(
            "tool_spec", {}
        )
        metric = request.get("metric") if isinstance(request, dict) else None
        if metric not in trusted_tools:
            sources.append(
                {
                    "tool_execution": tool_evaluation,
                    "context": {
                        **context,
                        "source_artifact": tool_evaluation.get("artifacts", {}).get(
                            "tool_execution"
                        ),
                    },
                }
            )
    return sources


def compact_aggregate_result(
    aggregate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip repeated provenance before sending aggregate evidence to an LLM."""

    if not aggregate:
        return None

    def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "episode_result_count": summary.get("episode_result_count"),
            "quality": {
                key: value.get("value")
                for key, value in summary.get("quality", {}).items()
            },
            "statistics": {
                key: {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key != "provenance"
                }
                for key, value in summary.get("statistics", {}).items()
            },
        }

    return {
        "schema_version": aggregate.get("schema_version"),
        "status": aggregate.get("status"),
        "source_count": aggregate.get("source_count"),
        "unique_episode_count": aggregate.get("unique_episode_count"),
        "input_issues": aggregate.get("input_issues", []),
        "metrics": [
            {
                "metric": metric.get("metric"),
                "value_kind": metric.get("value_kind"),
                "unit": metric.get("unit"),
                "cohorts": [
                    {
                        "role": cohort.get("role"),
                        "policy_names": cohort.get("policy_names", []),
                        "summary": compact_summary(cohort.get("summary", {})),
                        "passed_summary": (
                            compact_summary(cohort["passed_summary"])
                            if cohort.get("passed_summary")
                            else None
                        ),
                        "groups": {
                            dimension: [
                                {
                                    "value": group.get("value"),
                                    "summary": compact_summary(
                                        group.get("summary", {})
                                    ),
                                    "passed_summary": (
                                        compact_summary(group["passed_summary"])
                                        if group.get("passed_summary")
                                        else None
                                    ),
                                }
                                for group in groups
                            ]
                            for dimension, groups in cohort.get("groups", {}).items()
                        },
                    }
                    for cohort in metric.get("cohorts", [])
                ],
            }
            for metric in aggregate.get("metrics", [])
        ],
    }


def aggregate_round_results(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    sources = _aggregate_sources(round_plan, child_manifest, tool_evaluation)
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no episode ToolResult rows were available",
            "metrics": [],
        }
        write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


def aggregate_evaluation_results(
    round_runs: list[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    sources = [
        source
        for item in round_runs
        for source in _aggregate_sources(
            item["round_plan"], item["child_manifest"], item["tool_evaluation"]
        )
    ]
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no completed round ToolResult rows were available",
            "metrics": [],
        }
        write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


def _execution_vqa_video_contract(
    episode_dir: Path,
    *,
    execution_backend: str,
) -> tuple[bool, dict[str, Any], str]:
    """Validate backend-specific video evidence before it reaches VQA."""

    metadata_path = episode_dir / "episode.json"
    metadata: dict[str, Any] = {}
    metadata_error: str | None = None
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
            else:
                metadata_error = "episode.json is not a JSON object"
        except (OSError, json.JSONDecodeError) as exc:
            metadata_error = f"episode.json is unreadable: {type(exc).__name__}"
    else:
        metadata_error = "episode.json is missing"

    if not (episode_dir / "video.mp4").is_file():
        return False, metadata, "is missing video.mp4"
    if (episode_dir / "video.mp4").stat().st_size <= 0:
        return False, metadata, "has an empty video.mp4"
    if metadata_error:
        return False, metadata, metadata_error
    if (metadata.get("artifacts") or {}).get("video") != "video.mp4":
        return False, metadata, "does not declare artifacts.video=video.mp4"
    if execution_backend != "expert":
        return True, metadata, ""

    visual_capture = metadata.get("visual_capture") or {}
    if visual_capture.get("status") != "completed":
        return False, metadata, "does not declare a completed visual_capture"
    return True, metadata, ""


def _policy_episode_for_execution_vqa(
    child_manifest: dict[str, Any],
    child_dir: Path,
    *,
    execution_backend: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]] | None:
    """Select evidence from the backend that this round actually evaluated."""

    desired_policy = "expert" if execution_backend == "expert" else "act"
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    candidates = sorted(
        (
            episode
            for episode in trusted.get("episodes", [])
            if str(episode.get("policy_name", "")).casefold() == desired_policy
        ),
        key=lambda episode: (
            not _execution_vqa_video_contract(
                child_dir
                / "evaluation/telemetry"
                / str(episode.get("episode_dir") or ""),
                execution_backend=execution_backend,
            )[0],
            int(episode.get("seed") or 0),
            str(episode.get("episode_dir") or ""),
        ),
    )
    if not candidates:
        return None
    episode = candidates[0]
    episode_dir = child_dir / "evaluation/telemetry" / episode["episode_dir"]
    return episode_dir, episode, list(episode.get("tool_results", []))


def _same_telemetry_episode(
    candidate: dict[str, Any], representative: dict[str, Any]
) -> bool:
    """Match generated and Trusted Tool rows to one physical rollout."""

    candidate_dir = candidate.get("episode_dir")
    representative_dir = representative.get("episode_dir")
    if candidate_dir and representative_dir:
        return str(candidate_dir) == str(representative_dir)
    return (
        candidate.get("seed") == representative.get("seed")
        and str(candidate.get("policy_name", "")).casefold()
        == str(representative.get("policy_name", "")).casefold()
    )


def run_round_execution_vqa(
    *,
    repo_root: Path,
    child_manifest: dict[str, Any],
    child_dir: Path,
    tool_evaluation: dict[str, Any] | None,
    execution_dir: Path,
    provider: Any,
    model: str,
    round_plan: dict[str, Any] | None = None,
    reviewed_vqa_registry: Path | None = None,
) -> dict[str, Any]:
    """Run VQA on official-expert or ACT evidence without mixing their roles."""

    query = build_execution_vqa_query(
        task_name=(
            str((round_plan or {}).get("task_name") or child_manifest.get("task_name"))
            if (round_plan or {}).get("task_name") or child_manifest.get("task_name")
            else None
        ),
        template_id=(round_plan or {}).get("template_id"),
        sub_aspect=(round_plan or {}).get("sub_aspect"),
        tool_contract=(round_plan or {}).get("tool_request"),
        reviewed_registry_dir=reviewed_vqa_registry,
    )
    write_json(execution_dir / "execution_vqa_query.json", query)
    route = (round_plan or {}).get("route")
    execution_backend = round_execution_backend(round_plan or {"route": route})
    evidence_backend = "expert" if execution_backend == "expert" else "act"
    selected = _policy_episode_for_execution_vqa(
        child_manifest,
        child_dir,
        execution_backend=evidence_backend,
    )
    if selected is None:
        backend = "expert" if evidence_backend == "expert" else "ACT"
        result = {
            "schema_version": 1,
            "status": "skipped" if evidence_backend == "expert" else "failed",
            "reason": f"no completed {backend} telemetry episode was available",
            "evidence_conflict": False,
            "query": query,
        }
        write_json(
            execution_dir
            / (
                "execution_vqa_skipped.json"
                if evidence_backend == "expert"
                else "execution_vqa_error.json"
            ),
            result,
        )
        return result
    episode_dir, representative, numeric_results = selected
    representative_path = str(episode_dir.relative_to(repo_root))
    video_ready, metadata, video_reason = _execution_vqa_video_contract(
        episode_dir,
        execution_backend=evidence_backend,
    )
    if not video_ready:
        backend = "expert" if evidence_backend == "expert" else "ACT"
        result = {
            "schema_version": 1,
            "status": "skipped" if evidence_backend == "expert" else "failed",
            "reason": f"completed {backend} telemetry episode {video_reason}",
            "representative_episode": representative_path,
            "evidence_conflict": False,
            "query": query,
            "visual_capture": metadata.get("visual_capture"),
        }
        write_json(
            execution_dir
            / (
                "execution_vqa_skipped.json"
                if evidence_backend == "expert"
                else "execution_vqa_error.json"
            ),
            result,
        )
        return result
    known_tools = {item.get("tool") for item in numeric_results}
    desired_role = (
        "expert_validation"
        if evidence_backend == "expert"
        else "policy_under_evaluation"
    )
    for episode in (tool_evaluation or {}).get("episodes", []):
        if episode.get("role") != desired_role:
            continue
        if not _same_telemetry_episode(episode, representative):
            continue
        result = episode.get("result", {})
        if result.get("tool") not in known_tools:
            numeric_results.append(result)
            known_tools.add(result.get("tool"))
    try:
        scene_seed = (child_manifest.get("scene_validation") or {}).get("seed")
        representative_seed = representative.get("seed")
        reference_scene = child_dir / "evidence/initial_head.png"
        if (
            scene_seed is not None
            and representative_seed is not None
            and int(scene_seed) != int(representative_seed)
        ):
            # Never label an image from a skipped seed as the rollout's
            # reference scene. The rollout video remains valid evidence.
            reference_scene = None
        result = run_execution_vqa(
            provider=provider,
            model=model,
            video_path=episode_dir / "video.mp4",
            output_dir=execution_dir / "execution_vqa",
            numeric_tool_results=numeric_results,
            events_path=episode_dir / "events.jsonl",
            semantic_trace_path=episode_dir / "semantic_trace.npz",
            reference_scene=reference_scene,
            query=query,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "representative_episode": representative_path,
            "evidence_conflict": False,
            "query": query,
        }
        write_json(execution_dir / "execution_vqa_error.json", result)
        return result
    result["status"] = "passed"
    result["representative_episode"] = representative_path
    result["artifacts"] = {
        key: (
            str(Path(value).resolve().relative_to(repo_root))
            if isinstance(value, str)
            and Path(value).is_absolute()
            and Path(value).resolve().is_relative_to(repo_root)
            else value
        )
        for key, value in result.get("artifacts", {}).items()
    }
    write_json(execution_dir / "execution_vqa/execution_vqa.json", result)
    return result


def compact_execution_vqa(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not result:
        return None
    return {
        "status": result.get("status"),
        "model_requested": result.get("model_requested"),
        "representative_episode": result.get("representative_episode"),
        "evidence_conflict": bool(result.get("evidence_conflict")),
        "observation": result.get("observation"),
        "selected_frames": result.get("selection", {}).get("selected_frames", []),
        "artifacts": result.get("artifacts", {}),
        "reason": result.get("reason"),
        "query": result.get("query"),
    }


def summarize_round(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    tool_evaluation: dict[str, Any] | None = None,
    aggregate_result: dict[str, Any] | None = None,
    execution_vqa: dict[str, Any] | None = None,
    taskgen_returncode: int = 0,
) -> dict[str, Any]:
    capability_contract = validate_round_capability_contract(round_plan)
    scene = child_manifest.get("scene_validation", {})
    vision = child_manifest.get("vision_validation", {})
    act = child_manifest.get("act_evaluation", {})
    expert = scene.get("expert", {})
    positions = child_manifest.get("position_samples", {})
    position_metrics = positions.get("metrics", {})
    variant_samples = positions.get("samples", [])
    observed_bell_ids = sorted(
        {
            int(item["bell_id"])
            for item in variant_samples
            if isinstance(item, dict)
            and not isinstance(item.get("bell_id"), bool)
            and isinstance(item.get("bell_id"), int)
        }
    )
    clutter_counts = [
        int(item["clutter_count"])
        for item in variant_samples
        if isinstance(item, dict)
        and not isinstance(item.get("clutter_count"), bool)
        and isinstance(item.get("clutter_count"), int)
    ]
    policy_success = read_policy_success(child_dir / "evaluation/_result.txt")
    trusted_tools = compact_trusted_tools(child_manifest)
    is_official = round_plan.get("route") == "official"
    execution_backend = round_execution_backend(round_plan)
    uses_act = execution_backend in {"act", "both"}
    uses_expert = execution_backend in {"expert", "both"}
    if uses_act:
        actual_seeds = [int(value) for value in act.get("actual_seeds", [])]
    else:
        actual_seeds = [
            int(item["seed"])
            for item in scene.get("expert_batch", {}).get("episodes", [])
            if item.get("seed") is not None
        ]
    static = child_manifest.get("static_validation") or {}
    gate_status = {
        "variant_spec": (
            (child_manifest.get("capability_contract_validation") or {}).get(
                "status"
            )
            == "passed"
        ),
        "ast": bool((static.get("load_actors_ast") or {}).get("valid")),
        "render": bool(scene.get("render_success")),
        "rule": bool((scene.get("rule_check") or {}).get("passed")),
        "scene_variant": bool(positions.get("passed")),
        "vision": bool(vision.get("passed")),
        "expert": bool((scene.get("expert_batch") or expert).get("passed")),
        "act": bool((not uses_act and is_official) or act.get("passed")),
        "toolkit": bool(
            (child_manifest.get("trusted_tool_evaluation") or {}).get(
                "episode_count"
            )
        ),
        "planned_tool": bool(
            tool_evaluation and tool_evaluation.get("status") == "passed"
        ),
        "aggregate": bool(
            aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
        ),
        "execution_vqa": bool(
            execution_vqa
            and (
                execution_vqa.get("status") == "passed"
                or (
                    not uses_act
                    and execution_vqa.get("status") == "skipped"
                )
            )
        ),
    }
    required_gates = (
        list(capability_contract["required_gates"])
        if capability_contract is not None
        else []
    )
    required_gate_status = {
        "required": required_gates,
        "by_gate": {gate: bool(gate_status.get(gate, False)) for gate in required_gates},
    }
    required_gate_status["passed"] = all(
        required_gate_status["by_gate"].values()
    )
    if is_official:
        expert_batch = scene.get("expert_batch") or expert
        pipeline_passed = bool(
            child_manifest.get("status")
            == ("completed" if uses_act else "completed_without_act")
            and taskgen_returncode == 0
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and (not uses_expert or expert_batch.get("passed"))
            and (not uses_act or act.get("passed"))
            and child_manifest.get("trusted_tool_evaluation", {}).get("episode_count")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in {"passed", "skipped"}
        )
    else:
        # Generated rounds keep their expert, visual, and task-specific
        # position gates while ACT remains the policy under evaluation.
        pipeline_passed = bool(
            child_manifest.get("status") == "completed"
            and taskgen_returncode == 0
            and scene.get("rule_check", {}).get("passed")
            and vision.get("passed")
            and expert.get("passed")
            and positions.get("passed")
            and act.get("passed")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in {"passed", "skipped"}
        )
    if capability_contract is not None:
        pipeline_passed = bool(pipeline_passed and required_gate_status["passed"])
    return {
        "round_id": round_plan["round_id"],
        "variant_id": (
            round_plan.get("task_variant_id") or round_plan.get("template_id")
        ),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "capability_contract": round_plan.get("capability_contract"),
        "required_gate_status": required_gate_status,
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "taskgen_run_id": child_manifest.get("run_id"),
        "taskgen_returncode": taskgen_returncode,
        "execution": round_plan["execution"],
        "observations": {
            "execution_backend": {
                "expert": "expert",
                "act": "ACT",
                "both": "ACT+expert",
            }[execution_backend],
            "requested_seeds": [
                int(value) for value in round_plan["execution"].get("seeds", [])
            ],
            "actual_seeds": actual_seeds,
            "scene_alignment": bool(scene.get("rule_check", {}).get("passed")),
            "observed_color": vision.get("observed_color"),
            "bell_visible": vision.get("bell_visible"),
            "position_authority": vision.get("position_authority"),
            "expert_solvable": (
                bool((scene.get("expert_batch") or expert).get("passed"))
                if uses_expert or not is_official
                else None
            ),
            "act_pipeline_status": bool(act.get("passed")) if uses_act else None,
            "policy_success": policy_success if uses_act else None,
            "position_samples": positions.get("samples", []),
            "position_metrics": position_metrics,
            "controlled_axis": positions.get("controlled_axis"),
            "variant_samples": variant_samples,
            "variant_metrics": position_metrics,
            "observed_bell_ids": observed_bell_ids,
            "bell_instance_id": (
                observed_bell_ids[0] if len(observed_bell_ids) == 1 else None
            ),
            "scene_clutter": {
                "expected": bool(position_metrics.get("expected_clutter")),
                "counts": clutter_counts,
                "all_matched": position_metrics.get("all_clutter_matched"),
                "authority": (
                    "simulator_task_info:cluttered_table_info"
                    if clutter_counts
                    else None
                ),
            },
            "trusted_tools": trusted_tools,
            "planned_tool": compact_tool_evaluation(tool_evaluation),
            "aggregate": compact_aggregate_result(aggregate_result),
            "execution_vqa": compact_execution_vqa(execution_vqa),
            "required_gate_status": required_gate_status,
        },
        "pipeline_passed": pipeline_passed,
        "interpretation": (
            "任务路由与执行后端分别记录；ACT 策略结果和流水线状态分开报告，" "策略失败不会被误记为 pipeline failure。"
        ),
    }


def execute_round(
    repo_root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    provider: Any,
    toolgen_model: str,
    telemetry_profile: str = "balanced_v1",
    reviewed_tool_registry: Path | None = None,
    reviewed_vqa_registry: Path | None = None,
    tool_recovery_max_restarts: int = 1,
    inject_tool_exception_once: bool = False,
    registration_identity: dict[str, Any] | None = None,
    round_attempt_index: int = 1,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], int,]:
    if round_attempt_index < 1:
        raise ValueError("round_attempt_index must be positive")
    round_id = round_plan["round_id"]
    run_id_suffix = (
        "" if round_attempt_index == 1 else f"_attempt_{round_attempt_index:02d}"
    )
    command, run_id = build_taskgen_command(
        repo_root,
        evaluation_id,
        round_plan,
        text_model=text_model,
        vision_model=vision_model,
        base_url=base_url,
        gpu=gpu,
        max_reflections=max_reflections,
        telemetry_profile=telemetry_profile,
        registration_identity=registration_identity,
        run_id_suffix=run_id_suffix,
    )
    execution_dir = evaluation_dir / "execution" / round_id
    if round_attempt_index > 1:
        execution_dir = execution_dir / f"round_attempt_{round_attempt_index:02d}"
    write_json(
        execution_dir / "taskgen_command.json",
        {"command": command, "child_run_id": run_id},
    )
    update_manifest(
        evaluation_dir,
        status=f"executing_{round_id}",
        active_child_run_id=run_id,
    )
    returncode = run_logged(
        command,
        cwd=repo_root,
        log_path=execution_dir / "taskgen.log",
    )
    child_dir = repo_root / "mea/generated_tasks" / run_id
    child_manifest_path = child_dir / "manifest.json"
    if not child_manifest_path.is_file():
        raise RuntimeError(f"child TaskGen manifest 不存在: {child_manifest_path}")
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    if registration_identity is not None and child_manifest.get(
        "registration_identity"
    ) != registration_identity:
        raise RuntimeError(
            f"child registration identity mismatch: {run_id}"
        )
    write_json(
        execution_dir / "child_run.json",
        {
            "run_id": run_id,
            "returncode": returncode,
            "manifest_path": str(child_manifest_path.relative_to(repo_root)),
            "status": child_manifest.get("status"),
        },
    )
    recovery_summary: dict[str, Any] | None = None
    if (
        child_manifest.get("status")
        in {
            "completed",
            "completed_without_act",
        }
        and returncode == 0
    ):
        tool_kwargs: dict[str, Any] = {
            "provider": provider,
            "model": toolgen_model,
        }
        if reviewed_tool_registry is not None:
            tool_kwargs["reviewed_registry_dir"] = reviewed_tool_registry
        telemetry_dir = child_dir / "evaluation/telemetry"
        injected = False

        def execute_tool_attempt(
            attempt_dir: Path, attempt_index: int
        ) -> dict[str, Any]:
            nonlocal injected
            output_dir = (
                execution_dir / "planned_tool"
                if attempt_index == 1
                else execution_dir / f"planned_tool_retry_{attempt_index - 1:02d}"
            )
            write_json(
                attempt_dir / "tool_output.json",
                {"output_dir": str(output_dir.relative_to(repo_root))},
            )
            if inject_tool_exception_once and not injected:
                injected = True
                raise UnexpectedToolExecutionError(
                    "development fault injection before Tool analysis"
                )
            try:
                return execute_tool_request(
                    repo_root,
                    child_dir,
                    output_dir,
                    round_plan["tool_request"],
                    **tool_kwargs,
                )
            except ToolOrchestrationError:
                # Contract, routing, validation, and semantic input failures are
                # terminal. Retrying them would hide a real evaluation error.
                raise
            except Exception as exc:
                # Only exceptions outside the expected Tool contract are
                # classified as restartable runtime failures.
                raise UnexpectedToolExecutionError(
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        recovery = run_bounded_tool_recovery(
            execution_dir / "tool_recovery",
            logical_round_id=round_id,
            execute=execute_tool_attempt,
            telemetry_sha256=lambda: directory_tree_sha256(telemetry_dir),
            max_restarts=tool_recovery_max_restarts,
        )
        recovery_summary = {
            key: recovery.get(key)
            for key in (
                "status",
                "attempt_count",
                "restarts_used",
                "same_telemetry_reused",
                "telemetry_sha256",
                "failure_class",
                "action",
                "recovery_scope",
                "additional_act_rollouts_started_by_recovery",
                "policy_or_simulator_restarted",
                "provider_or_registry_work_may_repeat",
            )
        }
        tool_evaluation = recovery["result"]
    else:
        skip_reason = (
            f"child TaskGen exited with code {returncode}"
            if returncode != 0
            else "child TaskGen pipeline did not complete"
        )
        tool_evaluation = {
            "schema_version": 1,
            "status": "skipped",
            "requested_route": "auto",
            "route": None,
            "reference_tool": None,
            "tool_request": round_plan["tool_request"],
            "route_decision": {
                "status": "skipped",
                "requested_route": "auto",
                "resolved_route": None,
                "reason": skip_reason,
                "provider_required": None,
                "provider_called": False,
            },
            "source": {},
            "episodes": [],
            "validation": {"reason": skip_reason},
            "artifacts": {},
        }
        write_json(execution_dir / "planned_tool_skipped.json", tool_evaluation)
    aggregate_result = aggregate_round_results(
        round_plan,
        child_manifest,
        tool_evaluation,
        execution_dir / "aggregate_result.json",
    )
    execution_vqa = run_round_execution_vqa(
        repo_root=repo_root,
        child_manifest=child_manifest,
        child_dir=child_dir,
        tool_evaluation=tool_evaluation,
        execution_dir=execution_dir,
        provider=provider,
        model=vision_model,
        round_plan=round_plan,
        reviewed_vqa_registry=reviewed_vqa_registry,
    )
    round_summary = summarize_round(
        round_plan,
        child_manifest,
        child_dir,
        tool_evaluation,
        aggregate_result,
        execution_vqa,
        returncode,
    )
    round_summary["round_attempt_index"] = round_attempt_index
    round_summary["execution_artifact_dir"] = str(
        execution_dir.relative_to(repo_root)
    ).replace("\\", "/")
    round_summary.setdefault("observations", {})["tool_recovery"] = recovery_summary
    write_json(evaluation_dir / "summary" / f"{round_id}.json", round_summary)
    return child_manifest, child_dir, round_summary, tool_evaluation, returncode


def execute_round_stage_aware(
    repo_root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    provider: Any,
    toolgen_model: str,
    telemetry_profile: str = "balanced_v1",
    reviewed_tool_registry: Path | None = None,
    reviewed_vqa_registry: Path | None = None,
    tool_recovery_max_restarts: int = 0,
    round_recovery_max_restarts: int = 1,
    inject_tool_exception_once: bool = False,
    registration_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], int]:
    """Run one logical Agent round under the paper's whole-round controller.

    Existing local TaskGen and ToolGen repair loops stay inside their stages.
    Only an exhausted, explicitly typed unexpected Tool-execution exception is
    promoted to a whole-round restart.  Each retry gets a new child run id and
    append-only execution directory, so failed ACT evidence is never silently
    overwritten or reused as if it belonged to the replacement round.
    """

    round_id = str(round_plan["round_id"])
    attempt_results: dict[
        int, tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], int]
    ] = {}

    def taskgen_provider_called(child: dict[str, Any]) -> bool:
        """Return whether the materialized TaskGen stage used a provider.

        Generated BBH manifests record a non-empty ``provider.calls`` map;
        bounded click_bell manifests record ``called=false`` but their later
        visual self-check still invokes the vision provider.  Counting both
        avoids attributing a restarted round only to ToolGen.
        """

        provider_record = child.get("provider")
        if isinstance(provider_record, dict):
            explicit = provider_record.get("called")
            if isinstance(explicit, bool) and explicit:
                return True
            calls = provider_record.get("calls")
            if isinstance(calls, dict) and bool(calls):
                return True
        return bool(
            child.get("visual_self_reflection")
            or child.get("vision_validation")
        )

    def runtime_for_child(attempt_index: int) -> dict[str, Any]:
        suffix = "" if attempt_index == 1 else f"_attempt_{attempt_index:02d}"
        run_id = child_run_id(evaluation_id, round_id) + suffix
        manifest_path = repo_root / "mea/generated_tasks" / run_id / "manifest.json"
        child: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                child = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                child = {}
        act = child.get("act_evaluation")
        actual = act.get("actual_seeds") if isinstance(act, dict) else []
        act_count = len(actual) if isinstance(actual, list) else 0
        return {
            "provider_called": taskgen_provider_called(child),
            "simulator_called": bool(
                child.get("scene_validation")
                or child.get("act_evaluation")
                or child.get("expert_evaluation")
            ),
            "act_rollouts_started": act_count,
        }

    def execute_attempt(attempt_dir: Path, attempt_index: int) -> dict[str, Any]:
        del attempt_dir  # The controller owns its trace; Agent paths are canonical.
        try:
            result = execute_round(
                repo_root,
                evaluation_dir,
                evaluation_id,
                round_plan,
                text_model=text_model,
                vision_model=vision_model,
                base_url=base_url,
                gpu=gpu,
                max_reflections=max_reflections,
                provider=provider,
                toolgen_model=toolgen_model,
                telemetry_profile=telemetry_profile,
                reviewed_tool_registry=reviewed_tool_registry,
                reviewed_vqa_registry=reviewed_vqa_registry,
                tool_recovery_max_restarts=tool_recovery_max_restarts,
                inject_tool_exception_once=(
                    inject_tool_exception_once and attempt_index == 1
                ),
                registration_identity=registration_identity,
                round_attempt_index=attempt_index,
            )
        except BoundedRecoveryError as exc:
            execution_dir = evaluation_dir / "execution" / round_id
            if attempt_index > 1:
                execution_dir = execution_dir / f"round_attempt_{attempt_index:02d}"
            recovery_path = execution_dir / "tool_recovery/recovery_summary.json"
            recovery: dict[str, Any] = {}
            if recovery_path.is_file():
                loaded = json.loads(recovery_path.read_text(encoding="utf-8"))
                recovery = loaded if isinstance(loaded, dict) else {}
            if recovery.get("failure_class") != "unexpected_tool_execution_exception":
                raise
            raise StageFailure(
                "tool_execution",
                "unexpected_exception",
                str(exc),
                runtime=runtime_for_child(attempt_index),
                details={
                    "tool_recovery_summary": str(
                        recovery_path.relative_to(repo_root)
                    ).replace("\\", "/"),
                    "failed_child_run_id": child_run_id(evaluation_id, round_id)
                    + (
                        ""
                        if attempt_index == 1
                        else f"_attempt_{attempt_index:02d}"
                    ),
                },
            ) from exc
        attempt_results[attempt_index] = result
        child_manifest, _child_dir, _summary, tool_evaluation, _returncode = result
        route = tool_evaluation.get("route_decision") or {}
        runtime = runtime_for_child(attempt_index)
        execution_vqa = (
            (_summary.get("observations") or {}).get("execution_vqa") or {}
        )
        runtime["provider_called"] = bool(
            runtime["provider_called"]
            or route.get("provider_called")
            or execution_vqa.get("model_requested")
        )
        return {
            "status": "completed",
            "attempt_index": attempt_index,
            "child_run_id": child_manifest.get("run_id"),
            "execution_artifact_dir": _summary.get("execution_artifact_dir"),
            "pipeline_passed": bool(_summary.get("pipeline_passed")),
            "runtime": runtime,
        }

    round_identity = {
        "evaluation_id": evaluation_id,
        "round_plan": round_plan,
        "text_model": text_model,
        "vision_model": vision_model,
        "toolgen_model": toolgen_model,
        "telemetry_profile": telemetry_profile,
        "reviewed_tool_registry": (
            str(reviewed_tool_registry) if reviewed_tool_registry is not None else None
        ),
        "reviewed_vqa_registry": (
            str(reviewed_vqa_registry) if reviewed_vqa_registry is not None else None
        ),
        "registration_identity": registration_identity,
    }
    recovery = run_stage_aware_round_recovery(
        evaluation_dir / "execution" / round_id / "whole_round_recovery",
        logical_round_id=round_id,
        round_identity=round_identity,
        execute_round=execute_attempt,
        max_restarts=round_recovery_max_restarts,
    )
    final_attempt = int(recovery["attempts"][-1]["attempt_index"])
    if final_attempt not in attempt_results:
        raise WholeRoundRecoveryError(
            "round recovery completed without a materialized Agent result",
            summary=recovery,
        )
    result = attempt_results[final_attempt]
    result[2].setdefault("observations", {})["whole_round_recovery"] = {
        key: recovery.get(key)
        for key in (
            "status",
            "recovery_scope",
            "attempt_count",
            "restarts_used",
            "whole_round_restarted",
            "policy_or_simulator_restarted",
            "additional_round_attempts_started_by_recovery",
            "additional_act_rollouts_started_by_recovery",
            "runtime",
            "attempts",
        )
    }
    write_json(evaluation_dir / "summary" / f"{round_id}.json", result[2])
    return result


def _round_evidence(
    repo_root: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    round_summary: dict[str, Any],
    tool_evaluation: dict[str, Any],
) -> dict[str, Any]:
    static = child_manifest.get("static_validation", {})
    scene = child_manifest.get("scene_validation", {})
    vision = child_manifest.get("vision_validation", {})
    reflection = child_manifest.get("visual_self_reflection", {})
    retrieval = child_manifest.get("task_retrieval") or {}
    knowledge = child_manifest.get("knowledge_retrieval") or {}
    trusted_tool_evaluation = child_manifest.get("trusted_tool_evaluation") or {}
    child_relative = child_dir.relative_to(repo_root)
    task_module = str(child_manifest.get("task_module") or "")
    module_source = repo_root / (task_module.replace(".", "/") + ".py")
    generated_task_artifact = (
        str(module_source.relative_to(repo_root))
        if task_module and module_source.is_file()
        else str(child_relative / "task.py")
    )
    act_videos = sorted(
        str(path.relative_to(repo_root))
        for path in (child_dir / "evaluation").glob("episode*.mp4")
    )
    rollout_video_paths = {
        child_dir
        / "evaluation/telemetry"
        / str(episode.get("episode_dir") or "")
        / "video.mp4"
        for episode in trusted_tool_evaluation.get("episodes", [])
    }
    rollout_videos = sorted(
        str(path.relative_to(repo_root))
        for path in rollout_video_paths
        if path.is_file()
    )
    variant_spec_path = child_dir / "variant_spec.json"
    variant_spec = (
        json.loads(variant_spec_path.read_text(encoding="utf-8"))
        if variant_spec_path.is_file()
        else None
    )
    feedback_observations = {
        key: value
        for key, value in round_summary["observations"].items()
        if key
        not in {
            "trusted_tools",
            "planned_tool",
            "aggregate",
            "execution_vqa",
        }
    }

    round_execution_value = round_summary.get("execution_artifact_dir")
    round_execution = (
        Path(str(round_execution_value))
        if round_execution_value
        else Path("mea/evaluation_runs")
        / evaluation_id
        / "execution"
        / round_plan["round_id"]
    )
    execution_vqa_observation = round_summary["observations"].get("execution_vqa") or {}
    execution_vqa_artifacts = execution_vqa_observation.get("artifacts") or {}
    execution_vqa_artifact = execution_vqa_artifacts.get(
        "result"
    ) or execution_vqa_artifacts.get("execution_vqa")
    if not execution_vqa_artifact:
        if execution_vqa_observation.get("status") == "skipped":
            execution_vqa_artifact = str(round_execution / "execution_vqa_skipped.json")
        elif execution_vqa_observation.get("status") == "failed":
            execution_vqa_artifact = str(round_execution / "execution_vqa_error.json")
    return {
        "round_id": round_plan["round_id"],
        "child_run_id": child_manifest.get("run_id"),
        "variant_id": (
            round_plan.get("task_variant_id") or round_plan.get("template_id")
        ),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "capability_contract": round_plan.get("capability_contract"),
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "seeds": (
            round_summary["observations"].get("actual_seeds")
            or round_plan["execution"]["seeds"]
        ),
        "requested_seeds": round_plan["execution"]["seeds"],
        "num_episodes": round_plan["execution"]["num_episodes"],
        "task_retrieval": {
            "catalog_size": retrieval.get("catalog_size"),
            "selected_tasks": retrieval.get("selected_tasks", []),
            "reasoning": retrieval.get("reasoning"),
        },
        "knowledge_retrieval": {
            "selected_ids": knowledge.get("selected_ids", []),
            "context_character_count": knowledge.get("context_character_count"),
            "committed_index_current": knowledge.get("committed_index_current"),
        },
        "generation": {
            "variant_spec": variant_spec,
            "complete_method_generated": static.get("load_actors_ast", {}).get(
                "complete_method_generated"
            ),
            "generated_color": static.get("load_actors_ast", {}).get("generated_color"),
        },
        "visual_observation": {
            "render_success": scene.get("render_success"),
            "aligned": vision.get("aligned"),
            "target_actor": vision.get("target_actor"),
            "bell_visible": vision.get("bell_visible"),
            "observed_color": vision.get("observed_color"),
            "unexpected_changes": vision.get("unexpected_changes"),
            "confidence": vision.get("confidence"),
            "position_authority": vision.get("position_authority"),
        },
        "visual_self_reflection": {
            "passed": reflection.get("passed"),
            "max_repairs": reflection.get("max_repairs"),
            "repairs_used": reflection.get("repairs_used"),
            "final_attempt": reflection.get("final_attempt"),
            "attempt_count": len(reflection.get("attempts", [])),
        },
        "observations": {
            **feedback_observations,
            "pipeline_passed": round_summary["pipeline_passed"],
        },
        "tool_evaluation": tool_evaluation,
        "aggregate": round_summary["observations"].get("aggregate"),
        "execution_vqa": round_summary["observations"].get("execution_vqa"),
        "trusted_tool_evaluation": {
            "artifact": trusted_tool_evaluation.get("artifact"),
            "episode_count": trusted_tool_evaluation.get("episode_count"),
            "episodes": compact_trusted_tools(child_manifest),
        },
        "artifacts": {
            "generated_task": generated_task_artifact,
            "scene_image": str(child_relative / "evidence/initial_head.png"),
            "vision_result": str(child_relative / "validation/vision.json"),
            "position_samples": str(
                child_relative / "validation/position_samples.json"
            ),
            "reflection_summary": str(child_relative / "reflection/summary.json"),
            "act_videos": act_videos,
            "rollout_videos": rollout_videos,
            "act_result": str(child_relative / "evaluation/_result.txt"),
            "trusted_tools": trusted_tool_evaluation.get("artifact"),
            "planned_tool": tool_evaluation.get("artifacts", {}).get("tool_execution"),
            "aggregate": str(round_execution / "aggregate_result.json"),
            "execution_vqa": execution_vqa_artifact,
            "execution_vqa_query": str(round_execution / "execution_vqa_query.json"),
            "execution_vqa_montage": execution_vqa_artifacts.get("montage"),
            "execution_vqa_selection": execution_vqa_artifacts.get("selection"),
            "child_manifest": str(child_relative / "manifest.json"),
        },
    }


def build_evidence_bundle(
    repo_root: Path,
    evaluation_id: str,
    user_request: str,
    plan: dict[str, Any],
    round_runs: list[dict[str, Any]],
    evaluation_aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rounds = [
        _round_evidence(
            repo_root,
            evaluation_id,
            item["round_plan"],
            item["child_manifest"],
            item["child_dir"],
            item["round_summary"],
            item["tool_evaluation"],
        )
        for item in round_runs
    ]
    total_episodes = sum(item["num_episodes"] for item in rounds)
    weighted_success = 0.0
    measured_episodes = 0
    for item in rounds:
        rate = item["observations"].get("policy_success")
        if rate is not None:
            weighted_success += float(rate) * item["num_episodes"]
            measured_episodes += item["num_episodes"]
    policy_success = weighted_success / measured_episodes if measured_episodes else None
    position_rounds = [
        item for item in rounds if str(item["sub_aspect"]).startswith("object_position")
    ]
    position_metrics_by_round = {
        item["round_id"]: item["observations"].get("position_metrics", {})
        for item in position_rounds
    }
    sampled_xy: list[list[float]] = []
    for item in position_rounds:
        for sample in item["observations"].get("position_samples", []):
            position = sample.get("bell_position") or sample.get("block_position")
            if isinstance(position, list) and len(position) >= 2:
                sampled_xy.append([float(position[0]), float(position[1])])
    unique_xy = {(round(item[0], 8), round(item[1], 8)) for item in sampled_xy}
    position_metrics = (
        {
            "sample_count": len(sampled_xy),
            "unique_xy_count": len(unique_xy),
            "x_span": (
                max(item[0] for item in sampled_xy)
                - min(item[0] for item in sampled_xy)
            ),
            "y_span": (
                max(item[1] for item in sampled_xy)
                - min(item[1] for item in sampled_xy)
            ),
            "position_varied": len(unique_xy) > 1,
            "by_round": position_metrics_by_round,
        }
        if sampled_xy
        else {}
    )
    evaluation_relative = Path("mea/evaluation_runs") / evaluation_id
    completed_template_ids = [item["round_plan"]["template_id"] for item in round_runs]
    remaining_template_ids = [
        item
        for item in plan.get("requested_template_ids", [])
        if item not in completed_template_ids
    ]
    decision_artifacts = [
        str(evaluation_relative / f"plan/decision_after_round_{round_number}.json")
        for round_number in range(1, len(plan.get("round_decisions", [])) + 1)
    ]
    evidence_assessment_artifacts = [
        str(evaluation_relative / f"plan/evidence_after_round_{round_number}.json")
        for round_number in range(1, len(plan.get("round_decisions", [])) + 1)
    ]
    history_path = repo_root / evaluation_relative / "plan/history_retrieval.json"
    history_retrieval = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else {"status": "missing", "matches": []}
    )
    global_route_path = repo_root / evaluation_relative / "plan/global_query_route.json"
    global_route = (
        json.loads(global_route_path.read_text(encoding="utf-8"))
        if global_route_path.is_file()
        else None
    )
    execution_backends = sorted(
        {str(item["observations"].get("execution_backend") or "ACT") for item in rounds}
    )
    act_statuses = [item["observations"].get("act_pipeline_status") for item in rounds]
    measured_act_statuses = [bool(value) for value in act_statuses if value is not None]
    expert_statuses = [item["observations"].get("expert_solvable") for item in rounds]
    measured_expert_statuses = [
        bool(value) for value in expert_statuses if value is not None
    ]
    return {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "user_request": user_request,
        "plan": {
            "max_rounds": plan["max_rounds"],
            "executed_rounds": len(rounds),
            "planning_state": plan.get("planning_state"),
            "round_decisions": plan.get("round_decisions", []),
            "requested_template_ids": plan.get("requested_template_ids", []),
            "completed_template_ids": completed_template_ids,
            "remaining_template_ids": remaining_template_ids,
            "round_budget_remaining": max(int(plan["max_rounds"]) - len(rounds), 0),
        },
        "rounds": rounds,
        "observations": {
            "execution_backends": execution_backends,
            "scene_alignment": all(
                item["observations"]["scene_alignment"] for item in rounds
            ),
            "observed_color_by_round": [
                item["observations"]["observed_color"] for item in rounds
            ],
            "expert_solvable": (
                all(measured_expert_statuses) if measured_expert_statuses else None
            ),
            "act_pipeline_status": (
                all(measured_act_statuses) if measured_act_statuses else None
            ),
            "policy_success": policy_success,
            "policy_success_by_round": [
                item["observations"]["policy_success"] for item in rounds
            ],
            "position_varied": position_metrics.get("position_varied"),
            "position_metrics": position_metrics,
            "position_metrics_by_round": position_metrics_by_round,
            "pipeline_passed": all(
                item["observations"]["pipeline_passed"] for item in rounds
            ),
            "aggregate": compact_aggregate_result(evaluation_aggregate),
            "execution_vqa_conflict": any(
                bool(item.get("execution_vqa", {}).get("evidence_conflict"))
                for item in rounds
            ),
        },
        "total_episodes": total_episodes,
        "history_retrieval": history_retrieval,
        "global_query_route": (
            {
                "selection": global_route.get("selection"),
                "resolved": global_route.get("resolved"),
                "catalog_sha256": global_route.get("catalog_sha256"),
                "provider_called": global_route.get("provider_called"),
                "attempt_count": global_route.get("attempt_count"),
            }
            if global_route is not None
            else None
        ),
        "limitations": {
            "bounded_three_round_prototype": True,
            "few_episodes_are_not_a_generalization_benchmark": True,
            "policy_result_is_not_pipeline_status": True,
        },
        "artifacts": {
            "evaluation_plan": str(evaluation_relative / "plan/evaluation_plan.json"),
            "plan_decisions": decision_artifacts,
            "evidence_assessments": evidence_assessment_artifacts,
            "history_retrieval": str(
                evaluation_relative / "plan/history_retrieval.json"
            ),
            "global_query_route": (
                str(evaluation_relative / "plan/global_query_route.json")
                if global_route is not None
                else None
            ),
            "summary": str(evaluation_relative / "summary/summary.json"),
            "aggregate": str(evaluation_relative / "summary/aggregate_result.json"),
            "round_artifacts": [item["artifacts"] for item in rounds],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id")
    parser.add_argument(
        "--auto-route",
        action="store_true",
        help=(
            "Route the open query through the trusted ACT catalog. In this "
            "mode task, profile, and initial aspects are selected and "
            "validated before any task-specific planner runs."
        ),
    )
    parser.add_argument(
        "--task-name",
        default="beat_block_hammer",
        help=(
            "Canonical RoboTwin task identity. beat_block_hammer uses the "
            "bounded Plan/TaskGen flow; other schema-backed tasks use the "
            "unchanged official expert-probe route."
        ),
    )
    parser.add_argument(
        "--task-module",
        help="Optional Python module for an official schema-backed task.",
    )
    parser.add_argument(
        "--task-profile",
        choices=["official", "position_lr", "adaptive_properties", "fixed_suite"],
        default="official",
        help=(
            "official preserves the upstream task. position_lr enables the "
            "legacy bounded two-round click_bell profile. adaptive_properties "
            "uses model-selected position/object-instance aspects. fixed_suite "
            "executes the selected trusted templates in a frozen order."
        ),
    )
    parser.add_argument(
        "--planning-policy",
        choices=["dynamic_evidence_v1", "fixed_predeclared_v1"],
        default="dynamic_evidence_v1",
        help=(
            "For auto-routed click_bell, choose adaptive evidence routing or "
            "a frozen predeclared schedule over the same selected candidates."
        ),
    )
    parser.add_argument(
        "--generated-rounds",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=2,
        help="Round budget for a bounded click_bell generated profile.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=["expert", "act", "both"],
        help=(
            "Policy backend for schema-backed official tasks. Defaults to "
            "expert; both evaluates ACT and keeps expert as validation."
        ),
    )
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument(
        "--model-profile",
        choices=available_model_profiles(),
        default="legacy",
        help=(
            "Named per-stage model defaults. Individual --*-model arguments "
            "override the selected profile."
        ),
    )
    parser.add_argument("--planner-model")
    parser.add_argument("--taskgen-model")
    parser.add_argument("--toolgen-model")
    parser.add_argument("--vision-model")
    parser.add_argument("--feedback-model")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum visual diagnosis-driven CodeGen repairs per TaskGen run.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--history-database",
        type=Path,
        help=(
            "SQLite planning-history cache. Defaults to "
            "mea/evaluation_runs/history.sqlite3 under --repo-root."
        ),
    )
    parser.add_argument("--history-limit", type=int, default=3)
    parser.add_argument(
        "--reviewed-tool-registry",
        type=Path,
        help=(
            "Optional explicit reviewed generated-Tool registry. Exact "
            "contract/schema/hash matches may be reused across evaluations "
            "without a ToolGen provider call."
        ),
    )
    parser.add_argument(
        "--reviewed-vqa-registry",
        type=Path,
        help=(
            "Optional hash-pinned reviewed VQAQuerySpec registry. Matching "
            "entries may only select existing trusted visual phenomena."
        ),
    )
    parser.add_argument(
        "--tool-recovery-max-restarts",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Legacy local Tool-substage retry budget. The paper-aligned default "
            "is 0 so unexpected Tool execution is handled by whole-round recovery."
        ),
    )
    parser.add_argument(
        "--round-recovery-max-restarts",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Restart the whole evaluation round once only after an explicitly "
            "typed unexpected Tool-execution exception."
        ),
    )
    parser.add_argument(
        "--inject-tool-exception-once",
        action="store_true",
        help=(
            "Development-only unexpected Tool-execution fault. With the "
            "paper-aligned defaults it restarts the whole round and ACT once."
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable cross-evaluation planning retrieval and indexing.",
    )
    parser.add_argument(
        "--evidence-manifest",
        help="Repo-relative hash-pinned preregistration for a registered run.",
    )
    parser.add_argument(
        "--command-plan",
        help="Repo-relative inert fixed/dynamic command plan.",
    )
    parser.add_argument(
        "--registered-route",
        help="Repo-relative deterministic validated route produced by the command plan.",
    )
    parser.add_argument(
        "--registered-strategy",
        choices=["fixed_predeclared_v1", "dynamic_evidence_v1"],
        help="Strategy identity bound by the command plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes must be positive")
    if args.inject_tool_exception_once and not (
        args.tool_recovery_max_restarts == 1
        or args.round_recovery_max_restarts == 1
    ):
        raise SystemExit(
            "--inject-tool-exception-once requires a Tool-substage or whole-round restart"
        )
    if args.auto_route and args.task_module is not None:
        raise SystemExit(
            "--auto-route resolves a trusted task module; do not pass --task-module"
        )
    registered_values = (
        args.evidence_manifest,
        args.command_plan,
        args.registered_route,
        args.registered_strategy,
    )
    if any(value is not None for value in registered_values) and not all(
        value is not None for value in registered_values
    ):
        raise SystemExit(
            "registered execution requires --evidence-manifest, --command-plan, "
            "--registered-route, and --registered-strategy together"
        )
    if args.registered_strategy is not None and args.auto_route:
        raise SystemExit("registered execution forbids live --auto-route")
    if args.registered_strategy is not None and (
        args.tool_recovery_max_restarts != 0
        or args.round_recovery_max_restarts != 0
    ):
        raise SystemExit(
            "registered execution requires both recovery restart budgets to be 0"
        )
    if args.registered_strategy is not None and args.evaluation_id is None:
        raise SystemExit("registered execution requires an explicit --evaluation-id")
    repo_root = args.repo_root.expanduser().resolve()
    registered_execution: dict[str, Any] | None = None
    if args.registered_strategy is not None:
        try:
            registered_execution = load_registered_execution(
                repo_root,
                evidence_manifest_path=str(args.evidence_manifest),
                command_plan_path=str(args.command_plan),
                registered_route_path=str(args.registered_route),
                strategy=str(args.registered_strategy),
                evaluation_id=str(args.evaluation_id),
                observed_argv=list(sys.argv),
            )
        except StrategyPlanError as exc:
            raise SystemExit(f"registered execution preflight failed: {exc}") from exc
    models = resolve_model_profile(
        args.model_profile,
        {
            "planner": args.planner_model,
            "taskgen": args.taskgen_model,
            "toolgen": args.toolgen_model,
            "vision": args.vision_model,
            "feedback": args.feedback_model,
        },
    )
    history_path = (
        args.history_database.expanduser().resolve()
        if args.history_database
        else repo_root / "mea/evaluation_runs/history.sqlite3"
    )
    reviewed_tool_registry = (
        args.reviewed_tool_registry.expanduser().resolve()
        if args.reviewed_tool_registry is not None
        else None
    )
    reviewed_vqa_registry = (
        args.reviewed_vqa_registry.expanduser().resolve()
        if args.reviewed_vqa_registry is not None
        else None
    )
    provider = None
    global_catalog: dict[str, Any] | None = None
    global_route_result: dict[str, Any] | None = None
    global_history_retrieval: dict[str, Any] = {
        "schema_version": 1,
        "status": "disabled" if args.no_history else "empty",
        "candidates": [],
    }
    global_router: GlobalQueryRouter | None = None
    validated_proposal: dict[str, Any] | None = (
        registered_execution["validated_proposal"]
        if registered_execution is not None
        else None
    )
    routed_task_profile: str | None = (
        "adaptive_properties" if registered_execution is not None else None
    )

    if args.auto_route:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=models["planner"],
            vision_model=models["vision"],
            timeout=180.0,
        )
        global_catalog = build_act_catalog(repo_root)
        ready_tasks = [task["task_name"] for task in global_catalog.get("tasks", [])]
        if not ready_tasks:
            raise SystemExit("trusted ACT catalog has no checkpoint-ready tasks")
        global_history_context: list[dict[str, Any]] = []
        if not args.no_history:
            try:
                global_history_db = EvaluationHistoryDB(
                    history_path,
                    repo_root=repo_root,
                )
                global_history_retrieval = global_history_db.retrieve_similar_global(
                    args.request,
                    allowed_task_names=ready_tasks,
                    policy_name="ACT",
                    checkpoint_setting="demo_clean",
                    limit=args.history_limit,
                    exclude_evaluation_id=args.evaluation_id,
                )
                global_history_retrieval["status"] = "passed"
                global_history_context = list(
                    global_history_retrieval.get("candidates", [])
                )
            except Exception as exc:
                global_history_retrieval = {
                    "schema_version": 1,
                    "status": "failed",
                    "candidates": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        global_router = GlobalQueryRouter(
            provider,
            model=models["planner"],
            catalog=global_catalog,
        )
        global_route_result = global_router.route(
            args.request,
            history_context=global_history_context,
        )
        selection = global_route_result["selection"]
        if selection["decision"] == "unsupported":
            unsupported = finish_unsupported_global_route(
                repo_root,
                evaluation_id=args.evaluation_id,
                user_request=args.request,
                catalog=global_catalog,
                route_result=global_route_result,
                router=global_router,
                history_retrieval=global_history_retrieval,
            )
            print(json.dumps(unsupported, ensure_ascii=False, indent=2))
            return
        routed = route_to_planner_proposal(selection, global_catalog)
        args.task_name = routed["task_name"]
        routed_task_profile = routed["task_profile"]
        args.task_profile = (
            (
                "fixed_suite"
                if args.planning_policy == "fixed_predeclared_v1"
                else "adaptive_properties"
            )
            if args.task_name == "click_bell"
            else "official"
        )
        validated_proposal = routed["proposal"]

    legacy_click_bell = args.task_profile == "position_lr"
    adaptive_click_bell = args.task_profile == "adaptive_properties"
    fixed_click_bell = args.task_profile == "fixed_suite"
    bounded_click_bell = legacy_click_bell or adaptive_click_bell or fixed_click_bell
    if bounded_click_bell and args.task_name != "click_bell":
        raise SystemExit(
            "click_bell generated task profiles require --task-name click_bell"
        )
    if args.task_name == "beat_block_hammer" and args.task_profile != "official":
        raise SystemExit("beat_block_hammer does not use click_bell task profiles")
    if args.task_name == "beat_block_hammer" and args.execution_backend:
        raise SystemExit(
            "--execution-backend currently applies to schema-backed official "
            "tasks; beat_block_hammer keeps its bounded generated-task flow"
        )
    if bounded_click_bell and args.execution_backend not in {None, "act"}:
        raise SystemExit("click_bell generated profiles are ACT-only")
    if legacy_click_bell and args.generated_rounds not in {1, 2}:
        raise SystemExit("click_bell position_lr supports at most 2 rounds")
    execution_backend = (
        "act"
        if args.task_name == "beat_block_hammer" or bounded_click_bell
        else (args.execution_backend or "expert")
    )
    # The deterministic official planner can materialize --plan-only without
    # any provider credential. Full execution still creates the provider for
    # final Feedback (and for VQA when an ACT video exists).
    if provider is None and (
        args.task_name == "beat_block_hammer"
        or adaptive_click_bell
        or fixed_click_bell
        or not args.plan_only
    ):
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=models["planner"],
            vision_model=models["vision"],
            timeout=180.0,
        )
    if args.task_name == "beat_block_hammer":
        assert provider is not None
        planner = PlanAgentPrototype(
            repo_root,
            provider,
            model=models["planner"],
        )
    elif adaptive_click_bell:
        assert provider is not None
        planner = ClickBellAdaptivePlanAgent(
            repo_root,
            provider,
            model=models["planner"],
            start_seed=args.start_seed,
            num_episodes=args.num_episodes,
            telemetry_profile=args.telemetry_profile,
            max_rounds=args.generated_rounds,
        )
    elif fixed_click_bell:
        assert provider is not None
        planner = ClickBellFixedSuitePlanAgent(
            repo_root,
            provider,
            model=models["planner"],
            start_seed=args.start_seed,
            num_episodes=args.num_episodes,
            telemetry_profile=args.telemetry_profile,
            max_rounds=args.generated_rounds,
        )
    elif legacy_click_bell:
        planner = ClickBellPositionPlanAgent(
            repo_root,
            start_seed=args.start_seed,
            num_episodes=args.num_episodes,
            telemetry_profile=args.telemetry_profile,
            max_rounds=args.generated_rounds,
        )
    else:
        planner = OfficialTaskPlanAgent(
            repo_root,
            task_name=args.task_name,
            task_module=args.task_module,
            start_seed=args.start_seed,
            num_episodes=args.num_episodes,
            telemetry_profile=args.telemetry_profile,
            execution_backend=execution_backend,
        )
    history_database = None
    history_context: list[dict[str, Any]] = []
    history_retrieval: dict[str, Any] = {
        "schema_version": 1,
        "status": "disabled" if args.no_history else "empty",
        "candidates": [],
    }
    history_path = (
        args.history_database.expanduser().resolve()
        if args.history_database
        else repo_root / "mea/evaluation_runs/history.sqlite3"
    )
    if not args.no_history:
        try:
            history_database = EvaluationHistoryDB(
                history_path,
                repo_root=repo_root,
            )
            history_retrieval = history_database.retrieve_similar(
                args.request,
                task_name=args.task_name,
                policy_name=(
                    "ACT" if execution_backend in {"act", "both"} else "expert"
                ),
                checkpoint_setting="demo_clean",
                limit=args.history_limit,
                exclude_evaluation_id=args.evaluation_id,
            )
            history_retrieval["status"] = "passed"
            history_context = list(history_retrieval.get("candidates", []))
        except Exception as exc:
            history_retrieval = {
                "schema_version": 1,
                "status": "failed",
                "candidates": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
    planner_kwargs: dict[str, Any] = {
        "evaluation_id": args.evaluation_id,
        "history_context": history_context,
        "history_metadata": {
            key: value
            for key, value in history_retrieval.items()
            if key != "candidates"
        },
    }
    if validated_proposal is not None:
        planner_kwargs["validated_proposal"] = validated_proposal
    manifest = planner.plan(args.request, **planner_kwargs)
    evaluation_id = manifest["evaluation_id"]
    evaluation_dir = repo_root / "mea/evaluation_runs" / evaluation_id
    plan = manifest["plan"]
    candidate_suite = list(plan.get("requested_template_ids") or [])
    planning_policy = (
        "fixed_predeclared_v1"
        if fixed_click_bell
        else "dynamic_evidence_v1"
        if adaptive_click_bell
        else None
    )
    registration_identity: dict[str, Any] | None = None
    if registered_execution is not None:
        registration_identity = dict(
            registered_execution["registration_identity"]
        )
        if planning_policy != args.registered_strategy:
            raise RuntimeError(
                "registered strategy does not match resolved planner policy"
            )
        if candidate_suite != registered_execution["expected_candidate_suite"]:
            update_manifest(
                evaluation_dir,
                status="registration_failed",
                registration_identity=registration_identity,
                registration_failure="planner candidate suite differs from preregistration",
            )
            raise RuntimeError(
                "planner candidate suite differs from preregistered route"
            )
        write_json(
            evaluation_dir / "plan/registered_route.json",
            registered_execution["route"],
        )
    if (
        global_catalog is not None
        and global_route_result is not None
        and global_router is not None
    ):
        write_global_route_trace(
            evaluation_dir,
            catalog=global_catalog,
            route_result=global_route_result,
            router=global_router,
            history_retrieval=global_history_retrieval,
        )
    update_manifest(
        evaluation_dir,
        auto_route=args.auto_route,
        global_query_route_path=(
            "plan/global_query_route.json" if args.auto_route else None
        ),
        global_act_catalog_path=(
            "plan/global_act_catalog.json" if args.auto_route else None
        ),
        global_route_selection=(
            global_route_result["selection"]
            if global_route_result is not None
            else None
        ),
        model_profile=args.model_profile,
        resolved_models=models,
        history_database=(
            str(history_path.relative_to(repo_root))
            if history_path.is_relative_to(repo_root)
            else str(history_path)
        ),
        history_retrieval_status=history_retrieval.get("status"),
        task_name=args.task_name,
        task_module=args.task_module,
        task_profile=routed_task_profile or args.task_profile,
        generated_rounds=(args.generated_rounds if bounded_click_bell else None),
        telemetry_profile=args.telemetry_profile,
        execution_backend=execution_backend,
        planning_policy=planning_policy,
        candidate_suite_sha256=(
            canonical_sha256(candidate_suite) if candidate_suite else None
        ),
        reviewed_tool_registry=(
            str(reviewed_tool_registry.relative_to(repo_root))
            if reviewed_tool_registry is not None
            and reviewed_tool_registry.is_relative_to(repo_root)
            else str(reviewed_tool_registry)
            if reviewed_tool_registry is not None
            else None
        ),
        reviewed_vqa_registry=(
            str(reviewed_vqa_registry.relative_to(repo_root))
            if reviewed_vqa_registry is not None
            and reviewed_vqa_registry.is_relative_to(repo_root)
            else str(reviewed_vqa_registry)
            if reviewed_vqa_registry is not None
            else None
        ),
        tool_recovery={
            "schema_version": 1,
            "max_restarts": args.tool_recovery_max_restarts,
            "eligible_failure": "unexpected_tool_execution_exception",
            "reuses_recorded_telemetry": True,
            "restarts_policy_or_simulator": False,
            "development_fault_injection": bool(args.inject_tool_exception_once),
        },
        whole_round_recovery={
            "schema_version": 1,
            "max_restarts": args.round_recovery_max_restarts,
            "eligible_failure": "tool_execution/unexpected_exception",
            "reuses_recorded_telemetry": False,
            "restarts_policy_or_simulator": True,
            "policy_or_simulator_failures_are_not_retried": True,
            "development_fault_injection": bool(args.inject_tool_exception_once),
        },
        registration_identity=registration_identity,
        evidence_manifest=(
            str(args.evidence_manifest) if registration_identity is not None else None
        ),
        command_plan=(
            str(args.command_plan) if registration_identity is not None else None
        ),
        registered_route=(
            str(args.registered_route) if registration_identity is not None else None
        ),
    )

    if args.plan_only:
        update_manifest(evaluation_dir, status="planned_only")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    assert provider is not None

    round_runs: list[dict[str, Any]] = []
    try:
        executed_rounds = 0
        while executed_rounds < len(plan["rounds"]):
            round_plan = plan["rounds"][executed_rounds]
            (
                child_manifest,
                child_dir,
                round_summary,
                tool_evaluation,
                returncode,
            ) = execute_round_stage_aware(
                repo_root,
                evaluation_dir,
                evaluation_id,
                round_plan,
                text_model=models["taskgen"],
                vision_model=models["vision"],
                base_url=args.base_url,
                gpu=args.gpu,
                max_reflections=args.max_reflections,
                provider=provider,
                toolgen_model=models["toolgen"],
                telemetry_profile=args.telemetry_profile,
                reviewed_tool_registry=reviewed_tool_registry,
                reviewed_vqa_registry=reviewed_vqa_registry,
                tool_recovery_max_restarts=args.tool_recovery_max_restarts,
                round_recovery_max_restarts=args.round_recovery_max_restarts,
                inject_tool_exception_once=args.inject_tool_exception_once,
                registration_identity=registration_identity,
            )
            round_runs.append(
                {
                    "round_plan": round_plan,
                    "child_manifest": child_manifest,
                    "child_dir": child_dir,
                    "round_summary": round_summary,
                    "tool_evaluation": tool_evaluation,
                    "returncode": returncode,
                }
            )
            executed_rounds += 1

            plan, decision = planner.decide_next_round(
                evaluation_id=evaluation_id,
                user_request=args.request,
                current_plan=plan,
                observation_history=[item["round_summary"] for item in round_runs],
            )
            if decision["action"] == "stop":
                break

        evaluation_aggregate = aggregate_evaluation_results(
            round_runs,
            evaluation_dir / "summary/aggregate_result.json",
        )
        summary = {
            "schema_version": 2,
            "evaluation_id": evaluation_id,
            "status": (
                "completed"
                if round_runs
                and all(item["round_summary"]["pipeline_passed"] for item in round_runs)
                else "completed_with_pipeline_failure"
            ),
            "rounds": [item["round_summary"] for item in round_runs],
            "aggregate": compact_aggregate_result(evaluation_aggregate),
        }
        write_json(evaluation_dir / "summary/summary.json", summary)
        evidence = build_evidence_bundle(
            repo_root,
            evaluation_id,
            args.request,
            plan,
            round_runs,
            evaluation_aggregate,
        )
        write_json(evaluation_dir / "summary/evidence_bundle.json", evidence)
        update_manifest(
            evaluation_dir,
            status="generating_feedback",
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            summary=summary,
        )
        feedback = FeedbackAgent(
            repo_root,
            provider,
            model=models["feedback"],
        ).generate(
            evidence,
            output_dir=evaluation_dir / "feedback",
        )
        report_path = evaluation_dir / "evaluation_report.md"
        report_path.write_text(
            render_evaluation_report(evidence, feedback),
            encoding="utf-8",
        )
        update_manifest(
            evaluation_dir,
            status=summary["status"],
            lifecycle_status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            feedback_path="feedback/feedback.json",
            report_path="evaluation_report.md",
            child_run_ids=[item["child_manifest"].get("run_id") for item in round_runs],
            summary=summary,
            feedback=feedback,
        )
        history_index = {"status": "disabled" if args.no_history else "not_available"}
        if history_database is not None:
            try:
                history_index = {
                    "status": "passed",
                    **history_database.index_evaluation_dir(evaluation_dir),
                }
            except Exception as exc:
                history_index = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        update_manifest(evaluation_dir, history_index=history_index)
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "child_run_ids": [
                        item["child_manifest"].get("run_id") for item in round_runs
                    ],
                    "summary": summary,
                    "feedback": feedback,
                    "history_retrieval": {
                        "status": history_retrieval.get("status"),
                        "selected_count": len(history_context),
                    },
                    "history_index": history_index,
                    "report_path": str(report_path.relative_to(repo_root)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        update_manifest(
            evaluation_dir,
            status="failed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
