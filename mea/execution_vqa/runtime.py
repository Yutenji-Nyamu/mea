"""Execution-time VQA over one completed policy rollout."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.agent_evidence import round_execution_backend
from mea.planner.experiment_candidate import validate_experiment_candidate
from mea.round_tools import executed_runtime_task_schema
from mea.toolgen import build_tool_artifact_context

from .open_question import (
    OpenVQAQuestionAgent,
    load_run_local_vqa_questions,
    register_run_local_vqa_question,
)
from .prototype import run_execution_vqa
from .query import build_execution_vqa_query


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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

    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    candidates = sorted(
        (
            episode
            for episode in trusted.get("episodes", [])
            if (
                str(episode.get("policy_name", "")).casefold() == "expert"
                if execution_backend == "expert"
                else (
                    episode.get("role") == "policy_under_evaluation"
                    or str(episode.get("policy_name", "")).casefold()
                    == "act"
                )
            )
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
    return episode_dir, episode, _episode_numeric_tool_results(episode)


def _episode_numeric_tool_results(
    episode: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read both base Toolkit rows and the planned typed Tool row."""

    results = [
        deepcopy(dict(item))
        for item in episode.get("tool_results", [])
        if isinstance(item, Mapping)
    ]
    planned = episode.get("result")
    if isinstance(planned, Mapping):
        normalized = deepcopy(dict(planned))
        if normalized not in results:
            results.append(normalized)
    return results


def _canonical_episode_identity(value: Any) -> str | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    normalized = str(value).replace("\\", "/").rstrip("/")
    telemetry_marker = "evaluation/telemetry/"
    if telemetry_marker in normalized:
        normalized = normalized.split(telemetry_marker, 1)[1]
    return normalized.lstrip("./")


def _same_telemetry_episode(
    candidate: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> bool:
    """Match generated and Trusted Tool rows to one physical rollout."""

    candidate_dir = _canonical_episode_identity(
        candidate.get("episode_dir")
    )
    representative_dir = _canonical_episode_identity(
        representative.get("episode_dir")
    )
    if candidate_dir is not None and representative_dir is not None:
        return candidate_dir == representative_dir
    return (
        candidate.get("seed") == representative.get("seed")
        and str(candidate.get("policy_name", "")).casefold()
        == str(representative.get("policy_name", "")).casefold()
    )


def _legacy_round_requests_execution_vqa(
    round_plan: Mapping[str, Any] | None,
) -> bool:
    """Keep legacy VQA behavior unless the Proposal explicitly omits it."""

    semantic_needs = (round_plan or {}).get("semantic_need_execution")
    if not isinstance(semantic_needs, Mapping):
        return True
    vqa_need = semantic_needs.get("vqa_tool")
    return not (
        isinstance(vqa_need, Mapping)
        and vqa_need.get("requested") is False
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

    semantic_needs = (round_plan or {}).get("semantic_need_execution")
    vqa_need = (
        semantic_needs.get("vqa_tool")
        if isinstance(semantic_needs, Mapping)
        else None
    )
    open_vqa_bundle: dict[str, Any] | None = None
    if (
        isinstance(vqa_need, Mapping)
        and vqa_need.get("requested") is True
        and isinstance(vqa_need.get("description"), str)
        and vqa_need["description"].strip()
    ):
        candidate_value = (round_plan or {}).get("proposal") or (
            round_plan or {}
        ).get("experiment_candidate")
        candidate = (
            validate_experiment_candidate(candidate_value)
            if isinstance(candidate_value, Mapping)
            else None
        )
        if candidate is None:
            raise RuntimeError(
                "Query-induced VQA generation requires a typed Proposal"
            )
        runtime_schema = executed_runtime_task_schema(
            child_dir,
            task_name=str(candidate["base_task"]),
        )
        run_local_vqa_registry = (
            execution_dir.parent.parent / "vqa_registry"
        )
        artifact_context = build_tool_artifact_context(
            repo_root,
            task_name=str(candidate["base_task"]),
            proposal=candidate,
            task_artifact_summary=(
                child_manifest.get("task_artifact_summary")
                if isinstance(
                    child_manifest.get("task_artifact_summary"),
                    Mapping,
                )
                else None
            ),
            runtime_schema=runtime_schema,
            reusable_vqa_questions=load_run_local_vqa_questions(
                run_local_vqa_registry
            ),
        )
        vqa_agent = OpenVQAQuestionAgent(provider, model=model)
        open_vqa_bundle = vqa_agent.propose(
            artifact_context=artifact_context,
            vqa_need=vqa_need,
            template_id=(round_plan or {}).get("template_id"),
            tool_contract=(round_plan or {}).get("tool_request"),
            reviewed_registry_dir=reviewed_vqa_registry,
        )
        open_vqa_dir = execution_dir / "open_vqa_question"
        _write_json(open_vqa_dir / "question_bundle.json", open_vqa_bundle)
        if (
            open_vqa_bundle.get("status") in {"generated", "reused"}
            and isinstance(
                open_vqa_bundle.get("question_spec"),
                Mapping,
            )
        ):
            open_vqa_bundle["registration"] = (
                register_run_local_vqa_question(
                    run_local_vqa_registry,
                    open_vqa_bundle,
                    artifact_path=str(
                        (
                            open_vqa_dir / "question_bundle.json"
                        ).relative_to(repo_root)
                    ).replace("\\", "/"),
                )
            )
            _write_json(
                open_vqa_dir / "question_bundle.json",
                open_vqa_bundle,
            )
        if vqa_agent.last_prompt is not None:
            (open_vqa_dir / "prompt.md").write_text(
                vqa_agent.last_prompt,
                encoding="utf-8",
            )
        for index, response in enumerate(
            vqa_agent.last_responses,
            start=1,
        ):
            (open_vqa_dir / f"response_{index}.txt").write_text(
                response + "\n",
                encoding="utf-8",
            )
    proposal = ((round_plan or {}).get("tool_proposal") or {})
    proposal_vqa_explicit = bool(
        proposal.get("vqa_phenomenon_ids")
        or proposal.get("vqa_question_specs")
    )
    query = (
        open_vqa_bundle["query"]
        if open_vqa_bundle is not None
        else build_execution_vqa_query(
            task_name=(
                str(
                    (round_plan or {}).get("task_name")
                    or child_manifest.get("task_name")
                )
                if (round_plan or {}).get("task_name")
                or child_manifest.get("task_name")
                else None
            ),
            template_id=(round_plan or {}).get("template_id"),
            sub_aspect=(round_plan or {}).get("sub_aspect"),
            tool_contract=(round_plan or {}).get("tool_request"),
            proposed_phenomenon_ids=(
                proposal.get("vqa_phenomenon_ids")
                if proposal_vqa_explicit
                else None
            ),
            proposed_question_specs=(
                proposal.get("vqa_question_specs")
                if proposal_vqa_explicit
                else None
            ),
            reviewed_registry_dir=reviewed_vqa_registry,
        )
    )
    _write_json(execution_dir / "execution_vqa_query.json", query)
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
        _write_json(
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
        _write_json(
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
        if not reference_scene.is_file():
            # Native official rounds have rollout video/telemetry but no
            # TaskGen-owned initial-head image.  VQA can select its baseline
            # from the rollout instead of failing on a nonexistent artifact.
            reference_scene = None
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
        _write_json(execution_dir / "execution_vqa_error.json", result)
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
    _write_json(execution_dir / "execution_vqa/execution_vqa.json", result)
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


__all__ = ["compact_execution_vqa", "run_round_execution_vqa"]

