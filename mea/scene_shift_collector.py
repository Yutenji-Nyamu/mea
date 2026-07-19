"""Collect hash-bound scene-shift candidates from completed Agent runs.

This module is deliberately offline.  It discovers existing runtime artifacts,
checks the subset of their contracts needed by the scene-shift audit, and
reports exact missing/invalid inputs.  It never starts RoboTwin, ACT, or a
provider, and it never derives proxy labels from the VQA predictions being
evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.execution_vqa import (
    ExecutionVQAError,
    ExecutionVQAQueryError,
    validate_execution_vqa_query,
    validate_execution_vqa_response,
)
from mea.scene_shift_vqa_validation import (
    CONDITION_CONTRACTS,
    CONDITIONS,
    PROTOCOL as SUITE_PROTOCOL,
)


COLLECTION_PROTOCOL = "real_simulator_scene_shift_collection_v1"
_EVALUATION_ID = re.compile(r"eval_[A-Za-z0-9_.-]+")


class SceneShiftCollectionError(RuntimeError):
    """Raised for an invalid collector invocation rather than a bad run."""


class _ArtifactProblem(RuntimeError):
    def __init__(self, code: str, path: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.path = path
        self.detail = detail


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(
    root: Path,
    value: Any,
    *,
    code_prefix: str,
    scope: Path | None = None,
) -> Path:
    if not isinstance(value, str) or not value:
        raise _ArtifactProblem(
            f"{code_prefix}_path_invalid", str(value), "path must be a non-empty string"
        )
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        lexical = candidate.absolute()
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise _ArtifactProblem(
            f"{code_prefix}_path_escape", value, "path escapes repo root"
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise _ArtifactProblem(
                f"{code_prefix}_symlink", value, "path traverses a symlink"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise _ArtifactProblem(
            f"{code_prefix}_missing", value, "regular source file is missing"
        ) from exc
    if not resolved.is_relative_to(root):
        raise _ArtifactProblem(
            f"{code_prefix}_path_escape", value, "resolved path escapes repo root"
        )
    if scope is not None and not resolved.is_relative_to(scope):
        raise _ArtifactProblem(
            f"{code_prefix}_wrong_scope", value, "source is outside its owning run"
        )
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise _ArtifactProblem(
            f"{code_prefix}_missing", value, "regular source file is missing or empty"
        )
    return resolved


def _read_object(path: Path, *, code_prefix: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _ArtifactProblem(
            f"{code_prefix}_json_invalid",
            path.as_posix(),
            "source is not valid JSON",
        ) from exc
    if not isinstance(value, dict):
        raise _ArtifactProblem(
            f"{code_prefix}_json_invalid",
            path.as_posix(),
            "JSON source must be an object",
        )
    return value


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _valid_colors(colors: Any, count: Any) -> bool:
    return bool(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 1
        and isinstance(colors, list)
        and len(colors) == count
        and all(
            isinstance(color, list)
            and len(color) == 3
            and all(
                isinstance(component, (int, float))
                and not isinstance(component, bool)
                and math.isfinite(float(component))
                and 0.0 <= float(component) <= 1.0
                for component in color
            )
            for color in colors
        )
    )


def _is_finite_zero(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == 0.0
    )


def _scene_contract_passed(child: Mapping[str, Any], condition: str, seed: int) -> bool:
    contract = CONDITION_CONTRACTS[condition]
    scene = child.get("scene_validation")
    samples = child.get("position_samples")
    if not isinstance(scene, Mapping) or not isinstance(samples, Mapping):
        return False
    rule = scene.get("rule_check")
    if (
        scene.get("seed") != seed
        or scene.get("setup_success") is not True
        or scene.get("render_success") is not True
        or not isinstance(rule, Mapping)
        or rule.get("passed") is not True
        or samples.get("passed") is not True
        or samples.get("controlled_axis") != contract["controlled_axis"]
        or samples.get("variant_contract") != contract["changes"]
    ):
        return False
    randomization = scene.get("domain_randomization")
    if not isinstance(randomization, Mapping):
        return False
    if condition == "scene_background_texture.unseen":
        return bool(
            scene.get("eval_mode") is True
            and randomization.get("random_background") is True
            and _is_finite_zero(randomization.get("clean_background_rate"))
            and randomization.get("texture_split") == "unseen"
            and isinstance(randomization.get("wall_texture"), str)
            and randomization["wall_texture"].startswith("unseen/")
            and isinstance(randomization.get("table_texture"), str)
            and randomization["table_texture"].startswith("unseen/")
            and randomization.get("background_authority")
            == "simulator_task_info:texture_info"
        )
    authority = (
        "simulator_task_attributes:random_light,crazy_random_light_rate,"
        "crazy_random_light;simulator_light_components:get_color"
    )
    return bool(
        randomization.get("random_light") is True
        and _is_finite_zero(randomization.get("crazy_random_light_rate"))
        and randomization.get("crazy_random_light") is False
        and _valid_colors(
            randomization.get("direction_light_colors"),
            randomization.get("direction_light_count"),
        )
        and _valid_colors(
            randomization.get("point_light_colors"),
            randomization.get("point_light_count"),
        )
        and randomization.get("lighting_authority") == authority
    )


def _diagnostic(
    candidate: Mapping[str, Any],
    *,
    code: str,
    artifact: str,
    path: str | None,
    detail: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source_evaluation_id": candidate.get("source_evaluation_id"),
        "round_id": candidate.get("round_id"),
        "child_run_id": candidate.get("child_run_id"),
        "condition": candidate.get("condition"),
        "seed": candidate.get("seed"),
        "code": code,
        "artifact": artifact,
        "path": path,
        "detail": detail,
    }


def _candidate_id(evaluation_id: str, round_id: str, child_run_id: str | None) -> str:
    child = child_run_id or "missing_child"
    return f"{evaluation_id}__{round_id}__{child}"


def _collect_round(
    root: Path,
    evaluation_dir: Path,
    parent_path: Path,
    parent: Mapping[str, Any],
    round_value: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_id = str(parent["evaluation_id"])
    round_id = str(round_value.get("round_id") or "missing_round")
    condition = str(round_value.get("variant_id") or "")
    child_run_id = round_value.get("child_run_id") or round_value.get("taskgen_run_id")
    child_run_id = str(child_run_id) if isinstance(child_run_id, str) else None
    candidate: dict[str, Any] = {
        "candidate_id": _candidate_id(evaluation_id, round_id, child_run_id),
        "source_evaluation_id": evaluation_id,
        "round_id": round_id,
        "child_run_id": child_run_id,
        "condition": condition,
        "seed": None,
        "status": "incomplete",
        "sources": {},
        "expected_query": None,
        "diagnostic_codes": [],
    }
    diagnostics: list[dict[str, Any]] = []

    def add(code: str, artifact: str, path: Any, detail: str) -> None:
        diagnostics.append(
            _diagnostic(
                candidate,
                code=code,
                artifact=artifact,
                path=str(path) if path is not None else None,
                detail=detail,
            )
        )

    if condition not in CONDITION_CONTRACTS:
        add("condition_unsupported", "round", None, "round is not a supported scene shift")
        candidate["diagnostic_codes"] = ["condition_unsupported"]
        return candidate, diagnostics
    contract = CONDITION_CONTRACTS[condition]

    child_ids = parent.get("child_run_ids")
    child_is_bound = bool(
        child_run_id
        and (
            (
                isinstance(child_ids, list)
                and bool(child_ids)
                and child_run_id in child_ids
                and all(isinstance(item, str) and item for item in child_ids)
                and len(child_ids) == len(set(child_ids))
                and parent.get("active_child_run_id") == child_ids[-1]
            )
            or (
                child_ids is None
                and parent.get("active_child_run_id") == child_run_id
            )
        )
    )
    if not child_is_bound:
        add(
            "parent_child_binding_invalid",
            "evaluation_manifest",
            _relative(root, parent_path),
            (
                "round child requires a non-empty unique child_run_ids list, "
                "membership, and active_child_run_id equal to its final entry"
            ),
        )
    if (
        parent.get("telemetry_profile") != "balanced_v1"
        or not isinstance(parent.get("base_commit"), str)
        or not parent.get("base_commit")
    ):
        add(
            "evaluation_protocol_invalid",
            "evaluation_manifest",
            _relative(root, parent_path),
            "completed parent must bind balanced_v1 telemetry and a base commit",
        )

    child_path: Path | None = None
    child: dict[str, Any] | None = None
    if child_run_id is None or re.fullmatch(r"run_[A-Za-z0-9_.-]+", child_run_id) is None:
        add("child_run_id_invalid", "taskgen_manifest", None, "child run id is missing or invalid")
    else:
        expected_child = f"mea/generated_tasks/{child_run_id}/manifest.json"
        try:
            child_path = _repo_path(root, expected_child, code_prefix="taskgen_manifest")
            child = _read_object(child_path, code_prefix="taskgen_manifest")
        except _ArtifactProblem as exc:
            add(exc.code, "taskgen_manifest", exc.path, exc.detail)

    artifacts = round_value.get("artifacts")
    artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    execution_path_value = artifacts.get("execution_vqa") or (
        f"mea/evaluation_runs/{evaluation_id}/execution/{round_id}/"
        "execution_vqa/execution_vqa.json"
    )
    execution_path: Path | None = None
    execution: dict[str, Any] | None = None
    try:
        execution_path = _repo_path(
            root,
            execution_path_value,
            code_prefix="execution_vqa",
            scope=evaluation_dir,
        )
        execution = _read_object(execution_path, code_prefix="execution_vqa")
    except _ArtifactProblem as exc:
        add(exc.code, "execution_vqa", exc.path, exc.detail)

    episode_path: Path | None = None
    video_path: Path | None = None
    query_path: Path | None = None
    montage_path: Path | None = None
    episode: dict[str, Any] | None = None
    query: dict[str, Any] | None = None

    if execution is not None:
        if execution.get("schema_version") != 1 or execution.get("status") != "passed":
            add(
                "execution_vqa_not_passed",
                "execution_vqa",
                _relative(root, execution_path),
                "Execution VQA must be a completed passed artifact",
            )
        embedded_query = execution.get("query")
        try:
            normalized_query = validate_execution_vqa_query(embedded_query)
        except ExecutionVQAQueryError as exc:
            normalized_query = None
            add(
                "execution_vqa_query_contract_invalid",
                "execution_vqa",
                _relative(root, execution_path),
                str(exc),
            )
        expected_query = {
            "task_name": "click_bell",
            "template_id": contract["template_id"],
            "sub_aspect": contract["sub_aspect"],
            "primary_visibility_phenomenon_id": contract[
                "primary_visibility_phenomenon_id"
            ],
            "phenomenon_ids": list(contract["phenomenon_ids"]),
        }
        candidate["expected_query"] = expected_query
        if normalized_query is not None and (
            normalized_query.get("task_name") != "click_bell"
            or normalized_query.get("template_id") != contract["template_id"]
            or normalized_query.get("sub_aspect") != contract["sub_aspect"]
            or normalized_query.get("phenomenon_ids") != contract["phenomenon_ids"]
        ):
            add(
                "execution_vqa_query_mismatch",
                "execution_vqa",
                _relative(root, execution_path),
                "embedded query does not match the condition contract",
            )

        execution_artifacts = execution.get("artifacts")
        execution_artifacts = (
            dict(execution_artifacts) if isinstance(execution_artifacts, Mapping) else {}
        )
        query_value = execution_artifacts.get("query") or artifacts.get(
            "execution_vqa_query"
        )
        montage_value = execution_artifacts.get("montage") or artifacts.get(
            "execution_vqa_montage"
        )
        selection = execution.get("selection")
        selection = dict(selection) if isinstance(selection, Mapping) else {}
        montage_value = montage_value or selection.get("montage_path")
        video_value = selection.get("video_path")
        representative = execution.get("representative_episode")
        episode_value = (
            f"{representative}/episode.json"
            if isinstance(representative, str) and representative
            else None
        )
        for name, value, scope in (
            ("episode", episode_value, child_path.parent if child_path else None),
            ("video", video_value, child_path.parent if child_path else None),
            ("query", query_value, evaluation_dir),
            ("montage", montage_value, evaluation_dir),
        ):
            try:
                resolved = _repo_path(
                    root, value, code_prefix=name, scope=scope
                )
                if name == "episode":
                    episode_path = resolved
                    episode = _read_object(resolved, code_prefix="episode")
                elif name == "video":
                    video_path = resolved
                elif name == "query":
                    query_path = resolved
                    query = _read_object(resolved, code_prefix="query")
                else:
                    montage_path = resolved
            except _ArtifactProblem as exc:
                add(exc.code, name, exc.path, exc.detail)

        if query is not None and normalized_query is not None and query != normalized_query:
            add(
                "query_content_mismatch",
                "query",
                _relative(root, query_path),
                "hashed query differs from the embedded Execution VQA query",
            )
        if (
            video_path is not None
            and episode_path is not None
            and video_path.parent != episode_path.parent
        ):
            add(
                "video_episode_mismatch",
                "video",
                _relative(root, video_path),
                "selected video is not in the representative episode directory",
            )
        if montage_path is not None and selection.get("montage_path"):
            try:
                selected_montage = _repo_path(
                    root,
                    selection["montage_path"],
                    code_prefix="montage",
                    scope=evaluation_dir,
                )
                if selected_montage != montage_path:
                    add(
                        "montage_selection_mismatch",
                        "montage",
                        _relative(root, montage_path),
                        "selected montage differs from the hashed montage",
                    )
            except _ArtifactProblem as exc:
                add(exc.code, "montage", exc.path, exc.detail)

        selected_frames = selection.get("selected_frames")
        frame_ids = [
            item.get("frame_id")
            for item in selected_frames or []
            if isinstance(item, Mapping) and isinstance(item.get("frame_id"), str)
        ]
        if not frame_ids or len(frame_ids) != len(selected_frames or []):
            add(
                "execution_vqa_frames_invalid",
                "execution_vqa",
                _relative(root, execution_path),
                "selected frames must contain non-empty frame ids",
            )
        else:
            observation = execution.get("observation")
            # ExecutionVQA appends derived, evidence-level fields such as
            # ``evidence_conflict`` after validating the provider response.
            # Permit only that known extension, then revalidate the provider
            # contract.  The complete result remains hash-bound in the source
            # inventory, so unknown fields still fail closed.
            response_keys = {
                "phenomena",
                "confidence",
                "frame_ids",
                "numeric_consistency",
                "conflicts",
            }
            try:
                if not isinstance(observation, Mapping) or set(observation) not in (
                    response_keys,
                    response_keys | {"evidence_conflict"},
                ):
                    raise ExecutionVQAError(
                        "Execution VQA observation has invalid fields"
                    )
                normalized = validate_execution_vqa_response(
                    {key: observation.get(key) for key in response_keys},
                    allowed_frame_ids=frame_ids,
                    expected_phenomenon_ids=contract["phenomenon_ids"],
                )
                if "evidence_conflict" in observation and (
                    not isinstance(observation["evidence_conflict"], bool)
                    or observation["evidence_conflict"]
                    != normalized["evidence_conflict"]
                ):
                    raise ExecutionVQAError(
                        "Execution VQA evidence_conflict does not match response"
                    )
            except ExecutionVQAError as exc:
                add(
                    "execution_vqa_response_invalid",
                    "execution_vqa",
                    _relative(root, execution_path),
                    str(exc),
                )

    # If the result itself is missing, still inventory independently existing
    # inputs instead of falsely reporting the ACT episode/video/query/montage
    # as absent.  They remain incomplete because no passed VQA artifact binds
    # them together.
    if execution is None:
        query_value = artifacts.get("execution_vqa_query") or (
            f"mea/evaluation_runs/{evaluation_id}/execution/{round_id}/"
            "execution_vqa_query.json"
        )
        montage_value = artifacts.get("execution_vqa_montage") or (
            f"mea/evaluation_runs/{evaluation_id}/execution/{round_id}/"
            "execution_vqa/execution_montage.png"
        )
        for name, value in (("query", query_value), ("montage", montage_value)):
            try:
                resolved = _repo_path(
                    root, value, code_prefix=name, scope=evaluation_dir
                )
                if name == "query":
                    query_path = resolved
                    query = _read_object(resolved, code_prefix="query")
                else:
                    montage_path = resolved
            except _ArtifactProblem as exc:
                add(exc.code, name, exc.path, exc.detail)
        act = child.get("act_evaluation") if child is not None else None
        associations = act.get("video_associations") if isinstance(act, Mapping) else None
        expected_seeds = round_value.get("seeds")
        expected_seed = (
            expected_seeds[0]
            if isinstance(expected_seeds, list)
            and len(expected_seeds) == 1
            and isinstance(expected_seeds[0], int)
            and not isinstance(expected_seeds[0], bool)
            else None
        )
        matches: list[tuple[Path, Path, dict[str, Any]]] = []
        for association in associations or []:
            if not isinstance(association, Mapping):
                continue
            try:
                candidate_episode = _repo_path(
                    root,
                    f"{association.get('episode_dir')}/episode.json",
                    code_prefix="episode",
                    scope=child_path.parent if child_path else None,
                )
                candidate_video = _repo_path(
                    root,
                    association.get("video"),
                    code_prefix="video",
                    scope=child_path.parent if child_path else None,
                )
                candidate_metadata = _read_object(
                    candidate_episode, code_prefix="episode"
                )
            except _ArtifactProblem:
                continue
            if expected_seed is None or candidate_metadata.get("seed") == expected_seed:
                matches.append(
                    (candidate_episode, candidate_video, candidate_metadata)
                )
        if len(matches) == 1:
            episode_path, video_path, episode = matches[0]

    if episode is not None:
        seed = episode.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            add(
                "episode_seed_invalid",
                "episode",
                _relative(root, episode_path),
                "episode seed must be a non-negative integer",
            )
        else:
            candidate["seed"] = seed
        if (
            episode.get("task_name") != "click_bell"
            or str(episode.get("policy_name", "")).casefold() != "act"
            or episode.get("checkpoint_setting") != "demo_clean"
            or episode.get("telemetry_profile_id") != "balanced_v1"
            or not isinstance(episode.get("telemetry_profile_sha256"), str)
            or not episode.get("telemetry_profile_sha256")
            or not isinstance(episode.get("success"), bool)
        ):
            add(
                "episode_contract_invalid",
                "episode",
                _relative(root, episode_path),
                "episode is not a completed balanced_v1 click_bell ACT sample",
            )

    if child is not None:
        if (
            child.get("schema_version") != 1
            or child.get("status") != "completed"
            or child.get("failure") is not None
            or child.get("task_name") != "click_bell"
            or child.get("task_module") != "mea.tasks.click_bell"
            or child.get("mode") != "reuse"
            or child.get("generation_kind") != "bounded_variant_overlay"
            or child.get("variant_id") != contract["template_id"]
            or child.get("capability_id") != contract["capability_id"]
        ):
            add(
                "taskgen_contract_invalid",
                "taskgen_manifest",
                _relative(root, child_path),
                "child is not the required completed bounded scene variant",
            )
        seed = candidate.get("seed")
        if isinstance(seed, int) and not _scene_contract_passed(child, condition, seed):
            add(
                "scene_contract_invalid",
                "taskgen_manifest",
                _relative(root, child_path),
                "simulator-authoritative scene state does not match the condition",
            )
        act = child.get("act_evaluation")
        if not isinstance(act, Mapping):
            add(
                "act_evaluation_missing",
                "taskgen_manifest",
                _relative(root, child_path),
                "child has no ACT evaluation object",
            )
        elif isinstance(candidate.get("seed"), int):
            seed = int(candidate["seed"])
            associations = act.get("video_associations")
            episode_dir = _relative(root, episode_path.parent) if episode_path else None
            video_relative = _relative(root, video_path) if video_path else None
            matches = [
                item
                for item in associations or []
                if isinstance(item, Mapping)
                and item.get("episode_dir") == episode_dir
                and item.get("video") == video_relative
            ]
            if (
                act.get("passed") is not True
                or act.get("task_name") != "click_bell"
                or act.get("task_config") != "demo_clean"
                or act.get("checkpoint_setting") != "demo_clean"
                or seed not in (act.get("actual_seeds") or [])
                or len(matches) != 1
            ):
                add(
                    "act_evaluation_contract_invalid",
                    "taskgen_manifest",
                    _relative(root, child_path),
                    "ACT manifest does not bind the representative episode and video",
                )

    if child is not None and child.get("base_commit") != parent.get("base_commit"):
        add(
            "base_commit_mismatch",
            "taskgen_manifest",
            _relative(root, child_path),
            "parent and child base commits differ",
        )

    source_paths = {
        "taskgen_manifest": child_path,
        "evaluation_manifest": parent_path,
        "episode": episode_path,
        "video": video_path,
        "execution_vqa": execution_path,
        "query": query_path,
        "montage": montage_path,
    }
    for name, path in source_paths.items():
        if path is None:
            if not any(item["artifact"] == name for item in diagnostics):
                add(f"{name}_missing", name, None, f"{name} could not be resolved")
            continue
        candidate["sources"][name] = {
            "path": _relative(root, path),
            "sha256": _sha256(path),
        }

    # Update diagnostic seeds after the representative episode becomes known.
    for item in diagnostics:
        item["seed"] = candidate.get("seed")
    candidate["diagnostic_codes"] = sorted({item["code"] for item in diagnostics})
    if not diagnostics and len(candidate["sources"]) == 7:
        candidate["status"] = "ready"
    return candidate, diagnostics


def _placeholder_candidates(
    evaluation_id: str,
    templates: list[str],
    *,
    code: str,
    path: str | None,
    detail: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, condition in enumerate(templates, start=1):
        candidate = {
            "candidate_id": _candidate_id(evaluation_id, f"unresolved_{index}", None),
            "source_evaluation_id": evaluation_id,
            "round_id": None,
            "child_run_id": None,
            "condition": condition,
            "seed": None,
            "status": "incomplete",
            "sources": {},
            "expected_query": None,
            "diagnostic_codes": [code],
        }
        candidates.append(candidate)
        diagnostics.append(
            _diagnostic(
                candidate,
                code=code,
                artifact="evidence_bundle",
                path=path,
                detail=detail,
            )
        )
    return candidates, diagnostics


def _evaluation_problem(
    evaluation_id: str,
    *,
    code: str,
    path: str,
    detail: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = {
        "candidate_id": _candidate_id(evaluation_id, "unresolved", None),
        "source_evaluation_id": evaluation_id,
        "round_id": None,
        "child_run_id": None,
        "condition": None,
        "seed": None,
        "status": "incomplete",
        "sources": {},
        "expected_query": None,
        "diagnostic_codes": [code],
    }
    return candidate, _diagnostic(
        candidate,
        code=code,
        artifact="evaluation_manifest",
        path=path,
        detail=detail,
    )


def _label_suite_draft(
    candidates: list[dict[str, Any]],
    labels: Mapping[str, Any] | None,
    reviewer_id: str | None,
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    if labels is None:
        return None, "not_requested", []
    if not isinstance(labels, Mapping):
        raise SceneShiftCollectionError("labels must be a candidate_id -> phenomenon map")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        return None, "labels_incomplete", [
            {
                "code": "reviewer_id_missing",
                "candidate_id": None,
                "phenomenon_id": None,
                "detail": "reviewer_id is required when labels are supplied",
            }
        ]
    ready = [item for item in candidates if item["status"] == "ready"]
    issues: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    ready_ids = {item["candidate_id"] for item in ready}
    for extra in sorted(set(labels) - ready_ids):
        issues.append(
            {
                "code": "label_candidate_unknown",
                "candidate_id": str(extra),
                "phenomenon_id": None,
                "detail": "labels refer to a candidate that is not ready",
            }
        )
    for candidate in ready:
        candidate_labels = labels.get(candidate["candidate_id"])
        phenomena = candidate["expected_query"]["phenomenon_ids"]
        if not isinstance(candidate_labels, Mapping):
            issues.append(
                {
                    "code": "candidate_labels_missing",
                    "candidate_id": candidate["candidate_id"],
                    "phenomenon_id": None,
                    "detail": "candidate needs an external label for every phenomenon",
                }
            )
            continue
        unknown = sorted(set(candidate_labels) - set(phenomena))
        for phenomenon in unknown:
            issues.append(
                {
                    "code": "label_phenomenon_unknown",
                    "candidate_id": candidate["candidate_id"],
                    "phenomenon_id": str(phenomenon),
                    "detail": "label is outside the expected query contract",
                }
            )
        normalized_labels = []
        for phenomenon in phenomena:
            observed = candidate_labels.get(phenomenon)
            if not isinstance(observed, bool):
                issues.append(
                    {
                        "code": "label_missing_or_invalid",
                        "candidate_id": candidate["candidate_id"],
                        "phenomenon_id": phenomenon,
                        "detail": "external proxy label must be boolean",
                    }
                )
                continue
            normalized_labels.append(
                {
                    "phenomenon_id": phenomenon,
                    "observed": observed,
                    "label_source": "development_agent_proxy",
                    "reviewer_id": reviewer_id.strip(),
                }
            )
        if len(normalized_labels) == len(phenomena):
            cases.append(
                {
                    "id": candidate["candidate_id"],
                    "condition": candidate["condition"],
                    "seed": candidate["seed"],
                    "source_evaluation_id": candidate["source_evaluation_id"],
                    "sources": deepcopy(candidate["sources"]),
                    "expected_query": deepcopy(candidate["expected_query"]),
                    "labels": normalized_labels,
                }
            )
    if issues or not ready:
        return None, "labels_incomplete", issues
    identity_hash = _canonical_sha256(
        [(item["candidate_id"], item["seed"]) for item in ready]
    )[:16]
    return (
        {
            "schema_version": 1,
            "suite_id": f"sceneshiftvqa_collected_{identity_hash}",
            "protocol": SUITE_PROTOCOL,
            "reviewer": {
                "id": reviewer_id.strip(),
                "kind": "development_agent_proxy",
            },
            "cases": cases,
        },
        "emitted_unvalidated",
        [],
    )


def collect_scene_shift_candidates(
    repo_root: str | Path,
    *,
    evaluation_ids: list[str] | tuple[str, ...] | None = None,
    labels: Mapping[str, Any] | None = None,
    reviewer_id: str | None = None,
) -> dict[str, Any]:
    """Return deterministic candidates and diagnostics without making calls.

    ``labels`` is optional and must map ``candidate_id`` to a mapping of
    ``phenomenon_id -> bool``.  Those booleans must come from a caller review;
    VQA predictions are intentionally never used as labels.
    """

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise SceneShiftCollectionError(f"repo root is not a directory: {root}")
    evaluation_root = root / "mea/evaluation_runs"
    if not evaluation_root.is_dir() or evaluation_root.is_symlink():
        raise SceneShiftCollectionError(
            "mea/evaluation_runs must be a regular directory inside repo root"
        )
    selected: set[str] | None = None
    if evaluation_ids is not None:
        if (
            not isinstance(evaluation_ids, (list, tuple))
            or any(
                not isinstance(item, str) or _EVALUATION_ID.fullmatch(item) is None
                for item in evaluation_ids
            )
        ):
            raise SceneShiftCollectionError("evaluation_ids contain an invalid id")
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise SceneShiftCollectionError("evaluation_ids contain duplicates")
        selected = set(evaluation_ids)

    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    scanned = 0
    completed = 0
    relevant = 0
    seen_evaluations: set[str] = set()
    for evaluation_dir in sorted(evaluation_root.iterdir(), key=lambda item: item.name):
        if not evaluation_dir.name.startswith("eval_"):
            continue
        if selected is not None and evaluation_dir.name not in selected:
            continue
        seen_evaluations.add(evaluation_dir.name)
        scanned += 1
        if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
            if selected is not None:
                candidate, problem = _evaluation_problem(
                    evaluation_dir.name,
                    code="evaluation_path_invalid",
                    path=f"mea/evaluation_runs/{evaluation_dir.name}",
                    detail="requested evaluation is not a regular directory",
                )
                candidates.append(candidate)
                diagnostics.append(problem)
            continue
        parent_path = evaluation_dir / "manifest.json"
        try:
            parent_path = _repo_path(
                root,
                _relative(root, parent_path),
                code_prefix="evaluation_manifest",
                scope=evaluation_dir,
            )
            parent = _read_object(parent_path, code_prefix="evaluation_manifest")
        except _ArtifactProblem as exc:
            # A directory without a readable completed manifest is not a
            # completed evaluation and therefore is outside this collector.
            if selected is not None:
                candidate, problem = _evaluation_problem(
                    evaluation_dir.name,
                    code=exc.code,
                    path=exc.path,
                    detail=exc.detail,
                )
                candidates.append(candidate)
                diagnostics.append(problem)
            continue
        if (
            parent.get("evaluation_id") != evaluation_dir.name
            or parent.get("lifecycle_status") != "completed"
            or parent.get("status") != "completed"
        ):
            if selected is not None:
                candidate, problem = _evaluation_problem(
                    evaluation_dir.name,
                    code="evaluation_not_completed",
                    path=_relative(root, parent_path),
                    detail="requested evaluation is not a completed passing run",
                )
                candidates.append(candidate)
                diagnostics.append(problem)
            continue
        completed += 1
        if parent.get("task_name") != "click_bell":
            continue
        plan = parent.get("plan")
        requested = (
            list(plan.get("requested_template_ids") or [])
            if isinstance(plan, Mapping)
            else []
        )
        planned_conditions = [item for item in requested if item in CONDITION_CONTRACTS]
        evidence_value = parent.get("evidence_path") or "summary/evidence_bundle.json"
        try:
            evidence_path = _repo_path(
                root,
                str(Path("mea/evaluation_runs") / evaluation_dir.name / evidence_value)
                if not Path(str(evidence_value)).is_absolute()
                and not str(evidence_value).startswith("mea/evaluation_runs/")
                else evidence_value,
                code_prefix="evidence_bundle",
                scope=evaluation_dir,
            )
            evidence = _read_object(evidence_path, code_prefix="evidence_bundle")
        except _ArtifactProblem as exc:
            if planned_conditions:
                relevant += 1
                placeholders, missing = _placeholder_candidates(
                    evaluation_dir.name,
                    planned_conditions,
                    code=exc.code,
                    path=exc.path,
                    detail=exc.detail,
                )
                candidates.extend(placeholders)
                diagnostics.extend(missing)
            continue
        rounds = evidence.get("rounds")
        if not isinstance(rounds, list):
            rounds = []
        relevant_rounds = [
            item
            for item in rounds
            if isinstance(item, Mapping)
            and item.get("variant_id") in CONDITION_CONTRACTS
        ]
        if not relevant_rounds and planned_conditions:
            relevant += 1
            placeholders, missing = _placeholder_candidates(
                evaluation_dir.name,
                planned_conditions,
                code="scene_round_missing",
                path=_relative(root, evidence_path),
                detail="completed evaluation has no scene-shift round evidence",
            )
            candidates.extend(placeholders)
            diagnostics.extend(missing)
            continue
        if relevant_rounds:
            relevant += 1
        for round_value in relevant_rounds:
            candidate, round_diagnostics = _collect_round(
                root, evaluation_dir, parent_path, parent, round_value
            )
            candidates.append(candidate)
            diagnostics.extend(round_diagnostics)

    if selected is not None:
        missing_ids = sorted(selected - seen_evaluations)
        for evaluation_id in missing_ids:
            candidate, problem = _evaluation_problem(
                evaluation_id,
                code="evaluation_missing",
                path=f"mea/evaluation_runs/{evaluation_id}/manifest.json",
                detail="requested evaluation directory does not exist",
            )
            candidates.append(candidate)
            diagnostics.append(problem)

    candidates.sort(
        key=lambda item: (
            str(item.get("source_evaluation_id") or ""),
            str(item.get("round_id") or ""),
            str(item.get("child_run_id") or ""),
        )
    )
    diagnostics.sort(
        key=lambda item: (
            str(item.get("source_evaluation_id") or ""),
            str(item.get("round_id") or ""),
            str(item.get("code") or ""),
            str(item.get("artifact") or ""),
        )
    )
    counts: dict[str, dict[str, int]] = {}
    for condition in CONDITIONS:
        condition_candidates = [item for item in candidates if item["condition"] == condition]
        counts[condition] = {
            "candidate_count": len(condition_candidates),
            "ready_count": sum(item["status"] == "ready" for item in condition_candidates),
            "incomplete_count": sum(
                item["status"] != "ready" for item in condition_candidates
            ),
            "unique_seed_count": len(
                {
                    item["seed"]
                    for item in condition_candidates
                    if isinstance(item.get("seed"), int)
                }
            ),
        }
    duplicate_identities = [
        {"condition": condition, "seed": seed, "count": count}
        for (condition, seed), count in sorted(
            Counter(
                (item["condition"], item["seed"])
                for item in candidates
                if item["status"] == "ready" and isinstance(item.get("seed"), int)
            ).items()
        )
        if count > 1
    ]
    suite_draft, label_status, label_diagnostics = _label_suite_draft(
        candidates, labels, reviewer_id
    )
    inventory_sha256 = _canonical_sha256(candidates)
    return {
        "schema_version": 1,
        "protocol": COLLECTION_PROTOCOL,
        "mode": "offline_completed_artifact_collection",
        "evaluation_root": "mea/evaluation_runs",
        "evaluation_count_scanned": scanned,
        "completed_evaluation_count": completed,
        "relevant_evaluation_count": relevant,
        "candidate_count": len(candidates),
        "ready_candidate_count": sum(item["status"] == "ready" for item in candidates),
        "incomplete_candidate_count": sum(
            item["status"] != "ready" for item in candidates
        ),
        "counts_by_condition": counts,
        "duplicate_condition_seed_identities": duplicate_identities,
        "inventory_sha256": inventory_sha256,
        "candidates": candidates,
        "diagnostic_count": len(diagnostics),
        "diagnostics": diagnostics,
        "label_status": label_status,
        "label_diagnostics": label_diagnostics,
        "suite_draft": suite_draft,
        "suite_validated": False,
        "paper_table_eligible": False,
        "provider_called": False,
        "simulator_called": False,
        "act_called": False,
        "labels_inferred_from_vqa": False,
        "limitations": [
            "Collection inventories completed artifacts; it does not witness execution.",
            "A suite draft is unvalidated and requires external proxy labels for every phenomenon.",
            "Development-agent proxy labels and small samples are not paper-table evidence.",
        ],
    }


__all__ = [
    "COLLECTION_PROTOCOL",
    "SceneShiftCollectionError",
    "collect_scene_shift_candidates",
]
