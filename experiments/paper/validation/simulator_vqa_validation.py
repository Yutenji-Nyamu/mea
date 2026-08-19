"""Offline validation for real-simulator clean versus scene-clutter VQA.

This module never creates images, calls a model, or runs a simulator.  It
audits two completed Execution VQA artifacts, their source montages and query
files, the corresponding TaskGen simulator manifests, and explicit
``development_agent_proxy`` binary labels.
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


PROTOCOL = "real_simulator_clean_scene_clutter_v1"
CONDITIONS = ("clean", "scene_clutter")
CLUTTER_TEMPLATE = "robustness.scene_clutter.official_table"
CLUTTER_ASPECT = "robustness.scene_clutter"
CLEAN_TEMPLATE = "task_execution.official_baseline"
CLEAN_ASPECT = "task_execution.official_baseline"
CLUTTER_PHENOMENA = (
    "bell_visibly_pressed",
    "bell_target_selected_among_clutter",
)
CLEAN_PHENOMENA = ("bell_visibly_pressed",)

SUITE_KEYS = {"schema_version", "suite_id", "protocol", "reviewer", "cases"}
REVIEWER_KEYS = {"id", "kind"}
CASE_KEYS = {
    "id",
    "condition",
    "source_evaluation_id",
    "source_execution_vqa",
    "source_taskgen_manifest",
    "source_montage",
    "seed",
    "expected_query",
    "labels",
}
EXPECTED_QUERY_KEYS = {
    "task_name",
    "template_id",
    "sub_aspect",
    "phenomenon_ids",
}
LABEL_KEYS = {"phenomenon_id", "observed", "label_source", "reviewer_id"}


class SimulatorVQAValidationError(RuntimeError):
    """Raised when simulator/VQA provenance or proxy labels are invalid."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SimulatorVQAValidationError(f"{field} must be an object")
    return dict(value)


def _unique_strings(value: Any, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise SimulatorVQAValidationError(f"{field} must be a non-empty string list")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise SimulatorVQAValidationError(f"{field} must be unique")
    return normalized


def _safe_artifact(
    repo_root: Path,
    value: Any,
    *,
    field: str,
    require_nonempty: bool = True,
) -> Path:
    if not isinstance(value, str) or not value:
        raise SimulatorVQAValidationError(f"{field} must be a non-empty path")
    raw = Path(value).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    try:
        relative = candidate.relative_to(repo_root)
    except ValueError as exc:
        raise SimulatorVQAValidationError(f"{field} escapes repo root") from exc
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SimulatorVQAValidationError(f"{field} contains a symlink")
    if not candidate.is_file():
        raise SimulatorVQAValidationError(f"{field} is missing: {candidate}")
    if require_nonempty and candidate.stat().st_size <= 0:
        raise SimulatorVQAValidationError(f"{field} is empty: {candidate}")
    return candidate


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulatorVQAValidationError(f"invalid {field}: {path}: {exc}") from exc
    return _mapping(value, field=field)


def validate_simulator_vqa_suite(value: Any) -> dict[str, Any]:
    """Validate the pre-registered two-condition development suite."""

    if not isinstance(value, Mapping) or set(value) != SUITE_KEYS:
        raise SimulatorVQAValidationError(
            f"suite fields must be exactly {sorted(SUITE_KEYS)}"
        )
    if value.get("schema_version") != 1:
        raise SimulatorVQAValidationError("suite schema_version must be 1")
    suite_id = value.get("suite_id")
    if (
        not isinstance(suite_id, str)
        or re.fullmatch(r"simvqa_[A-Za-z0-9_]+", suite_id) is None
    ):
        raise SimulatorVQAValidationError("suite_id must begin with simvqa_")
    if value.get("protocol") != PROTOCOL:
        raise SimulatorVQAValidationError(f"suite protocol must be {PROTOCOL}")
    reviewer = _mapping(value.get("reviewer"), field="reviewer")
    if set(reviewer) != REVIEWER_KEYS:
        raise SimulatorVQAValidationError("reviewer must contain exactly id and kind")
    if not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip():
        raise SimulatorVQAValidationError("reviewer.id must be non-empty")
    if reviewer.get("kind") != "development_agent_proxy":
        raise SimulatorVQAValidationError(
            "this development suite requires development_agent_proxy labels"
        )

    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise SimulatorVQAValidationError(
            "suite must contain exactly one clean and one scene_clutter case"
        )
    seen_ids: set[str] = set()
    seen_conditions: set[str] = set()
    normalized_cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, field=f"cases[{index}]")
        if set(case) != CASE_KEYS:
            raise SimulatorVQAValidationError(
                f"cases[{index}] fields must be exactly {sorted(CASE_KEYS)}"
            )
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise SimulatorVQAValidationError("case id is missing or duplicate")
        seen_ids.add(case_id)
        condition = case.get("condition")
        if condition not in CONDITIONS or condition in seen_conditions:
            raise SimulatorVQAValidationError(
                "conditions must be unique clean and scene_clutter"
            )
        seen_conditions.add(condition)
        evaluation_id = case.get("source_evaluation_id")
        if (
            not isinstance(evaluation_id, str)
            or re.fullmatch(r"eval_[A-Za-z0-9_]+", evaluation_id) is None
        ):
            raise SimulatorVQAValidationError(
                f"{case_id}.source_evaluation_id must begin with eval_"
            )
        for field in (
            "source_execution_vqa",
            "source_taskgen_manifest",
            "source_montage",
        ):
            if not isinstance(case.get(field), str) or not case[field]:
                raise SimulatorVQAValidationError(f"{case_id}.{field} is invalid")
        seed = case.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SimulatorVQAValidationError(f"{case_id}.seed must be non-negative")

        expected_query = _mapping(
            case.get("expected_query"), field=f"{case_id}.expected_query"
        )
        if set(expected_query) != EXPECTED_QUERY_KEYS:
            raise SimulatorVQAValidationError(
                f"{case_id}.expected_query fields are invalid"
            )
        phenomenon_ids = _unique_strings(
            expected_query.get("phenomenon_ids"),
            field=f"{case_id}.expected_query.phenomenon_ids",
        )
        expected_contract = (
            {
                "task_name": "click_bell",
                "template_id": CLEAN_TEMPLATE,
                "sub_aspect": CLEAN_ASPECT,
                "phenomenon_ids": list(CLEAN_PHENOMENA),
            }
            if condition == "clean"
            else {
                "task_name": "click_bell",
                "template_id": CLUTTER_TEMPLATE,
                "sub_aspect": CLUTTER_ASPECT,
                "phenomenon_ids": list(CLUTTER_PHENOMENA),
            }
        )
        normalized_query = {
            **expected_query,
            "phenomenon_ids": phenomenon_ids,
        }
        if normalized_query != expected_contract:
            raise SimulatorVQAValidationError(
                f"{case_id}.expected_query does not match {condition} contract"
            )

        labels = case.get("labels")
        if not isinstance(labels, list) or len(labels) != len(phenomenon_ids):
            raise SimulatorVQAValidationError(
                f"{case_id}.labels must cover every expected phenomenon"
            )
        normalized_labels: list[dict[str, Any]] = []
        seen_phenomena: set[str] = set()
        for label_index, raw_label in enumerate(labels):
            label = _mapping(raw_label, field=f"{case_id}.labels[{label_index}]")
            if set(label) != LABEL_KEYS:
                raise SimulatorVQAValidationError(
                    f"{case_id}.labels[{label_index}] fields are invalid"
                )
            phenomenon_id = label.get("phenomenon_id")
            if phenomenon_id not in phenomenon_ids or phenomenon_id in seen_phenomena:
                raise SimulatorVQAValidationError(
                    f"{case_id}.labels contain missing, unknown, or duplicate phenomena"
                )
            seen_phenomena.add(phenomenon_id)
            if not isinstance(label.get("observed"), bool):
                raise SimulatorVQAValidationError(
                    f"{case_id}.{phenomenon_id} proxy label must be boolean"
                )
            if label.get("label_source") != "development_agent_proxy":
                raise SimulatorVQAValidationError(
                    f"{case_id}.{phenomenon_id} label source must be development_agent_proxy"
                )
            if label.get("reviewer_id") != reviewer["id"]:
                raise SimulatorVQAValidationError(
                    f"{case_id}.{phenomenon_id} reviewer does not match suite"
                )
            normalized_labels.append(dict(label))
        if seen_phenomena != set(phenomenon_ids):
            raise SimulatorVQAValidationError(
                f"{case_id}.labels do not exactly cover expected phenomena"
            )
        normalized_cases.append(
            {
                **case,
                "expected_query": normalized_query,
                "labels": normalized_labels,
            }
        )
    if seen_conditions != set(CONDITIONS):
        raise SimulatorVQAValidationError(
            "suite requires one clean and one scene_clutter case"
        )
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "protocol": PROTOCOL,
        "reviewer": {
            "id": reviewer["id"].strip(),
            "kind": "development_agent_proxy",
        },
        "cases": normalized_cases,
    }


def _relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _tracked_bell(scene: Mapping[str, Any]) -> dict[str, Any]:
    tracked = scene.get("tracked_actors")
    if not isinstance(tracked, list):
        raise SimulatorVQAValidationError(
            "TaskGen scene_validation.tracked_actors must be a list"
        )
    matches = [
        dict(item)
        for item in tracked
        if isinstance(item, Mapping) and item.get("id") == "bell"
    ]
    if len(matches) != 1:
        raise SimulatorVQAValidationError(
            "TaskGen scene_validation must contain exactly one tracked bell"
        )
    return matches[0]


def _validate_taskgen_condition(
    repo_root: Path,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    path = _safe_artifact(
        repo_root,
        case["source_taskgen_manifest"],
        field=f"{case['id']}.source_taskgen_manifest",
    )
    manifest = _read_json(path, field="TaskGen manifest")
    run_id = manifest.get("run_id")
    expected_relative = Path("mea/generated_tasks") / str(run_id) / "manifest.json"
    if _relative(repo_root, path) != expected_relative.as_posix():
        raise SimulatorVQAValidationError(
            f"{case['id']} TaskGen manifest path does not match run_id"
        )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("task_name") != "click_bell"
        or manifest.get("failure") is not None
    ):
        raise SimulatorVQAValidationError(
            f"{case['id']} TaskGen manifest is not a completed click_bell run"
        )
    act = _mapping(manifest.get("act_evaluation"), field="manifest.act_evaluation")
    if (
        act.get("passed") is not True
        or act.get("task_name") != "click_bell"
        or act.get("task_config") != "demo_clean"
        or act.get("checkpoint_setting") != "demo_clean"
        or act.get("num_episodes") != 1
        or act.get("actual_seeds") != [case["seed"]]
    ):
        raise SimulatorVQAValidationError(
            f"{case['id']} ACT condition must be one demo_clean episode at the declared seed"
        )
    scene = _mapping(
        manifest.get("scene_validation"), field="manifest.scene_validation"
    )
    if (
        scene.get("setup_success") is not True
        or scene.get("render_success") is not True
        or scene.get("seed") != case["seed"]
        or not _mapping(scene.get("rule_check"), field="scene.rule_check").get("passed")
    ):
        raise SimulatorVQAValidationError(
            f"{case['id']} simulator setup/render/rule evidence failed"
        )
    randomization = _mapping(
        scene.get("domain_randomization"), field="scene.domain_randomization"
    )
    if randomization.get("authority") != "simulator_task_info:cluttered_table_info":
        raise SimulatorVQAValidationError(
            f"{case['id']} clutter provenance is not simulator-authoritative"
        )
    clutter_objects = randomization.get("cluttered_objects")
    if not isinstance(clutter_objects, list):
        raise SimulatorVQAValidationError(
            f"{case['id']} cluttered_objects must be a list"
        )
    clutter_count = randomization.get("cluttered_object_count")
    if (
        isinstance(clutter_count, bool)
        or not isinstance(clutter_count, int)
        or clutter_count != len(clutter_objects)
    ):
        raise SimulatorVQAValidationError(
            f"{case['id']} clutter count does not match simulator records"
        )

    condition = case["condition"]
    if condition == "clean":
        condition_passed = bool(
            manifest.get("mode") == "official"
            and manifest.get("generation_kind") == "official_passthrough"
            and manifest.get("task_module") == "envs.click_bell"
            and randomization.get("cluttered_table") is False
            and randomization.get("clean_background_rate") == 1.0
            and clutter_count == 0
        )
    else:
        position_samples = _mapping(
            manifest.get("position_samples"), field="manifest.position_samples"
        )
        metrics = _mapping(
            position_samples.get("metrics"), field="position_samples.metrics"
        )
        contract = position_samples.get("variant_contract")
        condition_passed = bool(
            manifest.get("mode") == "reuse"
            and manifest.get("generation_kind") == "bounded_variant_overlay"
            and manifest.get("variant_id") == CLUTTER_TEMPLATE
            and manifest.get("capability_id") == CLUTTER_ASPECT
            and randomization.get("cluttered_table") is True
            and randomization.get("clean_background_rate") == 0.0
            and clutter_count >= 1
            and position_samples.get("passed") is True
            and metrics.get("expected_clutter") is True
            and metrics.get("all_clutter_matched") is True
            and contract
            == {
                "domain_randomization": {
                    "cluttered_table": True,
                    "clean_background_rate": 0.0,
                }
            }
        )
    if not condition_passed:
        raise SimulatorVQAValidationError(
            f"{case['id']} TaskGen manifest does not prove {condition} condition"
        )

    bell = _tracked_bell(scene)
    bell_position = bell.get("position")
    bell_quaternion = bell.get("quaternion")
    if (
        not isinstance(bell_position, list)
        or len(bell_position) != 3
        or not isinstance(bell_quaternion, list)
        or len(bell_quaternion) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in [*bell_position, *bell_quaternion]
        )
    ):
        raise SimulatorVQAValidationError(f"{case['id']} tracked bell pose is invalid")
    bell_id = _mapping(scene.get("task_attributes"), field="scene.task_attributes").get(
        "bell_id"
    )
    if isinstance(bell_id, bool) or not isinstance(bell_id, int):
        raise SimulatorVQAValidationError(f"{case['id']} bell_id is invalid")

    associations = act.get("video_associations")
    if not isinstance(associations, list) or len(associations) != 1:
        raise SimulatorVQAValidationError(
            f"{case['id']} requires exactly one ACT video association"
        )
    association = _mapping(associations[0], field="ACT video association")
    episode_dir = association.get("episode_dir")
    video = _safe_artifact(
        repo_root,
        association.get("video"),
        field=f"{case['id']}.source_video",
    )
    if not isinstance(episode_dir, str) or not episode_dir:
        raise SimulatorVQAValidationError(f"{case['id']} ACT episode_dir is invalid")
    episode_path = _safe_artifact(
        repo_root,
        f"{episode_dir}/episode.json",
        field=f"{case['id']}.source_episode",
    )
    episode = _read_json(episode_path, field="ACT episode")
    telemetry_profile_sha256 = episode.get("telemetry_profile_sha256")
    if (
        episode.get("task_name") != "click_bell"
        or str(episode.get("policy_name", "")).casefold() != "act"
        or episode.get("seed") != case["seed"]
        or episode.get("checkpoint_setting") != "demo_clean"
        or episode.get("telemetry_profile_id") != "balanced_v1"
        or not isinstance(telemetry_profile_sha256, str)
        or not telemetry_profile_sha256
        or not isinstance(episode.get("success"), bool)
    ):
        raise SimulatorVQAValidationError(
            f"{case['id']} ACT episode identity/configuration is invalid"
        )

    parent_path = _safe_artifact(
        repo_root,
        f"mea/evaluation_runs/{case['source_evaluation_id']}/manifest.json",
        field=f"{case['id']}.source_evaluation_manifest",
    )
    parent = _read_json(parent_path, field="parent evaluation manifest")
    base_commit = parent.get("base_commit")
    if (
        parent.get("evaluation_id") != case["source_evaluation_id"]
        or parent.get("status") != "completed"
        or parent.get("lifecycle_status") != "completed"
        or parent.get("active_child_run_id") != run_id
        or parent.get("task_name") != "click_bell"
        or parent.get("telemetry_profile") != "balanced_v1"
        or not isinstance(base_commit, str)
        or not base_commit
    ):
        raise SimulatorVQAValidationError(
            f"{case['id']} parent evaluation does not bind the declared child run"
        )
    return {
        "path": _relative(repo_root, path),
        "sha256": _file_sha256(path),
        "run_id": run_id,
        "episode_dir": episode_dir,
        "video": _relative(repo_root, video),
        "video_sha256": _file_sha256(video),
        "episode": _relative(repo_root, episode_path),
        "episode_sha256": _file_sha256(episode_path),
        "policy_success": episode["success"],
        "checkpoint_setting": episode["checkpoint_setting"],
        "telemetry_profile_id": episode["telemetry_profile_id"],
        "telemetry_profile_sha256": telemetry_profile_sha256,
        "parent_evaluation": {
            "path": _relative(repo_root, parent_path),
            "sha256": _file_sha256(parent_path),
            "base_commit": base_commit,
            "telemetry_profile": parent["telemetry_profile"],
            "active_child_run_id": parent["active_child_run_id"],
        },
        "bell_position": [float(value) for value in bell_position],
        "bell_quaternion": [float(value) for value in bell_quaternion],
        "bell_id": bell_id,
        "clutter_count": clutter_count,
        "clutter_authority": randomization["authority"],
    }


def _validate_execution_vqa_case(
    repo_root: Path,
    case: Mapping[str, Any],
    condition_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _safe_artifact(
        repo_root,
        case["source_execution_vqa"],
        field=f"{case['id']}.source_execution_vqa",
    )
    relative_path = Path(_relative(repo_root, path))
    expected_prefix = Path("mea/evaluation_runs") / case["source_evaluation_id"]
    if relative_path.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
        raise SimulatorVQAValidationError(
            f"{case['id']} Execution VQA path does not match evaluation_id"
        )
    artifact = _read_json(path, field="Execution VQA artifact")
    if artifact.get("schema_version") != 1 or artifact.get("status") != "passed":
        raise SimulatorVQAValidationError(
            f"{case['id']} Execution VQA artifact is not passed schema v1"
        )
    try:
        query = validate_execution_vqa_query(artifact.get("query"))
    except ExecutionVQAQueryError as exc:
        raise SimulatorVQAValidationError(
            f"{case['id']} Execution VQA query is invalid: {exc}"
        ) from exc
    expected_query = case["expected_query"]
    actual_query_contract = {
        "task_name": query["task_name"],
        "template_id": query["template_id"],
        "sub_aspect": query["sub_aspect"],
        "phenomenon_ids": query["phenomenon_ids"],
    }
    if actual_query_contract != expected_query:
        raise SimulatorVQAValidationError(
            f"{case['id']} embedded query does not match expected condition"
        )
    artifacts = _mapping(artifact.get("artifacts"), field="artifact.artifacts")
    artifact_montage = _safe_artifact(
        repo_root,
        artifacts.get("montage"),
        field=f"{case['id']}.artifact.montage",
    )
    suite_montage = _safe_artifact(
        repo_root,
        case["source_montage"],
        field=f"{case['id']}.source_montage",
    )
    selection = _mapping(artifact.get("selection"), field="artifact.selection")
    selection_video = _safe_artifact(
        repo_root,
        selection.get("video_path"),
        field=f"{case['id']}.selection.video_path",
    )
    condition_video = _safe_artifact(
        repo_root,
        condition_evidence["video"],
        field=f"{case['id']}.condition.video",
    )
    if selection_video != condition_video:
        raise SimulatorVQAValidationError(
            f"{case['id']} VQA selection video is not the validated ACT video"
        )
    selection_montage = _safe_artifact(
        repo_root,
        selection.get("montage_path"),
        field=f"{case['id']}.selection.montage_path",
    )
    if not (artifact_montage == suite_montage == selection_montage):
        raise SimulatorVQAValidationError(
            f"{case['id']} source montage paths do not identify one file"
        )
    query_path = _safe_artifact(
        repo_root,
        artifacts.get("query"),
        field=f"{case['id']}.artifact.query",
    )
    if _read_json(query_path, field="Execution VQA query artifact") != query:
        raise SimulatorVQAValidationError(
            f"{case['id']} embedded and source query artifacts differ"
        )
    representative_episode = artifact.get("representative_episode")
    if representative_episode != condition_evidence["episode_dir"]:
        raise SimulatorVQAValidationError(
            f"{case['id']} VQA episode is not the validated simulator condition episode"
        )

    selected_frames = selection.get("selected_frames")
    if not isinstance(selected_frames, list) or not selected_frames:
        raise SimulatorVQAValidationError(f"{case['id']} selection must contain frames")
    allowed_frame_ids = []
    for frame in selected_frames:
        frame_value = _mapping(frame, field="selected frame")
        frame_id = frame_value.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise SimulatorVQAValidationError(
                f"{case['id']} selected frame id is invalid"
            )
        allowed_frame_ids.append(frame_id)
    observation = _mapping(artifact.get("observation"), field="artifact.observation")
    response_keys = {
        "phenomena",
        "confidence",
        "frame_ids",
        "numeric_consistency",
        "conflicts",
    }
    try:
        normalized_observation = validate_execution_vqa_response(
            {key: observation.get(key) for key in response_keys},
            allowed_frame_ids=allowed_frame_ids,
            expected_phenomenon_ids=query["phenomenon_ids"],
        )
    except ExecutionVQAError as exc:
        raise SimulatorVQAValidationError(
            f"{case['id']} Execution VQA observation is invalid: {exc}"
        ) from exc
    predictions = {item["id"]: item for item in normalized_observation["phenomena"]}
    rows: list[dict[str, Any]] = []
    for label in case["labels"]:
        prediction = predictions[label["phenomenon_id"]]
        rows.append(
            {
                "case_id": case["id"],
                "condition": case["condition"],
                "phenomenon_id": label["phenomenon_id"],
                "label_source": "development_agent_proxy",
                "proxy_observed": label["observed"],
                "predicted_observed": prediction["observed"],
                "confidence": float(prediction["confidence"]),
                "covered": isinstance(prediction["observed"], bool),
                "correct": prediction["observed"] == label["observed"],
            }
        )
    return (
        {
            "path": _relative(repo_root, path),
            "sha256": _file_sha256(path),
            "query_path": _relative(repo_root, query_path),
            "query_sha256": _canonical_sha256(query),
            "montage_path": _relative(repo_root, suite_montage),
            "montage_sha256": _file_sha256(suite_montage),
            "representative_episode": representative_episode,
        },
        rows,
    )


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["correct"] for row in rows)
    total = len(rows)
    return {
        "value": correct / total if total else None,
        "correct": correct,
        "total": total,
    }


def summarize_simulator_vqa_suite(
    repo_root: str | Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit provenance and aggregate proxy accuracy without new model calls."""

    root = Path(repo_root).expanduser().resolve()
    normalized = validate_simulator_vqa_suite(suite)
    evidence_by_condition: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for case in normalized["cases"]:
        condition_evidence = _validate_taskgen_condition(root, case)
        vqa_evidence, case_rows = _validate_execution_vqa_case(
            root, case, condition_evidence
        )
        evidence_by_condition[case["condition"]] = condition_evidence
        rows.extend(case_rows)
        cases.append(
            {
                "id": case["id"],
                "condition": case["condition"],
                "seed": case["seed"],
                "source_evaluation_id": case["source_evaluation_id"],
                "taskgen": condition_evidence,
                "execution_vqa": vqa_evidence,
                "labels": case_rows,
            }
        )

    clean = evidence_by_condition["clean"]
    clutter = evidence_by_condition["scene_clutter"]
    pose_equal = clean["bell_position"] == clutter["bell_position"]
    quaternion_equal = clean["bell_quaternion"] == clutter["bell_quaternion"]
    target_identity = {
        "same_seed": normalized["cases"][0]["seed"] == normalized["cases"][1]["seed"],
        "same_bell_position": pose_equal,
        "same_bell_quaternion": quaternion_equal,
        "same_bell_id": clean["bell_id"] == clutter["bell_id"],
    }
    target_identity["passed"] = all(target_identity.values())
    if not target_identity["passed"]:
        raise SimulatorVQAValidationError(
            "clean/clutter target identity differs; pair is not controlled"
        )
    protocol_identity = {
        "same_base_commit": (
            clean["parent_evaluation"]["base_commit"]
            == clutter["parent_evaluation"]["base_commit"]
        ),
        "same_checkpoint_setting": (
            clean["checkpoint_setting"] == clutter["checkpoint_setting"]
        ),
        "same_telemetry_profile_id": (
            clean["telemetry_profile_id"] == clutter["telemetry_profile_id"]
        ),
        "same_telemetry_profile_sha256": (
            clean["telemetry_profile_sha256"]
            == clutter["telemetry_profile_sha256"]
        ),
    }
    protocol_identity["passed"] = all(protocol_identity.values())
    if not protocol_identity["passed"]:
        raise SimulatorVQAValidationError(
            "clean/clutter execution protocol differs; pair is not controlled"
        )

    by_condition = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "evaluation_count": 1,
            "label_count": len(condition_rows),
            "accuracy": _accuracy(condition_rows),
            "coverage": {
                "value": sum(row["covered"] for row in condition_rows)
                / len(condition_rows),
                "covered": sum(row["covered"] for row in condition_rows),
                "total": len(condition_rows),
            },
            "auroc": None,
            "auroc_unavailable_reason": "n_equals_1_for_condition",
        }
    label_sources = Counter(row["label_source"] for row in rows)
    return {
        "schema_version": 1,
        "suite_id": normalized["suite_id"],
        "protocol": PROTOCOL,
        "mode": "offline_completed_artifact_audit",
        "conditions": list(CONDITIONS),
        "evaluations_per_condition": 1,
        "target_identity": target_identity,
        "protocol_identity": protocol_identity,
        "accuracy": _accuracy(rows),
        "by_condition": by_condition,
        "auroc": None,
        "auroc_unavailable_reason": "n_equals_1_per_condition",
        "label_source_counts": dict(sorted(label_sources.items())),
        "reviewer": deepcopy(normalized["reviewer"]),
        "human_reviewer_count": 0,
        "paper_table_eligible": False,
        "paper_table_ineligible_reason": (
            "development_agent_proxy_labels_and_n_equals_1_per_condition"
        ),
        "provider_called": False,
        "simulator_called": False,
        "offline_aggregator_provider_called": False,
        "offline_aggregator_simulator_called": False,
        "image_proxy_used": False,
        "source_scope": "real_simulator_completed_rollouts",
        "cases": cases,
        "rows": rows,
        "limitations": [
            "The labels are development-agent proxies, not human gold.",
            "There is one completed rollout per condition, so AUROC is undefined.",
            "This is a functional Tables 7-8 smoke, not paper-level evidence.",
            "Artifact hashes are computed during this post-hoc audit, not frozen in a preregistered evidence manifest.",
            "Checkpoint equality is the recorded setting, not a checkpoint-file content hash.",
            "provider_called/simulator_called describe this offline aggregator, not the historical source evaluations.",
        ],
    }
