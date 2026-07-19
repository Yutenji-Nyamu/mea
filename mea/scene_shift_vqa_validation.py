"""Offline audit for simulator-native texture/light Execution VQA cases.

The protocol is deliberately evidence-only: it never starts RoboTwin, ACT, or
the VQA provider.  A suite is accepted only when every case already has a
completed TaskGen run, ACT episode/video, and Execution VQA artifact, and when
all source files are bound by caller-supplied SHA-256 hashes.
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


PROTOCOL = "real_simulator_texture_lighting_binary_v1"
CONDITIONS = ("scene_background_texture.unseen", "scene_lighting.static_random")
CONDITION_CONTRACTS: dict[str, dict[str, Any]] = {
    "scene_background_texture.unseen": {
        "capability_id": "scene_background_texture",
        "controlled_axis": "scene_background_texture",
        "template_id": "scene_background_texture.unseen",
        "sub_aspect": "scene_background_texture",
        "primary_visibility_phenomenon_id": (
            "bell_visible_with_unseen_background_texture"
        ),
        "phenomenon_ids": [
            "bell_visibly_pressed",
            "bell_visible_with_unseen_background_texture",
        ],
        "changes": {
            "domain_randomization": {
                "random_background": True,
                "clean_background_rate": 0.0,
            }
        },
    },
    "scene_lighting.static_random": {
        "capability_id": "scene_lighting",
        "controlled_axis": "scene_lighting",
        "template_id": "scene_lighting.static_random",
        "sub_aspect": "scene_lighting",
        "primary_visibility_phenomenon_id": "bell_visible_under_random_lighting",
        "phenomenon_ids": [
            "bell_visibly_pressed",
            "bell_visible_under_random_lighting",
        ],
        "changes": {
            "domain_randomization": {
                "random_light": True,
                "crazy_random_light_rate": 0.0,
            }
        },
    },
}

SUITE_KEYS = {"schema_version", "suite_id", "protocol", "reviewer", "cases"}
REVIEWER_KEYS = {"id", "kind"}
CASE_KEYS = {
    "id",
    "condition",
    "seed",
    "source_evaluation_id",
    "sources",
    "expected_query",
    "labels",
}
SOURCE_KEYS = {
    "taskgen_manifest",
    "evaluation_manifest",
    "episode",
    "video",
    "execution_vqa",
    "query",
    "montage",
}
REF_KEYS = {"path", "sha256"}
EXPECTED_QUERY_KEYS = {
    "task_name",
    "template_id",
    "sub_aspect",
    "primary_visibility_phenomenon_id",
    "phenomenon_ids",
}
LABEL_KEYS = {"phenomenon_id", "observed", "label_source", "reviewer_id"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SceneShiftVQAValidationError(RuntimeError):
    """Raised when a scene-shift suite or its evidence is not trustworthy."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneShiftVQAValidationError(f"{field} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        return _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            field=field,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneShiftVQAValidationError(
            f"invalid {field}: {path}: {exc}"
        ) from exc


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _resolve_hashed_ref(
    root: Path,
    value: Any,
    *,
    field: str,
) -> tuple[Path, dict[str, str]]:
    reference = _mapping(value, field=field)
    if set(reference) != REF_KEYS:
        raise SceneShiftVQAValidationError(
            f"{field} must contain exactly path and sha256"
        )
    raw_path = reference.get("path")
    expected_hash = reference.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\\" in raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
    ):
        raise SceneShiftVQAValidationError(
            f"{field}.path must be a repo-relative POSIX path"
        )
    if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
        raise SceneShiftVQAValidationError(
            f"{field}.sha256 must be a lowercase SHA-256 digest"
        )
    lexical = root / raw_path
    current = root
    for part in Path(raw_path).parts:
        current = current / part
        if current.is_symlink():
            raise SceneShiftVQAValidationError(f"{field}.path contains a symlink")
    candidate = lexical.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SceneShiftVQAValidationError(f"{field}.path escapes repo root") from exc
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise SceneShiftVQAValidationError(f"{field}.path is missing or empty")
    actual_hash = _sha256(candidate)
    if actual_hash != expected_hash:
        raise SceneShiftVQAValidationError(f"{field}.sha256 does not match content")
    return candidate, {"path": _relative(root, candidate), "sha256": actual_hash}


def _resolve_embedded_artifact(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SceneShiftVQAValidationError(f"{field} must be a non-empty path")
    raw = Path(value).expanduser()
    lexical = raw if raw.is_absolute() else root / raw
    try:
        lexical_relative = lexical.absolute().relative_to(root)
    except ValueError as exc:
        raise SceneShiftVQAValidationError(f"{field} escapes repo root") from exc
    current = root
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise SceneShiftVQAValidationError(f"{field} contains a symlink")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SceneShiftVQAValidationError(f"{field} escapes repo root") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SceneShiftVQAValidationError(f"{field} is missing or empty")
    return resolved


def _strings(value: Any, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise SceneShiftVQAValidationError(f"{field} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise SceneShiftVQAValidationError(f"{field} must not contain duplicates")
    return list(value)


def validate_scene_shift_vqa_suite(value: Any) -> dict[str, Any]:
    """Validate the preregistered suite contract without opening artifacts."""

    suite = _mapping(value, field="suite")
    if set(suite) != SUITE_KEYS:
        raise SceneShiftVQAValidationError(
            f"suite fields must be exactly {sorted(SUITE_KEYS)}"
        )
    if suite.get("schema_version") != 1:
        raise SceneShiftVQAValidationError("suite schema_version must be 1")
    suite_id = suite.get("suite_id")
    if (
        not isinstance(suite_id, str)
        or re.fullmatch(r"sceneshiftvqa_[A-Za-z0-9_]+", suite_id) is None
    ):
        raise SceneShiftVQAValidationError("suite_id must begin with sceneshiftvqa_")
    if suite.get("protocol") != PROTOCOL:
        raise SceneShiftVQAValidationError(f"suite protocol must be {PROTOCOL}")

    reviewer = _mapping(suite.get("reviewer"), field="reviewer")
    if set(reviewer) != REVIEWER_KEYS:
        raise SceneShiftVQAValidationError("reviewer must contain exactly id and kind")
    if not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip():
        raise SceneShiftVQAValidationError("reviewer.id must be non-empty")
    if reviewer.get("kind") != "development_agent_proxy":
        raise SceneShiftVQAValidationError(
            "scene-shift smoke labels must be development_agent_proxy"
        )

    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < 4:
        raise SceneShiftVQAValidationError(
            "suite requires at least two completed cases per condition"
        )
    ids: set[str] = set()
    condition_seeds: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    primary_labels: dict[str, set[bool]] = {condition: set() for condition in CONDITIONS}
    normalized_cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, field=f"cases[{index}]")
        if set(case) != CASE_KEYS:
            raise SceneShiftVQAValidationError(
                f"cases[{index}] fields must be exactly {sorted(CASE_KEYS)}"
            )
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise SceneShiftVQAValidationError("case id is missing or duplicate")
        ids.add(case_id)
        condition = case.get("condition")
        if condition not in CONDITION_CONTRACTS:
            raise SceneShiftVQAValidationError(
                f"{case_id}.condition must be a supported click_bell scene shift"
            )
        seed = case.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SceneShiftVQAValidationError(f"{case_id}.seed must be non-negative")
        if (condition, seed) in condition_seeds:
            raise SceneShiftVQAValidationError(
                f"{case_id} duplicates a condition/seed identity"
            )
        condition_seeds.add((condition, seed))
        counts[condition] += 1
        evaluation_id = case.get("source_evaluation_id")
        if (
            not isinstance(evaluation_id, str)
            or re.fullmatch(r"eval_[A-Za-z0-9_]+", evaluation_id) is None
        ):
            raise SceneShiftVQAValidationError(
                f"{case_id}.source_evaluation_id must begin with eval_"
            )
        sources = _mapping(case.get("sources"), field=f"{case_id}.sources")
        if set(sources) != SOURCE_KEYS:
            raise SceneShiftVQAValidationError(
                f"{case_id}.sources must contain every required hashed artifact"
            )
        for source_name, reference in sources.items():
            ref = _mapping(reference, field=f"{case_id}.sources.{source_name}")
            if set(ref) != REF_KEYS:
                raise SceneShiftVQAValidationError(
                    f"{case_id}.sources.{source_name} must contain path and sha256"
                )
            if not isinstance(ref.get("path"), str) or not ref["path"]:
                raise SceneShiftVQAValidationError(
                    f"{case_id}.sources.{source_name}.path is invalid"
                )
            if (
                not isinstance(ref.get("sha256"), str)
                or SHA256_RE.fullmatch(ref["sha256"]) is None
            ):
                raise SceneShiftVQAValidationError(
                    f"{case_id}.sources.{source_name}.sha256 is required"
                )

        expected = _mapping(
            case.get("expected_query"), field=f"{case_id}.expected_query"
        )
        if set(expected) != EXPECTED_QUERY_KEYS:
            raise SceneShiftVQAValidationError(
                f"{case_id}.expected_query fields are invalid"
            )
        phenomena = _strings(
            expected.get("phenomenon_ids"),
            field=f"{case_id}.expected_query.phenomenon_ids",
        )
        contract = CONDITION_CONTRACTS[condition]
        expected_contract = {
            "task_name": "click_bell",
            "template_id": contract["template_id"],
            "sub_aspect": contract["sub_aspect"],
            "primary_visibility_phenomenon_id": contract[
                "primary_visibility_phenomenon_id"
            ],
            "phenomenon_ids": contract["phenomenon_ids"],
        }
        normalized_expected = {**expected, "phenomenon_ids": phenomena}
        if normalized_expected != expected_contract:
            raise SceneShiftVQAValidationError(
                f"{case_id}.expected_query does not match its condition contract"
            )

        labels = case.get("labels")
        if not isinstance(labels, list) or len(labels) != len(phenomena):
            raise SceneShiftVQAValidationError(
                f"{case_id}.labels must cover every expected phenomenon"
            )
        seen_phenomena: set[str] = set()
        normalized_labels: list[dict[str, Any]] = []
        for label_index, raw_label in enumerate(labels):
            label = _mapping(
                raw_label, field=f"{case_id}.labels[{label_index}]"
            )
            if set(label) != LABEL_KEYS:
                raise SceneShiftVQAValidationError(
                    f"{case_id}.labels[{label_index}] fields are invalid"
                )
            phenomenon_id = label.get("phenomenon_id")
            if phenomenon_id not in phenomena or phenomenon_id in seen_phenomena:
                raise SceneShiftVQAValidationError(
                    f"{case_id}.labels contain unknown or duplicate phenomena"
                )
            seen_phenomena.add(phenomenon_id)
            if not isinstance(label.get("observed"), bool):
                raise SceneShiftVQAValidationError(
                    f"{case_id}.{phenomenon_id} proxy label must be boolean"
                )
            if label.get("label_source") != "development_agent_proxy":
                raise SceneShiftVQAValidationError(
                    f"{case_id}.{phenomenon_id} label source must be development_agent_proxy"
                )
            if label.get("reviewer_id") != reviewer["id"]:
                raise SceneShiftVQAValidationError(
                    f"{case_id}.{phenomenon_id} reviewer does not match suite"
                )
            if phenomenon_id == contract["primary_visibility_phenomenon_id"]:
                primary_labels[condition].add(label["observed"])
            normalized_labels.append(dict(label))
        if seen_phenomena != set(phenomena):
            raise SceneShiftVQAValidationError(
                f"{case_id}.labels do not exactly cover expected phenomena"
            )
        normalized_cases.append(
            {
                **case,
                "sources": deepcopy(sources),
                "expected_query": normalized_expected,
                "labels": normalized_labels,
            }
        )
    for condition in CONDITIONS:
        if counts[condition] < 2:
            raise SceneShiftVQAValidationError(
                f"{condition} requires at least two completed cases"
            )
        if primary_labels[condition] != {False, True}:
            raise SceneShiftVQAValidationError(
                f"{condition} primary visibility labels require both true and false"
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


def _validate_taskgen_case(
    root: Path,
    case: Mapping[str, Any],
    resolved: Mapping[str, tuple[Path, dict[str, str]]],
) -> dict[str, Any]:
    case_id = str(case["id"])
    condition = str(case["condition"])
    contract = CONDITION_CONTRACTS[condition]
    manifest_path = resolved["taskgen_manifest"][0]
    manifest = _read_json(manifest_path, field=f"{case_id} TaskGen manifest")
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or _relative(root, manifest_path)
        != f"mea/generated_tasks/{run_id}/manifest.json"
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} TaskGen manifest path does not match run_id"
        )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("failure") is not None
        or manifest.get("task_name") != "click_bell"
        or manifest.get("task_module") != "mea.tasks.click_bell"
        or manifest.get("mode") != "reuse"
        or manifest.get("generation_kind") != "bounded_variant_overlay"
        or manifest.get("variant_id") != contract["template_id"]
        or manifest.get("capability_id") != contract["capability_id"]
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} is not a completed bounded {condition} TaskGen run"
        )
    scene = _mapping(
        manifest.get("scene_validation"), field=f"{case_id}.scene_validation"
    )
    if (
        scene.get("seed") != case["seed"]
        or scene.get("eval_mode") is not True
        or scene.get("setup_success") is not True
        or scene.get("render_success") is not True
        or _mapping(scene.get("rule_check"), field=f"{case_id}.rule_check").get(
            "passed"
        )
        is not True
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} lacks a passing simulator scene at its declared seed"
        )
    randomization = _mapping(
        scene.get("domain_randomization"),
        field=f"{case_id}.scene_validation.domain_randomization",
    )
    if condition == "scene_background_texture.unseen":
        scene_condition_passed = bool(
            randomization.get("random_background") is True
            and randomization.get("clean_background_rate") == 0.0
            and randomization.get("texture_split") == "unseen"
            and isinstance(randomization.get("wall_texture"), str)
            and randomization["wall_texture"].startswith("unseen/")
            and isinstance(randomization.get("table_texture"), str)
            and randomization["table_texture"].startswith("unseen/")
            and randomization.get("background_authority")
            == "simulator_task_info:texture_info"
        )
        authority = randomization.get("background_authority")
        simulator_state = {
            "texture_split": randomization.get("texture_split"),
            "wall_texture": randomization.get("wall_texture"),
            "table_texture": randomization.get("table_texture"),
            "authority": authority,
        }
    else:
        lighting_authority = (
            "simulator_task_attributes:random_light,crazy_random_light_rate,"
            "crazy_random_light;simulator_light_components:get_color"
        )
        scene_condition_passed = bool(
            randomization.get("random_light") is True
            and randomization.get("crazy_random_light_rate") == 0.0
            and randomization.get("crazy_random_light") is False
            and _valid_colors(
                randomization.get("direction_light_colors"),
                randomization.get("direction_light_count"),
            )
            and _valid_colors(
                randomization.get("point_light_colors"),
                randomization.get("point_light_count"),
            )
            and randomization.get("lighting_authority") == lighting_authority
        )
        authority = randomization.get("lighting_authority")
        simulator_state = {
            "crazy_random_light_rate": randomization.get(
                "crazy_random_light_rate"
            ),
            "crazy_random_light": randomization.get("crazy_random_light"),
            "direction_light_colors": randomization.get("direction_light_colors"),
            "point_light_colors": randomization.get("point_light_colors"),
            "authority": authority,
        }
    if not scene_condition_passed:
        raise SceneShiftVQAValidationError(
            f"{case_id} lacks simulator-authoritative {condition} state"
        )
    samples = _mapping(
        manifest.get("position_samples"), field=f"{case_id}.position_samples"
    )
    if (
        samples.get("passed") is not True
        or samples.get("controlled_axis") != contract["controlled_axis"]
        or samples.get("variant_contract") != contract["changes"]
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} TaskGen sample contract does not bind the scene shift"
        )

    act = _mapping(manifest.get("act_evaluation"), field=f"{case_id}.act_evaluation")
    actual_seeds = act.get("actual_seeds")
    if (
        act.get("passed") is not True
        or act.get("task_name") != "click_bell"
        or act.get("task_config") != "demo_clean"
        or act.get("checkpoint_setting") != "demo_clean"
        or not isinstance(actual_seeds, list)
        or case["seed"] not in actual_seeds
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} lacks a completed ACT episode at its declared seed"
        )
    episode_path = resolved["episode"][0]
    video_path = resolved["video"][0]
    episode_dir = _relative(root, episode_path.parent)
    associations = act.get("video_associations")
    matching = [
        item
        for item in associations or []
        if isinstance(item, Mapping)
        and item.get("episode_dir") == episode_dir
        and item.get("video") == _relative(root, video_path)
    ]
    if len(matching) != 1 or video_path.parent != episode_path.parent:
        raise SceneShiftVQAValidationError(
            f"{case_id} ACT episode/video association does not match hashed sources"
        )
    episode = _read_json(episode_path, field=f"{case_id} ACT episode")
    telemetry_hash = episode.get("telemetry_profile_sha256")
    if (
        episode.get("task_name") != "click_bell"
        or str(episode.get("policy_name", "")).casefold() != "act"
        or episode.get("seed") != case["seed"]
        or episode.get("checkpoint_setting") != "demo_clean"
        or episode.get("telemetry_profile_id") != "balanced_v1"
        or not isinstance(telemetry_hash, str)
        or not telemetry_hash
        or not isinstance(episode.get("success"), bool)
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} ACT episode identity or telemetry is invalid"
        )
    return {
        "run_id": run_id,
        "base_commit": manifest.get("base_commit"),
        "checkpoint_setting": act["checkpoint_setting"],
        "telemetry_profile_id": episode["telemetry_profile_id"],
        "telemetry_profile_sha256": telemetry_hash,
        "episode_dir": episode_dir,
        "video_path": _relative(root, video_path),
        "act_success": episode["success"],
        "simulator_state": simulator_state,
    }


def _validate_parent_evaluation(
    root: Path,
    case: Mapping[str, Any],
    resolved: Mapping[str, tuple[Path, dict[str, str]]],
    taskgen: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(case["id"])
    path = resolved["evaluation_manifest"][0]
    expected = f"mea/evaluation_runs/{case['source_evaluation_id']}/manifest.json"
    if _relative(root, path) != expected:
        raise SceneShiftVQAValidationError(
            f"{case_id} evaluation manifest path does not match evaluation_id"
        )
    manifest = _read_json(path, field=f"{case_id} evaluation manifest")
    base_commit = manifest.get("base_commit")
    child_run_ids = manifest.get("child_run_ids")
    if child_run_ids is None:
        # Compatibility with older one-round evaluations that predate the
        # append-only child list. In that schema the active child was the only
        # child, so exact equality remains the binding authority.
        child_bound = manifest.get("active_child_run_id") == taskgen["run_id"]
    else:
        child_bound = bool(
            isinstance(child_run_ids, list)
            and child_run_ids
            and all(isinstance(item, str) and item for item in child_run_ids)
            and len(child_run_ids) == len(set(child_run_ids))
            and taskgen["run_id"] in child_run_ids
            and manifest.get("active_child_run_id") == child_run_ids[-1]
        )
    if (
        manifest.get("evaluation_id") != case["source_evaluation_id"]
        or manifest.get("status") != "completed"
        or manifest.get("lifecycle_status") != "completed"
        or not child_bound
        or manifest.get("task_name") != "click_bell"
        or manifest.get("telemetry_profile") != "balanced_v1"
        or not isinstance(base_commit, str)
        or not base_commit
        or base_commit != taskgen["base_commit"]
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} parent evaluation does not bind the TaskGen child/protocol"
        )
    return {
        "base_commit": base_commit,
        "child_binding": (
            "child_run_ids_membership"
            if child_run_ids is not None
            else "legacy_active_child"
        ),
    }


def _validate_execution_vqa_case(
    root: Path,
    case: Mapping[str, Any],
    resolved: Mapping[str, tuple[Path, dict[str, str]]],
    taskgen: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = str(case["id"])
    evaluation_prefix = Path("mea/evaluation_runs") / case["source_evaluation_id"]
    for source_name in ("execution_vqa", "query", "montage"):
        relative = Path(_relative(root, resolved[source_name][0]))
        if relative.parts[: len(evaluation_prefix.parts)] != evaluation_prefix.parts:
            raise SceneShiftVQAValidationError(
                f"{case_id} {source_name} is outside its evaluation directory"
            )
    artifact = _read_json(
        resolved["execution_vqa"][0], field=f"{case_id} Execution VQA"
    )
    if artifact.get("schema_version") != 1 or artifact.get("status") != "passed":
        raise SceneShiftVQAValidationError(
            f"{case_id} Execution VQA must be a completed passed artifact"
        )
    try:
        query = validate_execution_vqa_query(artifact.get("query"))
    except ExecutionVQAQueryError as exc:
        raise SceneShiftVQAValidationError(
            f"{case_id} embedded Execution VQA query is invalid: {exc}"
        ) from exc
    expected = case["expected_query"]
    query_contract = {
        "task_name": query["task_name"],
        "template_id": query["template_id"],
        "sub_aspect": query["sub_aspect"],
        "primary_visibility_phenomenon_id": expected[
            "primary_visibility_phenomenon_id"
        ],
        "phenomenon_ids": query["phenomenon_ids"],
    }
    if query_contract != expected:
        raise SceneShiftVQAValidationError(
            f"{case_id} embedded query does not match the preregistered condition"
        )
    if _read_json(resolved["query"][0], field=f"{case_id} query source") != query:
        raise SceneShiftVQAValidationError(
            f"{case_id} embedded and hashed query artifacts differ"
        )
    artifacts = _mapping(artifact.get("artifacts"), field=f"{case_id}.artifacts")
    artifact_result = _resolve_embedded_artifact(
        root, artifacts.get("result"), field=f"{case_id}.artifacts.result"
    )
    artifact_query = _resolve_embedded_artifact(
        root, artifacts.get("query"), field=f"{case_id}.artifacts.query"
    )
    artifact_montage = _resolve_embedded_artifact(
        root, artifacts.get("montage"), field=f"{case_id}.artifacts.montage"
    )
    if (
        artifact_result != resolved["execution_vqa"][0]
        or artifact_query != resolved["query"][0]
        or artifact_montage != resolved["montage"][0]
    ):
        raise SceneShiftVQAValidationError(
            f"{case_id} VQA artifact does not bind hashed result/query/montage paths"
        )
    selection = _mapping(artifact.get("selection"), field=f"{case_id}.selection")
    selection_video = _resolve_embedded_artifact(
        root, selection.get("video_path"), field=f"{case_id}.selection.video_path"
    )
    selection_montage = _resolve_embedded_artifact(
        root,
        selection.get("montage_path"),
        field=f"{case_id}.selection.montage_path",
    )
    if selection_video != resolved["video"][0]:
        raise SceneShiftVQAValidationError(
            f"{case_id} VQA selection is not the hashed ACT video"
        )
    if selection_montage != resolved["montage"][0]:
        raise SceneShiftVQAValidationError(
            f"{case_id} VQA selection is not the hashed montage"
        )
    if artifact.get("representative_episode") != taskgen["episode_dir"]:
        raise SceneShiftVQAValidationError(
            f"{case_id} VQA representative episode is not the hashed ACT episode"
        )
    selected_frames = selection.get("selected_frames")
    if not isinstance(selected_frames, list) or not selected_frames:
        raise SceneShiftVQAValidationError(f"{case_id} VQA has no selected frames")
    frame_ids: list[str] = []
    for raw_frame in selected_frames:
        frame = _mapping(raw_frame, field=f"{case_id}.selected_frame")
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise SceneShiftVQAValidationError(
                f"{case_id} selected frame id is invalid"
            )
        frame_ids.append(frame_id)
    observation = _mapping(
        artifact.get("observation"), field=f"{case_id}.observation"
    )
    try:
        normalized = validate_execution_vqa_response(
            {
                key: observation.get(key)
                for key in (
                    "phenomena",
                    "confidence",
                    "frame_ids",
                    "numeric_consistency",
                    "conflicts",
                )
            },
            allowed_frame_ids=frame_ids,
            expected_phenomenon_ids=query["phenomenon_ids"],
        )
    except ExecutionVQAError as exc:
        raise SceneShiftVQAValidationError(
            f"{case_id} Execution VQA response is invalid: {exc}"
        ) from exc
    predictions = {item["id"]: item for item in normalized["phenomena"]}
    rows = []
    for label in case["labels"]:
        prediction = predictions[label["phenomenon_id"]]
        rows.append(
            {
                "case_id": case_id,
                "condition": case["condition"],
                "seed": case["seed"],
                "phenomenon_id": label["phenomenon_id"],
                "primary_visibility": (
                    label["phenomenon_id"]
                    == expected["primary_visibility_phenomenon_id"]
                ),
                "label_source": "development_agent_proxy",
                "proxy_observed": label["observed"],
                "predicted_observed": prediction["observed"],
                "confidence": float(prediction["confidence"]),
                "correct": prediction["observed"] == label["observed"],
            }
        )
    return (
        {
            "model": _mapping(
                artifact.get("provider_metadata"),
                field=f"{case_id}.provider_metadata",
            ).get("model"),
            "representative_episode": artifact["representative_episode"],
        },
        rows,
    )


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    total = len(rows)
    return {"value": correct / total if total else None, "correct": correct, "total": total}


def summarize_scene_shift_vqa_suite(
    repo_root: str | Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit completed evidence and aggregate proxy labels without new calls."""

    root = Path(repo_root).expanduser().resolve()
    normalized = validate_scene_shift_vqa_suite(suite)
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for case in normalized["cases"]:
        resolved = {
            name: _resolve_hashed_ref(
                root,
                reference,
                field=f"{case['id']}.sources.{name}",
            )
            for name, reference in case["sources"].items()
        }
        taskgen = _validate_taskgen_case(root, case, resolved)
        parent = _validate_parent_evaluation(root, case, resolved, taskgen)
        vqa, case_rows = _validate_execution_vqa_case(root, case, resolved, taskgen)
        identity = {
            "base_commit": parent["base_commit"],
            "checkpoint_setting": taskgen["checkpoint_setting"],
            "telemetry_profile_id": taskgen["telemetry_profile_id"],
            "telemetry_profile_sha256": taskgen["telemetry_profile_sha256"],
        }
        identities.append(identity)
        rows.extend(case_rows)
        cases.append(
            {
                "id": case["id"],
                "condition": case["condition"],
                "seed": case["seed"],
                "source_evaluation_id": case["source_evaluation_id"],
                "source_hashes": {name: item[1] for name, item in resolved.items()},
                "taskgen": taskgen,
                "execution_vqa": vqa,
                "labels": case_rows,
            }
        )
    if any(identity != identities[0] for identity in identities[1:]):
        raise SceneShiftVQAValidationError(
            "suite cases do not share one commit/checkpoint/telemetry protocol"
        )
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_cases = [case for case in cases if case["condition"] == condition]
        condition_rows = [row for row in rows if row["condition"] == condition]
        primary_rows = [row for row in condition_rows if row["primary_visibility"]]
        by_condition[condition] = {
            "evaluation_count": len(condition_cases),
            "label_count": len(condition_rows),
            "accuracy": _accuracy(condition_rows),
            "primary_visibility_accuracy": _accuracy(primary_rows),
            "primary_visibility_label_balance": dict(
                sorted(
                    Counter(str(row["proxy_observed"]).lower() for row in primary_rows).items()
                )
            ),
            "auroc": None,
            "auroc_unavailable_reason": (
                "development_proxy_smoke_has_fewer_than_10_cases"
                if len(condition_cases) < 10
                else "not_computed_by_offline_functional_protocol"
            ),
        }
    return {
        "schema_version": 1,
        "suite_id": normalized["suite_id"],
        "protocol": PROTOCOL,
        "mode": "offline_completed_artifact_audit",
        "conditions": list(CONDITIONS),
        "protocol_identity": identities[0],
        "accuracy": _accuracy(rows),
        "by_condition": by_condition,
        "auroc": None,
        "auroc_unavailable_reason": "small_development_proxy_functional_suite",
        "reviewer": deepcopy(normalized["reviewer"]),
        "label_source_counts": dict(
            sorted(Counter(row["label_source"] for row in rows).items())
        ),
        "human_reviewer_count": 0,
        "paper_table_eligible": False,
        "paper_table_ineligible_reason": (
            "development_agent_proxy_labels_and_small_functional_suite"
        ),
        "provider_called": False,
        "simulator_called": False,
        "act_called": False,
        "image_proxy_used": False,
        "source_scope": "real_simulator_completed_act_execution_vqa",
        "cases": cases,
        "rows": rows,
        "limitations": [
            "This command audits existing cases and does not run RoboTwin, ACT, or VQA.",
            "Labels are development-agent proxies, not human gold or majority vote.",
            "At least two positive/negative cases per condition is a functional smoke, not paper-scale evidence.",
            "AUROC remains null for this small development protocol.",
        ],
    }
