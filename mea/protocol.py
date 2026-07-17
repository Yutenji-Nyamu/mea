"""Small, resumable ACT-only protocol helpers.

The protocol layer repeats complete Agent evaluations.  It does not replace
the Agent, RoboTwin evaluator, or the strict Easy/Hard paired runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


AGILE_BUDGETS = (1, 3, 5)


class ProtocolError(RuntimeError):
    """Raised when a protocol run or artifact violates its contract."""


def validate_budget(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} must be one of {AGILE_BUDGETS}")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        normalized = int(value)
    else:
        raise ProtocolError(f"{name} must be one of {AGILE_BUDGETS}")
    if normalized not in AGILE_BUDGETS:
        raise ProtocolError(f"{name} must be one of {AGILE_BUDGETS}")
    return normalized


def validate_run_id(value: str) -> str:
    normalized = str(value).strip()
    if not re.fullmatch(r"protocol_[A-Za-z0-9_]+", normalized):
        raise ProtocolError(
            "run_id must contain only letters, digits, or underscores and "
            "begin with 'protocol_'"
        )
    return normalized


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_repetition_schedule(
    *, repetitions: int, episodes: int, start_seed: int
) -> list[dict[str, Any]]:
    repetition_count = validate_budget(repetitions, name="repetitions")
    episode_count = validate_budget(episodes, name="episodes")
    seed = int(start_seed)
    if seed < 0:
        raise ProtocolError("start_seed must be non-negative")
    return [
        {
            "index": index + 1,
            "start_seed": seed + index * episode_count,
            "requested_episodes": episode_count,
            "status": "pending",
            "attempts": [],
        }
        for index in range(repetition_count)
    ]


def build_expected_sample_identities(
    *, variant_ids: Sequence[str], episodes: int, start_seed: int
) -> list[dict[str, Any]]:
    """Build the frozen cartesian sample set for a generated repetition.

    A generated evaluation may deliberately reuse a RoboTwin seed across
    variants.  The protocol identity is therefore ``(variant_id, seed)`` and
    never the raw seed alone.
    """

    episode_count = validate_budget(episodes, name="episodes")
    normalized = [str(value).strip() for value in variant_ids]
    if not normalized or any(not value for value in normalized):
        raise ProtocolError("variant_ids must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ProtocolError("variant_ids must not contain duplicates")
    seed = int(start_seed)
    if seed < 0:
        raise ProtocolError("start_seed must be non-negative")
    return [
        {"variant_id": variant_id, "seed": seed + offset}
        for variant_id in normalized
        for offset in range(episode_count)
    ]


def evaluation_id_for_attempt(
    run_id: str, repetition_index: int, attempt_index: int
) -> str:
    normalized = validate_run_id(run_id).removeprefix("protocol_")
    return (
        f"eval_protocol_{normalized}_rep_{int(repetition_index):03d}"
        f"_attempt_{int(attempt_index):02d}"
    )


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object: {path}")
    return value


def _act_episode_metadata(
    repo_root: Path,
    child_run_id: str,
    child_manifest: Mapping[str, Any],
    *,
    expected_task_name: str | None,
    variant_id: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    child_dir = repo_root / "mea/generated_tasks" / child_run_id
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    seen_paths: set[Path] = set()
    for episode in trusted.get("episodes", []):
        if str(episode.get("policy_name", "")).casefold() != "act":
            continue
        relative = episode.get("episode_dir")
        if not isinstance(relative, str) or not relative:
            issues.append(f"{child_run_id}: ACT episode is missing episode_dir")
            continue
        metadata_path = child_dir / "evaluation/telemetry" / relative / "episode.json"
        metadata_path = metadata_path.resolve()
        if not metadata_path.is_relative_to(child_dir.resolve()):
            issues.append(f"{child_run_id}: episode path escapes child run")
            continue
        if metadata_path in seen_paths:
            issues.append(f"{child_run_id}: duplicate ACT episode {relative}")
            continue
        seen_paths.add(metadata_path)
        try:
            metadata = _read_object(metadata_path, label="ACT episode metadata")
        except ProtocolError as exc:
            issues.append(str(exc))
            continue
        if str(metadata.get("policy_name", "")).casefold() != "act":
            issues.append(f"{child_run_id}: episode metadata is not ACT: {relative}")
            continue
        seed = metadata.get("seed")
        success = metadata.get("success")
        policy_steps = metadata.get("policy_steps")
        physics_steps = metadata.get("physics_steps")
        durations = {
            "simulation_duration_seconds": metadata.get(
                "simulation_duration_seconds"
            ),
            "wall_duration_seconds": metadata.get("wall_duration_seconds"),
        }
        invalid_fields: list[str] = []
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            invalid_fields.append("seed")
        if not isinstance(success, bool):
            invalid_fields.append("success")
        for field, value in (
            ("policy_steps", policy_steps),
            ("physics_steps", physics_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                invalid_fields.append(field)
        for field, value in durations.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                invalid_fields.append(field)
        if expected_task_name and metadata.get("task_name") != expected_task_name:
            invalid_fields.append("task_name")
        if episode.get("seed") != seed:
            invalid_fields.append("trusted_seed_mismatch")
        if episode.get("success") is not success:
            invalid_fields.append("trusted_success_mismatch")
        if metadata.get("error") is not None and metadata.get("error") != "":
            invalid_fields.append("error")
        if invalid_fields:
            issues.append(
                f"{child_run_id}: invalid ACT episode {relative}: "
                + ", ".join(invalid_fields)
            )
            continue
        rows.append(
            {
                "child_run_id": child_run_id,
                "variant_id": variant_id,
                "episode_dir": str(metadata_path.parent.relative_to(repo_root)),
                "seed": seed,
                "sample_identity": (
                    {"variant_id": variant_id, "seed": seed}
                    if variant_id is not None
                    else None
                ),
                "success": success,
                "policy_steps": policy_steps,
                "physics_steps": physics_steps,
                "simulation_duration_seconds": durations[
                    "simulation_duration_seconds"
                ],
                "rollout_wall_duration_seconds": durations[
                    "wall_duration_seconds"
                ],
            }
        )
    return rows, issues


def _number_sum(rows: list[dict[str, Any]], field: str) -> float | int:
    values = [row.get(field) for row in rows]
    valid = [
        value
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    total = sum(valid)
    if all(isinstance(value, int) for value in valid):
        return int(total)
    return float(total)


def _summarize_variants(
    rows: list[dict[str, Any]],
    expected_identities: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if expected_identities is None:
        return {}
    expected_counts = Counter(str(item["variant_id"]) for item in expected_identities)
    result: dict[str, dict[str, Any]] = {}
    for variant_id in sorted(expected_counts):
        selected = [row for row in rows if row.get("variant_id") == variant_id]
        requested = expected_counts[variant_id]
        observed = len(selected)
        successes = sum(row.get("success") is True for row in selected)
        result[variant_id] = {
            "requested_policy_episodes": requested,
            "observed_policy_episodes": observed,
            "coverage": observed / requested if requested else 0.0,
            "successes": successes,
            "success_rate": successes / observed if observed else None,
            "policy_steps": _number_sum(selected, "policy_steps"),
            "physics_steps": _number_sum(selected, "physics_steps"),
            "simulation_duration_seconds": _number_sum(
                selected, "simulation_duration_seconds"
            ),
            "rollout_wall_duration_seconds": _number_sum(
                selected, "rollout_wall_duration_seconds"
            ),
            "actual_seeds": [row["seed"] for row in selected],
        }
    return result


def collect_evaluation_measurement(
    repo_root: str | Path,
    *,
    evaluation_id: str,
    requested_episodes: int,
    returncode: int,
    agent_wall_duration_seconds: float,
    expected_sample_identities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    evaluation_root = (root / "mea/evaluation_runs").resolve()
    evaluation_dir = (evaluation_root / evaluation_id).resolve()
    if not evaluation_dir.is_relative_to(evaluation_root):
        raise ProtocolError("evaluation path escapes evaluation_runs")
    manifest_path = evaluation_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    issues: list[str] = []
    if manifest_path.is_file():
        try:
            manifest = _read_object(manifest_path, label="evaluation manifest")
        except ProtocolError as exc:
            issues.append(str(exc))
    else:
        issues.append(f"evaluation manifest is missing: {manifest_path}")

    rows: list[dict[str, Any]] = []
    child_statuses: list[dict[str, Any]] = []
    expected_task_name = (
        str(manifest.get("task_name")) if manifest.get("task_name") else None
    )
    expected_identities: list[dict[str, Any]] | None = None
    variant_by_child: dict[str, str] = {}
    if expected_sample_identities is not None:
        expected_identities = []
        for value in expected_sample_identities:
            variant_id = str(value.get("variant_id") or "").strip()
            seed = value.get("seed")
            if (
                not variant_id
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
            ):
                raise ProtocolError("invalid expected sample identity")
            expected_identities.append({"variant_id": variant_id, "seed": seed})
        summary_path = evaluation_dir / "summary/summary.json"
        if summary_path.is_file():
            try:
                evaluation_summary = _read_object(
                    summary_path, label="evaluation summary"
                )
                for round_summary in evaluation_summary.get("rounds") or []:
                    child_run_id = round_summary.get("taskgen_run_id")
                    variant_id = round_summary.get("variant_id")
                    if not child_run_id or not variant_id:
                        issues.append(
                            "generated round summary is missing taskgen_run_id or "
                            "variant_id"
                        )
                        continue
                    child_run_id = str(child_run_id)
                    variant_id = str(variant_id)
                    if child_run_id in variant_by_child:
                        issues.append(
                            f"duplicate generated-round mapping for {child_run_id}"
                        )
                    variant_by_child[child_run_id] = variant_id
            except ProtocolError as exc:
                issues.append(str(exc))
        else:
            issues.append(f"evaluation summary is missing: {summary_path}")
    generated_root = (root / "mea/generated_tasks").resolve()
    for child_run_id in manifest.get("child_run_ids") or []:
        child_dir = (generated_root / str(child_run_id)).resolve()
        if not child_dir.is_relative_to(generated_root):
            issues.append(f"child run path escapes generated_tasks: {child_run_id}")
            continue
        child_path = child_dir / "manifest.json"
        if not child_path.is_file():
            issues.append(f"child manifest is missing: {child_path}")
            continue
        try:
            child = _read_object(child_path, label="child manifest")
        except ProtocolError as exc:
            issues.append(str(exc))
            continue
        child_statuses.append(
            {
                "run_id": child_run_id,
                "status": child.get("status"),
                "failure": child.get("failure"),
            }
        )
        if child.get("status") != "completed":
            issues.append(
                f"{child_run_id}: child status is not completed: {child.get('status')}"
            )
        child_task_name = child.get("task_name")
        if expected_task_name and child_task_name != expected_task_name:
            issues.append(
                f"{child_run_id}: task mismatch {child_task_name!r} != "
                f"{expected_task_name!r}"
            )
        mapped_variant = variant_by_child.get(str(child_run_id))
        if expected_identities is not None:
            child_variant = str(child.get("variant_id") or "").strip()
            if not child_variant:
                issues.append(
                    f"{child_run_id}: child manifest is missing variant_id"
                )
            elif mapped_variant != child_variant:
                issues.append(
                    f"{child_run_id}: generated-round variant mismatch "
                    f"{mapped_variant!r} != {child_variant!r}"
                )
        child_rows, child_issues = _act_episode_metadata(
            root,
            str(child_run_id),
            child,
            expected_task_name=expected_task_name,
            variant_id=mapped_variant,
        )
        rows.extend(child_rows)
        issues.extend(child_issues)

    seeds = [row["seed"] for row in rows]
    actual_identities = [
        row["sample_identity"] for row in rows if row.get("sample_identity")
    ]
    duplicate_identities: list[dict[str, Any]] = []
    missing_identities: list[dict[str, Any]] = []
    unexpected_identities: list[dict[str, Any]] = []
    if expected_identities is None and len(seeds) != len(set(seeds)):
        issues.append("duplicate ACT seeds were observed inside the evaluation")
    if expected_identities is not None:
        expected_keys = [
            (item["variant_id"], item["seed"]) for item in expected_identities
        ]
        actual_keys = [
            (item["variant_id"], item["seed"]) for item in actual_identities
        ]
        expected_counts = Counter(expected_keys)
        actual_counts = Counter(actual_keys)
        duplicate_identities = [
            {"variant_id": variant_id, "seed": seed}
            for (variant_id, seed), count in sorted(actual_counts.items())
            if count > 1
        ]
        missing_identities = [
            {"variant_id": variant_id, "seed": seed}
            for variant_id, seed in sorted((expected_counts - actual_counts).elements())
        ]
        unexpected_identities = [
            {"variant_id": variant_id, "seed": seed}
            for variant_id, seed in sorted((actual_counts - expected_counts).elements())
        ]
        if any(row.get("variant_id") is None for row in rows):
            issues.append("one or more ACT episodes lack an explicit variant mapping")
        if duplicate_identities:
            issues.append("duplicate (variant_id, seed) samples were observed")
        if missing_identities:
            issues.append("expected generated samples are missing")
        if unexpected_identities:
            issues.append("unexpected generated samples were observed")

    successes = sum(row.get("success") is True for row in rows)
    observed = len(rows)
    lifecycle = manifest.get("lifecycle_status")
    evaluation_status = manifest.get("status")
    requested_count = (
        len(expected_identities)
        if expected_identities is not None
        else int(requested_episodes)
    )
    if observed != requested_count:
        issues.append(
            f"expected {requested_count} ACT episodes, observed {observed}"
        )
    completed = bool(
        returncode == 0
        and lifecycle == "completed"
        and evaluation_status == "completed"
        and not issues
    )
    failure_stage = None
    if not completed:
        if not manifest:
            failure_stage = "agent_startup"
        elif any(item.get("status") == "failed" for item in child_statuses):
            failure_stage = "taskgen_or_execution"
        elif evaluation_status in {"failed", "completed_with_pipeline_failure"}:
            failure_stage = "agent_or_feedback"
        elif returncode != 0:
            failure_stage = "agent_process"
        else:
            failure_stage = "artifact_validation"

    return {
        "schema_version": 2 if expected_identities is not None else 1,
        "evaluation_id": evaluation_id,
        "completed": completed,
        "returncode": int(returncode),
        "evaluation_status": evaluation_status,
        "lifecycle_status": lifecycle,
        "failure_stage": failure_stage,
        "evaluation_failure": manifest.get("failure"),
        "agent_wall_duration_seconds": float(agent_wall_duration_seconds),
        "samples": {
            "requested_policy_episodes": requested_count,
            "observed_policy_episodes": observed,
            "coverage": observed / requested_count if requested_count else 0.0,
            "successes": successes,
            "success_rate": successes / observed if observed else None,
            "policy_steps": _number_sum(rows, "policy_steps"),
            "physics_steps": _number_sum(rows, "physics_steps"),
            "simulation_duration_seconds": _number_sum(
                rows, "simulation_duration_seconds"
            ),
            "rollout_wall_duration_seconds": _number_sum(
                rows, "rollout_wall_duration_seconds"
            ),
            "actual_seeds": seeds,
            "actual_sample_identities": actual_identities,
            "duplicate_sample_identities": duplicate_identities,
            "missing_sample_identities": missing_identities,
            "unexpected_sample_identities": unexpected_identities,
            "by_variant": _summarize_variants(rows, expected_identities),
        },
        "episodes": rows,
        "artifact_issues": issues,
        "artifacts": {
            "evaluation_manifest": str(manifest_path.relative_to(root)),
            "evaluation_report": (
                str((evaluation_dir / "evaluation_report.md").relative_to(root))
                if (evaluation_dir / "evaluation_report.md").is_file()
                else None
            ),
        },
    }


def summarize_protocol(manifest: Mapping[str, Any]) -> dict[str, Any]:
    repetitions = list(manifest.get("repetitions") or [])
    terminal_attempts = [
        repetition["attempts"][-1]
        for repetition in repetitions
        if repetition.get("attempts")
    ]
    measurements = [
        attempt.get("measurement") or {}
        for attempt in terminal_attempts
        if attempt.get("status") == "completed"
    ]
    samples = [measurement.get("samples") or {} for measurement in measurements]
    observed = sum(int(item.get("observed_policy_episodes") or 0) for item in samples)
    successes = sum(int(item.get("successes") or 0) for item in samples)
    all_attempts = [
        attempt
        for repetition in repetitions
        for attempt in (repetition.get("attempts") or [])
    ]
    failure_counts = Counter(
        str((attempt.get("measurement") or {}).get("failure_stage") or "unknown")
        for attempt in all_attempts
        if attempt.get("status") in {"failed", "interrupted"}
    )
    config = manifest.get("config") or {}
    expected_variant_ids = [
        str(value) for value in (config.get("expected_variant_ids") or [])
    ]
    variant_multiplier = len(expected_variant_ids) or 1
    requested_total = (
        int(config.get("repetitions") or 0)
        * int(config.get("episodes") or 0)
        * variant_multiplier
    )
    actual_seeds = [
        seed
        for measurement in measurements
        for seed in (measurement.get("samples") or {}).get("actual_seeds", [])
    ]
    seed_counts = Counter(actual_seeds)
    duplicate_actual_seeds = sorted(
        seed for seed, count in seed_counts.items() if count > 1
    )
    actual_sample_identities = [
        identity
        for sample in samples
        for identity in sample.get("actual_sample_identities", [])
    ]
    identity_counts = Counter(
        (str(item.get("variant_id")), item.get("seed"))
        for item in actual_sample_identities
    )
    duplicate_sample_identities = [
        {"variant_id": variant_id, "seed": seed}
        for (variant_id, seed), count in sorted(identity_counts.items())
        if count > 1
    ]
    base_status = (
        "completed"
        if repetitions and all(item.get("status") == "completed" for item in repetitions)
        else "completed_with_failures"
        if repetitions
        and all(item.get("status") in {"completed", "failed"} for item in repetitions)
        else "in_progress"
    )
    protocol_violations = (
        duplicate_sample_identities
        if expected_variant_ids
        else duplicate_actual_seeds
    )
    status = (
        "completed_with_protocol_violation"
        if base_status == "completed" and protocol_violations
        else base_status
    )
    variants: dict[str, dict[str, Any]] = {}
    for variant_id in expected_variant_ids:
        entries = [
            (sample.get("by_variant") or {}).get(variant_id, {})
            for sample in samples
        ]
        variant_observed = sum(
            int(item.get("observed_policy_episodes") or 0) for item in entries
        )
        variant_successes = sum(int(item.get("successes") or 0) for item in entries)
        variant_requested = int(config.get("repetitions") or 0) * int(
            config.get("episodes") or 0
        )
        variants[variant_id] = {
            "requested_policy_episodes": variant_requested,
            "observed_policy_episodes": variant_observed,
            "coverage": (
                variant_observed / variant_requested if variant_requested else 0.0
            ),
            "successes": variant_successes,
            "success_rate": (
                variant_successes / variant_observed if variant_observed else None
            ),
            "policy_steps": sum(float(item.get("policy_steps") or 0) for item in entries),
            "physics_steps": sum(
                float(item.get("physics_steps") or 0) for item in entries
            ),
            "simulation_duration_seconds": sum(
                float(item.get("simulation_duration_seconds") or 0)
                for item in entries
            ),
            "rollout_wall_duration_seconds": sum(
                float(item.get("rollout_wall_duration_seconds") or 0)
                for item in entries
            ),
        }
    return {
        "schema_version": 2 if expected_variant_ids else 1,
        "run_id": manifest.get("run_id"),
        "status": status,
        "valid_for_comparison": status == "completed",
        "act_only": True,
        "requested_repetitions": len(repetitions),
        "completed_repetitions": sum(
            item.get("status") == "completed" for item in repetitions
        ),
        "failed_repetitions": sum(item.get("status") == "failed" for item in repetitions),
        "interrupted_repetitions": sum(
            item.get("status") == "interrupted" for item in repetitions
        ),
        "pending_repetitions": sum(item.get("status") == "pending" for item in repetitions),
        "requested_policy_episodes": requested_total,
        "observed_policy_episodes": observed,
        "coverage": observed / requested_total if requested_total else 0.0,
        "actual_seeds": actual_seeds,
        "duplicate_actual_seeds": duplicate_actual_seeds,
        "sample_identity_fields": (
            ["variant_id", "seed"] if expected_variant_ids else ["seed"]
        ),
        "actual_sample_identities": actual_sample_identities,
        "duplicate_sample_identities": duplicate_sample_identities,
        "variants": variants,
        "successes": successes,
        "pooled_success_rate": successes / observed if observed else None,
        "policy_steps": sum(float(item.get("policy_steps") or 0) for item in samples),
        "physics_steps": sum(float(item.get("physics_steps") or 0) for item in samples),
        "agent_wall_duration_seconds": sum(
            float(measurement.get("agent_wall_duration_seconds") or 0)
            for measurement in measurements
        ),
        "attempt_count": len(all_attempts),
        "total_attempt_wall_duration_seconds": sum(
            float((attempt.get("measurement") or {}).get("agent_wall_duration_seconds") or 0)
            for attempt in all_attempts
        ),
        "rollout_wall_duration_seconds": sum(
            float(item.get("rollout_wall_duration_seconds") or 0)
            for item in samples
        ),
        "failure_stage_counts": dict(sorted(failure_counts.items())),
        "smoke_only": len(repetitions) == 1,
        "limitations": [
            "Budgets 1/3/5 are agile development checks, not the paper's full protocol.",
            "Only ACT is evaluated in this agile protocol.",
            "Pooled descriptive statistics do not establish significance.",
            "Resume remains repetition-granular, not variant-granular.",
        ],
    }


def render_protocol_report(
    manifest: Mapping[str, Any], summary: Mapping[str, Any]
) -> str:
    config = manifest.get("config") or {}
    lines = [
        "# MEA Agile ACT Protocol Report",
        "",
        f"- run id: `{manifest.get('run_id')}`",
        f"- task: `{config.get('task_name')}`",
        f"- task profile: `{config.get('task_profile', 'official')}`",
        "- policy: `ACT`",
        f"- repetitions: `{config.get('repetitions')}`",
        f"- episodes per repetition: `{config.get('episodes')}`",
        f"- variants: `{config.get('expected_variant_ids') or ['official']}`",
        f"- status: `{summary.get('status')}`",
        f"- valid for comparison: `{str(bool(summary.get('valid_for_comparison'))).lower()}`",
        "",
        "## Repetitions",
        "",
        "| repetition | status | evaluation | episodes | success rate | coverage | policy steps | Agent wall s |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for repetition in manifest.get("repetitions") or []:
        attempt = (repetition.get("attempts") or [{}])[-1]
        measurement = attempt.get("measurement") or {}
        samples = measurement.get("samples") or {}
        lines.append(
            "| {index} | {status} | `{evaluation}` | {episodes} | {success} | "
            "{coverage:.3f} | {steps} | {wall:.3f} |".format(
                index=repetition.get("index"),
                status=repetition.get("status"),
                evaluation=attempt.get("evaluation_id") or "-",
                episodes=samples.get("observed_policy_episodes") or 0,
                success=(
                    "-"
                    if samples.get("success_rate") is None
                    else f"{float(samples['success_rate']):.3f}"
                ),
                coverage=float(samples.get("coverage") or 0),
                steps=samples.get("policy_steps") or 0,
                wall=float(measurement.get("agent_wall_duration_seconds") or 0),
            )
        )
    pooled = summary.get("pooled_success_rate")
    variants = summary.get("variants") or {}
    if variants:
        lines.extend(
            [
                "",
                "## Variants",
                "",
                "| variant | episodes | coverage | success rate | policy steps | rollout wall s |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant_id, value in variants.items():
            rate = value.get("success_rate")
            lines.append(
                "| `{variant}` | {observed}/{requested} | {coverage:.3f} | "
                "{rate} | {steps} | {wall:.3f} |".format(
                    variant=variant_id,
                    observed=value.get("observed_policy_episodes") or 0,
                    requested=value.get("requested_policy_episodes") or 0,
                    coverage=float(value.get("coverage") or 0),
                    rate="-" if rate is None else f"{float(rate):.3f}",
                    steps=value.get("policy_steps") or 0,
                    wall=float(value.get("rollout_wall_duration_seconds") or 0),
                )
            )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- completed repetitions: `{summary.get('completed_repetitions')}`",
            f"- observed/requested episodes: `{summary.get('observed_policy_episodes')}/{summary.get('requested_policy_episodes')}`",
            f"- pooled success rate: `{'unavailable' if pooled is None else f'{float(pooled):.3f}'}`",
            f"- policy steps: `{summary.get('policy_steps')}`",
            f"- Agent wall time: `{float(summary.get('agent_wall_duration_seconds') or 0):.3f} s`",
            f"- all-attempt wall time: `{float(summary.get('total_attempt_wall_duration_seconds') or 0):.3f} s`",
            f"- duplicate actual seeds: `{summary.get('duplicate_actual_seeds') or []}`",
            f"- duplicate composite identities: `{summary.get('duplicate_sample_identities') or []}`",
            f"- failure stages: `{summary.get('failure_stage_counts') or {}}`",
            "",
            "## Scope",
            "",
            "This is an ACT-only agile protocol run. Budget 1 is a smoke test; "
            "budgets 3 and 5 remain descriptive development checks and do not "
            "reproduce the paper's full 10-repeat experiment.",
            "",
        ]
    )
    return "\n".join(lines)
