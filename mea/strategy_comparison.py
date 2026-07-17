"""Strict ACT fixed-suite versus adaptive-strategy aggregation.

This is intentionally an artifact-only reader.  It never starts RoboTwin and
therefore cannot accidentally spend an ACT rollout while reports are rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class StrategyComparisonError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyComparisonError(
            f"cannot read JSON artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise StrategyComparisonError(f"artifact must be an object: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise StrategyComparisonError(f"artifact path must be relative: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise StrategyComparisonError(f"artifact path escapes repo: {relative}")
    return path


def _duration_seconds(manifest: Mapping[str, Any]) -> float | None:
    try:
        started = datetime.fromisoformat(str(manifest["created_at"]))
        finished = datetime.fromisoformat(str(manifest["execution_finished_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max((finished - started).total_seconds(), 0.0)


def _strategy_run(
    root: Path,
    relative: str,
    *,
    expected_policy: str,
) -> dict[str, Any]:
    evaluation_dir = _inside(root, relative)
    manifest = _read_object(evaluation_dir / "manifest.json")
    summary = _read_object(evaluation_dir / "summary/summary.json")
    if manifest.get("lifecycle_status") != "completed" or manifest.get(
        "status"
    ) not in {
        "completed",
        "completed_with_pipeline_failure",
    }:
        raise StrategyComparisonError(f"evaluation is not complete: {relative}")
    if summary.get("status") != "completed":
        raise StrategyComparisonError(f"evaluation pipeline did not pass: {relative}")
    planner = (
        manifest.get("planner") if isinstance(manifest.get("planner"), dict) else {}
    )
    planning_policy = manifest.get("planning_policy") or planner.get("planning_policy")
    if planning_policy != expected_policy:
        raise StrategyComparisonError(
            f"planning policy mismatch for {relative}: {planning_policy!r}"
        )
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise StrategyComparisonError(f"evaluation has no plan object: {relative}")
    policy = plan.get("policy")
    if not isinstance(policy, dict) or str(policy.get("name", "")).casefold() != "act":
        raise StrategyComparisonError(f"evaluation is not ACT-only: {relative}")
    candidates = plan.get("requested_template_ids")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) != len(set(candidates))
        or any(not isinstance(item, str) or not item for item in candidates)
    ):
        raise StrategyComparisonError(f"invalid frozen candidate suite: {relative}")
    suite_sha256 = _canonical_sha256(candidates)
    recorded_suite_sha256 = manifest.get("candidate_suite_sha256")
    if recorded_suite_sha256 != suite_sha256:
        raise StrategyComparisonError(f"candidate suite hash mismatch: {relative}")
    user_request = manifest.get("user_request")
    if not isinstance(user_request, str) or not user_request.strip():
        raise StrategyComparisonError(f"evaluation has no query identity: {relative}")
    global_route_selection = manifest.get("global_route_selection")
    if global_route_selection is not None and not isinstance(
        global_route_selection, dict
    ):
        raise StrategyComparisonError(f"invalid global route identity: {relative}")

    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise StrategyComparisonError(f"evaluation has no completed rounds: {relative}")
    samples: list[dict[str, Any]] = []
    for round_value in rounds:
        if (
            not isinstance(round_value, dict)
            or round_value.get("pipeline_passed") is not True
        ):
            raise StrategyComparisonError(f"round pipeline did not pass: {relative}")
        variant_id = round_value.get("variant_id")
        child_id = round_value.get("taskgen_run_id")
        if (
            variant_id not in candidates
            or not isinstance(child_id, str)
            or re.fullmatch(r"run_[A-Za-z0-9_.-]+", child_id) is None
        ):
            raise StrategyComparisonError(f"invalid round identity: {relative}")
        child_dir = _inside(root, f"mea/generated_tasks/{child_id}")
        child = _read_object(child_dir / "manifest.json")
        if child.get("run_id") != child_id or child.get("status") != "completed":
            raise StrategyComparisonError(
                f"child manifest identity/status mismatch: {child_id}"
            )
        trusted = child.get("trusted_tool_evaluation")
        episodes = trusted.get("episodes") if isinstance(trusted, dict) else None
        if not isinstance(episodes, list):
            raise StrategyComparisonError(f"child has no trusted episodes: {child_id}")
        act_episodes = [
            item
            for item in episodes
            if isinstance(item, dict)
            and str(item.get("policy_name", "")).casefold() == "act"
        ]
        if not act_episodes:
            raise StrategyComparisonError(f"child has no ACT episode: {child_id}")
        for episode_ref in act_episodes:
            episode_dir = episode_ref.get("episode_dir")
            if not isinstance(episode_dir, str) or not episode_dir:
                raise StrategyComparisonError(f"invalid ACT episode ref: {child_id}")
            telemetry_root = (child_dir / "evaluation/telemetry").resolve()
            episode_path = _inside(telemetry_root, f"{episode_dir}/episode.json")
            episode = _read_object(episode_path)
            if str(episode.get("policy_name", "")).casefold() != "act":
                raise StrategyComparisonError(
                    f"episode is not ACT: {child_id}/{episode_dir}"
                )
            if episode.get("task_name") != manifest.get("task_name"):
                raise StrategyComparisonError(
                    f"episode task mismatch: {child_id}/{episode_dir}"
                )
            seed = episode.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise StrategyComparisonError(
                    f"invalid episode seed: {child_id}/{episode_dir}"
                )
            if episode.get("error") not in {None, ""}:
                raise StrategyComparisonError(
                    f"episode contains execution error: {child_id}"
                )
            if not isinstance(episode.get("success"), bool):
                raise StrategyComparisonError(
                    f"episode success must be an explicit boolean: {child_id}"
                )
            for field in ("policy_steps", "physics_steps"):
                if isinstance(episode.get(field), bool) or not isinstance(
                    episode.get(field), int
                ):
                    raise StrategyComparisonError(f"invalid {field}: {child_id}")
            wall = episode.get("wall_duration_seconds")
            if isinstance(wall, bool) or not isinstance(wall, (int, float)):
                raise StrategyComparisonError(
                    f"invalid rollout wall duration: {child_id}"
                )
            samples.append(
                {
                    "variant_id": variant_id,
                    "seed": seed,
                    "success": episode["success"],
                    "policy_steps": episode["policy_steps"],
                    "physics_steps": episode["physics_steps"],
                    "rollout_wall_seconds": float(wall),
                    "episode": str(
                        episode_path.relative_to(root)
                    ),
                }
            )
    identities = [(item["variant_id"], item["seed"]) for item in samples]
    if len(identities) != len(set(identities)):
        raise StrategyComparisonError(f"duplicate sample identities: {relative}")
    variant_counts = Counter(item["variant_id"] for item in samples)
    if any(count != 1 for count in variant_counts.values()):
        raise StrategyComparisonError(
            f"N=1 strategy run must contain one seed per executed variant: {relative}"
        )
    if expected_policy == "fixed_predeclared_v1" and (
        set(variant_counts) != set(candidates)
        or len(samples) != len(candidates)
    ):
        raise StrategyComparisonError(
            f"fixed strategy does not cover its complete frozen candidate suite: {relative}"
        )
    return {
        "evaluation_dir": relative,
        "evaluation_id": manifest.get("evaluation_id"),
        "task_name": manifest.get("task_name"),
        "task_profile": manifest.get("task_profile"),
        "telemetry_profile": manifest.get("telemetry_profile"),
        "base_commit": manifest.get("base_commit"),
        "planning_policy": planning_policy,
        "user_request": user_request.strip(),
        "user_request_sha256": _canonical_sha256(user_request.strip()),
        "global_route_selection": global_route_selection,
        "policy": policy,
        "candidate_suite": candidates,
        "candidate_suite_sha256": suite_sha256,
        "process_wall_seconds": _duration_seconds(manifest),
        "samples": samples,
    }


def _totals(run: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(run["samples"])
    return {
        "act_rollouts": len(samples),
        "successes": sum(item["success"] for item in samples),
        "policy_steps": sum(item["policy_steps"] for item in samples),
        "physics_steps": sum(item["physics_steps"] for item in samples),
        "rollout_wall_seconds": sum(item["rollout_wall_seconds"] for item in samples),
        "process_wall_seconds": run["process_wall_seconds"],
    }


def compare_fixed_dynamic(
    repo_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    if config.get("schema_version") != 1:
        raise StrategyComparisonError("comparison config schema_version must be 1")
    fixed = _strategy_run(
        root,
        str(config.get("fixed_evaluation_dir") or ""),
        expected_policy="fixed_predeclared_v1",
    )
    dynamic = _strategy_run(
        root,
        str(config.get("dynamic_evaluation_dir") or ""),
        expected_policy="dynamic_evidence_v1",
    )
    comparable_fields = (
        "task_name",
        "telemetry_profile",
        "policy",
        "candidate_suite",
        "user_request",
        "global_route_selection",
    )
    mismatch = [field for field in comparable_fields if fixed[field] != dynamic[field]]
    if mismatch:
        raise StrategyComparisonError(f"strategy identity mismatch: {mismatch}")
    if fixed["base_commit"] != dynamic["base_commit"]:
        raise StrategyComparisonError("strategy base commits differ")
    fixed_map = {(item["variant_id"], item["seed"]): item for item in fixed["samples"]}
    dynamic_map = {
        (item["variant_id"], item["seed"]): item for item in dynamic["samples"]
    }
    unknown = sorted(set(dynamic_map) - set(fixed_map))
    if unknown:
        raise StrategyComparisonError(
            f"dynamic samples are outside fixed suite: {unknown}"
        )
    overlap = [
        {
            "variant_id": identity[0],
            "seed": identity[1],
            "fixed_success": fixed_map[identity]["success"],
            "dynamic_success": dynamic_map[identity]["success"],
            "exact_success_agreement": (
                fixed_map[identity]["success"] == dynamic_map[identity]["success"]
            ),
        }
        for identity in sorted(dynamic_map)
    ]
    return {
        "schema_version": 1,
        "protocol": "click_bell_act_efficiency_mechanism_fixed_vs_dynamic_n1_v2",
        "claim_scope": "table1_efficiency_mechanism_facing_n1_micro_pilot",
        "paper_table_eligible": False,
        "table2_consistency": None,
        "table2_unavailable_reason": "n1_has_no_trial_distribution",
        "identity": {
            "task_name": fixed["task_name"],
            "policy": fixed["policy"],
            "telemetry_profile": fixed["telemetry_profile"],
            "base_commit": fixed["base_commit"],
            "candidate_suite": fixed["candidate_suite"],
            "candidate_suite_sha256": fixed["candidate_suite_sha256"],
            "user_request_sha256": fixed["user_request_sha256"],
            "global_route_selection": fixed["global_route_selection"],
        },
        "strategies": {
            "fixed_predeclared_v1": {**fixed, "totals": _totals(fixed)},
            "dynamic_evidence_v1": {**dynamic, "totals": _totals(dynamic)},
        },
        "overlap": overlap,
        "overlap_exact_success_agreement_rate": (
            sum(item["exact_success_agreement"] for item in overlap) / len(overlap)
            if overlap
            else None
        ),
        "rollout_savings": len(fixed["samples"]) - len(dynamic["samples"]),
        "limitations": [
            "N=1 cannot estimate variance or the paper Table 2 consistency metric.",
            "This is one ACT task and one frozen candidate suite, not full RoboTwin.",
            "This custom fixed-vs-dynamic pair tests an efficiency mechanism; it is not the paper Table 1 standard-benchmark comparison.",
            "Checkpoint equality is logical policy configuration equality; checkpoint file content hashes are not yet recorded.",
        ],
    }


__all__ = [
    "StrategyComparisonError",
    "compare_fixed_dynamic",
]
