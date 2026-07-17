"""Auditable ACT three-task pilot aggregation.

This module intentionally does not claim to reproduce paper Tables 1--2.  It
compares one direct official ACT episode with one complete MEA Agent protocol
episode at the same task/seed and reports the missing statistical evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class BenchmarkPilotError(RuntimeError):
    pass


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise BenchmarkPilotError(f"artifact is missing or escapes repo: {relative}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkPilotError(f"cannot read artifact {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkPilotError(f"artifact must be a JSON object: {relative}")
    return value


def _episode_measurement(
    root: Path, relative: str, *, task_name: str, seed: int
) -> dict[str, Any]:
    value = _read_json(root, relative)
    required = {
        "success": bool,
        "policy_steps": int,
        "physics_steps": int,
        "simulation_duration_seconds": (int, float),
        "wall_duration_seconds": (int, float),
    }
    if value.get("task_name") != task_name or value.get("seed") != seed:
        raise BenchmarkPilotError(f"episode task/seed mismatch: {relative}")
    if str(value.get("policy_name", "")).casefold() != "act":
        raise BenchmarkPilotError(f"episode is not ACT: {relative}")
    if value.get("error") not in {None, ""}:
        raise BenchmarkPilotError(f"episode contains an execution error: {relative}")
    for field, expected in required.items():
        item = value.get(field)
        if isinstance(item, bool) and field != "success":
            raise BenchmarkPilotError(f"invalid {field}: {relative}")
        if not isinstance(item, expected):
            raise BenchmarkPilotError(f"invalid {field}: {relative}")
    return {
        "episode": relative,
        "task_name": task_name,
        "seed": seed,
        "success": value["success"],
        "policy_steps": value["policy_steps"],
        "physics_steps": value["physics_steps"],
        "simulation_duration_seconds": float(
            value["simulation_duration_seconds"]
        ),
        "rollout_wall_seconds": float(value["wall_duration_seconds"]),
    }


def _agent_route(
    root: Path, protocol_dir: str, *, task_name: str, seed: int
) -> dict[str, Any]:
    prefix = protocol_dir.rstrip("/")
    summary = _read_json(root, f"{prefix}/summary/protocol_summary.json")
    manifest = _read_json(root, f"{prefix}/protocol_manifest.json")
    if not summary.get("valid_for_comparison"):
        raise BenchmarkPilotError(f"Agent protocol is not valid: {protocol_dir}")
    episodes = [
        episode
        for repetition in manifest.get("repetitions") or []
        for attempt in (repetition.get("attempts") or [])[-1:]
        for episode in (attempt.get("measurement") or {}).get("episodes", [])
        if episode.get("seed") == seed
    ]
    if len(episodes) != 1:
        raise BenchmarkPilotError(
            f"expected one Agent episode for {task_name}/{seed}, found {len(episodes)}"
        )
    measured = _episode_measurement(
        root,
        f"{episodes[0]['episode_dir']}/episode.json",
        task_name=task_name,
        seed=seed,
    )
    measured.update(
        {
            "process_wall_seconds": float(
                summary.get("agent_wall_duration_seconds") or 0
            ),
            "protocol_summary": f"{prefix}/summary/protocol_summary.json",
        }
    )
    return measured


def aggregate_three_task_pilot(
    repo_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise BenchmarkPilotError("the pilot requires exactly three tasks")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for item in tasks:
        task_name = str(item.get("task_name") or "")
        seed = item.get("seed")
        identity = (task_name, seed)
        if not task_name or isinstance(seed, bool) or not isinstance(seed, int):
            raise BenchmarkPilotError("task_name/seed identity is invalid")
        if identity in identities:
            raise BenchmarkPilotError(f"duplicate task/seed identity: {identity}")
        identities.add(identity)
        validity = _read_json(root, str(item.get("direct_validity_artifact")))
        if validity.get("valid_for_comparison") is not True:
            raise BenchmarkPilotError(f"direct route is not protocol-valid: {identity}")
        direct = _episode_measurement(
            root,
            str(item.get("direct_episode")),
            task_name=task_name,
            seed=seed,
        )
        direct["process_wall_seconds"] = None
        agent = _agent_route(
            root,
            str(item.get("agent_protocol_dir")),
            task_name=task_name,
            seed=seed,
        )
        rows.append(
            {
                "task_name": task_name,
                "seed": seed,
                "routes": {
                    "direct_official_act": direct,
                    "mea_agent_official": agent,
                },
                "agreement": {
                    "binary_same_seed_exact": direct["success"] == agent["success"],
                    "policy_steps_same_seed_exact": (
                        direct["policy_steps"] == agent["policy_steps"]
                    ),
                },
            }
        )
    routes: dict[str, dict[str, Any]] = {}
    for route in ("direct_official_act", "mea_agent_official"):
        values = [row["routes"][route] for row in rows]
        routes[route] = {
            "tasks": len(values),
            "successes": sum(item["success"] is True for item in values),
            "policy_steps": sum(item["policy_steps"] for item in values),
            "physics_steps": sum(item["physics_steps"] for item in values),
            "rollout_wall_seconds": sum(
                item["rollout_wall_seconds"] for item in values
            ),
            "process_wall_seconds": (
                sum(float(item["process_wall_seconds"]) for item in values)
                if all(item["process_wall_seconds"] is not None for item in values)
                else None
            ),
        }
    binary_agreements = [
        row["agreement"]["binary_same_seed_exact"] for row in rows
    ]
    return {
        "schema_version": 1,
        "protocol": "act_three_task_n1_pilot_v1",
        "claim_scope": "instrumentation_smoke_not_paper_tables",
        "task_seed_identity_fields": ["task_name", "seed"],
        "routes": routes,
        "tasks": rows,
        "agreement": {
            "binary_same_seed_exact_count": sum(binary_agreements),
            "binary_same_seed_exact_rate": sum(binary_agreements) / len(rows),
            "table2_consistency": None,
            "unavailable_reason": "n1_has_no_variance",
        },
        "smoke_only": True,
        "paper_table_eligible": False,
        "limitations": [
            "N=1 has no variance and cannot reproduce paper Table 2.",
            "policy_steps are policy-call counts, not asserted to equal the paper's sample counter.",
            "cached direct paired runs do not record a comparable outer process wall time.",
        ],
    }


def render_pilot_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ACT three-task N=1 pilot",
        "",
        "This is an instrumentation smoke, not a reproduction of paper Tables 1--2.",
        "",
        "| task | seed | direct success/steps/wall s | MEA success/steps/wall s | binary agreement |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in summary.get("tasks") or []:
        direct = row["routes"]["direct_official_act"]
        agent = row["routes"]["mea_agent_official"]
        lines.append(
            f"| {row['task_name']} | {row['seed']} | "
            f"{direct['success']}/{direct['policy_steps']}/{direct['rollout_wall_seconds']:.3f} | "
            f"{agent['success']}/{agent['policy_steps']}/{agent['rollout_wall_seconds']:.3f} | "
            f"{row['agreement']['binary_same_seed_exact']} |"
        )
    lines.extend(
        [
            "",
            f"Binary agreement: `{summary['agreement']['binary_same_seed_exact_count']}/3`.",
            "Table-2 consistency: `unavailable (N=1 has no variance)`.",
            "",
        ]
    )
    return "\n".join(lines)
