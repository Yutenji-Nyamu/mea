"""Resolve and execute a Plan Agent ToolSpec over recorded trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mea.toolkit.tools import TOOL_CATALOG, TrajectoryView

from .prototype import ToolGenPrototype


class ToolOrchestrationError(RuntimeError):
    """Raised when a ToolSpec or its runtime inputs violate the contract."""


CONTACT_METRIC = "hammer_block_contact_ever"
CONTACT_QUESTION = "蓝色方块场景中，锤子是否与方块发生过严格物理接触？"
CONTACT_REQUIRED_SIGNALS = [
    "hammer_block_contact_intervals",
    "physics_step_index",
]
CONTACT_OUTPUT_CONTRACT = {
    "value_type": "boolean",
    "unit": None,
    "passed_rule": "equals_value",
    "evidence_rule": "first_physical_contact_physics_step_or_empty",
}
CONTACT_VALIDATION_REQUIREMENTS = {
    "force_codegen": {
        "min_episodes": 2,
        "distinct_reference_values": True,
        "required_reference_values": [False, True],
    },
    "reuse": {
        "min_episodes": 1,
        "distinct_reference_values": False,
        "required_reference_values": [],
    },
}
TOOL_SPEC_KEYS = {
    "schema_version",
    "task_name",
    "metric",
    "question",
    "route",
    "reference_tool",
    "required_signals",
    "output_contract",
    "validation_requirements",
}


def contact_tool_spec(route: str) -> dict[str, Any]:
    """Return the exact first-version contact ToolSpec for a route."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": CONTACT_METRIC,
        "question": CONTACT_QUESTION,
        "route": route,
        "reference_tool": CONTACT_METRIC,
        "required_signals": list(CONTACT_REQUIRED_SIGNALS),
        "output_contract": dict(CONTACT_OUTPUT_CONTRACT),
        "validation_requirements": {
            **CONTACT_VALIDATION_REQUIREMENTS[route],
            "required_reference_values": list(
                CONTACT_VALIDATION_REQUIREMENTS[route][
                    "required_reference_values"
                ]
            ),
        },
    }


def validate_tool_spec(
    value: Any,
    *,
    expected_route: str | None = None,
) -> dict[str, Any]:
    """Validate the intentionally narrow ToolSpec emitted by the Plan Agent."""

    if not isinstance(value, dict):
        raise ToolOrchestrationError("ToolSpec 必须是 JSON object")
    keys = set(value)
    if keys != TOOL_SPEC_KEYS:
        missing = sorted(TOOL_SPEC_KEYS - keys)
        extra = sorted(keys - TOOL_SPEC_KEYS)
        raise ToolOrchestrationError(
            f"ToolSpec fields 不匹配，missing={missing}, extra={extra}"
        )
    route = value.get("route")
    if route not in {"reuse", "force_codegen"}:
        raise ToolOrchestrationError("ToolSpec.route 只允许 reuse 或 force_codegen")
    if expected_route is not None and route != expected_route:
        raise ToolOrchestrationError(
            f"ToolSpec.route 必须是本轮约定的 {expected_route}"
        )
    expected = contact_tool_spec(route)
    for field in TOOL_SPEC_KEYS - {"route"}:
        if value.get(field) != expected[field]:
            raise ToolOrchestrationError(
                f"第一版 ToolSpec.{field} 必须等于已验证 contact contract"
            )
    return expected


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _role(policy_name: Any) -> str:
    normalized = str(policy_name or "").casefold()
    if normalized == "act":
        return "policy_under_evaluation"
    if normalized == "expert":
        return "expert_validation"
    return "validation_control"


def _result_projection(result: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "tool": result.get("tool"),
        "value": result.get("value"),
        "unit": result.get("unit"),
        "passed": result.get("passed"),
        "evidence_steps": list(result.get("evidence_steps", [])),
        "details": dict(result.get("details", {})),
    }
    if result.get("tool_sha256"):
        projected["tool_sha256"] = result["tool_sha256"]
    return projected


def _discover_episodes(
    child_run_dir: Path,
    reference_tool: str,
) -> list[dict[str, Any]]:
    telemetry_root = child_run_dir / "evaluation/telemetry"
    episodes: list[dict[str, Any]] = []
    for metadata_path in sorted(telemetry_root.glob("*/episode_*/episode.json")):
        episode_dir = metadata_path.parent
        try:
            trajectory = TrajectoryView(episode_dir)
            reference = TOOL_CATALOG[reference_tool]["function"](trajectory)
        except Exception as exc:
            raise ToolOrchestrationError(
                f"无法加载 telemetry episode {episode_dir}: {exc}"
            ) from exc
        if trajectory.metadata.get("error") is not None:
            raise ToolOrchestrationError(
                f"不接受带 error 的 telemetry episode: {episode_dir}"
            )
        if (
            trajectory.metadata.get("task_name") != "beat_block_hammer"
            or trajectory.schema.get("task_name") != "beat_block_hammer"
        ):
            raise ToolOrchestrationError(
                f"metadata/schema task 不匹配: {episode_dir}"
            )
        episodes.append(
            {
                "episode_dir_path": episode_dir,
                "policy_name": trajectory.metadata.get("policy_name"),
                "seed": trajectory.metadata.get("seed"),
                "role": _role(trajectory.metadata.get("policy_name")),
                "reference_result": reference,
            }
        )
    if not episodes:
        raise ToolOrchestrationError(
            f"没有在 {telemetry_root} 下发现完整 telemetry episode"
        )
    episodes.sort(
        key=lambda item: (
            {"policy_under_evaluation": 0, "expert_validation": 1}.get(
                item["role"], 2
            ),
            int(item.get("seed") or -1),
            str(item["episode_dir_path"]),
        )
    )
    return episodes


def _resolve(
    repo_root: Path,
    child_run_dir: Path,
    tool_spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes = _discover_episodes(child_run_dir, tool_spec["reference_tool"])
    reference_values = [
        item["reference_result"].get("value") for item in episodes
    ]
    requirements = tool_spec["validation_requirements"]
    if len(episodes) < int(requirements["min_episodes"]):
        raise ToolOrchestrationError(
            "telemetry episode 数量不足以满足 ToolSpec validation"
        )
    if requirements["distinct_reference_values"] and len(
        set(reference_values)
    ) < 2:
        raise ToolOrchestrationError(
            "telemetry 缺少 ToolGen differential gate 所需的不同 reference 输出"
        )
    required_values = set(requirements["required_reference_values"])
    if not required_values.issubset(set(reference_values)):
        raise ToolOrchestrationError(
            "telemetry 缺少 contact Tool 所需的 reference false/true 对照"
        )
    resolved = {
        "schema_version": 1,
        "tool_spec": tool_spec,
        "resolved_route": tool_spec["route"],
        "resolved_tool_name": (
            tool_spec["reference_tool"]
            if tool_spec["route"] == "reuse"
            else f"generated_{tool_spec['metric']}"
        ),
        "resolved_episodes": [
            {
                "role": item["role"],
                "policy_name": item["policy_name"],
                "seed": item["seed"],
                "reference_value": item["reference_result"].get("value"),
                "episode_dir": _relative(item["episode_dir_path"], repo_root),
            }
            for item in episodes
        ],
    }
    return resolved, episodes


def execute_tool_spec(
    repo_root: str | Path,
    child_run_dir: str | Path,
    output_dir: str | Path,
    tool_spec: dict[str, Any],
    *,
    provider: Any | None = None,
    model: str | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Execute reuse or force-codegen and emit one normalized envelope."""

    repo = Path(repo_root).expanduser().resolve()
    child = Path(child_run_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ToolOrchestrationError(f"tool output directory 已存在: {destination}")
    spec = validate_tool_spec(tool_spec)
    resolved, episodes = _resolve(repo, child, spec)
    destination.mkdir(parents=True)
    _write_json(destination / "tool_spec.json", spec)
    _write_json(destination / "resolved_tool_spec.json", resolved)

    if spec["route"] == "reuse":
        normalized_episodes = [
            {
                "episode_dir": _relative(item["episode_dir_path"], repo),
                "policy_name": item["policy_name"],
                "seed": item["seed"],
                "role": item["role"],
                "result": _result_projection(item["reference_result"]),
            }
            for item in episodes
        ]
        first_result = normalized_episodes[0]["result"]
        execution = {
            "schema_version": 1,
            "status": "passed",
            "route": "reuse",
            "reference_tool": spec["reference_tool"],
            "tool_spec": spec,
            "source": {
                "scope": "trusted_catalog",
                "tool": spec["reference_tool"],
                "reference_tool": spec["reference_tool"],
                "tool_sha256": first_result.get("tool_sha256"),
                "artifact": None,
            },
            "episodes": normalized_episodes,
            "validation": {
                "provider_called": False,
                "catalog_tool_found": True,
                "episode_count": len(normalized_episodes),
                "required_reference_values_observed": True,
            },
            "artifacts": {
                "tool_spec": _relative(destination / "tool_spec.json", repo),
                "resolved_tool_spec": _relative(
                    destination / "resolved_tool_spec.json", repo
                ),
            },
        }
    else:
        if provider is None or not model:
            raise ToolOrchestrationError(
                "force_codegen route 必须提供 provider 和 model"
            )
        generated_dir = destination / "generated"
        try:
            manifest = ToolGenPrototype(repo, provider, model=model).generate(
                spec["question"],
                reference_tool=spec["reference_tool"],
                episode_dirs=[item["episode_dir_path"] for item in episodes],
                output_dir=generated_dir,
                tool_name=resolved["resolved_tool_name"],
                max_attempts=max_attempts,
            )
        except Exception as exc:
            raise ToolOrchestrationError(f"planned ToolGen failed: {exc}") from exc
        raw_results = json.loads(
            (generated_dir / "execution_results.json").read_text(encoding="utf-8")
        )
        normalized_episodes = []
        validation_rows = []
        episode_lookup = {
            str(item["episode_dir_path"].resolve()): item for item in episodes
        }
        for raw in raw_results:
            episode = episode_lookup.get(str(Path(raw["episode_dir"]).resolve()))
            if episode is None:
                raise ToolOrchestrationError(
                    f"ToolGen 返回未知 episode: {raw['episode_dir']}"
                )
            normalized_episodes.append(
                {
                    "episode_dir": _relative(episode["episode_dir_path"], repo),
                    "policy_name": episode["policy_name"],
                    "seed": episode["seed"],
                    "role": episode["role"],
                    "result": _result_projection(raw["generated_result"]),
                }
            )
            validation_rows.append(
                {
                    "episode_dir": _relative(episode["episode_dir_path"], repo),
                    "deterministic": bool(raw.get("deterministic")),
                    "oracle_agreement": bool(raw.get("oracle_agreement")),
                    "artifacts_unchanged": bool(raw.get("artifacts_unchanged")),
                }
            )
        all_gates_passed = all(
            row[gate]
            for row in validation_rows
            for gate in (
                "deterministic",
                "oracle_agreement",
                "artifacts_unchanged",
            )
        )
        if not all_gates_passed:
            raise ToolOrchestrationError("ToolGen result 未通过 deterministic gates")
        execution = {
            "schema_version": 1,
            "status": "passed",
            "route": "force_codegen",
            "reference_tool": spec["reference_tool"],
            "tool_spec": spec,
            "source": {
                "scope": "run_local_generated",
                "tool": manifest["registration"]["tool"],
                "reference_tool": spec["reference_tool"],
                "tool_sha256": manifest["registration"]["tool_sha256"],
                "artifact": _relative(generated_dir / "generated_tool.py", repo),
            },
            "episodes": normalized_episodes,
            "validation": {
                "provider_called": True,
                "successful_attempt": manifest.get("successful_attempt"),
                "all_gates_passed": all_gates_passed,
                "episodes": validation_rows,
            },
            "artifacts": {
                "tool_spec": _relative(destination / "tool_spec.json", repo),
                "resolved_tool_spec": _relative(
                    destination / "resolved_tool_spec.json", repo
                ),
                "toolgen_manifest": _relative(
                    generated_dir / "manifest.json", repo
                ),
                "registration": _relative(
                    generated_dir / "registration.json", repo
                ),
                "execution_results": _relative(
                    generated_dir / "execution_results.json", repo
                ),
                "generated_tool": _relative(
                    generated_dir / "generated_tool.py", repo
                ),
            },
        }

    _write_json(destination / "tool_execution.json", execution)
    execution["artifacts"]["tool_execution"] = _relative(
        destination / "tool_execution.json", repo
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution
