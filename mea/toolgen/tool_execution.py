"""Episode discovery and execution for trusted or generated Rule Tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mea.toolkit.tools import TOOL_CATALOG, TrajectoryView

from .metric_oracle import _metric_semantic_differences
from .prototype import ToolGenPrototype, execute_generated_tool
from .registry import telemetry_schema_compatibility
from .targets import COMPOSITE_TARGETS, evaluate_target_oracle, target_definition
from .tool_contracts import ToolOrchestrationError, validate_tool_spec

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
    if normalized in {"act", "smolvla", "hy-vla", "hyvla"}:
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
    target_metric: str,
    reference_tool: str | None,
    task_name: str,
) -> list[dict[str, Any]]:
    telemetry_root = child_run_dir / "evaluation/telemetry"
    episodes: list[dict[str, Any]] = []
    for metadata_path in sorted(telemetry_root.glob("*/episode_*/episode.json")):
        episode_dir = metadata_path.parent
        try:
            trajectory = TrajectoryView(episode_dir)
            if (
                reference_tool in TOOL_CATALOG
                and target_metric == reference_tool
            ):
                # Preserve Trusted Tool provenance for reuse.  The generic
                # oracle projection intentionally strips tool/version/hash for
                # differential comparison, but those fields belong in the
                # user-facing reuse result and source envelope.
                oracle_result = TOOL_CATALOG[reference_tool]["function"](
                    trajectory
                )
            else:
                oracle_result = evaluate_target_oracle(
                    target_metric,
                    trajectory,
                    reference_tool=reference_tool,
                )
        except Exception as exc:
            raise ToolOrchestrationError(
                f"无法加载 telemetry episode {episode_dir}: {exc}"
            ) from exc
        if trajectory.metadata.get("error") is not None:
            raise ToolOrchestrationError(
                f"不接受带 error 的 telemetry episode: {episode_dir}"
            )
        if (
            trajectory.metadata.get("task_name") != task_name
            or trajectory.schema.get("task_name") != task_name
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
                "oracle_result": oracle_result,
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
    episodes = _discover_episodes(
        child_run_dir,
        tool_spec["metric"],
        tool_spec["reference_tool"],
        tool_spec["task_name"],
    )
    reference_values = [
        item["oracle_result"].get("value") for item in episodes
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
                "oracle_value": item["oracle_result"].get("value"),
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
    _precreated_destination: bool = False,
) -> dict[str, Any]:
    """Execute reuse or force-codegen and emit one normalized envelope."""

    repo = Path(repo_root).expanduser().resolve()
    child = Path(child_run_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not _precreated_destination:
        raise ToolOrchestrationError(
            f"tool output directory 已存在: {destination}"
        )
    spec = validate_tool_spec(tool_spec)
    if _precreated_destination and not destination.is_dir():
        raise ToolOrchestrationError(
            f"precreated tool output directory does not exist: {destination}"
        )
    resolved, episodes = _resolve(repo, child, spec)
    destination.mkdir(parents=True, exist_ok=_precreated_destination)
    _write_json(destination / "tool_spec.json", spec)
    _write_json(destination / "resolved_tool_spec.json", resolved)

    if spec["route"] == "reuse":
        if spec["reference_tool"] not in TOOL_CATALOG:
            raise ToolOrchestrationError(
                "reuse route 必须解析到 Trusted catalog tool"
            )
        normalized_episodes = [
            {
                "episode_dir": _relative(item["episode_dir_path"], repo),
                "policy_name": item["policy_name"],
                "seed": item["seed"],
                "role": item["role"],
                "result": _result_projection(item["oracle_result"]),
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
                target_metric=spec["metric"],
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
            raise ToolOrchestrationError(
                "ToolGen result 未通过 deterministic gates"
            )
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
                "property_validation": _relative(
                    generated_dir / "property_validation.json", repo
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


def _same_projection(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Use the same semantic oracle contract as initial Tool validation.

    Independent typed oracles may attach richer diagnostic details than the
    provider-authored Tool.  Only value/unit/pass/evidence and the declared
    operation/reason define executable equivalence; auxiliary diagnostics do
    not change the metric result.
    """

    return not _metric_semantic_differences(left, right)


def _execute_registry_match(
    repo_root: Path,
    destination: Path,
    spec: dict[str, Any],
    match: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    route: str,
    source_scope: str,
    registration_id_field: str,
    registry_artifact_key: str,
    oracle_evaluator: Any | None = None,
) -> dict[str, Any]:
    """Execute one exact generated-code registry match without a provider."""

    registration = match["registration"]
    source_path = match["source_path"]
    # Reviewed lookup returns the exact text whose hash was checked.  Run-local
    # matches retain their legacy path-backed behavior.
    source = match.get("source")
    if not isinstance(source, str):
        source = source_path.read_text(encoding="utf-8")
    current_schema = telemetry_schema_compatibility(
        [episode["episode_dir_path"] for episode in episodes],
        required_signals=spec.get("required_signals", []),
    )
    registered_schema_hash = registration.get("telemetry_schema_sha256")
    if registered_schema_hash is None:
        registered_schema_hash = registration.get(
            "telemetry_schema_compatibility", {}
        ).get("compatibility_sha256")
    if current_schema["compatibility_sha256"] != registered_schema_hash:
        raise ToolOrchestrationError(
            f"{route} Tool current telemetry schema no longer matches"
        )
    normalized_episodes: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for episode in episodes:
        generated = execute_generated_tool(
            source,
            episode["episode_dir_path"],
            tool_name=registration["tool_id"],
        )
        repeated = execute_generated_tool(
            source,
            episode["episode_dir_path"],
            tool_name=registration["tool_id"],
        )
        trajectory = TrajectoryView(episode["episode_dir_path"])
        expected = (
            oracle_evaluator(trajectory)
            if oracle_evaluator is not None
            else evaluate_target_oracle(
                spec["metric"],
                trajectory,
                reference_tool=spec["reference_tool"],
            )
        )
        agreement = _same_projection(generated, expected)
        deterministic = generated == repeated
        if not deterministic or not agreement:
            raise ToolOrchestrationError(
                f"{route} Tool failed deterministic/oracle revalidation"
            )
        normalized_episodes.append(
            {
                "episode_dir": _relative(episode["episode_dir_path"], repo_root),
                "policy_name": episode["policy_name"],
                "seed": episode["seed"],
                "role": episode["role"],
                "result": _result_projection(generated),
            }
        )
        validation_rows.append(
            {
                "episode_dir": _relative(episode["episode_dir_path"], repo_root),
                "deterministic": deterministic,
                "oracle_agreement": agreement,
            }
        )
    resolved = {
        "schema_version": 1,
        "tool_spec": spec,
        "resolved_route": route,
        "resolved_tool_name": registration["tool_id"],
        registration_id_field: registration["registration_id"],
        "resolved_episodes": [
            {
                "role": item["role"],
                "policy_name": item["policy_name"],
                "seed": item["seed"],
                "oracle_value": item["oracle_result"].get("value"),
                "episode_dir": _relative(item["episode_dir_path"], repo_root),
            }
            for item in episodes
        ],
    }
    _write_json(destination / "tool_spec.json", spec)
    _write_json(destination / "resolved_tool_spec.json", resolved)
    execution = {
        "schema_version": 1,
        "status": "passed",
        "route": route,
        "reference_tool": spec["reference_tool"],
        "tool_spec": spec,
        "source": {
            "scope": source_scope,
            "tool": registration["tool_id"],
            "reference_tool": spec["reference_tool"],
            "tool_sha256": registration["code_sha256"],
            "registration_id": registration["registration_id"],
            "artifact": _relative(source_path, repo_root),
        },
        "episodes": normalized_episodes,
        "validation": {
            "provider_called": False,
            "registry_match": True,
            "integrity_hashes_matched": True,
            "current_telemetry_revalidated": True,
            "current_telemetry_schema_sha256": current_schema[
                "compatibility_sha256"
            ],
            "current_episode_count": len(validation_rows),
            "persistent_registry_scope": route
            == "reviewed_persistent_reuse",
            "all_gates_passed": all(
                item[gate]
                for item in validation_rows
                for gate in ("deterministic", "oracle_agreement")
            ),
            "episodes": validation_rows,
        },
        "artifacts": {
            "tool_spec": _relative(destination / "tool_spec.json", repo_root),
            "resolved_tool_spec": _relative(
                destination / "resolved_tool_spec.json", repo_root
            ),
            "registration": _relative(match["registration_path"], repo_root),
            "generated_tool": _relative(source_path, repo_root),
        },
    }
    if route == "reviewed_persistent_reuse":
        execution["validation"]["review_manifest_approved"] = (
            match.get("review_manifest", {}).get("decision") == "approved"
        )
    execution["artifacts"][registry_artifact_key] = _relative(
        match["registry_dir"] / "index.json", repo_root
    )
    review_path = match.get("review_manifest_path")
    if isinstance(review_path, Path):
        execution["artifacts"]["review_manifest"] = _relative(
            review_path, repo_root
        )
    _write_json(destination / "tool_execution.json", execution)
    execution["artifacts"]["tool_execution"] = _relative(
        destination / "tool_execution.json", repo_root
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution


def _execute_run_local_match(
    repo_root: Path,
    destination: Path,
    spec: dict[str, Any],
    match: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve the established evaluation-local reuse envelope."""

    return _execute_registry_match(
        repo_root,
        destination,
        spec,
        match,
        episodes,
        route="run_local_reuse",
        source_scope="run_local_registry",
        registration_id_field="run_local_registration_id",
        registry_artifact_key="run_local_registry",
    )


def _execute_reviewed_match(
    repo_root: Path,
    destination: Path,
    spec: dict[str, Any],
    match: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reuse explicitly reviewed code while retaining current-data gates."""

    return _execute_registry_match(
        repo_root,
        destination,
        spec,
        match,
        episodes,
        route="reviewed_persistent_reuse",
        source_scope="reviewed_persistent_registry",
        registration_id_field="reviewed_registration_id",
        registry_artifact_key="reviewed_registry",
    )
