"""Rule Tool materialization services for one executed evaluation round.

The functions in this module own ToolGen work that depends on artifacts from
an already executed TaskGen/policy round.  They deliberately do not know about
the Agent CLI, planner selection, simulator launch, or paper protocols.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.tool_results import episode_tool_results
from mea.planner.experiment_candidate import validate_experiment_candidate
from mea.toolgen import (
    OpenToolRequestAgent,
    compatible_reviewed_tool_requests,
    compatible_run_local_tool_requests,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _episode_path_candidates(
    child_dir: Path,
    value: Any,
) -> list[Path]:
    if not isinstance(value, str) or not value.strip():
        return []
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return [raw.resolve()]
    return [
        (child_dir / raw).resolve(),
        (child_dir / "evaluation" / "telemetry" / raw).resolve(),
    ]


def executed_policy_episode_dirs(child_dir: Path) -> list[Path]:
    """Discover recorded episodes without naming a policy backend.

    Native runtime artifacts are authoritative when available.  The wildcard
    fallback supports existing ACT and SmolVLA bundles, whose historical
    directory labels differ from their actual policy name.
    """

    root = child_dir.expanduser().resolve()
    candidates: list[Path] = []
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "executed policy manifest is invalid"
            ) from exc
        method_runtime = (
            manifest.get("method_runtime")
            if isinstance(manifest, Mapping)
            else None
        )
        rollout = (
            method_runtime.get("rollout")
            if isinstance(method_runtime, Mapping)
            else None
        )
        if isinstance(rollout, Mapping):
            artifacts = rollout.get("artifacts")
            if isinstance(artifacts, Mapping):
                for key in ("telemetry_episode", "episode_dir"):
                    candidates.extend(
                        _episode_path_candidates(root, artifacts.get(key))
                    )
            episode = rollout.get("episode")
            if isinstance(episode, Mapping):
                candidates.extend(
                    _episode_path_candidates(
                        root,
                        episode.get("episode_dir"),
                    )
                )
                semantic = episode.get("semantic_telemetry")
                if isinstance(semantic, Mapping):
                    candidates.extend(
                        _episode_path_candidates(
                            root,
                            semantic.get("episode_dir"),
                        )
                    )
    telemetry_roots = (
        root / "evaluation" / "telemetry",
        root / "telemetry",
    )
    for telemetry_root in telemetry_roots:
        if telemetry_root.is_dir():
            candidates.extend(
                path.parent
                for path in telemetry_root.glob("*/episode_*/schema.json")
            )
            candidates.extend(
                path.parent
                for path in telemetry_root.glob("episode_*/schema.json")
            )
    unique: dict[str, Path] = {}
    for candidate in candidates:
        episode_dir = candidate.resolve()
        try:
            episode_dir.relative_to(root)
        except ValueError:
            # A native manifest may contain an absolute server path.  Accept
            # it only when it resolves back into this child bundle.
            continue
        if (episode_dir / "schema.json").is_file():
            unique[str(episode_dir)] = episode_dir
    return [unique[key] for key in sorted(unique)]


def executed_runtime_task_schema(
    child_dir: Path,
    *,
    task_name: str,
) -> dict[str, Any]:
    episode_dirs = executed_policy_episode_dirs(child_dir)
    if not episode_dirs:
        raise RuntimeError(
            "open ToolGen requires an executed policy telemetry schema"
        )
    schemas = [
        json.loads((path / "schema.json").read_text(encoding="utf-8"))
        for path in episode_dirs
    ]
    canonical = {
        json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for schema in schemas
    }
    if len(canonical) != 1:
        raise RuntimeError(
            "executed policy episodes expose inconsistent telemetry schemas"
        )
    schema = schemas[0]
    if schema.get("task_name") != task_name:
        raise RuntimeError("executed telemetry schema changed the bound task")
    return schema


def materialize_open_world_tool_request(
    repo_root: Path,
    execution_dir: Path,
    *,
    round_plan: Mapping[str, Any],
    child_dir: Path,
    provider: Any,
    toolgen_model: str,
    reviewed_tool_registry: Path | None = None,
) -> dict[str, Any]:
    """Run ToolGen after TaskGen/policy using the schema actually recorded."""

    candidate = round_plan.get("proposal") or round_plan.get(
        "experiment_candidate"
    )
    if not isinstance(candidate, Mapping):
        raise RuntimeError(
            "deferred open ToolGen requires a typed Proposal"
        )
    candidate = validate_experiment_candidate(candidate)
    rule_tool_need = candidate["rule_tool_need"]
    if rule_tool_need is None:
        raise RuntimeError(
            "deferred open Rule ToolGen requires rule_tool_need"
        )
    runtime_schema = executed_runtime_task_schema(
        child_dir,
        task_name=str(candidate["base_task"]),
    )
    episode_dirs = executed_policy_episode_dirs(child_dir)
    run_local_registry = execution_dir.parent.parent / "tool_registry"
    reusable_tool_requests = compatible_run_local_tool_requests(
        run_local_registry,
        task_name=str(candidate["base_task"]),
        episode_dirs=episode_dirs,
        include_derived_observables=True,
    )
    if reviewed_tool_registry is not None:
        reusable_tool_requests.extend(
            compatible_reviewed_tool_requests(
                reviewed_tool_registry,
                task_name=str(candidate["base_task"]),
                episode_dirs=episode_dirs,
            )
        )
    child_manifest = json.loads(
        (child_dir / "manifest.json").read_text(encoding="utf-8")
    )
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    already_measured_metrics = {
        str(result["tool"])
        for episode in trusted.get("episodes", [])
        if isinstance(episode, Mapping)
        for result in (
            episode.get("tool_results")
            if isinstance(episode.get("tool_results"), list)
            else [episode.get("result")]
        )
        if isinstance(result, Mapping)
        and isinstance(result.get("tool"), str)
        and str(result["tool"]).strip()
    }
    outcome_metric = trusted.get("outcome_metric")
    if isinstance(outcome_metric, str) and outcome_metric.strip():
        already_measured_metrics.add(outcome_metric.strip())
    tool_agent = OpenToolRequestAgent(
        repo_root,
        provider,
        model=toolgen_model,
    )
    bundle = tool_agent.propose(
        source_query=str(candidate["source_query"]),
        semantic_concern=str(candidate["semantic_concern"]),
        tool_need=str(rule_tool_need["description"]),
        task_name=str(candidate["base_task"]),
        generated_checker_semantics=bool(
            candidate["checker_need"] is not None
        ),
        runtime_schema=runtime_schema,
        reusable_tool_requests=reusable_tool_requests,
        forbidden_metric_ids=already_measured_metrics,
        proposal=candidate,
        task_artifact_summary=(
            child_manifest.get("task_artifact_summary")
            if isinstance(
                child_manifest.get("task_artifact_summary"),
                Mapping,
            )
            else None
        ),
        derived_observable_oracle_broker=(
            child_manifest.get("derived_observable_oracle_broker")
            if isinstance(
                child_manifest.get("derived_observable_oracle_broker"),
                Mapping,
            )
            else None
        ),
        allow_unsupported=True,
    )
    artifact_dir = execution_dir / "open_tool_request"
    _write_json(artifact_dir / "runtime_schema.json", runtime_schema)
    _write_json(artifact_dir / "tool_request_bundle.json", bundle)
    if tool_agent.last_prompt is not None:
        (artifact_dir / "prompt.md").write_text(
            tool_agent.last_prompt,
            encoding="utf-8",
        )
    for index, response in enumerate(tool_agent.last_responses, start=1):
        (artifact_dir / f"response_{index}.txt").write_text(
            response + "\n",
            encoding="utf-8",
        )
    return bundle


def reuse_bound_child_checker_tool(
    repo_root: Path,
    child_manifest: dict[str, Any],
    output_dir: Path,
    tool_request: dict[str, Any],
) -> dict[str, Any] | None:
    """Reuse an executed generated checker as the exact planned Rule Tool."""

    trusted = child_manifest.get("trusted_tool_evaluation")
    if (
        child_manifest.get("generation_kind")
        not in {
            "provider_scene_checker_codegen",
            "generic_provider_scene_checker_codegen",
        }
        or not isinstance(trusted, Mapping)
        or tool_request.get("schema_version") != 1
        or "metric_spec" in tool_request
        or trusted.get("outcome_metric") != tool_request.get("metric")
        or trusted.get("outcome_authority")
        != "llm_generated_python_ast_validated"
        or not isinstance(trusted.get("tool_retrieval"), Mapping)
        or trusted["tool_retrieval"].get("route")
        != "bound_llm_generated_checker"
    ):
        return None

    metric = str(tool_request["metric"])
    episodes = trusted.get("episodes")
    binding = trusted.get("outcome_binding")
    source_artifact = trusted.get("artifact")
    if (
        tool_request.get("task_name") != child_manifest.get("task_name")
        or not isinstance(binding, Mapping)
        or binding.get("metric") != metric
        or binding.get("authority") != "llm_generated_python_ast_validated"
        or binding.get("task_module") != child_manifest.get("task_module")
        or binding.get("module_sha256")
        != child_manifest.get("candidate_module_sha256")
        or not isinstance(source_artifact, str)
        or not source_artifact.strip()
        or not isinstance(episodes, list)
        or not episodes
        or trusted.get("episode_count") != len(episodes)
    ):
        raise RuntimeError(
            "provider checker metric matched the ToolProposal but its trusted "
            "execution binding is incomplete"
        )
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise RuntimeError("provider checker Tool episode must be an object")
        results = episode_tool_results(episode)
        if len(results) != 1:
            raise RuntimeError(
                "provider checker Tool episode must contain exactly one result"
            )
        result = results[0]
        details = result.get("details")
        if (
            result.get("tool") != metric
            or not isinstance(result.get("value"), bool)
            or not isinstance(result.get("passed"), bool)
            or result.get("passed") != result.get("value")
            or episode.get("role") != "policy_under_evaluation"
            or not isinstance(details, Mapping)
            or details.get("authority")
            != "llm_generated_python_ast_validated"
            or details.get("task_module") != child_manifest.get("task_module")
            or details.get("module_sha256") != binding.get("module_sha256")
        ):
            raise RuntimeError(
                "provider checker ToolResult does not match its "
                "task/module authority"
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    tool_execution_path = output_dir / "tool_execution.json"
    request_path = output_dir / "tool_request.json"
    route_path = output_dir / "route_decision.json"
    route_decision = {
        "schema_version": 1,
        "status": "resolved",
        "matching_policy": "exact_bound_child_metric",
        "requested_route": "auto",
        "resolved_route": "bound_child_trusted_checker",
        "task_name": tool_request.get("task_name"),
        "metric": metric,
        "exact_match": True,
        "matched_registry": "child_task_checker",
        "reference_tool": metric,
        "provider_required": False,
        "provider_called": False,
        "reason": (
            "the executed provider-written task checker already produced this "
            "exact metric for the same bound policy episode"
        ),
    }

    def relative(path: Path) -> str:
        return path.relative_to(repo_root).as_posix()

    evaluation = {
        "schema_version": 1,
        "status": "passed",
        "requested_route": "auto",
        "route": "bound_child_trusted_checker",
        "reference_tool": metric,
        "tool_request": deepcopy(tool_request),
        "route_decision": route_decision,
        "source": {
            "scope": "bound_child_task_checker",
            "artifact": source_artifact,
            "aggregate_artifact": trusted.get("aggregate_artifact"),
            "authority": trusted.get("outcome_authority"),
        },
        "episodes": deepcopy(episodes),
        "validation": {
            "status": "passed",
            "provider_called": False,
            "exact_metric_match": True,
            "episode_count": len(episodes),
            "authority": trusted.get("outcome_authority"),
        },
        "artifacts": {
            "tool_request": relative(request_path),
            "route_decision": relative(route_path),
            "tool_execution": relative(tool_execution_path),
            "source_execution": trusted.get("artifact"),
        },
    }
    _write_json(request_path, tool_request)
    _write_json(route_path, route_decision)
    _write_json(tool_execution_path, evaluation)
    return evaluation


__all__ = [
    "executed_policy_episode_dirs",
    "executed_runtime_task_schema",
    "materialize_open_world_tool_request",
    "reuse_bound_child_checker_tool",
]
