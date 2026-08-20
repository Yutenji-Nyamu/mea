"""Route, reuse, generate, and register one semantic Rule Tool request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .metric_spec import (
    MetricSpecError,
    build_task_code_context,
    evaluate_metric_spec,
    execute_metric_spec,
    metric_spec_tool_spec,
)
from .registry import (
    RunLocalRegistryError,
    find_run_local_registration,
    infer_registry_dir,
    public_registration_summary,
    register_run_local_tool,
)
from .router import ToolRouterError, route_tool_request
from .targets import COMPOSITE_TARGETS
from .tool_contracts import (
    CONTACT_METRIC,
    ToolOrchestrationError,
    _composite_tool_spec,
    _generic_trusted_tool_spec,
    contact_tool_spec,
)
from .tool_execution import (
    _discover_episodes,
    _execute_run_local_match,
    _relative,
    _result_projection,
    _role,
    _write_json,
    execute_tool_spec,
)

def _resolved_spec_from_request(
    tool_request: dict[str, Any],
    resolved_route: str,
) -> dict[str, Any]:
    """Translate a semantic request into the legacy internal execution spec."""

    metric = tool_request["metric"]
    question = tool_request["question"]
    if resolved_route == "reuse":
        if metric == CONTACT_METRIC:
            spec = contact_tool_spec("reuse")
        else:
            spec = _generic_trusted_tool_spec(
                metric, question, tool_request["task_name"]
            )
    elif resolved_route == "force_codegen":
        if metric not in COMPOSITE_TARGETS:
            raise ToolOrchestrationError(
                f"no executable composite ToolSpec for metric: {metric}"
            )
        spec = _composite_tool_spec(
            metric,
            question,
            tool_request["task_name"],
            route="force_codegen",
        )
    else:
        raise ToolOrchestrationError(
            f"automatic Tool route is not executable: {resolved_route}"
        )
    spec["question"] = question
    return spec


def _execute_typed_metric_request(
    repo: Path,
    child_run_dir: Path,
    destination: Path,
    request: dict[str, Any],
    decision: dict[str, Any],
    *,
    registry_root: Path | None,
    task_proposal: dict[str, Any] | None,
    provider: Any | None,
    model: str | None,
    max_attempts: int,
    fixture_episode_dirs: Iterable[str | Path],
    oracle_evaluator: Any | None,
) -> dict[str, Any]:
    """Generate one ToolProposal metric and gate it on the round's telemetry."""

    telemetry_root = child_run_dir / "evaluation/telemetry"
    episode_dirs = [
        path.parent
        for path in sorted(telemetry_root.rglob("episode_*/episode.json"))
    ]
    if not episode_dirs:
        raise ToolOrchestrationError(
            f"no complete telemetry episode found under {telemetry_root}"
        )
    typed_tool_spec = metric_spec_tool_spec(
        task_name=request["task_name"],
        metric=request["metric"],
        question=request["question"],
        metric_spec=request["metric_spec"],
    )
    try:
        context = build_task_code_context(
            child_run_dir,
            task_proposal=task_proposal,
            repo_root=repo,
        )
        raw = execute_metric_spec(
            task_name=request["task_name"],
            metric=request["metric"],
            question=request["question"],
            metric_spec=request["metric_spec"],
            episode_dirs=episode_dirs,
            fixture_episode_dirs=fixture_episode_dirs,
            oracle_evaluator=oracle_evaluator,
            output_dir=destination / "typed_metric_spec",
            task_code_context=context,
            registry_dir=registry_root,
            provider=provider,
            model=model,
            max_attempts=max_attempts,
        )
    except MetricSpecError as exc:
        raise ToolOrchestrationError(f"typed MetricSpec execution failed: {exc}") from exc

    actual_route = str(raw["route"])
    decision["resolved_route"] = actual_route
    if actual_route == "semantic_library_reuse":
        decision["matched_registry"] = "generated_tool_library"
        decision["reason"] = (
            "the semantic key matched a generated Tool and the source was "
            "revalidated on the current telemetry"
        )
    elif actual_route == "provider_python_codegen":
        decision["matched_registry"] = "query_induced_python_tool"
        decision["reason"] = (
            "registry miss triggered provider Python generation; static, "
            "deterministic-runtime, artifact, live-telemetry, and applicable "
            "numeric-oracle gates passed"
        )
    decision["provider_called"] = bool(raw.get("provider_called"))
    _write_json(destination / "route_decision.json", decision)

    normalized_episodes = [
        {
            "episode_dir": _relative(Path(row["episode_dir"]), repo),
            "policy_name": row.get("policy_name"),
            "seed": row.get("seed"),
            "role": _role(row.get("policy_name")),
            "result": _result_projection(row["generated_result"]),
        }
        for row in raw["episodes"]
    ]
    generated_source = Path(
        raw.get("source_path")
        or destination / "typed_metric_spec/generated_tool.py"
    ).expanduser().resolve()
    registration = raw.get("registration")
    validation_rows = [
        *(raw.get("episodes") or []),
        *(raw.get("fixtures") or []),
    ]
    oracle_agreements = [
        row.get("oracle_agreement")
        for row in validation_rows
        if isinstance(row, dict)
    ]
    oracle_agreement = (
        all(value is True for value in oracle_agreements)
        if oracle_agreements
        and all(value is not None for value in oracle_agreements)
        else None
    )
    validation_authority = raw.get("validation_authority")
    # A typed MetricSpec is executed by provider-written Python but checked by
    # the separate trusted interpreter below. That interpreter is an
    # independent numeric oracle just as a caller-owned evaluator is. A
    # free-form derived observable without either authority has runtime gates,
    # but no numeric oracle evidence.
    independent_numeric_oracle = validation_authority in {
        "caller_supplied_independent_numeric_oracle",
        "typed_metric_spec_interpreter",
    }
    execution = {
        "schema_version": 1,
        "status": "passed",
        "requested_route": "auto",
        "route": actual_route,
        "reference_tool": None,
        "tool_spec": raw["tool_spec"],
        "tool_request": request,
        "route_decision": decision,
        "source": {
            "scope": (
                "generated_tool_library"
                if actual_route == "semantic_library_reuse"
                else "run_local_generated"
            ),
            "tool": request["metric"],
            "reference_tool": None,
            "artifact": (
                _relative(generated_source, repo)
                if generated_source.is_file()
                else None
            ),
            "registration_id": (
                registration.get("registration_id")
                if isinstance(registration, dict)
                else None
            ),
        },
        "episodes": normalized_episodes,
        "validation": {
            "provider_called": bool(raw.get("provider_called")),
            "typed_metric_spec": True,
            "python_generated_by_model": actual_route == "provider_python_codegen",
            "authority": validation_authority,
            "oracle_kind": (
                "trusted_typed_metric_interpreter"
                if validation_authority == "typed_metric_spec_interpreter"
                else "caller_supplied_numeric_oracle"
                if validation_authority
                == "caller_supplied_independent_numeric_oracle"
                else "no_numeric_oracle"
            ),
            "task_code_context_consumed": bool(
                raw.get("task_code_context_consumed")
            ),
            "episode_count": len(normalized_episodes),
            "current_telemetry_revalidated": bool(
                raw.get("current_telemetry_revalidated")
            ),
            "current_episode_count": int(
                raw.get("current_episode_count") or 0
            ),
            "validation_gates_passed": True,
            "independent_numeric_oracle": independent_numeric_oracle,
            "oracle_agreement": oracle_agreement,
        },
        "artifacts": {
            "tool_request": _relative(destination / "tool_request.json", repo),
            "catalog_snapshot": _relative(
                destination / "catalog_snapshot.json", repo
            ),
            "route_decision": _relative(
                destination / "route_decision.json", repo
            ),
            "metric_spec_execution": _relative(
                destination / "typed_metric_spec/execution.json", repo
            ),
        },
    }
    _write_json(destination / "resolved_tool_spec.json", execution["tool_spec"])
    execution["artifacts"]["resolved_tool_spec"] = _relative(
        destination / "resolved_tool_spec.json", repo
    )
    execution["artifacts"]["tool_execution"] = _relative(
        destination / "tool_execution.json", repo
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution

def _register_generated_for_evaluation(
    repo: Path,
    child_run_dir: str | Path,
    destination: Path,
    registry_root: Path,
    spec: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    generated_dir = destination / "generated"
    manifest_path = generated_dir / "manifest.json"
    generation_registration_path = generated_dir / "registration.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_registration = json.loads(
        generation_registration_path.read_text(encoding="utf-8")
    )
    episodes = _discover_episodes(
        Path(child_run_dir).expanduser().resolve(),
        spec["metric"],
        spec["reference_tool"],
        spec["task_name"],
    )
    try:
        match = register_run_local_tool(
            registry_root,
            tool_spec=spec,
            episode_dirs=[item["episode_dir_path"] for item in episodes],
            source_path=generated_dir / "generated_tool.py",
            tool_id=generation_registration.get("tool"),
        )
    except RunLocalRegistryError as exc:
        raise ToolOrchestrationError(
            f"failed to register generated Tool for this evaluation: {exc}"
        ) from exc
    registration = match["registration"]
    # Keep the per-generation result readable while the shared library owns
    # the reusable source.
    _write_json(generation_registration_path, registration)
    manifest["registration"] = registration
    _write_json(manifest_path, manifest)
    execution["source"].update(
        {
            "registration_id": registration["registration_id"],
            "registration_scope": "generated_artifact_library",
        }
    )
    execution["run_local_registration"] = public_registration_summary(match)
    execution["artifacts"].update(
        {
            "generated_tool_library": _relative(registry_root / "index.json", repo),
            "generated_tool_spec": _relative(
                match["registration_path"], repo
            ),
            "generated_tool_artifact": _relative(match["source_path"], repo),
        }
    )
    return execution


def execute_tool_request(
    repo_root: str | Path,
    child_run_dir: str | Path,
    output_dir: str | Path,
    tool_request: dict[str, Any],
    *,
    provider: Any | None = None,
    model: str | None = None,
    max_attempts: int = 2,
    run_local_registry_dir: str | Path | None = None,
    task_proposal: dict[str, Any] | None = None,
    fixture_episode_dirs: Iterable[str | Path] = (),
    oracle_evaluator: Any | None = None,
) -> dict[str, Any]:
    """Automatically route and execute one route-free semantic Tool request."""

    repo = Path(repo_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ToolOrchestrationError(
            f"tool output directory already exists: {destination}"
        )
    try:
        routing = route_tool_request(tool_request)
    except ToolRouterError as exc:
        raise ToolOrchestrationError(f"invalid tool_request: {exc}") from exc

    request = routing["tool_request"]
    snapshot = routing["catalog_snapshot"]
    decision = routing["route_decision"]
    if decision["status"] != "resolved":
        destination.mkdir(parents=True)
        decision["provider_called"] = False
        _write_json(destination / "tool_request.json", request)
        _write_json(destination / "catalog_snapshot.json", snapshot)
        _write_json(destination / "route_decision.json", decision)
        raise ToolOrchestrationError(
            "automatic Tool Router found no supported exact metric match"
        )

    destination.mkdir(parents=True)
    _write_json(destination / "tool_request.json", request)
    _write_json(destination / "catalog_snapshot.json", snapshot)
    _write_json(destination / "route_decision.json", decision)
    registry_root = (
        Path(run_local_registry_dir).expanduser().resolve()
        if run_local_registry_dir is not None
        else infer_registry_dir(destination)
    )
    if decision["resolved_route"] in {
        "typed_metric_spec_compile",
        "typed_metric_spec_execute",
    }:
        try:
            return _execute_typed_metric_request(
                repo,
                Path(child_run_dir).expanduser().resolve(),
                destination,
                request,
                decision,
                registry_root=registry_root,
                task_proposal=task_proposal,
                provider=provider,
                model=model,
                max_attempts=max_attempts,
                fixture_episode_dirs=fixture_episode_dirs,
                oracle_evaluator=oracle_evaluator,
            )
        except Exception as exc:
            decision["status"] = "execution_failed"
            attempt_root = destination / "typed_metric_spec/attempts"
            attempt_count = len(
                list(attempt_root.glob("attempt_*/validation.json"))
            )
            decision["provider_called"] = attempt_count > 0
            decision["provider_attempt_count"] = attempt_count
            decision["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _write_json(destination / "route_decision.json", decision)
            raise

    spec = _resolved_spec_from_request(request, decision["resolved_route"])
    run_local_match = None
    registry_episodes: list[dict[str, Any]] | None = None
    if decision["resolved_route"] == "force_codegen":
        if registry_root is not None:
            try:
                registry_episodes = _discover_episodes(
                    Path(child_run_dir).expanduser().resolve(),
                    spec["metric"],
                    spec["reference_tool"],
                    spec["task_name"],
                )
            except ToolOrchestrationError:
                # Preserve the established execution-failure audit.  The normal
                # force-codegen path below will report the telemetry error.
                registry_episodes = None

        if registry_root is not None and registry_episodes is not None:
            try:
                run_local_match = find_run_local_registration(
                    registry_root,
                    tool_spec=spec,
                    episode_dirs=[
                        item["episode_dir_path"] for item in registry_episodes
                    ],
                )
            except RunLocalRegistryError as exc:
                decision["run_local_lookup"] = {
                    "status": "invalid_registry",
                    "message": str(exc),
                }
            if run_local_match is None and "run_local_lookup" not in decision:
                decision["run_local_lookup"] = {
                    "status": "miss",
                    "registry_dir": _relative(registry_root, repo),
                }

        lookup_audit = {
            key: decision[key]
            for key in ("run_local_lookup",)
            if key in decision
        }
        if run_local_match is not None:
            routing = route_tool_request(
                request,
                run_local_registration=public_registration_summary(
                    run_local_match
                ),
            )
            snapshot = routing["catalog_snapshot"]
            decision = routing["route_decision"]
        decision.update(lookup_audit)
        _write_json(destination / "catalog_snapshot.json", snapshot)
        _write_json(destination / "route_decision.json", decision)
    try:
        if run_local_match is not None and registry_episodes is not None:
            execution = _execute_run_local_match(
                repo,
                destination,
                spec,
                run_local_match,
                registry_episodes,
            )
        else:
            execution = execute_tool_spec(
                repo,
                child_run_dir,
                destination,
                spec,
                provider=provider,
                model=model,
                max_attempts=max_attempts,
                _precreated_destination=True,
            )
            if (
                decision["resolved_route"] == "force_codegen"
                and registry_root is not None
            ):
                execution = _register_generated_for_evaluation(
                    repo,
                    child_run_dir,
                    destination,
                    registry_root,
                    spec,
                    execution,
                )
    except Exception as exc:
        decision["status"] = "execution_failed"
        decision["provider_called"] = bool(
            decision["provider_required"] and provider is not None
        )
        decision["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(destination / "route_decision.json", decision)
        raise

    decision["provider_called"] = bool(
        execution.get("validation", {}).get("provider_called")
    )
    _write_json(destination / "route_decision.json", decision)

    resolved_path = destination / "resolved_tool_spec.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["requested_route"] = "auto"
    resolved["route_decision"] = decision
    _write_json(resolved_path, resolved)

    execution["requested_route"] = "auto"
    execution["route"] = decision["resolved_route"]
    execution["tool_request"] = request
    execution["route_decision"] = decision
    execution["artifacts"].update(
        {
            "tool_request": _relative(destination / "tool_request.json", repo),
            "catalog_snapshot": _relative(
                destination / "catalog_snapshot.json", repo
            ),
            "route_decision": _relative(
                destination / "route_decision.json", repo
            ),
        }
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution
