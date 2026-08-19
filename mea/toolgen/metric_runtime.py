"""Generate, validate, execute, and register one MetricSpec Rule Tool."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mea.providers.json_response import extract_json_response
from mea.toolkit.tools import TrajectoryView

from .metric_codegen import (
    _provider_codegen_prompt,
    _validate_derived_signal_access,
    build_task_code_context,
    compile_metric_spec_source,
)
from .metric_evaluation import evaluate_metric_spec
from .metric_oracle import (
    _metric_semantic_differences,
    _metric_semantic_projection,
    _semantic_review_prompt,
    _validate_external_oracle_result,
    _validate_semantic_review,
)
from .metric_schema import (
    MetricSpecError,
    _CORE_ARTIFACTS,
    _canonical,
    _write_json,
    metric_spec_tool_spec,
    validate_metric_spec,
)

def _validate_metric_source(
    *,
    source_text: str,
    metric: str,
    spec: Mapping[str, Any],
    episodes: list[Path],
    trajectories: list[TrajectoryView],
    oracle_evaluator: Callable[[TrajectoryView], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Run static, determinism, semantic, and artifact-preservation gates."""

    from mea.toolgen.prototype import (
        ToolGenError,
        execute_generated_tool,
        validate_generated_tool,
    )

    try:
        validate_generated_tool(source_text)
        if spec["operation"] == "derived_observable":
            _validate_derived_signal_access(source_text, spec)
    except ToolGenError as exc:
        raise MetricSpecError(f"generated Python failed the static gate: {exc}") from exc
    except MetricSpecError:
        raise
    rows: list[dict[str, Any]] = []
    values: list[Any] = []
    for episode, trajectory in zip(episodes, trajectories):
        before = {
            name: (episode / name).stat().st_mtime_ns
            for name in _CORE_ARTIFACTS
            if (episode / name).is_file()
        }
        try:
            first = execute_generated_tool(source_text, episode, tool_name=metric)
            second = execute_generated_tool(source_text, episode, tool_name=metric)
        except ToolGenError as exc:
            raise MetricSpecError(
                f"generated Python failed on real telemetry: {exc}"
            ) from exc
        generated = {
            key: first.get(key)
            for key in ("value", "unit", "passed", "evidence_steps", "details")
        }
        if spec["operation"] == "derived_observable":
            oracle = (
                _validate_external_oracle_result(
                    oracle_evaluator(trajectory),
                    spec=spec,
                    trajectory=trajectory,
                )
                if oracle_evaluator is not None
                else _validate_external_oracle_result(
                    generated,
                    spec=spec,
                    trajectory=trajectory,
                )
            )
        else:
            oracle = evaluate_metric_spec(spec, trajectory)
        deterministic = _canonical(first) == _canonical(second)
        semantic_differences = (
            _metric_semantic_differences(generated, oracle)
            if oracle_evaluator is not None
            or spec["operation"] != "derived_observable"
            else []
        )
        oracle_agreement = (
            not semantic_differences
            if oracle_evaluator is not None
            or spec["operation"] != "derived_observable"
            else None
        )
        semantic_contract_valid = not semantic_differences
        after = {
            name: (episode / name).stat().st_mtime_ns
            for name in _CORE_ARTIFACTS
            if (episode / name).is_file()
        }
        if (
            not deterministic
            or not semantic_contract_valid
            or before != after
        ):
            raise MetricSpecError(
                "generated Python validation failed: "
                + _canonical(
                    {
                        "deterministic": deterministic,
                        "oracle_agreement": oracle_agreement,
                        "semantic_contract_valid": semantic_contract_valid,
                        "artifacts_unchanged": before == after,
                        "semantic_differences": semantic_differences,
                        "expected": _metric_semantic_projection(oracle),
                        "actual": _metric_semantic_projection(generated),
                    }
                )
            )
        values.append(oracle.get("value"))
        rows.append(
            {
                "episode_dir": str(episode),
                "policy_name": trajectory.metadata.get("policy_name"),
                "seed": trajectory.metadata.get("seed"),
                "generated_result": first,
                "oracle_projection": oracle,
                "deterministic": deterministic,
                "oracle_agreement": oracle_agreement,
                "semantic_contract_valid": semantic_contract_valid,
                "validation_authority": (
                    "caller_supplied_independent_numeric_oracle"
                    if oracle_evaluator is not None
                    else "toolgen_semantic_review_runtime"
                    if spec["operation"] == "derived_observable"
                    else "typed_metric_spec_interpreter"
                ),
                "artifacts_unchanged": before == after,
            }
        )
    return rows, values


def execute_metric_spec(
    *,
    task_name: str,
    metric: str,
    question: str,
    metric_spec: Mapping[str, Any],
    episode_dirs: Iterable[str | Path],
    output_dir: str | Path,
    fixture_episode_dirs: Iterable[str | Path] = (),
    oracle_evaluator: Callable[[TrajectoryView], Mapping[str, Any]] | None = None,
    task_code_context: Mapping[str, Any] | None = None,
    registry_dir: str | Path | None = None,
    provider: Any | None = None,
    model: str | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Generate, validate, and optionally register one Query-induced Tool."""

    from mea.toolgen.prototype import (
        ToolGenError,
        extract_generated_tool,
        validate_generated_tool,
    )
    from mea.toolgen.registry import (
        find_run_local_registration,
        public_registration_summary,
        register_run_local_tool,
    )

    spec = validate_metric_spec(metric_spec)
    tool_spec = metric_spec_tool_spec(
        task_name=task_name,
        metric=metric,
        question=question,
        metric_spec=spec,
    )
    context = deepcopy(dict(task_code_context)) if task_code_context else None
    if context is not None and context.get("task_name") != task_name:
        raise MetricSpecError("TaskGen code context belongs to a different task")
    episodes = [Path(item).expanduser().resolve() for item in episode_dirs]
    if not episodes or len(set(episodes)) != len(episodes):
        raise MetricSpecError(
            "MetricSpec validation needs at least one unique telemetry episode"
        )
    fixtures = [
        Path(item).expanduser().resolve()
        for item in fixture_episode_dirs
    ]
    if len(set(fixtures)) != len(fixtures) or set(fixtures).intersection(episodes):
        raise MetricSpecError(
            "fixture and live telemetry episode paths must be unique"
        )
    if spec["operation"] == "derived_observable" and (
        bool(fixtures) != (oracle_evaluator is not None)
    ):
        raise MetricSpecError(
            "caller-supplied derived_observable fixtures and numeric oracle "
            "must be provided together"
        )
    if spec["operation"] != "derived_observable" and (
        fixtures or oracle_evaluator is not None
    ):
        raise MetricSpecError(
            "caller-supplied fixtures/oracle are only valid for "
            "derived_observable"
        )
    validation_episodes = [*fixtures, *episodes]
    validation_trajectories = [
        TrajectoryView(path) for path in validation_episodes
    ]
    trajectories = validation_trajectories[len(fixtures) :]
    for trajectory in validation_trajectories:
        if (
            trajectory.metadata.get("task_name") != task_name
            or trajectory.schema.get("task_name") != task_name
        ):
            raise MetricSpecError("MetricSpec episode task/schema does not match")
    if spec["operation"] in {
        "derived_observable",
        "minimum_distance",
        "terminal_minimum_distance",
        "terminal_signal_component",
        "terminal_signal_difference",
    }:
        required_signals = (
            set(spec["required_signals"])
            if spec["operation"] == "derived_observable"
            else {*spec["left_signals"], spec["right_signal"]}
            if spec["operation"] == "terminal_minimum_distance"
            else {spec["left_signal"], spec["right_signal"]}
            if spec["operation"]
            in {"minimum_distance", "terminal_signal_difference"}
            else {spec["signal"]}
        )
        for trajectory in validation_trajectories:
            missing = sorted(required_signals - set(trajectory.trace))
            if missing:
                raise MetricSpecError(
                    f"MetricSpec signals are absent from TaskSchema telemetry: {missing}"
                )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise MetricSpecError(f"MetricSpec output already exists: {destination}")
    destination.mkdir(parents=True)
    _write_json(destination / "metric_spec.json", spec)
    _write_json(destination / "tool_spec.json", tool_spec)
    if context is not None:
        _write_json(destination / "task_code_context.json", context)

    def validate_source(
        source_text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
        validation_rows, values = _validate_metric_source(
            source_text=source_text,
            metric=metric,
            spec=spec,
            episodes=validation_episodes,
            trajectories=validation_trajectories,
            oracle_evaluator=oracle_evaluator,
        )
        return (
            validation_rows[len(fixtures) :],
            validation_rows[: len(fixtures)],
            values,
        )

    registry_match = None
    if registry_dir is not None:
        registry_match = find_run_local_registration(
            registry_dir, tool_spec=tool_spec, episode_dirs=episodes
        )
    semantic_review: dict[str, Any] | None = None
    automatic_derived_validation = (
        spec["operation"] == "derived_observable"
        and oracle_evaluator is None
    )
    if registry_match is not None and automatic_derived_validation:
        stored_review = (
            registry_match["registration"]
            .get("validation", {})
            .get("semantic_review")
        )
        try:
            semantic_review = _validate_semantic_review(stored_review)
        except MetricSpecError:
            # A legacy derived Tool without this review contract is not an
            # exact match for the new mainline validation authority.
            registry_match = None
    if registry_match is not None:
        source_path = registry_match["source_path"]
        route = "semantic_library_reuse"
        generation: dict[str, Any] | None = None
        rows, fixture_rows, values = validate_source(
            source_path.read_text(encoding="utf-8")
        )
    elif provider is not None:
        if not isinstance(model, str) or not model.strip():
            raise MetricSpecError("provider Python codegen requires model")
        attempts_dir = destination / "attempts"
        attempts_dir.mkdir()
        failures: list[dict[str, Any]] = []
        rows = []
        fixture_rows = []
        values = []
        source_text: str | None = None
        previous_candidate_source: str | None = None
        successful_attempt: int | None = None
        attempt_limit = max(1, min(int(max_attempts), 3))
        for attempt_index in range(attempt_limit):
            attempt_dir = attempts_dir / f"attempt_{attempt_index}"
            attempt_dir.mkdir()
            candidate: str | None = None
            prompt = _provider_codegen_prompt(
                repo_root=Path(__file__).resolve().parents[2],
                metric=metric,
                question=question,
                metric_spec=spec,
                trajectory=trajectories[0],
                task_code_context=context,
                previous_error=(
                    failures[-1]["message"] if failures else None
                ),
                previous_source=(
                    previous_candidate_source if failures else None
                ),
            )
            (attempt_dir / "prompt.md").write_text(prompt, encoding="utf-8")
            try:
                failure_stage = "provider_call"
                response = provider.text(
                    prompt,
                    model=model.strip(),
                    system=(
                        "Return exactly one Python code fence containing the "
                        "complete generated_tool(trajectory) function."
                    ),
                    max_tokens=1800,
                    temperature=0.0,
                )
                (attempt_dir / "response.txt").write_text(
                    response + "\n", encoding="utf-8"
                )
                failure_stage = "response_parse"
                candidate = extract_generated_tool(response)
                (attempt_dir / "generated_tool.py").write_text(
                    candidate, encoding="utf-8"
                )
                candidate_review = None
                failure_stage = "generated_source_validation"
                if automatic_derived_validation:
                    validate_generated_tool(candidate)
                    _validate_derived_signal_access(candidate, spec)
                    review_prompt = _semantic_review_prompt(
                        metric=metric,
                        metric_spec=spec,
                        source_text=candidate,
                    )
                    (attempt_dir / "review_prompt.md").write_text(
                        review_prompt,
                        encoding="utf-8",
                    )
                    failure_stage = "semantic_review"
                    review_response = provider.text(
                        review_prompt,
                        model=model.strip(),
                        system="Return only strict ToolGen semantic-review JSON.",
                        max_tokens=500,
                        temperature=0.0,
                    )
                    (attempt_dir / "review_response.txt").write_text(
                        review_response + "\n",
                        encoding="utf-8",
                    )
                    candidate_review = _validate_semantic_review(
                        extract_json_response(review_response)
                    )
                    _write_json(
                        attempt_dir / "semantic_review.json",
                        candidate_review,
                    )
                failure_stage = "live_validation_oracle"
                (
                    candidate_rows,
                    candidate_fixture_rows,
                    candidate_values,
                ) = validate_source(
                    candidate
                )
            except Exception as exc:
                if isinstance(candidate, str) and candidate.strip():
                    previous_candidate_source = candidate
                failure = {
                    "attempt_index": attempt_index,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "failure_stage": failure_stage,
                    "required": (
                        "The complete generated_tool must satisfy the typed "
                        "MetricSpec, static checks, and live telemetry oracle."
                    ),
                    "provider": deepcopy(
                        dict(getattr(provider, "last_metadata", {}))
                    ),
                }
                failures.append(failure)
                _write_json(
                    attempt_dir / "validation.json",
                    {"valid": False, **failure},
                )
                continue
            source_text = candidate
            rows = candidate_rows
            fixture_rows = candidate_fixture_rows
            values = candidate_values
            successful_attempt = attempt_index
            semantic_review = candidate_review
            _write_json(
                attempt_dir / "validation.json",
                {
                    "valid": True,
                    "episode_count": len(rows),
                    "deterministic": True,
                    "oracle_agreement": (
                        True if oracle_evaluator is not None else None
                    ),
                    "semantic_contract_valid": True,
                    "semantic_review": semantic_review,
                    "artifacts_unchanged": True,
                },
            )
            break
        if source_text is None:
            raise MetricSpecError(
                "provider failed to generate a valid Python Tool: "
                + " | ".join(item["message"] for item in failures)
            )
        source_path = destination / "generated_tool.py"
        source_path.write_text(source_text, encoding="utf-8")
        route = "provider_python_codegen"
        generation = {
            "successful_attempt": successful_attempt,
            "attempt_count": len(failures) + 1,
            "failures": failures,
            "model_requested": model.strip(),
            "provider": deepcopy(dict(getattr(provider, "last_metadata", {}))),
            "semantic_review": semantic_review,
        }
    else:
        if spec["operation"] == "derived_observable":
            raise MetricSpecError(
                "derived_observable registry miss requires a provider"
            )
        source_path = destination / "generated_tool.py"
        source_path.write_text(compile_metric_spec_source(spec), encoding="utf-8")
        try:
            validate_generated_tool(source_path.read_text(encoding="utf-8"))
        except ToolGenError as exc:  # pragma: no cover - compiler invariant guard
            raise MetricSpecError(
                f"compiled MetricSpec failed the ToolGen static gate: {exc}"
            ) from exc
        route = "typed_metric_spec_compile"
        generation = None
        rows, fixture_rows, values = validate_source(
            source_path.read_text(encoding="utf-8")
        )
    finite_values = [float(item) for item in values if isinstance(item, (int, float))]
    if any(not math.isfinite(item) for item in finite_values):
        raise MetricSpecError("MetricSpec oracle produced a non-finite value")

    registration = None
    if registry_match is None and registry_dir is not None:
        registration = register_run_local_tool(
            registry_dir,
            tool_spec=tool_spec,
            episode_dirs=episodes,
            source_path=source_path,
            tool_id=metric,
            semantic_review=semantic_review,
        )
    elif registry_match is not None:
        registration = registry_match
    result = {
        "schema_version": 1,
        "status": "passed",
        "route": route,
        "provider_called": generation is not None,
        "generation": generation,
        "source_path": str(source_path),
        "tool_spec": tool_spec,
        "task_code_context_consumed": context is not None,
        "validation_authority": (
            "toolgen_semantic_review_runtime"
            if automatic_derived_validation
            else "caller_supplied_independent_numeric_oracle"
            if spec["operation"] == "derived_observable"
            else "typed_metric_spec_interpreter"
        ),
        "semantic_review": semantic_review,
        "fixtures": fixture_rows,
        "episodes": rows,
        "registration": (
            public_registration_summary(registration) if registration else None
        ),
        "limitations": [
            (
                "provider-defined derived observable over declared telemetry"
                if spec["operation"] == "derived_observable"
                else f"typed semantic oracle: {spec['operation']}"
            ),
            (
                "provider-generated Python"
                if generation is not None
                else "reused validated generated Python"
                if route == "semantic_library_reuse"
                else "provider-free compatibility compiler"
            ),
            (
                "semantic review plus declared-signal, deterministic, finite-"
                "result, evidence-step, and artifact-immutability gates; no "
                "success or reward authority"
                if automatic_derived_validation
                else "output is checked twice against the caller-supplied "
                "numeric oracle on fixtures and live episodes"
                if spec["operation"] == "derived_observable"
                else "output is checked twice against the trusted "
                "interpreter on each live episode; live values need not differ"
            ),
        ],
    }
    _write_json(destination / "execution.json", result)
    return result
