"""Catalog-independent RoboTwin TaskGen orchestration.

The backend consumes one runtime ``ExperimentCandidate`` and a thin task
adapter.  The adapter describes the official task program and exposes the
small simulator-specific hooks needed to validate generated code.  It does
not enumerate aspects, variants, metrics, or planner routes.

Task reuse is exact and semantic. A miss invokes the provider with at most one
local regeneration after static, fixture, render, or expert diagnosis. Policy
execution remains outside this module.
"""

from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.planner.proposal_execution import (
    ProposalExecutionError,
    validate_taskgen_candidate_execution,
)
from mea.planner.semantic_coverage import (
    SemanticCoverageError,
    build_implementation_trace,
)
from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    resolve_robotwin_task_context,
)

from .attempts import TaskGenerationStageError
from .generic_contracts import (
    ExactTaskLookup,
    GenericRoboTwinTaskAdapter,
    GenericTaskGenError,
    GenericTaskGenHooks,
)
from .generic_request import (
    _canonical_sha256,
    _core_prompt,
    _need_description,
    _normalize_adapter,
    _read_generation_context,
    _resolve_repo_file,
    _semantic_field_access_guide,
    _text,
    _validate_exact_match,
    generic_task_semantic_key,
)
from .generic_validation import (
    _GENERIC_READ_ONLY_METHOD_CALLS,
    _official_class,
    _official_task_methods,
    _validate_preservation_feasibility,
    build_generic_task_subclass_module,
    validate_generic_task_methods,
)
from .provider_scene_checker import (
    TextProvider,
    compose_prompt,
    run_provider_codegen,
    text_sha256,
    validate_provider_run_id,
    write_candidate_artifacts,
)
from .semantic_review import (
    CheckerSemanticReviewError,
    review_generated_checker,
    validate_checker_semantic_review_binding,
)


_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _model_names_from_source(
    source_path: Path, *, class_name: str
) -> set[str]:
    _source, class_node = _official_class(
        source_path, class_name=class_name
    )
    result: set[str] = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "modelname"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value
            ):
                result.add(keyword.value.value)
    return result


def _discover_task_documents(
    repo_root: Path, *, task_name: str
) -> tuple[str, ...]:
    candidates = (
        f"mea/knowledge/tasks/{task_name}.md",
        f"envs/{task_name}/README.Agent.md",
        f"envs/{task_name}.README.Agent.md",
    )
    return tuple(
        relative for relative in candidates if (repo_root / relative).is_file()
    )


def _discover_asset_descriptions(
    repo_root: Path,
    *,
    source_path: Path,
    class_name: str,
    task_schema: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    model_names = _model_names_from_source(
        source_path, class_name=class_name
    )
    for actor in (
        task_schema.get("tracked_actors", [])
        if isinstance(task_schema, Mapping)
        else []
    ):
        if isinstance(actor, Mapping):
            scene_name = actor.get("scene_name")
            if isinstance(scene_name, str) and scene_name:
                model_names.add(scene_name)
    paths: list[str] = []
    for model_name in sorted(model_names):
        directory = repo_root / "description/objects_description" / model_name
        if directory.is_dir():
            paths.extend(
                path.relative_to(repo_root).as_posix()
                for path in sorted(directory.glob("*.json"))
                if path.is_file()
            )
    return tuple(paths)


def discover_generic_robotwin_task_identity(
    repo_root: str | Path,
    task_name: str,
    *,
    runtime_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover one executable RoboTwin base without a task-name registry.

    This hook-free identity is the authority boundary shared by routing and
    TaskGen.  Official source makes the task discoverable.  A reviewed
    TaskSchema accelerates TaskGen; when absent, one fresh runtime probe may
    provide the actor/telemetry authority instead.
    """

    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        raise GenericTaskGenError("task_name is not a RoboTwin identifier")
    root = Path(repo_root).expanduser().resolve()
    relative_source = f"envs/{task_name}.py"
    source_path = _resolve_repo_file(
        root, relative_source, label="official task source"
    )
    _official_class(source_path, class_name=task_name)
    try:
        task_context = resolve_robotwin_task_context(
            root,
            task_name,
            runtime_probe=runtime_probe,
        )
    except RoboTwinTaskContextError as exc:
        raise GenericTaskGenError(str(exc)) from exc
    schema = task_context.task_schema
    return {
        "schema_version": 1,
        "task_name": task_name,
        "official_source": relative_source,
        "official_class": task_name,
        "task_schema": deepcopy(schema) if schema is not None else None,
        "task_context": task_context.to_dict(),
        "documentation_paths": list(
            _discover_task_documents(root, task_name=task_name)
        ),
        "asset_paths": list(
            _discover_asset_descriptions(
                root,
                source_path=source_path,
                class_name=task_name,
                task_schema=schema,
            )
        ),
    }


def _candidate_requires_official_core_conjunct(
    candidate: Mapping[str, Any],
) -> bool:
    """Recognize a Proposal that retains the official task goal.

    The generated checker may add an experimental condition, but it must not
    copy the official predicate.  A direct call to the runtime-provided
    untouched method is the only supported composition boundary.
    """

    checker_need = candidate.get("checker_need")
    if not isinstance(checker_need, Mapping):
        return False
    fragments = [str(checker_need.get("description") or "")]
    intent = candidate.get("evaluation_intent")
    if isinstance(intent, Mapping):
        preserved = intent.get("preserved_conditions")
        if isinstance(preserved, list):
            fragments.extend(str(item) for item in preserved)
    text = " ".join(fragments).casefold()
    return any(
        marker in text
        for marker in (
            "official core predicate",
            "official task goal",
            "official goal",
            "official success",
            "official check_success",
            "untouched official",
            "官方任务目标",
            "官方目标",
            "官方成功",
            "官方 check_success",
        )
    )


def load_generic_robotwin_task_adapter(
    repo_root: str | Path,
    task_name: str,
    *,
    checker_fixtures: CheckerFixtureValidator,
    preflight_candidate: PreflightCandidate,
    resolve_metric: ResolveMetric,
    resolve_checker_contract: ResolveCheckerContract,
    prompt_constraints: str = "",
    runtime_probe: Mapping[str, Any] | None = None,
) -> GenericRoboTwinTaskAdapter:
    """Discover a thin adapter for any source/context-backed RoboTwin task.

    The factory has no task-name, concern, template, or metric registry.
    Semantic fixtures and simulator preflight remain explicit injected hooks;
    the factory never marks either gate as passed by default.
    """

    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        raise GenericTaskGenError("task_name is not a RoboTwin identifier")
    if not callable(checker_fixtures):
        raise GenericTaskGenError(
            "checker_fixtures must be an explicit callable"
        )
    if not callable(preflight_candidate):
        raise GenericTaskGenError(
            "preflight_candidate must be an explicit callable"
        )
    if not callable(resolve_metric) or not callable(
        resolve_checker_contract
    ):
        raise GenericTaskGenError(
            "metric and checker contract resolvers must be callable"
        )
    root = Path(repo_root).expanduser().resolve()
    identity = discover_generic_robotwin_task_identity(
        root,
        task_name,
        runtime_probe=runtime_probe,
    )
    relative_source = str(identity["official_source"])
    source_path = _resolve_repo_file(
        root, relative_source, label="official task source"
    )
    raw_schema = identity["task_schema"]
    if not isinstance(raw_schema, Mapping):
        raise GenericTaskGenError(
            "TaskContext has no simulator-authoritative actor/telemetry "
            "schema; run one official reset context probe before TaskGen"
        )
    schema = deepcopy(dict(raw_schema))
    readme = root / "mea/taskgen/README.Agent.md"
    if not readme.is_file():
        raise GenericTaskGenError(
            "TaskGen README.Agent.md is unavailable"
        )

    def validate(
        methods: Mapping[str, str],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        report = validate_generic_task_methods(
            methods,
            official_source=source_path,
            official_class=task_name,
            scene_need=candidate.get("scene_need"),
            required_method_changes={
                "load_actors": candidate.get("scene_need") is not None,
                "check_success": candidate.get("checker_need") is not None,
            },
            require_official_core_conjunct=(
                _candidate_requires_official_core_conjunct(candidate)
            ),
        )
        try:
            raw_fixtures = checker_fixtures(methods, candidate)
        except GenericTaskGenError:
            raise
        except Exception as exc:
            raise GenericTaskGenError(
                "checker fixture hook failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(raw_fixtures, list):
            raise GenericTaskGenError(
                "checker fixture hook must return a list"
            )
        report["checker_fixtures"] = [
            deepcopy(dict(item))
            if isinstance(item, Mapping)
            else item
            for item in raw_fixtures
        ]
        report["checker_fixture_count"] = len(raw_fixtures)
        return report

    module_name = Path(relative_source).with_suffix("").as_posix().replace(
        "/", "."
    )
    hooks = GenericTaskGenHooks(
        validate_methods=validate,
        build_module=lambda methods, candidate: (
            build_generic_task_subclass_module(
                methods,
                official_module=module_name,
                official_class=task_name,
                emit_overrides={
                    "load_actors": (
                        candidate.get("scene_need") is not None
                    ),
                    "check_success": (
                        candidate.get("checker_need") is not None
                    ),
                },
            )
        ),
        preflight_candidate=preflight_candidate,
        resolve_metric=resolve_metric,
        resolve_checker_contract=resolve_checker_contract,
        prompt_constraints=prompt_constraints,
    )
    return GenericRoboTwinTaskAdapter(
        task_name=str(identity["task_name"]),
        official_source=relative_source,
        official_class=str(identity["official_class"]),
        task_schema=schema,
        documentation_paths=tuple(identity["documentation_paths"]),
        asset_paths=tuple(identity["asset_paths"]),
        hooks=hooks,
        task_context=deepcopy(identity["task_context"]),
    )


def _normalize_validation(
    value: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    methods: Mapping[str, str],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenericTaskGenError("method validation hook must return an object")
    report = deepcopy(dict(value))
    if report.get("valid") is not True:
        raise GenericTaskGenError("method validation did not return valid=true")
    policy = report.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        raise GenericTaskGenError("method validation lacks an AST policy id")
    fixtures = report.get("checker_fixtures")
    if not fixtures and isinstance(preflight.get("checker_fixtures"), list):
        fixtures = deepcopy(preflight["checker_fixtures"])
        report["checker_fixtures"] = fixtures
    if (
        not isinstance(fixtures, list)
        or not fixtures
        or any(
            not isinstance(item, Mapping) or item.get("passed") is not True
            for item in fixtures
        )
    ):
        raise GenericTaskGenError(
            "method validation requires passing checker fixtures"
        )
    if preflight.get("render_passed") is not True:
        raise GenericTaskGenError("render preflight did not pass")
    if preflight.get("expert_passed") is not True:
        raise GenericTaskGenError("expert preflight did not pass")
    scene_requested = candidate.get("scene_need") is not None
    expected_scene_state = "changed" if scene_requested else "preserved"
    scene_report = preflight.get("scene_change")
    if isinstance(scene_report, Mapping):
        scene_alignment_passed = bool(
            scene_report.get("passed") is True
            and scene_report.get("expected_state") == expected_scene_state
        )
        scene_alignment_authority = str(
            scene_report.get("authority") or "simulator_scene_comparison"
        )
    elif scene_requested:
        scene_alignment_passed = (
            preflight.get("scene_change_passed") is True
        )
        scene_alignment_authority = "legacy_scene_change_gate"
    else:
        method_provenance = report.get("method_provenance")
        scene_alignment_passed = bool(
            isinstance(method_provenance, Mapping)
            and method_provenance.get("load_actors")
            == "official_reused"
        )
        scene_alignment_authority = "exact_official_load_actors_reuse"
    if not scene_alignment_passed:
        raise GenericTaskGenError(
            "preflight did not verify the expected scene state "
            f"{expected_scene_state!r} relative to the official control"
        )
    report["scene_sha256"] = text_sha256(methods["load_actors"])
    report["success_sha256"] = text_sha256(methods["check_success"])
    if candidate.get("checker_need") is not None:
        try:
            report["checker_semantic_review"] = (
                validate_checker_semantic_review_binding(
                    report.get("checker_semantic_review"),
                    candidate=candidate,
                    checker_sha256=report["success_sha256"],
                )
            )
        except CheckerSemanticReviewError as exc:
            raise GenericTaskGenError(str(exc)) from exc
        report["checker_semantic_review_required"] = True
    else:
        if report.get("checker_semantic_review") is not None:
            raise GenericTaskGenError(
                "official checker reuse must not carry a generated-checker "
                "semantic review"
            )
        report["checker_semantic_review"] = None
        report["checker_semantic_review_required"] = False
    report["checker_fixture_count"] = len(fixtures)
    report["preflight"] = deepcopy(dict(preflight))
    report["scene_alignment"] = {
        "passed": True,
        "expected_state": expected_scene_state,
        "authority": scene_alignment_authority,
    }
    report["model_written_python"] = True
    report["restricted_success_spec_compiler_used"] = False
    try:
        implementation_trace = build_implementation_trace(
            candidate,
            taskgen_validation=report,
        )
    except SemanticCoverageError as exc:
        raise GenericTaskGenError(
            f"invalid semantic implementation trace: {exc}"
        ) from exc
    if implementation_trace is not None:
        report["implementation_trace"] = implementation_trace
        if implementation_trace["repair_required"]:
            raise GenericTaskGenError(
                "generated TaskGen artifact does not implement the direct "
                "EvaluationIntent; regenerate once or explicitly classify "
                "the candidate as diagnostic_proxy/unsupported"
            )
    return report


class GenericRoboTwinTaskGenBackend:
    """Reuse or generate one TaskGen artifact without a task/aspect catalog."""

    def __init__(
        self,
        repo_root: str | Path,
        provider: TextProvider,
        *,
        model: str,
        find_exact: ExactTaskLookup | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = _text(model, field="model")
        self.find_exact = find_exact

    def materialize(
        self,
        candidate: Mapping[str, Any],
        adapter: GenericRoboTwinTaskAdapter,
        *,
        run_id: str,
        max_regenerations: int = 1,
        ablation_switches: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an exact match or one fully preflighted generated artifact."""

        run_id = validate_provider_run_id(
            run_id, error_type=GenericTaskGenError
        )
        try:
            normalized_candidate = validate_experiment_candidate(candidate)
        except ExperimentCandidateError as exc:
            raise GenericTaskGenError(
                f"invalid ExperimentCandidate: {exc}"
            ) from exc
        try:
            normalized_candidate = validate_taskgen_candidate_execution(
                normalized_candidate,
                allowed_change_roots=("load_actors", "check_success"),
            )
        except ProposalExecutionError as exc:
            raise GenericTaskGenError(str(exc)) from exc
        _validate_preservation_feasibility(normalized_candidate)
        if (
            normalized_candidate["scene_need"] is None
            and normalized_candidate["checker_need"] is None
        ):
            raise GenericTaskGenError(
                "generic TaskGen requires a scene or checker need; "
                "Tool-only candidates must bypass TaskGen"
            )
        normalized_adapter = _normalize_adapter(adapter)
        semantic_key = generic_task_semantic_key(
            normalized_candidate, adapter, repo_root=self.repo_root
        )
        semantic_hash = _canonical_sha256(semantic_key)
        lookup_query = {
            "schema_version": 1,
            "semantic_key": deepcopy(semantic_key),
            "semantic_key_sha256": semantic_hash,
        }
        if ablation_switches is not None and (
            not isinstance(ablation_switches, Mapping)
            or any(
                not isinstance(value, bool)
                for value in ablation_switches.values()
            )
        ):
            raise GenericTaskGenError(
                "ablation_switches must map component names to booleans"
            )
        # Every Table 3 arm, including its all-enabled control, must generate
        # independently. Otherwise a prior cell can silently supply its code.
        reuse_allowed = ablation_switches is None
        match = (
            self.find_exact(deepcopy(lookup_query))
            if self.find_exact is not None and reuse_allowed
            else None
        )
        if match is not None:
            try:
                implementation_trace = build_implementation_trace(
                    normalized_candidate
                )
            except SemanticCoverageError as exc:
                raise GenericTaskGenError(
                    f"invalid semantic implementation trace: {exc}"
                ) from exc
            return {
                "schema_version": 1,
                "status": "reused",
                "route": "exact_generated_task_reuse",
                "candidate": normalized_candidate,
                "semantic_key": semantic_key,
                "semantic_key_sha256": semantic_hash,
                "provider_required": False,
                "provider_call_count": 0,
                "implementation_trace": implementation_trace,
                "exact_match": _validate_exact_match(
                    match,
                    semantic_key=semantic_key,
                    semantic_key_sha256=semantic_hash,
                ),
            }

        rag_context = _read_generation_context(
            self.repo_root, adapter=normalized_adapter
        )
        official_source_path = _resolve_repo_file(
            self.repo_root,
            normalized_adapter["official_source"],
            label="adapter official source",
        )
        official_methods = _official_task_methods(
            official_source_path,
            class_name=normalized_adapter["official_class"],
        )
        prompt, prompt_context = compose_prompt(
            core_contract=_core_prompt(
                normalized_candidate,
                normalized_adapter,
                prompt_constraints=adapter.hooks.prompt_constraints,
            ),
            rag_context=rag_context,
            repo_root=self.repo_root,
            ablation_switches=ablation_switches,
            error_type=GenericTaskGenError,
        )
        attempt_root = (
            self.repo_root / "mea/generated_task_attempts" / run_id
        )
        validation_counter = 0
        accepted_module: dict[str, str] = {}

        def validate(methods: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal validation_counter
            validation_counter += 1
            provider_methods = {
                name: str(methods[name])
                for name in ("load_actors", "check_success")
            }
            method_needs = {
                "load_actors": "scene_need",
                "check_success": "checker_need",
            }
            typed_methods = {
                name: (
                    provider_methods[name]
                    if normalized_candidate[need] is not None
                    else official_methods[name]
                )
                for name, need in method_needs.items()
            }
            method_provenance = {
                name: (
                    "provider_generated"
                    if normalized_candidate[need] is not None
                    else "official_reused"
                )
                for name, need in method_needs.items()
            }
            official_reused_methods = [
                name
                for name in ("load_actors", "check_success")
                if method_provenance[name] == "official_reused"
            ]
            checker_semantic_review = None
            try:
                raw_validation = adapter.hooks.validate_methods(
                    typed_methods, normalized_candidate
                )
                module_source = adapter.hooks.build_module(
                    typed_methods, normalized_candidate
                )
                if not isinstance(module_source, str) or not module_source.strip():
                    raise GenericTaskGenError(
                        "module builder must return non-empty Python source"
                    )
                compile(
                    module_source,
                    f"<generic-taskgen-{run_id}-attempt-{validation_counter}>",
                    "exec",
                )
                attempt_dir = (
                    attempt_root / f"attempt_{validation_counter:02d}"
                )
                (attempt_dir / "candidate_task.py").write_text(
                    module_source, encoding="utf-8"
                )
                if normalized_candidate["checker_need"] is not None:
                    try:
                        checker_semantic_review = (
                            review_generated_checker(
                                provider=self.provider,
                                model=self.model,
                                candidate=normalized_candidate,
                                task_context=normalized_adapter[
                                    "task_context"
                                ],
                                method_provenance=method_provenance,
                                generated_scene=typed_methods["load_actors"],
                                official_checker=official_methods[
                                    "check_success"
                                ],
                                generated_checker=typed_methods[
                                    "check_success"
                                ],
                                attempt_dir=attempt_dir,
                            )
                        )
                    except CheckerSemanticReviewError as exc:
                        raise GenericTaskGenError(
                            str(exc),
                            runtime={
                                "semantic_review_provider_calls": (
                                    exc.provider_calls
                                )
                            },
                        ) from exc
                preflight = adapter.hooks.preflight_candidate(
                    attempt_dir, module_source, normalized_candidate
                )
                validation = _normalize_validation(
                    {
                        **deepcopy(dict(raw_validation)),
                        "method_provenance": method_provenance,
                        "official_reused_methods": (
                            official_reused_methods
                        ),
                        "checker_semantic_review": (
                            checker_semantic_review
                        ),
                        "semantic_review_provider_calls": (
                            1 if checker_semantic_review is not None else 0
                        ),
                    },
                    candidate=normalized_candidate,
                    methods=typed_methods,
                    preflight=preflight,
                )
            except GenericTaskGenError as exc:
                if checker_semantic_review is not None:
                    runtime = dict(exc.runtime)
                    runtime["semantic_review_provider_calls"] = 1
                    raise GenericTaskGenError(
                        str(exc),
                        runtime=runtime,
                    ) from exc
                raise
            except Exception as exc:
                raise TaskGenerationStageError(
                    "task_generation",
                    "unclassified_exception",
                    "generic TaskGen validation hook failed: "
                    f"{type(exc).__name__}: {exc}",
                    runtime=(
                        {"provider_calls": 2}
                        if checker_semantic_review is not None
                        else {"provider_calls": 1}
                    ),
                ) from exc
            accepted_module["source"] = module_source
            return validation

        generated = run_provider_codegen(
            attempt_root=attempt_root,
            proposal=normalized_candidate,
            prompt=prompt,
            provider=self.provider,
            model=self.model,
            validate=validate,
            error_type=GenericTaskGenError,
            max_regenerations=max_regenerations,
        )
        if "source" not in accepted_module:
            raise GenericTaskGenError("accepted TaskGen module source is missing")
        metric = _need_description(
            adapter.hooks.resolve_metric(normalized_candidate),
            field="generated metric",
        )
        checker_contract = adapter.hooks.resolve_checker_contract(
            normalized_candidate
        )
        if not isinstance(checker_contract, Mapping):
            raise GenericTaskGenError(
                "checker contract hook must return an object"
            )
        run_dir = self.repo_root / "mea/generated_tasks" / run_id
        manifest = write_candidate_artifacts(
            run_dir=run_dir,
            task_name=normalized_adapter["task_name"],
            proposal=normalized_candidate,
            prompt=prompt,
            prompt_context=prompt_context,
            generated=generated,
            module_source=accepted_module["source"],
            model=self.model,
            metric=metric,
            checker_contract=checker_contract,
        )
        resolution = {
            "schema_version": 1,
            "status": "generated",
            "route": "generic_provider_scene_checker_codegen",
            "candidate": normalized_candidate,
            "adapter": normalized_adapter,
            "semantic_key": semantic_key,
            "semantic_key_sha256": semantic_hash,
            "provider_required": True,
            "provider_call_count": generated["attempt_summary"]["runtime"][
                "provider_calls"
            ],
            "local_regeneration_count": generated["attempt_summary"][
                "regenerations_used"
            ],
            "run_dir": str(run_dir),
            "candidate_manifest": manifest,
            "validation": deepcopy(dict(generated["validation"])),
            "implementation_trace": deepcopy(
                generated["validation"].get("implementation_trace")
            ),
        }
        (run_dir / "generic_taskgen_resolution.json").write_text(
            json.dumps(
                resolution,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return resolution


__all__ = [
    "ExactTaskLookup",
    "GenericRoboTwinTaskAdapter",
    "GenericRoboTwinTaskGenBackend",
    "GenericTaskGenError",
    "GenericTaskGenHooks",
    "build_generic_task_subclass_module",
    "discover_generic_robotwin_task_identity",
    "generic_task_semantic_key",
    "load_generic_robotwin_task_adapter",
    "validate_generic_task_methods",
]
