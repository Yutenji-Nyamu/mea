"""Shared mechanics for provider-written RoboTwin scene/checker dialects."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .attempts import (
    REPAIR_SUCCESS_SPEC,
    TaskGenerationRecoveryError,
    TaskGenerationStageError,
    run_bounded_task_generation,
)

_RUN_ID_PATTERN = re.compile(r"run_[A-Za-z0-9_]+")


def validate_provider_run_id(
    run_id: Any,
    *,
    error_type: type[Exception] = ValueError,
) -> str:
    """Validate the package/path identifier shared by every provider dialect."""

    value = str(run_id)
    if not _RUN_ID_PATTERN.fullmatch(value):
        raise error_type("run_id must be an importable run_* package name")
    return value


class TextProvider(Protocol):
    def text(self, prompt: str, **kwargs: Any) -> str:
        ...


_FORBIDDEN_AST_NODES = (
    ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda,
    ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
    ast.Delete, ast.Yield, ast.YieldFrom, ast.Await,
)
_FORBIDDEN_NAMES = {
    "__import__", "breakpoint", "builtins", "compile", "delattr", "eval",
    "exec", "getattr", "globals", "importlib", "input", "locals", "open",
    "os", "pathlib", "requests", "setattr", "shutil", "socket",
    "subprocess", "sys", "urllib",
}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _attribute_parts(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return None
    parts.append(cursor.id)
    return tuple(reversed(parts))


def validate_method_ast(
    source: str,
    method_name: str,
    *,
    safe_direct_calls: set[str],
    safe_module_calls: set[tuple[str, ...]],
    safe_method_calls: set[str],
    allowed_private_attributes: set[str],
    error_type: type[Exception],
) -> ast.Module:
    """Validate one provider method under a task-dialect capability boundary."""

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise error_type(f"{method_name} syntax error: {exc}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise error_type(
            f"{method_name} must contain exactly one function definition"
        )
    function = tree.body[0]
    if (
        function.name != method_name
        or function.decorator_list
        or function.args.posonlyargs
        or len(function.args.args) != 1
        or function.args.args[0].arg != "self"
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
        or function.args.defaults
    ):
        raise error_type(
            f"{method_name} must be an undecorated method with only self"
        )
    nodes = list(ast.walk(tree))
    if len(nodes) > 500:
        raise error_type(f"{method_name} exceeds the bounded AST size")
    for node in nodes:
        if isinstance(node, _FORBIDDEN_AST_NODES):
            raise error_type(
                f"{method_name} contains forbidden AST node "
                f"{type(node).__name__}"
            )
        if isinstance(node, ast.Name) and (
            node.id in _FORBIDDEN_NAMES
            or "__" in node.id
            or (node.id.startswith("_") and node.id != "_")
        ):
            raise error_type(f"{method_name} contains forbidden name {node.id!r}")
        if isinstance(node, ast.Attribute) and (
            "__" in node.attr
            or (
                node.attr.startswith("_")
                and node.attr not in allowed_private_attributes
            )
        ):
            raise error_type(
                f"{method_name} contains forbidden attribute {node.attr!r}"
            )
        if isinstance(node, ast.keyword) and node.arg is None:
            raise error_type(
                f"{method_name} may not expand keyword dictionaries"
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in safe_direct_calls:
                    raise error_type(
                        f"{method_name} call {node.func.id!r} is not allowed"
                    )
            elif isinstance(node.func, ast.Attribute):
                parts = _attribute_parts(node.func)
                if (
                    parts not in safe_module_calls
                    and node.func.attr not in safe_method_calls
                ):
                    raise error_type(
                        f"{method_name} call "
                        f"{'.'.join(parts or (node.func.attr,))!r} is not allowed"
                    )
            else:
                raise error_type(f"{method_name} contains an indirect call")
    return tree


def validate_ablation_switches(
    value: Mapping[str, Any] | None, *, error_type: type[Exception]
) -> dict[str, bool]:
    switches = (
        {"rag": True, "visual_self_check": True, "readme_agent": True}
        if value is None else dict(value)
    )
    expected = {"rag", "visual_self_check", "readme_agent"}
    if set(switches) != expected or any(
        not isinstance(switches[key], bool) for key in expected
    ):
        raise error_type(
            "ablation switches must be exactly rag, visual_self_check, "
            "and readme_agent booleans"
        )
    return {key: switches[key] for key in sorted(expected)}


def compose_prompt(
    *,
    core_contract: str,
    rag_context: str,
    repo_root: Path,
    ablation_switches: Mapping[str, Any] | None,
    error_type: type[Exception],
) -> tuple[str, dict[str, Any]]:
    switches = validate_ablation_switches(
        ablation_switches, error_type=error_type
    )
    sections = [core_contract]
    readme_path = repo_root / "mea/taskgen/README.Agent.md"
    if not readme_path.is_file():
        readme_path = Path(__file__).with_name("README.Agent.md")
    if switches["readme_agent"]:
        sections.append(
            "README.AGENT CONTEXT:\n"
            + readme_path.read_text(encoding="utf-8").strip()
        )
    if switches["rag"]:
        sections.append("RETRIEVED ROBOTWIN API AND TASK CONTEXT:\n" + rag_context)
    prompt = "\n\n".join(sections).strip() + "\n"
    return prompt, {
        "switches": switches,
        "components": {
            "core_contract": True,
            "rag_context": switches["rag"],
            "readme_agent_context": switches["readme_agent"],
            "visual_self_check": switches["visual_self_check"],
        },
        "readme_agent_ref": (
            "mea/taskgen/README.Agent.md" if switches["readme_agent"] else None
        ),
        "prompt_sha256": text_sha256(prompt),
    }


def parse_method_pair(
    response: str, *, error_type: type[Exception]
) -> dict[str, str]:
    normalized = response.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3 and lines[0].strip() in {"```", "```json"}:
            normalized = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise error_type("provider response must be one JSON object") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"load_actors", "check_success"}
        or any(not isinstance(value[key], str) for key in value)
    ):
        raise error_type(
            "provider response must contain exactly string fields "
            "load_actors and check_success"
        )
    return value


def retrieve_class_methods(
    source_path: str | Path,
    *,
    class_name: str,
    method_names: tuple[str, ...],
    error_type: type[Exception],
) -> str:
    """Retrieve exact official methods as compact code context for TaskGen."""

    path = Path(source_path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise error_type(f"RAG source is unavailable: {path}") from exc
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise error_type(f"RAG source is invalid Python: {path}") from exc
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise error_type(f"RAG class {class_name!r} is missing: {path}")
    by_name = {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    sections: list[str] = []
    for name in method_names:
        node = by_name.get(name)
        segment = (
            ast.get_source_segment(source, node)
            if node is not None
            else None
        )
        if not segment:
            raise error_type(f"RAG method {name!r} is missing: {path}")
        sections.append(textwrap.dedent(segment).strip())
    return "\n\n".join(sections) + "\n"


def _compatibility_attempt_records(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in summary["attempts"]:
        attempt = dict(raw)
        failure = attempt.get("failure")
        result = attempt.get("result")
        failure_diagnosis = (
            failure.get("diagnosis")
            if isinstance(failure, Mapping)
            and isinstance(failure.get("diagnosis"), Mapping)
            else {}
        )
        accepted = attempt.get("status") == "accepted"
        records.append(
            {
                "attempt": int(attempt["attempt_index"]),
                "status": "validated" if accepted else "validation_failed",
                "diagnosis": (
                    None
                    if accepted
                    else str((failure or {}).get("message") or "validation failed")
                ),
                "provider_metadata": dict(
                    (
                        result.get("provider_metadata")
                        if isinstance(result, Mapping)
                        else None
                    )
                    or failure_diagnosis.get("provider_metadata")
                    or {}
                ),
            }
        )
    return records


def _write_compatibility_attempts(
    destination: Path,
    *,
    attempt_root: Path,
    summary: Mapping[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for record in _compatibility_attempt_records(summary):
        number = int(record["attempt"])
        source = attempt_root / f"attempt_{number:02d}" / "provider_response.txt"
        if not source.is_file():
            raise ValueError(f"provider attempt response is missing: {source}")
        (destination / f"attempt_{number:02d}_response.txt").write_text(
            source.read_text(encoding="utf-8").rstrip("\n") + "\n",
            encoding="utf-8",
        )
    _write_json(destination / "attempts.json", _compatibility_attempt_records(summary))


def _write_codegen_failure_artifacts(
    run_dir: Path,
    *,
    proposal: Mapping[str, Any],
    prompt: str,
    model: str,
    attempt_root: Path,
    summary: Mapping[str, Any],
) -> None:
    if run_dir.exists():
        raise ValueError(f"candidate run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "proposal_prompt.md").write_text(prompt, encoding="utf-8")
    _write_json(run_dir / "proposal.json", dict(proposal))
    _write_compatibility_attempts(
        run_dir / "provider_attempts",
        attempt_root=attempt_root,
        summary=summary,
    )
    final_failure = summary["attempts"][-1].get("failure") or {}
    _write_json(
        run_dir / "failure_manifest.json",
        {
            "schema_version": 1,
            "status": "codegen_validation_failed",
            "run_id": run_dir.name,
            "task_name": proposal["task_name"],
            "model_requested": model,
            "provider_call_count": summary["runtime"]["provider_calls"],
            "local_regeneration_count": summary["regenerations_used"],
            "final_diagnosis": str(
                final_failure.get("message") or "validation failed"
            ),
            "act_rollouts_completed": 0,
        },
    )


def run_provider_codegen(
    *,
    attempt_root: str | Path,
    proposal: Mapping[str, Any],
    prompt: str,
    provider: TextProvider,
    model: str,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
    error_type: type[Exception],
    max_regenerations: int = 1,
    failure_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Use the existing TaskGenerationAttempt controller for provider retries."""

    attempt_path = Path(attempt_root)
    state: dict[str, Any] = {"attempt_root": str(attempt_path)}
    diagnosis: str | None = None
    previous_methods: dict[str, str] | None = None

    def execute_attempt(
        attempt_dir: Path, _index: int, requested_action: str | None
    ) -> dict[str, Any]:
        nonlocal diagnosis, previous_methods
        current_prompt = prompt
        checker_only_repair = bool(
            requested_action == REPAIR_SUCCESS_SPEC
            and diagnosis
            and previous_methods is not None
        )
        if requested_action is not None and diagnosis:
            current_prompt += (
                "\n\nLOCAL VALIDATION FAILED ON THE PREVIOUS RESPONSE:\n"
                + diagnosis
            )
            if checker_only_repair:
                current_prompt += (
                    "\nThe previous load_actors method already passed the "
                    "same-seed simulator setup and render gates. Copy that "
                    "method exactly and repair only check_success; changing "
                    "the accepted scene would invalidate a checker-local "
                    "repair.\n\nPREVIOUS METHOD PAIR:\n"
                    + json.dumps(
                        previous_methods,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                )
            else:
                current_prompt += "\nRegenerate both complete methods once."
            current_prompt += " Do not return a patch or explanation."
        (attempt_dir / "provider_prompt.md").write_text(
            current_prompt, encoding="utf-8"
        )
        _write_json(attempt_dir / "proposal.json", dict(proposal))
        response = provider.text(
            current_prompt,
            model=model,
            system=(
                "Return one strict JSON object containing the two requested "
                "complete Python methods."
            ),
            max_tokens=5000,
            temperature=0.0,
        )
        (attempt_dir / "provider_response.txt").write_text(
            response + "\n", encoding="utf-8"
        )
        provider_metadata = dict(
            getattr(provider, "last_metadata", {}) or {}
        )
        try:
            accepted_previous_methods = (
                dict(previous_methods) if checker_only_repair else None
            )
            raw_methods = parse_method_pair(response, error_type=error_type)
            methods = dict(raw_methods)
            if checker_only_repair:
                assert accepted_previous_methods is not None
                accepted_scene = accepted_previous_methods["load_actors"]
                provider_changed_scene = (
                    methods["load_actors"] != accepted_scene
                )
                methods["load_actors"] = accepted_scene
                _write_json(
                    attempt_dir / "repair_scope.json",
                    {
                        "schema_version": 1,
                        "scope": "checker_only",
                        "preserved_method": "load_actors",
                        "regenerated_method": "check_success",
                        "provider_changed_preserved_method": (
                            provider_changed_scene
                        ),
                        "provider_scene_output_ignored": provider_changed_scene,
                        "authority": (
                            "previous_same_seed_simulator_setup_and_render"
                        ),
                    },
                )
            previous_methods = dict(methods)
            validation = validate(methods)
        except error_type as exc:
            diagnosis = str(exc)
            checker_validation_failure = diagnosis.startswith(
                (
                    "generated checker failed live negative/positive fixtures",
                    "generated checker contradicts TaskSchema success contract",
                )
            )
            raise TaskGenerationStageError(
                "success_spec" if checker_validation_failure else "scene_codegen",
                "invalid_spec"
                if checker_validation_failure
                else "invalid_candidate",
                diagnosis,
                runtime={"provider_calls": 1},
                diagnosis={"provider_metadata": provider_metadata},
            ) from exc
        state.update(
            {
                "methods": methods,
                "validation": validation,
                "response": response,
                "provider_metadata": provider_metadata,
            }
        )
        _write_json(attempt_dir / "static_validation.json", validation)
        return {
            "status": "accepted",
            "provider_metadata": state["provider_metadata"],
            "runtime": {"provider_calls": 1},
        }

    try:
        summary = run_bounded_task_generation(
            attempt_path,
            proposal_identity=proposal,
            execute_attempt=execute_attempt,
            max_regenerations=max_regenerations,
        )
    except TaskGenerationRecoveryError as exc:
        if failure_run_dir is not None:
            _write_codegen_failure_artifacts(
                Path(failure_run_dir),
                proposal=proposal,
                prompt=prompt,
                model=model,
                attempt_root=attempt_path,
                summary=exc.summary,
            )
        raise error_type(str(exc)) from exc
    state["attempt_summary"] = summary
    return state


def write_candidate_artifacts(
    *,
    run_dir: str | Path,
    task_name: str,
    proposal: Mapping[str, Any],
    prompt: str,
    prompt_context: Mapping[str, Any],
    generated: Mapping[str, Any],
    module_source: str,
    model: str,
    metric: str,
    checker_contract: Mapping[str, Any],
    compatibility_attempt_directory: bool = False,
) -> dict[str, Any]:
    """Write the one candidate schema shared by BBH and ClickBell dialects."""

    root = Path(run_dir)
    if root.exists():
        raise ValueError(f"candidate run directory already exists: {root}")
    compile(module_source, str(root / "task.py"), "exec")
    root.mkdir(parents=True, exist_ok=False)
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "task.py").write_text(module_source, encoding="utf-8")
    (root / "proposal_prompt.md").write_text(prompt, encoding="utf-8")
    (root / "provider_response.txt").write_text(
        str(generated["response"]) + "\n", encoding="utf-8"
    )
    _write_json(root / "proposal.json", dict(proposal))
    validation = dict(generated["validation"])
    fixtures = [dict(item) for item in validation["checker_fixtures"]]
    method_provenance = dict(
        validation.get("method_provenance")
        or {
            "load_actors": "provider_generated",
            "check_success": "provider_generated",
        }
    )
    official_reused_methods = list(
        validation.get("official_reused_methods") or []
    )
    _write_json(root / "checker_fixtures.json", fixtures)
    summary = dict(generated["attempt_summary"])
    if compatibility_attempt_directory:
        _write_compatibility_attempts(
            root / "provider_attempts",
            attempt_root=Path(str(generated["attempt_root"])),
            summary=summary,
        )
    else:
        _write_json(root / "provider_attempts.json", summary)
    run_id = root.name
    manifest = {
        "schema_version": 1,
        "status": "fixture_validated_candidate_not_production_accepted",
        "run_id": run_id,
        "task_name": task_name,
        "task_module": f"mea.generated_tasks.{run_id}.task",
        "proposal_sha256": _canonical_sha256(proposal),
        "module_sha256": _file_sha256(root / "task.py"),
        "scene_method_sha256": validation["scene_sha256"],
        "success_method_sha256": validation["success_sha256"],
        "codegen_provenance": {
            "source_kind": "provider_response_python",
            "provider_called": True,
            "generated_by_model": True,
            "method_provenance": method_provenance,
            "official_reused_methods": official_reused_methods,
            "model_requested": model,
            "provider_metadata": dict(generated["provider_metadata"]),
            "provider_call_count": summary["runtime"]["provider_calls"],
            "local_regeneration_count": summary["regenerations_used"],
            "local_regeneration_limit": summary["max_regenerations"],
            "restricted_success_spec_compiler_used": False,
            "ast_policy": validation["policy"],
            "taskgen_ablation_switches": dict(prompt_context["switches"]),
            "prompt_components": dict(prompt_context["components"]),
            "prompt_sha256": prompt_context["prompt_sha256"],
            "readme_agent_ref": prompt_context["readme_agent_ref"],
        },
        "checker_contract": {
            **dict(checker_contract),
            "metric": metric,
            "authority": (
                "official_task_method_reused"
                if "check_success" in official_reused_methods
                else "llm_generated_python_ast_validated"
            ),
            "official_success": (
                "check_success" in official_reused_methods
            ),
            "fixture_count": len(fixtures),
            "fixture_pass_count": sum(1 for item in fixtures if item["passed"]),
        },
        "live_boundary": {
            "act_rollouts_completed": 0,
            "expert_or_simulator_probes_completed": 0,
            "production_accepted": False,
            "candidate_task_module_is_importable": True,
        },
    }
    _write_json(root / "candidate_manifest.json", manifest)
    return manifest


__all__ = [
    "TextProvider", "compose_prompt", "parse_method_pair",
    "retrieve_class_methods",
    "run_provider_codegen", "text_sha256", "validate_ablation_switches",
    "validate_method_ast", "validate_provider_run_id",
    "write_candidate_artifacts",
]
