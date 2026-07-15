"""Bounded ToolGen prototype over offline RoboTwin trajectories."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mea.toolkit.tools import TOOL_CATALOG, TrajectoryView

from .examples import EXAMPLE_CATALOG


class ToolGenError(RuntimeError):
    """Raised when generation or differential validation fails."""


CORE_ARTIFACTS = (
    "episode.json",
    "schema.json",
    "states.csv",
    "semantic_trace.npz",
    "events.jsonl",
)

RESULT_KEYS = ("value", "unit", "passed", "evidence_steps", "details")

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

ALLOWED_TRAJECTORY_ATTRIBUTES = {
    "contact_intervals",
    "events",
    "hammer_block_contacts",
    "metadata",
    "policy_states",
    "schema",
    "success_events",
    "trace",
}

FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "os",
    "pathlib",
    "requests",
    "setattr",
    "socket",
    "subprocess",
    "sys",
    "vars",
    "__import__",
}

FORBIDDEN_ATTRIBUTES = {
    "append",
    "clear",
    "dump",
    "dumps",
    "fromfile",
    "genfromtxt",
    "glob",
    "load",
    "load_library",
    "loadtxt",
    "loads",
    "memmap",
    "mkdir",
    "open",
    "pop",
    "read_bytes",
    "read_text",
    "remove",
    "rename",
    "replace",
    "resolve",
    "rmdir",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "sort",
    "tofile",
    "unlink",
    "update",
    "write_bytes",
    "write_text",
}

FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.For,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)

ALLOWED_NUMPY_CHAINS = {
    ("abs",),
    ("all",),
    ("any",),
    ("argmax",),
    ("argmin",),
    ("asarray",),
    ("clip",),
    ("diff",),
    ("isfinite",),
    ("linalg",),
    ("linalg", "norm"),
    ("max",),
    ("mean",),
    ("min",),
    ("sum",),
    ("where",),
}

ALLOWED_VALUE_ATTRIBUTES = {
    "astype",
    "copy",
    "dtype",
    "get",
    "shape",
    "size",
    "tolist",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(episode_dir: Path) -> dict[str, str]:
    missing = [
        name for name in CORE_ARTIFACTS if not (episode_dir / name).is_file()
    ]
    if missing:
        raise ToolGenError(f"trajectory 缺少 core artifacts: {missing}")
    return {
        name: _sha256(episode_dir / name)
        for name in CORE_ARTIFACTS
    }


def _validate_episode_for_toolgen(episode_dir: Path) -> TrajectoryView:
    _artifact_hashes(episode_dir)
    trajectory = TrajectoryView(episode_dir)
    if trajectory.metadata.get("error") is not None:
        raise ToolGenError("ToolGen 不接受带 episode error 的不完整 trajectory")
    metadata_task = trajectory.metadata.get("task_name")
    schema_task = trajectory.schema.get("task_name")
    if metadata_task != "beat_block_hammer" or schema_task != metadata_task:
        raise ToolGenError(
            "第一版 ToolGen 只接受 metadata/schema 一致的 beat_block_hammer"
        )
    return trajectory


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_for_node(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def extract_generated_tool(response: str) -> str:
    """Extract exactly one complete ``generated_tool`` function."""

    fenced = re.findall(
        r"```(?:python)?\s*(.*?)```",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for candidate in fenced + [response.strip()]:
        source = textwrap.dedent(candidate).strip()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "generated_tool"
        ]
        if len(functions) == 1:
            return _source_for_node(source, functions[0]).strip() + "\n"
    raise ToolGenError("GPT 响应中没有且仅有一个完整 generated_tool()")


def _assignment_targets(target: ast.AST) -> list[ast.AST]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return [item for value in target.elts for item in _assignment_targets(value)]
    return [target]


def _attribute_chain(node: ast.Attribute) -> tuple[ast.AST, tuple[str, ...]]:
    values = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        values.append(current.attr)
        current = current.value
    return current, tuple(reversed(values))


def validate_generated_tool(source: str) -> dict[str, Any]:
    """Validate the narrow, offline-only generated function contract."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolGenError(f"generated tool 语法错误: {exc}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ToolGenError("generated module 只能包含一个 function")
    function = tree.body[0]
    if function.name != "generated_tool":
        raise ToolGenError("function 必须命名为 generated_tool")
    if function.decorator_list:
        raise ToolGenError("generated_tool 不允许 decorator")
    if function.returns is not None:
        raise ToolGenError("generated_tool 不允许 type annotation")
    args = function.args
    if (
        len(args.posonlyargs) + len(args.args) != 1
        or (args.posonlyargs + args.args)[0].arg != "trajectory"
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
        or args.defaults
    ):
        raise ToolGenError("generated_tool 必须只有一个 trajectory 参数")
    if (args.posonlyargs + args.args)[0].annotation is not None:
        raise ToolGenError("trajectory 参数不允许 type annotation")

    function_count = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    if function_count != 1:
        raise ToolGenError("generated_tool 不允许 nested helper function")

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ToolGenError(f"不允许 AST node: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ToolGenError(f"不允许 name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise ToolGenError(f"不允许 attribute: {node.attr}")
            root, chain = _attribute_chain(node)
            if isinstance(root, ast.Name) and root.id == "trajectory":
                if (
                    len(chain) != 1
                    or chain[0] not in ALLOWED_TRAJECTORY_ATTRIBUTES
                ):
                    raise ToolGenError(
                        f"TrajectoryView 未公开 attribute chain: {'.'.join(chain)}"
                    )
            elif isinstance(root, ast.Name) and root.id == "np":
                if chain not in ALLOWED_NUMPY_CHAINS:
                    raise ToolGenError(
                        f"NumPy attribute chain 未列入 allowlist: {'.'.join(chain)}"
                    )
            elif len(chain) != 1 or chain[0] not in ALLOWED_VALUE_ATTRIBUTES:
                raise ToolGenError(
                    f"value attribute chain 未列入 allowlist: {'.'.join(chain)}"
                )
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                for item in _assignment_targets(target):
                    if not isinstance(item, ast.Name):
                        raise ToolGenError("只允许给局部变量赋值")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "generated_tool":
                raise ToolGenError("generated_tool 不允许递归")
            if node.func.id not in SAFE_BUILTINS:
                raise ToolGenError(f"不允许调用 global function: {node.func.id}")

    compile(tree, "<generated_tool>", "exec")
    return {
        "valid": True,
        "function_name": "generated_tool",
        "source_sha256": _source_hash(source),
        "ast_node_count": sum(1 for _ in ast.walk(tree)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _validate_payload(
    raw: Any,
    trajectory: TrajectoryView,
) -> dict[str, Any]:
    value = _jsonable(raw)
    if not isinstance(value, dict):
        raise ToolGenError("generated_tool 返回值必须是 dict")
    missing = [key for key in RESULT_KEYS if key not in value]
    extra = sorted(set(value) - set(RESULT_KEYS))
    if missing or extra:
        raise ToolGenError(f"result contract 不匹配: missing={missing}, extra={extra}")
    if value["unit"] is not None and not isinstance(value["unit"], str):
        raise ToolGenError("unit 必须是 str 或 None")
    if value["passed"] is not None and not isinstance(value["passed"], bool):
        raise ToolGenError("passed 必须是 bool 或 None")
    if not isinstance(value["details"], dict):
        raise ToolGenError("details 必须是 dict")
    steps = value["evidence_steps"]
    if not isinstance(steps, list) or any(
        not isinstance(step, int) or isinstance(step, bool) for step in steps
    ):
        raise ToolGenError("evidence_steps 必须是 int list")
    available_steps = set(
        trajectory.trace["physics_step"].astype(np.int64).tolist()
    )
    unknown = [step for step in steps if step not in available_steps]
    if unknown:
        raise ToolGenError(f"evidence_steps 不在 trace 中: {unknown}")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ToolGenError(f"result 不是有限 JSON value: {exc}") from exc
    return value


def _load_function(source: str):
    validation = validate_generated_tool(source)
    globals_dict = {"__builtins__": SAFE_BUILTINS, "np": np}
    locals_dict: dict[str, Any] = {}
    exec(compile(source, "<generated_tool>", "exec"), globals_dict, locals_dict)
    return locals_dict["generated_tool"], validation


def _execute_generated_tool_in_process(
    source: str,
    episode_dir: str | Path,
    *,
    tool_name: str,
) -> dict[str, Any]:
    """Execute one validated function on a fresh trajectory view."""

    function, validation = _load_function(source)
    trajectory = TrajectoryView(episode_dir)
    payload = _validate_payload(function(trajectory), trajectory)
    return {
        "tool": tool_name,
        "version": 1,
        "generated": True,
        "tool_sha256": validation["source_sha256"],
        **payload,
    }


def execute_generated_tool(
    source: str,
    episode_dir: str | Path,
    *,
    tool_name: str,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Execute generated code in a separate Python process with a timeout."""

    validate_generated_tool(source)
    command = [
        sys.executable,
        "-m",
        "mea.toolgen.worker",
        "--episode-dir",
        str(Path(episode_dir).expanduser().resolve()),
        "--tool-name",
        tool_name,
    ]
    try:
        result = subprocess.run(
            command,
            input=source,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).resolve().parents[2],
            timeout=max(0.1, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolGenError(
            f"generated Tool execution timeout: {timeout_seconds}s"
        ) from exc
    try:
        message = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ToolGenError(
            "generated Tool worker 返回了无效 JSON: "
            + result.stderr[-1000:]
        ) from exc
    if result.returncode != 0 or not message.get("ok"):
        raise ToolGenError(
            "generated Tool worker failed: "
            + str(message.get("error") or result.stderr[-1000:])
        )
    return message["result"]


def _reference_projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": _jsonable(result.get("value")),
        "unit": result.get("unit"),
        "passed": result.get("passed"),
        "evidence_steps": _jsonable(result.get("evidence_steps", [])),
        "details": _jsonable(result.get("details", {})),
    }


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def retrieve_examples(
    user_request: str,
    reference_tool: str,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Select a tiny, deterministic source-level few-shot set."""

    if reference_tool not in EXAMPLE_CATALOG:
        raise ToolGenError(f"没有 standalone example: {reference_tool}")
    text = f"{reference_tool} {user_request}".lower()
    ranked = []
    for name, item in EXAMPLE_CATALOG.items():
        matches = [tag for tag in item["tags"] if str(tag).lower() in text]
        score = len(matches) + (100 if name == reference_tool else 0)
        ranked.append((score, name, matches, item))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "name": name,
            "description": item["description"],
            "matched_tags": matches,
            "source_sha256": _source_hash(inspect.getsource(item["function"])),
            "source": inspect.getsource(item["function"]),
        }
        for _, name, matches, item in ranked[: max(1, int(limit))]
    ]


def _prompt(
    repo_root: Path,
    user_request: str,
    reference_tool: str,
    examples: list[dict[str, Any]],
    diagnostic: str | None,
) -> str:
    contract = (repo_root / "mea/toolgen/README.Agent.md").read_text(
        encoding="utf-8"
    )
    example_text = "\n\n".join(
        f"VERIFIED EXAMPLE {item['name']}:\n```python\n{item['source'].strip()}\n```"
        for item in examples
    )
    repair = (
        f"\nPREVIOUS ATTEMPT FAILED:\n{diagnostic}\nRegenerate the whole function.\n"
        if diagnostic
        else ""
    )
    description = TOOL_CATALOG[reference_tool]["description"]
    return f"""You are the ToolGen code agent for an offline RoboTwin trajectory.

USER REQUEST:
{user_request}

TARGET ORACLE:
- reference tool: {reference_tool}
- semantics: {description}
- this is a force-codegen plumbing test; do not choose reuse.

OUTPUT CONTRACT AND AVAILABLE DATA:
{contract}

RETRIEVED VERIFIED EXAMPLES:
{example_text}
{repair}
Output exactly one Python fenced block containing the complete
`def generated_tool(trajectory):` function and nothing else.
"""


def _verify_examples(
    examples: list[dict[str, Any]],
    episode_dirs: list[Path],
) -> list[dict[str, Any]]:
    validations = []
    for example in examples:
        function = EXAMPLE_CATALOG[example["name"]]["function"]
        source = inspect.getsource(function)
        episode_results = []
        for episode_dir in episode_dirs:
            trajectory = TrajectoryView(episode_dir)
            payload = _validate_payload(function(trajectory), trajectory)
            trusted = TOOL_CATALOG[example["name"]]["function"](
                TrajectoryView(episode_dir)
            )
            expected = _reference_projection(trusted)
            agreement = _equal(payload, expected)
            episode_results.append(
                {"episode_dir": str(episode_dir), "agreement": agreement}
            )
            if not agreement:
                raise ToolGenError(
                    f"few-shot example 与 Trusted Tool 不一致: {example['name']}"
                )
        validations.append(
            {
                "name": example["name"],
                "source_sha256": _source_hash(source),
                "episodes": episode_results,
            }
        )
    return validations


class ToolGenPrototype:
    """Generate, gate, and execute one run-local offline tool."""

    def __init__(self, repo_root: str | Path, provider: Any, *, model: str):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = model

    def generate(
        self,
        user_request: str,
        *,
        reference_tool: str,
        episode_dirs: list[str | Path],
        output_dir: str | Path,
        tool_name: str | None = None,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        if reference_tool not in TOOL_CATALOG:
            raise ToolGenError(f"未知 reference tool: {reference_tool}")
        if reference_tool not in EXAMPLE_CATALOG:
            raise ToolGenError(
                f"第一版 ToolGen 尚无可执行 few-shot example: {reference_tool}"
            )
        episodes = [Path(path).expanduser().resolve() for path in episode_dirs]
        if len(episodes) < 2:
            raise ToolGenError("differential gate 至少需要两个 episode")
        if len(set(episodes)) != len(episodes):
            raise ToolGenError("differential gate 不允许重复 episode path")
        for episode in episodes:
            _validate_episode_for_toolgen(episode)
        oracle_values = [
            _jsonable(
                TOOL_CATALOG[reference_tool]["function"](
                    _validate_episode_for_toolgen(episode)
                ).get("value")
            )
            for episode in episodes
        ]
        unique_oracle_values = {
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            for value in oracle_values
        }
        if len(unique_oracle_values) < 2:
            raise ToolGenError(
                "differential gate 要求 reference oracle 至少有两个不同输出"
            )
        if (
            reference_tool == "hammer_block_contact_ever"
            and set(oracle_values) != {False, True}
        ):
            raise ToolGenError(
                "contact ToolGen 必须同时提供 physical-contact 正例和负例"
            )
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists():
            raise ToolGenError(f"output directory 已存在: {destination}")
        destination.mkdir(parents=True)
        attempts_dir = destination / "attempts"
        attempts_dir.mkdir()
        tool_name = tool_name or f"generated_{reference_tool}"
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", tool_name):
            raise ToolGenError(f"非法 tool_name: {tool_name}")
        max_attempts = max(1, min(int(max_attempts), 3))

        examples = retrieve_examples(user_request, reference_tool)
        retrieval = {
            "mode": "deterministic_source_example_retrieval",
            "reference_tool": reference_tool,
            "selected_examples": [
                {key: value for key, value in item.items() if key != "source"}
                for item in examples
            ],
        }
        _write_json(destination / "request.json", {
            "user_request": user_request,
            "reference_tool": reference_tool,
            "tool_name": tool_name,
            "episode_dirs": [str(path) for path in episodes],
        })
        _write_json(destination / "retrieval.json", retrieval)
        example_validation = _verify_examples(examples, episodes)
        _write_json(destination / "example_validation.json", example_validation)

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "generating",
            "created_at": datetime.now().astimezone().isoformat(),
            "base_commit": _git_head(self.repo_root),
            "generator_source_sha256": _sha256(Path(__file__)),
            "contract_sha256": _sha256(
                self.repo_root / "mea/toolgen/README.Agent.md"
            ),
            "model_requested": self.model,
            "reference_tool": reference_tool,
            "tool_name": tool_name,
            "max_attempts": max_attempts,
            "example_validation": example_validation,
        }
        _write_json(destination / "manifest.json", manifest)
        diagnostic = None
        failures = []

        for attempt_index in range(max_attempts):
            attempt_dir = attempts_dir / f"attempt_{attempt_index}"
            attempt_dir.mkdir()
            prompt = _prompt(
                self.repo_root,
                user_request,
                reference_tool,
                examples,
                diagnostic,
            )
            (attempt_dir / "prompt.md").write_text(prompt, encoding="utf-8")
            try:
                response = self.provider.text(
                    prompt,
                    model=self.model,
                    system=(
                        "Return exactly one Python code fence containing the complete "
                        "generated_tool(trajectory) function."
                    ),
                    max_tokens=1800,
                    temperature=0.0,
                )
            except Exception as exc:
                failure = {
                    "attempt_index": attempt_index,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "provider": dict(
                        getattr(self.provider, "last_metadata", {})
                    ),
                }
                failures.append(failure)
                _write_json(
                    attempt_dir / "validation.json",
                    {"valid": False, **failure},
                )
                diagnostic = json.dumps(
                    failure, ensure_ascii=False, indent=2
                )
                continue
            (attempt_dir / "response.txt").write_text(
                response + "\n", encoding="utf-8"
            )
            provider_metadata = dict(
                getattr(self.provider, "last_metadata", {})
            )
            try:
                source = extract_generated_tool(response)
                (attempt_dir / "generated_tool.py").write_text(
                    source, encoding="utf-8"
                )
                static_validation = validate_generated_tool(source)
                episode_results = []
                for episode in episodes:
                    hashes_before = _artifact_hashes(episode)
                    first = execute_generated_tool(
                        source, episode, tool_name=tool_name
                    )
                    second = execute_generated_tool(
                        source, episode, tool_name=tool_name
                    )
                    deterministic = _equal(first, second)
                    reference = TOOL_CATALOG[reference_tool]["function"](
                        TrajectoryView(episode)
                    )
                    expected = _reference_projection(reference)
                    generated_payload = {
                        key: first.get(key) for key in RESULT_KEYS
                    }
                    agreement = _equal(generated_payload, expected)
                    hashes_after = _artifact_hashes(episode)
                    artifacts_unchanged = hashes_before == hashes_after
                    result = {
                        "episode_dir": str(episode),
                        "policy_name": TrajectoryView(episode).metadata.get(
                            "policy_name"
                        ),
                        "seed": TrajectoryView(episode).metadata.get("seed"),
                        "generated_result": first,
                        "trusted_projection": expected,
                        "deterministic": deterministic,
                        "oracle_agreement": agreement,
                        "artifacts_unchanged": artifacts_unchanged,
                        "artifact_sha256": hashes_after,
                    }
                    episode_results.append(result)
                    if not deterministic:
                        raise ToolGenError("generated Tool 同一输入两次输出不一致")
                    if not agreement:
                        raise ToolGenError(
                            "generated Tool 与 Trusted Tool oracle 不一致: "
                            + json.dumps(result, ensure_ascii=False)[:3000]
                        )
                    if not artifacts_unchanged:
                        raise ToolGenError("generated Tool 修改了 trajectory artifact")

                validation = {
                    "valid": True,
                    "attempt_index": attempt_index,
                    "static": static_validation,
                    "provider": provider_metadata,
                    "episodes": episode_results,
                }
                _write_json(attempt_dir / "validation.json", validation)
                (destination / "generated_tool.py").write_text(
                    source, encoding="utf-8"
                )
                _write_json(destination / "execution_results.json", episode_results)
                registration = {
                    "schema_version": 1,
                    "scope": "run_local",
                    "status": "validated",
                    "tool": tool_name,
                    "source": "generated_tool.py",
                    "tool_sha256": static_validation["source_sha256"],
                    "reference_tool": reference_tool,
                    "validated_episode_count": len(episode_results),
                }
                _write_json(destination / "registration.json", registration)
                manifest.update(
                    {
                        "status": "passed",
                        "completed_at": datetime.now().astimezone().isoformat(),
                        "successful_attempt": attempt_index,
                        "tool_sha256": static_validation["source_sha256"],
                        "provider": provider_metadata,
                        "failures": failures,
                        "registration": registration,
                    }
                )
                _write_json(destination / "manifest.json", manifest)
                return manifest
            except Exception as exc:
                failure = {
                    "attempt_index": attempt_index,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "provider": provider_metadata,
                }
                failures.append(failure)
                _write_json(attempt_dir / "validation.json", {
                    "valid": False,
                    **failure,
                })
                diagnostic = json.dumps(failure, ensure_ascii=False, indent=2)

        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now().astimezone().isoformat(),
                "failures": failures,
            }
        )
        _write_json(destination / "manifest.json", manifest)
        raise ToolGenError(
            f"ToolGen 在 {max_attempts} 次尝试后仍失败: {failures[-1]['message']}"
        )
