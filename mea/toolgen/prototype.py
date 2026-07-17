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
from .targets import (
    BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
    PICKUP_TO_CONTACT_METRIC,
    evaluate_target_oracle,
    target_definition,
)


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

ALLOWED_TRAJECTORY_CHAINS = {
    ("metadata", "get"),
    ("schema", "get"),
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
    ("inf",),
    ("linalg",),
    ("linalg", "norm"),
    ("max",),
    ("mean",),
    ("min",),
    ("nanargmin",),
    ("nanmin",),
    ("sqrt",),
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


_validate_bbh_episode_for_toolgen = _validate_episode_for_toolgen


def _validate_episode_for_toolgen(
    episode_dir: Path,
    *,
    supported_task_names: set[str] | None = None,
) -> TrajectoryView:
    """Load a complete episode and enforce the selected target's task family."""

    if supported_task_names is None or supported_task_names == {"beat_block_hammer"}:
        return _validate_bbh_episode_for_toolgen(episode_dir)
    _artifact_hashes(episode_dir)
    trajectory = TrajectoryView(episode_dir)
    if trajectory.metadata.get("error") is not None:
        raise ToolGenError("ToolGen does not accept an episode with an error")
    metadata_task = trajectory.metadata.get("task_name")
    if (
        ("*" not in supported_task_names and metadata_task not in supported_task_names)
        or trajectory.schema.get("task_name") != metadata_task
    ):
        raise ToolGenError(
            "ToolGen requires matching metadata/schema for a target-supported task"
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
                    not (
                        len(chain) == 1
                        and chain[0] in ALLOWED_TRAJECTORY_ATTRIBUTES
                    )
                    and chain not in ALLOWED_TRAJECTORY_CHAINS
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


def _clone_trajectory(episode_dir: Path) -> TrajectoryView:
    trajectory = TrajectoryView(episode_dir)
    trajectory.trace = {
        key: value.copy() for key, value in trajectory.trace.items()
    }
    trajectory.events = json.loads(json.dumps(trajectory.events))
    trajectory.schema = json.loads(json.dumps(trajectory.schema))
    trajectory.metadata = json.loads(json.dumps(trajectory.metadata))
    trajectory.policy_states = json.loads(json.dumps(trajectory.policy_states))
    return trajectory


def _target_property_scenarios(
    target_metric: str,
    episode_dirs: list[Path],
    *,
    reference_tool: str | None,
) -> list[dict[str, Any]]:
    """Build read-only, in-memory counterexamples for target edge semantics."""

    if target_metric != PICKUP_TO_CONTACT_METRIC:
        return []

    no_pickup = _clone_trajectory(episode_dirs[0])
    no_pickup.trace["hammer_position"][:, 2] = float(
        no_pickup.trace["hammer_position"][0, 2]
    )
    for event in no_pickup.events:
        if (
            event.get("type") == "contact_interval"
            and set(event.get("actors", [])) == {"020_hammer", "box"}
        ):
            event["physical_contact"] = False
            event["first_physical_policy_step"] = None
            event["first_physical_physics_step"] = None
            event["first_physical_simulation_time_seconds"] = None

    numeric_episode = next(
        (
            episode
            for episode in episode_dirs
            if isinstance(
                evaluate_target_oracle(
                    target_metric,
                    TrajectoryView(episode),
                    reference_tool=reference_tool,
                ).get("value"),
                (int, float),
            )
        ),
        None,
    )
    if numeric_episode is None:
        raise ToolGenError("property gate 缺少可构造 early-contact 的 numeric trajectory")
    contact_before_pickup = _clone_trajectory(numeric_episode)
    contacts = [
        event
        for event in contact_before_pickup.events
        if event.get("type") == "contact_interval"
        and set(event.get("actors", [])) == {"020_hammer", "box"}
        and event.get("physical_contact", False)
    ]
    if not contacts:
        raise ToolGenError("property gate numeric trajectory 缺少 strict contact event")
    first = min(
        contacts,
        key=lambda event: event["first_physical_physics_step"],
    )
    first["first_physical_policy_step"] = int(
        contact_before_pickup.trace["policy_step"][0]
    )
    first["first_physical_physics_step"] = int(
        contact_before_pickup.trace["physics_step"][0]
    )
    first["first_physical_simulation_time_seconds"] = float(
        contact_before_pickup.trace["simulation_time_seconds"][0]
    )

    return [
        {"name": "pickup_not_observed", "trajectory": no_pickup},
        {
            "name": "contact_precedes_pickup",
            "trajectory": contact_before_pickup,
        },
    ]


def _execute_on_trajectory(
    source: str,
    trajectory: TrajectoryView,
    *,
    tool_name: str,
) -> dict[str, Any]:
    function, validation = _load_function(source)
    payload = _validate_payload(function(trajectory), trajectory)
    return {
        "tool": tool_name,
        "version": 1,
        "generated": True,
        "tool_sha256": validation["source_sha256"],
        **payload,
    }


def retrieve_examples(
    user_request: str,
    reference_tool: str | None = None,
    *,
    target_metric: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Select a tiny, deterministic source-level few-shot set."""

    target_metric = target_metric or reference_tool
    if not target_metric:
        raise ToolGenError("ToolGen target_metric 不能为空")
    try:
        definition = target_definition(
            target_metric,
            reference_tool=reference_tool,
        )
    except KeyError as exc:
        raise ToolGenError(str(exc)) from exc
    required = list(definition["supporting_examples"])
    missing = [name for name in required if name not in EXAMPLE_CATALOG]
    if missing:
        raise ToolGenError(f"缺少 standalone supporting examples: {missing}")
    text = f"{target_metric} {reference_tool or ''} {user_request}".lower()
    ranked = []
    for name, item in EXAMPLE_CATALOG.items():
        matches = [tag for tag in item["tags"] if str(tag).lower() in text]
        required_rank = required.index(name) if name in required else None
        score = len(matches) + (
            10000 - (required_rank * 100)
            if required_rank is not None
            else 0
        )
        ranked.append((score, name, matches, item))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    effective_limit = max(len(required), max(1, int(limit)))
    return [
        {
            "name": name,
            "description": item["description"],
            "matched_tags": matches,
            "source_sha256": _source_hash(inspect.getsource(item["function"])),
            "source": inspect.getsource(item["function"]),
        }
        for _, name, matches, item in ranked[:effective_limit]
    ]


def _prompt(
    repo_root: Path,
    user_request: str,
    target_metric: str,
    reference_tool: str | None,
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
    definition = target_definition(
        target_metric,
        reference_tool=reference_tool,
    )
    target_contract = json.dumps(
        {
            key: value
            for key, value in definition.items()
            if key not in {"supporting_examples"}
        },
        ensure_ascii=False,
        indent=2,
    )
    if definition["oracle_kind"] == "composite_trusted_tools":
        target_guidance = """- for pickup, use the first trace sample whose hammer Z rise from the initial
  sample is greater than or equal to schema.pickup_height_threshold_m.
- for contact, use only the earliest strict physical hammer-block contact.
- compute duration from trace `simulation_time_seconds` and the contact event's
  `first_physical_simulation_time_seconds`; do not invent a schema key. If a
  timestep is needed, the only valid key is `physics_timestep_seconds`.
- return passed=None because this target is a descriptive measurement.
- preserve the contract's null semantics, details keys, reason strings, and
  ascending unique physics-step evidence exactly.
- set details.duration_physics_steps to null whenever ordering_valid is false;
  for contact-before-pickup, evidence must still be sorted ascending."""
        target_guidance += """
- `.append()` is forbidden by the AST gate. Build evidence without mutation,
  for example `sorted([step for step in [pickup_step, contact_step] if step is
  not None])`."""
    elif target_metric == BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC:
        target_guidance = """- select the active arm from initial bell_position X:
  negative selects left_tcp_position, otherwise right_tcp_position.
- access every recorded array only as `trajectory.trace["field_name"]`;
  `trajectory.semantic_trace` does not exist and must never be used.
- compute Euclidean XY distance to bell_contact_position at every trace row.
- prefer the supported finite reduction `np.argmin(np.where(np.isfinite(d), d, np.inf))`.
- return the finite minimum in meters with passed=None.
- evidence is the physics step at the minimum; details must contain exactly
  active_arm, min_error_physics_step, and simulation_time_seconds."""
    else:
        target_guidance = (
            "- preserve the exact result semantics demonstrated by the "
            "reference example."
        )
    return f"""You are the ToolGen code agent for an offline RoboTwin trajectory.

USER REQUEST:
{user_request}

TARGET ORACLE:
- target metric: {target_metric}
- exact reference tool: {reference_tool or 'none; validated by a private composition oracle'}
- contract: {target_contract}
- generate this target directly; do not call a Trusted Tool and do not choose reuse.
{target_guidance}

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
        reference_tool: str | None = None,
        target_metric: str | None = None,
        episode_dirs: list[str | Path],
        output_dir: str | Path,
        tool_name: str | None = None,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        target_metric = target_metric or reference_tool
        if not target_metric:
            raise ToolGenError("ToolGen target_metric 不能为空")
        try:
            definition = target_definition(
                target_metric,
                reference_tool=reference_tool,
            )
        except KeyError as exc:
            raise ToolGenError(str(exc)) from exc
        episodes = [Path(path).expanduser().resolve() for path in episode_dirs]
        if len(episodes) < 2:
            raise ToolGenError("differential gate 至少需要两个 episode")
        if len(set(episodes)) != len(episodes):
            raise ToolGenError("differential gate 不允许重复 episode path")
        supported_task_names = set(definition.get("supported_task_names", []))
        for episode in episodes:
            _validate_episode_for_toolgen(
                episode, supported_task_names=supported_task_names
            )
        oracle_values = [
            _jsonable(
                evaluate_target_oracle(
                    target_metric,
                    _validate_episode_for_toolgen(
                        episode, supported_task_names=supported_task_names
                    ),
                    reference_tool=reference_tool,
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
            target_metric == "hammer_block_contact_ever"
            and set(oracle_values) != {False, True}
        ):
            raise ToolGenError(
                "contact ToolGen 必须同时提供 physical-contact 正例和负例"
            )
        if target_metric == PICKUP_TO_CONTACT_METRIC:
            numeric_values = [
                value
                for value in oracle_values
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]
            if None not in oracle_values or not numeric_values:
                raise ToolGenError(
                    "composite ToolGen 必须同时提供 null 负例和 numeric 正例"
                )
            if any(not math.isfinite(float(value)) or value < 0 for value in numeric_values):
                raise ToolGenError("composite ToolGen oracle 必须是非负有限秒数")
        property_scenarios = _target_property_scenarios(
            target_metric,
            episodes,
            reference_tool=reference_tool,
        )
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists():
            raise ToolGenError(f"output directory 已存在: {destination}")
        destination.mkdir(parents=True)
        attempts_dir = destination / "attempts"
        attempts_dir.mkdir()
        tool_name = tool_name or f"generated_{target_metric}"
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", tool_name):
            raise ToolGenError(f"非法 tool_name: {tool_name}")
        max_attempts = max(1, min(int(max_attempts), 3))

        examples = retrieve_examples(
            user_request,
            reference_tool,
            target_metric=target_metric,
        )
        retrieval = {
            "mode": "deterministic_source_example_retrieval",
            "target_metric": target_metric,
            "reference_tool": reference_tool,
            "oracle_kind": definition["oracle_kind"],
            "selected_examples": [
                {key: value for key, value in item.items() if key != "source"}
                for item in examples
            ],
        }
        _write_json(destination / "request.json", {
            "user_request": user_request,
            "target_metric": target_metric,
            "reference_tool": reference_tool,
            "tool_name": tool_name,
            "episode_dirs": [str(path) for path in episodes],
        })
        _write_json(destination / "retrieval.json", retrieval)
        example_validation = _verify_examples(examples, episodes)
        _write_json(destination / "example_validation.json", example_validation)

        manifest: dict[str, Any] = {
            "schema_version": 2,
            "status": "generating",
            "created_at": datetime.now().astimezone().isoformat(),
            "base_commit": _git_head(self.repo_root),
            "generator_source_sha256": _sha256(Path(__file__)),
            "contract_sha256": _sha256(
                self.repo_root / "mea/toolgen/README.Agent.md"
            ),
            "model_requested": self.model,
            "target_metric": target_metric,
            "reference_tool": reference_tool,
            "oracle_kind": definition["oracle_kind"],
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
                target_metric,
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
                    expected = evaluate_target_oracle(
                        target_metric,
                        TrajectoryView(episode),
                        reference_tool=reference_tool,
                    )
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
                        "oracle_projection": expected,
                        "trusted_projection": (
                            expected
                            if definition["oracle_kind"] == "exact_trusted_tool"
                            else None
                        ),
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
                            "generated Tool 与 validation oracle 不一致: "
                            + json.dumps(result, ensure_ascii=False)[:3000]
                        )
                    if not artifacts_unchanged:
                        raise ToolGenError("generated Tool 修改了 trajectory artifact")

                property_results = []
                for scenario in property_scenarios:
                    trajectory = scenario["trajectory"]
                    first = _execute_on_trajectory(
                        source,
                        trajectory,
                        tool_name=tool_name,
                    )
                    second = _execute_on_trajectory(
                        source,
                        trajectory,
                        tool_name=tool_name,
                    )
                    generated_payload = {
                        key: first.get(key) for key in RESULT_KEYS
                    }
                    expected = evaluate_target_oracle(
                        target_metric,
                        trajectory,
                        reference_tool=reference_tool,
                    )
                    deterministic = _equal(first, second)
                    agreement = _equal(generated_payload, expected)
                    result = {
                        "scenario": scenario["name"],
                        "generated_result": first,
                        "oracle_projection": expected,
                        "deterministic": deterministic,
                        "oracle_agreement": agreement,
                    }
                    property_results.append(result)
                    if not deterministic or not agreement:
                        raise ToolGenError(
                            "generated Tool 未通过 counterfactual property gate: "
                            + json.dumps(result, ensure_ascii=False)[:3000]
                        )

                validation = {
                    "valid": True,
                    "attempt_index": attempt_index,
                    "static": static_validation,
                    "provider": provider_metadata,
                    "episodes": episode_results,
                    "property_scenarios": property_results,
                }
                _write_json(attempt_dir / "validation.json", validation)
                (destination / "generated_tool.py").write_text(
                    source, encoding="utf-8"
                )
                _write_json(destination / "execution_results.json", episode_results)
                _write_json(
                    destination / "property_validation.json",
                    property_results,
                )
                registration = {
                    "schema_version": 2,
                    "scope": "run_local",
                    "status": "validated",
                    "tool": tool_name,
                    "source": "generated_tool.py",
                    "tool_sha256": static_validation["source_sha256"],
                    "target_metric": target_metric,
                    "reference_tool": reference_tool,
                    "oracle_kind": definition["oracle_kind"],
                    "validated_episode_count": len(episode_results),
                    "validated_property_scenario_count": len(property_results),
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
