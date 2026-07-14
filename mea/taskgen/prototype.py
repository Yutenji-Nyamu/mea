"""Small, auditable TaskGen vertical slice.

The prototype deliberately asks the model to generate the complete
``load_actors`` method.  The surrounding module is a deterministic wrapper so
that the canonical task class and imports remain stable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.retrieval import TaskRetriever


class TaskGenError(RuntimeError):
    """Raised when proposal, generation, or validation fails."""


BLUE_REFERENCE_REQUEST = "把 beat_block_hammer 任务中的红色方块改成蓝色，其他行为保持不变。"

PROTECTED_PATHS = (
    "envs/beat_block_hammer.py",
    "policy/ACT/eval.sh",
    "script/eval_policy.py",
)

ALLOWED_GLOBAL_NAMES = {
    "ValueError",
    "abs",
    "all",
    "any",
    "bool",
    "cos",
    "create_actor",
    "create_box",
    "dict",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "np",
    "pow",
    "print",
    "rand_pose",
    "range",
    "sapien",
    "set",
    "sin",
    "str",
    "sum",
    "tuple",
}

BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}

BANNED_ATTRIBUTE_ROOTS = {
    "builtins",
    "ctypes",
    "importlib",
    "marshal",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"


def extract_json_response(response: str) -> dict[str, Any]:
    """Parse a strict JSON object, tolerating a Markdown JSON fence."""
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(response.strip())
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise TaskGenError("GPT 没有返回可解析的 JSON object")


def _source_for_node(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def extract_load_actors(response: str) -> str:
    """Extract exactly one complete ``load_actors`` method from a response."""
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + [response.strip()]

    for candidate in candidates:
        source = textwrap.dedent(candidate).strip()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_actors"
        ]
        if len(methods) == 1:
            return textwrap.dedent(_source_for_node(source, methods[0])).strip() + "\n"

    raise TaskGenError("GPT 响应中没有且仅有一个完整 load_actors() 方法")


def validate_variant_spec(spec: dict[str, Any], task_name: str) -> dict[str, Any]:
    """Validate and normalize the first prototype's appearance VariantSpec."""
    if spec.get("task_name") != task_name:
        raise TaskGenError(
            f"VariantSpec task_name 必须是 {task_name!r}，实际为 {spec.get('task_name')!r}"
        )

    changes = spec.get("changes")
    if not isinstance(changes, dict):
        raise TaskGenError("VariantSpec.changes 必须是 object")
    block = changes.get("block")
    if not isinstance(block, dict):
        raise TaskGenError("第一版要求 VariantSpec.changes.block")

    color = block.get("color")
    if not isinstance(color, list) or len(color) != 3:
        raise TaskGenError("block.color 必须是三个通道的 list")
    normalized_color = [float(channel) for channel in color]
    if any(channel < 0.0 or channel > 1.0 for channel in normalized_color):
        raise TaskGenError("block.color 通道必须在 [0, 1]")

    normalized = {
        "task_name": task_name,
        "intent": str(spec.get("intent") or "change_object_appearance"),
        "generation_mode": str(spec.get("generation_mode") or "force_codegen"),
        "changes": {
            "block": {
                "position_mode": str(block.get("position_mode") or "official_random"),
                "yaw_mode": str(block.get("yaw_mode") or "official_random"),
                "scale": float(block.get("scale", 1.0)),
                "color": normalized_color,
            }
        },
        "preserve": list(
            spec.get("preserve")
            or [
                "official_position_sampling",
                "official_yaw_sampling",
                "play_once",
                "check_success",
                "checkpoint",
            ]
        ),
    }

    if normalized["changes"]["block"]["position_mode"] not in {
        "official_random",
        "fixed",
    }:
        raise TaskGenError("不支持的 position_mode")
    if normalized["changes"]["block"]["yaw_mode"] not in {
        "official_random",
        "fixed",
    }:
        raise TaskGenError("不支持的 yaw_mode")
    if normalized["changes"]["block"]["scale"] <= 0:
        raise TaskGenError("block.scale 必须为正数")
    return normalized


def _assigned_names(function: ast.FunctionDef) -> set[str]:
    names = {argument.arg for argument in function.args.args}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _literal_create_box_color(function: ast.FunctionDef) -> list[float] | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or _call_name(node) != "create_box":
            continue
        for keyword in node.keywords:
            if keyword.arg != "color":
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, (list, tuple)) and len(value) == 3:
                return [float(channel) for channel in value]
    return None


def validate_load_actors(method_source: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Apply a small static policy to the model-generated complete method."""
    try:
        tree = ast.parse(textwrap.dedent(method_source))
    except SyntaxError as exc:
        raise TaskGenError(f"load_actors 语法错误: {exc}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TaskGenError("响应必须只包含一个函数定义")
    function = tree.body[0]
    if function.name != "load_actors":
        raise TaskGenError("函数名必须是 load_actors")
    if [arg.arg for arg in function.args.args] != ["self"]:
        raise TaskGenError("load_actors 参数必须只有 self")
    if function.decorator_list:
        raise TaskGenError("load_actors 不允许 decorator")

    banned_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Try,
        ast.With,
    )
    for node in ast.walk(function):
        if isinstance(node, banned_nodes):
            raise TaskGenError(f"load_actors 包含禁止语法: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise TaskGenError("禁止访问 dunder attribute")
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in BANNED_ATTRIBUTE_ROOTS:
                raise TaskGenError(f"禁止访问模块: {root.id}")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in BANNED_CALLS or name == "super":
                raise TaskGenError(f"禁止调用: {name}")

    assigned = _assigned_names(function)
    unresolved = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in assigned
        and node.id not in ALLOWED_GLOBAL_NAMES
    }
    if unresolved:
        raise TaskGenError(f"load_actors 使用了未批准的全局名: {sorted(unresolved)}")

    calls = [_call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)]
    required_calls = {
        "add_prohibit_area",
        "append",
        "create_actor",
        "create_box",
        "rand_pose",
        "set_mass",
    }
    missing = sorted(required_calls - set(calls))
    if missing:
        raise TaskGenError(f"load_actors 缺少 BBH 必要调用: {missing}")

    expected_color = spec["changes"]["block"]["color"]
    generated_color = _literal_create_box_color(function)
    if generated_color is None:
        raise TaskGenError("未找到 create_box(..., color=<literal>)")
    if any(abs(a - b) > 1e-6 for a, b in zip(generated_color, expected_color)):
        raise TaskGenError(
            f"生成颜色 {generated_color} 与 VariantSpec {expected_color} 不一致"
        )

    node_count = sum(1 for _ in ast.walk(function))
    if node_count > 500:
        raise TaskGenError(f"load_actors AST 过大: {node_count} nodes")

    return {
        "valid": True,
        "node_count": node_count,
        "calls": sorted(set(name for name in calls if name)),
        "generated_color": generated_color,
        "complete_method_generated": True,
        "calls_super": False,
    }


def compile_overlay(spec: dict[str, Any]) -> dict[str, Any]:
    block = spec["changes"]["block"]
    return {
        "mea": {
            "enabled": True,
            "block": {
                "position_mode": block["position_mode"],
                "yaw_mode": block["yaw_mode"],
                "scale": block["scale"],
                "color": block["color"],
            },
        }
    }


def build_generated_module(method_source: str) -> str:
    method = textwrap.indent(textwrap.dedent(method_source).strip(), "    ")
    return (
        '"""TaskGen output: canonical BeatBlockHammer with a generated load_actors."""\n\n'
        "import numpy as np\n"
        "import sapien\n\n"
        "from envs.beat_block_hammer import beat_block_hammer as OfficialBeatBlockHammer\n"
        "from envs.utils import create_actor, create_box, rand_pose\n\n\n"
        "class beat_block_hammer(OfficialBeatBlockHammer):\n"
        f"{method}\n"
    )


def _proposal_prompt(user_request: str, task_name: str) -> str:
    return f"""你是 ManipEvalAgent 的 Task Proposal Agent。

用户请求：{user_request}
规范任务名：{task_name}

请把请求转换为严格 JSON，不要输出 Markdown。第一版只处理 beat_block_hammer 的方块变式。
必须返回：
{{
  "task_name": "beat_block_hammer",
  "intent": "change_object_appearance",
  "generation_mode": "force_codegen",
  "changes": {{
    "block": {{
      "position_mode": "official_random",
      "yaw_mode": "official_random",
      "scale": 1.0,
      "color": [0.0, 0.2, 1.0]
    }}
  }},
  "preserve": [
    "official_position_sampling",
    "official_yaw_sampling",
    "play_once",
    "check_success",
    "checkpoint"
  ]
}}

颜色使用 [0,1] RGB。除非用户明确要求，否则保持官方随机位置、随机 yaw、尺度、专家轨迹和成功判定不变。
"""


def _extract_method_from_file(path: Path, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if not methods:
        raise TaskGenError(f"{path} 中没有 {method_name}")
    return textwrap.dedent(_source_for_node(source, methods[0]))


def _codegen_prompt(
    repo_root: Path,
    user_request: str,
    spec: dict[str, Any],
    retrieved_tasks: list[str],
) -> str:
    agent_readme = (repo_root / "mea/taskgen/README.Agent.md").read_text(encoding="utf-8")
    official_task = (repo_root / "envs/beat_block_hammer.py").read_text(encoding="utf-8")
    configurable_task = (repo_root / "mea/tasks/beat_block_hammer.py").read_text(encoding="utf-8")
    retrieved_sections = []
    for task_name in retrieved_tasks:
        if task_name == "beat_block_hammer":
            continue
        source_path = repo_root / "envs" / f"{task_name}.py"
        retrieved_sections.append(
            f"### {source_path.relative_to(repo_root)}\n"
            f"```python\n{source_path.read_text(encoding='utf-8')}\n```"
        )
    retrieved_context = (
        "\n\n".join(retrieved_sections)
        if retrieved_sections
        else "No additional example was selected."
    )

    return f"""You are the TaskGen code agent for RoboTwin 2.0.

USER REQUEST:
{user_request}

VALIDATED VARIANT SPEC:
{json.dumps(spec, ensure_ascii=False, indent=2)}

Your output will be inserted into a thin subclass of the official
``envs.beat_block_hammer.beat_block_hammer`` class.

OUTPUT CONTRACT:
1. Output exactly one Python fenced code block.
2. The block must contain the complete ``def load_actors(self):`` method and nothing else.
3. Generate the complete method body yourself. Do not call ``super()``.
4. Recreate every actor, official pose sampling/rejection rule, mass setting,
   and prohibited area used by BeatBlockHammer.
5. Apply only the validated requested change. For this spec, use a literal RGB
   tuple in ``create_box(..., color=...)``.
6. Preserve actor attribute names ``self.hammer`` and ``self.block`` because
   inherited ``play_once`` and ``check_success`` depend on them.
7. Available globals are: ``np``, ``sapien``, ``create_actor``, ``create_box``,
   ``rand_pose``, and ordinary safe builtins. Do not import anything.
8. Do not access files, network, processes, environment variables, or dynamic imports.

README.AGENT:
{agent_readme}

OFFICIAL TASK SOURCE (authoritative behavior):
```python
{official_task}
```

EXISTING CONFIGURABLE BBH SOURCE (reference implementation; resolve the spec
to literal values in your generated method):
```python
{configurable_task}
```

GPT-RETRIEVED TASK SOURCES:
{retrieved_context}
"""


class TaskGenPrototype:
    """Generate and package one TaskGen run using an OpenAI-compatible provider."""

    def __init__(self, repo_root: str | Path, provider: Any, *, model: str):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = model

    def _base_hashes(self) -> dict[str, str]:
        return {
            relative: _sha256(self.repo_root / relative)
            for relative in PROTECTED_PATHS
        }

    def generate(
        self,
        user_request: str,
        *,
        task_name: str = "beat_block_hammer",
        mode: str = "force_codegen",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"force_codegen", "reuse"}:
            raise TaskGenError(f"不支持的 generation mode: {mode}")

        run_id = run_id or make_run_id()
        if not re.fullmatch(r"run_[A-Za-z0-9_]+", run_id):
            raise TaskGenError("run_id 必须是合法 Python package 名并以 run_ 开头")

        run_dir = self.repo_root / "mea/generated_tasks" / run_id
        if run_dir.exists():
            raise TaskGenError(f"run directory 已存在: {run_dir}")
        generation_dir = run_dir / "generation"
        validation_dir = run_dir / "validation"
        evidence_dir = run_dir / "evidence"
        evaluation_dir = run_dir / "evaluation"
        for directory in (generation_dir, validation_dir, evidence_dir, evaluation_dir):
            directory.mkdir(parents=True, exist_ok=False)
        (run_dir / "__init__.py").write_text("", encoding="utf-8")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "proposing",
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": user_request,
            "task_name": task_name,
            "mode": mode,
            "base_commit": _git_head(self.repo_root),
            "protected_hashes_before": self._base_hashes(),
            "provider": {"model_requested": self.model},
        }
        _write_json(run_dir / "request.json", {"user_request": user_request})
        _write_json(run_dir / "manifest.json", manifest)

        proposal_prompt = _proposal_prompt(user_request, task_name)
        (generation_dir / "proposal_prompt.md").write_text(proposal_prompt, encoding="utf-8")
        proposal_response = self.provider.text(
            proposal_prompt,
            model=self.model,
            system="只输出满足 schema 的 JSON object。",
            max_tokens=1200,
            temperature=0.0,
        )
        (generation_dir / "proposal_response.txt").write_text(
            proposal_response + "\n", encoding="utf-8"
        )
        provider_calls = {
            "proposal": dict(getattr(self.provider, "last_metadata", {}))
        }
        spec = validate_variant_spec(extract_json_response(proposal_response), task_name)
        spec["generation_mode"] = mode
        _write_json(run_dir / "variant_spec.json", spec)

        overlay = compile_overlay(spec)
        import yaml

        (run_dir / "overlay.yml").write_text(
            yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        validation: dict[str, Any] = {"variant_spec": {"valid": True}}
        task_module = "mea.tasks.beat_block_hammer"
        task_retrieval = None

        if mode == "force_codegen":
            manifest["status"] = "retrieving_task_sources"
            _write_json(run_dir / "manifest.json", manifest)
            task_retrieval = TaskRetriever(
                self.repo_root,
                self.provider,
                model=self.model,
            ).select(
                user_request,
                task_name,
                spec,
                output_dir=generation_dir,
            )
            provider_calls["retrieval"] = dict(
                getattr(self.provider, "last_metadata", {})
            )

            manifest["status"] = "generating"
            manifest["task_retrieval"] = task_retrieval
            _write_json(run_dir / "manifest.json", manifest)
            code_prompt = _codegen_prompt(
                self.repo_root,
                user_request,
                spec,
                task_retrieval["selected_tasks"],
            )
            (generation_dir / "code_prompt.md").write_text(code_prompt, encoding="utf-8")
            code_response = self.provider.text(
                code_prompt,
                model=self.model,
                system=(
                    "Return exactly one Python code fence containing the complete "
                    "load_actors(self) method."
                ),
                max_tokens=4096,
                temperature=0.0,
            )
            (generation_dir / "code_response.txt").write_text(
                code_response + "\n", encoding="utf-8"
            )
            provider_calls["codegen"] = dict(
                getattr(self.provider, "last_metadata", {})
            )
            method_source = extract_load_actors(code_response)
            (generation_dir / "load_actors.py.txt").write_text(
                method_source, encoding="utf-8"
            )
            validation["load_actors_ast"] = validate_load_actors(method_source, spec)

            module_source = build_generated_module(method_source)
            compile(module_source, str(run_dir / "task.py"), "exec")
            (run_dir / "task.py").write_text(module_source, encoding="utf-8")
            task_module = f"mea.generated_tasks.{run_id}.task"

        hashes_after = self._base_hashes()
        protected_unchanged = hashes_after == manifest["protected_hashes_before"]
        validation["protected_diff"] = {
            "valid": protected_unchanged,
            "hashes_after": hashes_after,
        }
        if not protected_unchanged:
            raise TaskGenError("TaskGen 修改了受保护的官方文件")

        _write_json(validation_dir / "static.json", validation)
        manifest.update(
            {
                "status": "generated",
                "task_module": task_module,
                "overlay": str((run_dir / "overlay.yml").relative_to(self.repo_root)),
                "static_validation": validation,
                "task_retrieval": task_retrieval,
                "provider": {
                    "model_requested": self.model,
                    "calls": provider_calls,
                    "last_metadata": dict(getattr(self.provider, "last_metadata", {})),
                },
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        return manifest
