"""Small, auditable task-library retriever used before TaskGen code generation."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


class TaskRetrievalError(RuntimeError):
    """Raised when the task catalog or GPT selection is invalid."""


def _extract_json_response(response: str) -> dict[str, Any]:
    source = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", source, re.DOTALL)
    if fenced:
        source = fenced.group(1).strip()
    start = source.find("{")
    end = source.rfind("}")
    if start < 0 or end < start:
        raise TaskRetrievalError("retrieval response 中没有 JSON object")
    try:
        value = json.loads(source[start : end + 1])
    except json.JSONDecodeError as exc:
        raise TaskRetrievalError("retrieval response 不是合法 JSON") from exc
    if not isinstance(value, dict):
        raise TaskRetrievalError("retrieval response 必须是 JSON object")
    return value


def discover_task_catalog(repo_root: str | Path) -> list[dict[str, Any]]:
    """Discover concrete RoboTwin task files without reading their source into GPT."""

    root = Path(repo_root).expanduser().resolve()
    envs_dir = root / "envs"
    catalog: list[dict[str, Any]] = []
    for path in sorted(envs_dir.glob("*.py")):
        if path.name.startswith("_") or path.stem in {"utils"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            raise TaskRetrievalError(f"无法解析 task file: {path}") from exc
        matching_classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == path.stem
        ]
        if not matching_classes:
            continue
        methods = {
            node.name
            for node in matching_classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        catalog.append(
            {
                "task_name": path.stem,
                "source_path": str(path.relative_to(root)),
                "has_load_actors": "load_actors" in methods,
                "has_check_success": "check_success" in methods,
            }
        )
    if not catalog:
        raise TaskRetrievalError(f"没有在 {envs_dir} 中发现 task files")
    return catalog


def validate_task_selection(
    selection: dict[str, Any],
    *,
    canonical_task: str,
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise TaskRetrievalError("TaskSelection 必须是 JSON object")
    selected = selection.get("selected_tasks")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 3:
        raise TaskRetrievalError("selected_tasks 必须包含 1 到 3 个 task names")
    if any(not isinstance(name, str) or not name for name in selected):
        raise TaskRetrievalError("selected_tasks 中每项必须是非空字符串")
    if len(selected) != len(set(selected)):
        raise TaskRetrievalError("selected_tasks 不能重复")
    if selected[0] != canonical_task:
        raise TaskRetrievalError("canonical task 必须排在 selected_tasks 第一位")
    available = {item["task_name"] for item in catalog}
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise TaskRetrievalError(f"选择了不存在的 task files: {unknown}")
    reasoning = selection.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise TaskRetrievalError("reasoning 必须是非空字符串")
    return {
        "catalog_size": len(catalog),
        "selected_tasks": selected,
        "reasoning": reasoning.strip(),
        "selected_sources": [f"envs/{name}.py" for name in selected],
    }


def _retrieval_prompt(
    repo_root: Path,
    user_request: str,
    canonical_task: str,
    variant_spec: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> str:
    instructions = (repo_root / "mea/retrieval/README.Agent.md").read_text(
        encoding="utf-8"
    )
    task_names = [item["task_name"] for item in catalog]
    return f"""You are the task-source retrieval agent for MEA.

USER REQUEST:
{user_request}

CANONICAL TASK:
{canonical_task}

VALIDATED VARIANT SPEC:
{json.dumps(variant_spec, ensure_ascii=False, indent=2)}

AVAILABLE ROBOTWIN TASK NAMES ({len(task_names)}):
{json.dumps(task_names, ensure_ascii=False, indent=2)}

RETRIEVAL RULES:
{instructions}

Return strict JSON and no Markdown:
{{
  "selected_tasks": ["{canonical_task}", "blocks_ranking_rgb"],
  "reasoning": "The canonical task defines authoritative behavior; the RGB task demonstrates color construction."
}}
"""


class TaskRetriever:
    """Ask GPT to select a few source files from the 50-task catalog."""

    def __init__(self, repo_root: str | Path, provider: Any, *, model: str):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = model

    def select(
        self,
        user_request: str,
        canonical_task: str,
        variant_spec: dict[str, Any],
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        catalog = discover_task_catalog(self.repo_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "task_catalog.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        prompt = _retrieval_prompt(
            self.repo_root,
            user_request,
            canonical_task,
            variant_spec,
            catalog,
        )
        (output_dir / "retrieval_prompt.md").write_text(prompt, encoding="utf-8")
        response = self.provider.text(
            prompt,
            model=self.model,
            system="Return exactly one strict TaskSelection JSON object.",
            max_tokens=600,
            temperature=0.0,
        )
        (output_dir / "retrieval_response.txt").write_text(
            response + "\n", encoding="utf-8"
        )
        selection = validate_task_selection(
            _extract_json_response(response),
            canonical_task=canonical_task,
            catalog=catalog,
        )
        selection["provider_metadata"] = dict(
            getattr(self.provider, "last_metadata", {})
        )
        (output_dir / "retrieval.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return selection
