"""Small deterministic Task/Asset/Documentation RAG for MEA TaskGen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mea.knowledge import build_knowledge_index_data, source_symbol_text


class KnowledgeRetrievalError(RuntimeError):
    """Raised when the compact knowledge selection is invalid."""


def _document_map(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    documents = index.get("documents")
    if not isinstance(documents, list):
        raise KnowledgeRetrievalError("knowledge index 缺少 documents")
    result = {item.get("id"): item for item in documents}
    if None in result or len(result) != len(documents):
        raise KnowledgeRetrievalError("knowledge document id 缺失或重复")
    return result


def select_document_ids(
    user_request: str,
    task_name: str,
    variant_spec: dict[str, Any],
) -> list[str]:
    """Select only knowledge that changes the quality of this generation."""

    if task_name not in {"beat_block_hammer", "click_bell"}:
        raise KnowledgeRetrievalError(
            f"Documentation RAG has no supported task card for {task_name!r}"
        )
    selected = [f"task.{task_name}", "api.scene_creation"]
    if task_name == "click_bell":
        # Both position and instance proposals depend on the same existing
        # functional-point/instance asset contract.
        selected.append("asset.050_bell")
        return selected
    request = user_request.lower()
    block = variant_spec.get("changes", {}).get("block", {})
    hammer_terms = ("hammer", "锤", "replace hammer", "functional point")
    if any(term in request for term in hammer_terms) and not block.get("color"):
        selected.append("asset.020_hammer")
    return selected


class KnowledgeRetriever:
    """Retrieve compact cards plus method-level source examples."""

    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root).expanduser().resolve()

    def select(
        self,
        user_request: str,
        task_name: str,
        variant_spec: dict[str, Any],
        retrieved_tasks: list[str],
        *,
        output_dir: Path,
        max_characters: int = 8000,
    ) -> dict[str, Any]:
        index = build_knowledge_index_data(self.repo_root)
        committed_path = self.repo_root / "mea/knowledge/index.json"
        committed = (
            json.loads(committed_path.read_text(encoding="utf-8"))
            if committed_path.is_file()
            else None
        )
        documents = _document_map(index)
        snapshots = {
            item.get("task_name"): item for item in index.get("agent_readmes", [])
        }
        agent_readme = snapshots.get(task_name)
        if not isinstance(agent_readme, dict):
            raise KnowledgeRetrievalError(
                f"knowledge index lacks README.Agent snapshot for {task_name!r}"
            )
        selected_ids = select_document_ids(user_request, task_name, variant_spec)
        selected: list[dict[str, Any]] = []
        context_sections: list[str] = [
            "## README.Agent freshness snapshot\n\n"
            + json.dumps(agent_readme, ensure_ascii=False, indent=2)
        ]
        for document_id in selected_ids:
            if document_id not in documents:
                raise KnowledgeRetrievalError(
                    f"选择了不存在的 knowledge document: {document_id}"
                )
            metadata = documents[document_id]
            content = (self.repo_root / metadata["path"]).read_text(
                encoding="utf-8"
            )
            selected.append(
                {
                    "id": document_id,
                    "kind": metadata["kind"],
                    "path": metadata["path"],
                    "title": metadata["title"],
                    "tags": metadata["tags"],
                    "character_count": len(content),
                    "source_symbols": metadata.get("source_symbols", []),
                    "source_files": metadata.get("source_files", []),
                    "reason": {
                        "task_contract": "task contract",
                        "asset_contract": "required existing asset contract",
                        "api": "required scene API",
                    }.get(metadata["kind"], "retrieved generation context"),
                }
            )
            context_sections.append(
                f"## {document_id}\nSource: `{metadata['path']}`\n\n{content.strip()}"
            )

        examples: list[dict[str, Any]] = []
        for related_task in retrieved_tasks:
            if related_task == task_name:
                continue
            relative = f"envs/{related_task}.py"
            method = source_symbol_text(
                self.repo_root, relative, f"{related_task}.load_actors"
            )
            example_id = f"example.{related_task}.load_actors"
            examples.append(
                {
                    "id": example_id,
                    "path": relative,
                    "symbol": f"{related_task}.load_actors",
                    "character_count": len(method),
                    "reason": "TaskRetriever selected a method-level construction example",
                }
            )
            context_sections.append(
                f"## {example_id}\nSource: `{relative}:{related_task}.load_actors`\n\n```python\n{method.rstrip()}\n```"
            )

        context = "\n\n".join(context_sections) + "\n"
        if len(context) > max_characters:
            raise KnowledgeRetrievalError(
                f"knowledge context 过大: {len(context)} > {max_characters} characters"
            )
        result = {
            "schema_version": 1,
            "scope": index["scope"],
            "selected_documents": selected,
            "selected_examples": examples,
            "selected_ids": [item["id"] for item in selected]
            + [item["id"] for item in examples],
            "context_character_count": len(context),
            "max_characters": max_characters,
            "committed_index_current": committed == index,
            "agent_readme_snapshot": agent_readme,
            "agent_readme_current": committed == index,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "knowledge_catalog.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "knowledge_retrieval.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "agent_readme_snapshot.json").write_text(
            json.dumps(agent_readme, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "knowledge_context.md").write_text(
            context, encoding="utf-8"
        )
        result["context"] = context
        return result
