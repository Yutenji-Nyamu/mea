"""Live-provider, zero-ACT matched micro-ablation with separate proxy review.

This is intentionally smaller than paper Table 3.  Generation calls a real
provider under the schedule's declared switches.  Review is a later,
append-only development-agent proxy step, so provider output never grades
itself and an unreviewed run cannot expose a success rate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.module_ablation_execution import _validate_execution_schedule
from mea.module_ablation_protocol import _canonical_sha256


class LiveModuleAblationError(RuntimeError):
    """The live micro-ablation or its proxy review violates the contract."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise LiveModuleAblationError(f"append-only artifact already exists: {path}") from exc


def _safe_new_directory(repo_root: Path, output_dir: Path) -> Path:
    root = repo_root.expanduser().resolve()
    candidate = output_dir.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise LiveModuleAblationError("output directory must stay inside repository") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise LiveModuleAblationError("output directory contains a symlink")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root) or resolved == root:
        raise LiveModuleAblationError("output directory must be a repository child")
    if resolved.exists():
        raise LiveModuleAblationError(f"output directory already exists: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _selected_items(
    repo_root: Path,
    schedule: Mapping[str, Any],
    item_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    items = _validate_execution_schedule(repo_root, schedule)
    if item_ids is None:
        selected = items
    else:
        requested = list(item_ids)
        if not requested or len(requested) != len(set(requested)):
            raise LiveModuleAblationError("item ids must be a unique non-empty list")
        by_id = {item["schedule_item_id"]: item for item in items}
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise LiveModuleAblationError(f"unknown schedule item ids: {missing}")
        selected = [by_id[item_id] for item_id in requested]
    if not selected:
        raise LiveModuleAblationError("no schedule items selected")
    return selected


def _prompt(item: Mapping[str, Any]) -> str:
    switches = dict(item["module_switches"])
    identity = dict(item["input_identity"])
    component = str(item["component"])
    retrieved = identity.pop("retrieved_context", None)
    readme = identity.pop("readme_agent_guidance", None)
    payload: dict[str, Any] = {
        "component": component,
        "condition": item["condition"],
        "matched_input": identity,
        "active_module_switches": switches,
        "output_contract": {
            "candidate": "one concise implementation proposal",
            "self_reported_checks": "object of booleans",
            "limitations": "list of strings",
        },
    }
    if switches.get("rag"):
        payload["retrieved_context"] = retrieved or (
            "Use the trusted RoboTwin task/tool contracts supplied by the caller."
        )
    if component == "taskgen" and switches.get("readme_agent"):
        payload["readme_agent_guidance"] = readme or (
            "Preserve simulator APIs, task identity, and deterministic seed handling."
        )
    if component == "taskgen":
        payload["visual_self_check_contract"] = (
            "required after generation"
            if switches.get("visual_self_check")
            else "disabled for this condition"
        )
    return (
        "Produce strict JSON for this matched development micro-ablation. "
        "Do not claim simulator or ACT execution.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    )


def generate_live_module_ablation(
    repo_root: str | Path,
    schedule: Mapping[str, Any],
    *,
    output_dir: str | Path,
    provider: Any,
    model: str,
    schedule_item_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate real-provider candidates; deliberately produce no scores."""

    root = Path(repo_root).expanduser().resolve()
    selected = _selected_items(root, schedule, schedule_item_ids)
    mismatched_models = [
        item["schedule_item_id"]
        for item in selected
        if (item.get("execution_identity") or {}).get("provider_model") != model
    ]
    if mismatched_models:
        raise LiveModuleAblationError(
            f"provider model differs from schedule for {mismatched_models}"
        )
    output = _safe_new_directory(root, Path(output_dir))
    rows: list[dict[str, Any]] = []
    for item in selected:
        item_dir = output / "items" / item["schedule_item_id"]
        item_dir.mkdir(parents=True, exist_ok=False)
        prompt = _prompt(item)
        prompt_path = item_dir / "prompt.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        response = provider.text(
            prompt,
            model=model,
            max_tokens=900,
            temperature=0.0,
        )
        if not isinstance(response, str) or not response.strip():
            raise LiveModuleAblationError("provider returned an empty candidate")
        response_path = item_dir / "candidate.txt"
        response_path.write_text(response.strip() + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "kind": "table3_live_candidate_v1",
            "schedule_contract_sha256": schedule["schedule_contract_sha256"],
            "schedule_item_id": item["schedule_item_id"],
            "schedule_item_sha256": item["schedule_item_sha256"],
            "component": item["component"],
            "condition": item["condition"],
            "input_identity_sha256": item["input_identity_sha256"],
            "execution_identity_sha256": item["execution_identity_sha256"],
            "applied_module_switches": item["module_switches"],
            "prompt_sha256": _file_sha256(prompt_path),
            "candidate_sha256": _file_sha256(response_path),
            "provider_calls": 1,
            "provider_metadata": dict(getattr(provider, "last_metadata", {})),
            "review_status": "awaiting_development_agent_proxy",
            "success": None,
            "simulator_called": False,
            "act_rollouts_started": 0,
            "paper_table_eligible": False,
        }
        manifest_path = item_dir / "manifest.json"
        _write_new(manifest_path, manifest)
        rows.append(
            {
                "schedule_item_id": item["schedule_item_id"],
                "component": item["component"],
                "condition": item["condition"],
                "manifest": manifest_path.relative_to(output).as_posix(),
                "manifest_sha256": _file_sha256(manifest_path),
                "success": None,
            }
        )
    run_manifest = {
        "schema_version": 1,
        "kind": "table3_live_provider_micro_generation_v1",
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "status": "awaiting_development_agent_proxy_review",
        "selected_item_count": len(rows),
        "items": rows,
        "runtime": {
            "provider_calls_expected": len(rows),
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "success_rates": None,
        "paper_table_eligible": False,
        "claim_scope": "live-provider matched micro-ablation; not paper Table 3",
    }
    run_manifest["items_sha256"] = _canonical_sha256(rows)
    _write_new(output / "generation_manifest.json", run_manifest)
    return run_manifest


def review_live_module_ablation(
    run_dir: str | Path,
    review: Mapping[str, Any],
) -> dict[str, Any]:
    """Append proxy labels after candidate generation and expose matched rates."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "generation_manifest.json"
    try:
        generation = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveModuleAblationError(f"cannot read generation manifest: {exc}") from exc
    if generation.get("kind") != "table3_live_provider_micro_generation_v1":
        raise LiveModuleAblationError("unsupported generation manifest")
    if review.get("schema_version") != 1 or review.get("reviewer") != "development_agent_proxy":
        raise LiveModuleAblationError(
            "review must declare schema_version=1 and reviewer=development_agent_proxy"
        )
    labels = review.get("labels")
    if not isinstance(labels, Mapping):
        raise LiveModuleAblationError("review labels must be an object")
    expected_ids = [item["schedule_item_id"] for item in generation["items"]]
    if set(labels) != set(expected_ids):
        raise LiveModuleAblationError("review labels must exactly match generated items")
    reviewed: list[dict[str, Any]] = []
    for item in generation["items"]:
        item_id = item["schedule_item_id"]
        label = labels[item_id]
        if not isinstance(label, Mapping) or set(label) != {"success", "rationale"}:
            raise LiveModuleAblationError(f"invalid proxy label contract: {item_id}")
        success = label["success"]
        rationale = label["rationale"]
        if not isinstance(success, bool) or not isinstance(rationale, str) or not rationale.strip():
            raise LiveModuleAblationError(f"invalid proxy label value: {item_id}")
        candidate_manifest = root / item["manifest"]
        if _file_sha256(candidate_manifest) != item["manifest_sha256"]:
            raise LiveModuleAblationError(f"candidate manifest hash mismatch: {item_id}")
        outcome = {
            "schema_version": 1,
            "kind": "table3_development_agent_proxy_outcome_v1",
            "schedule_item_id": item_id,
            "candidate_manifest_sha256": item["manifest_sha256"],
            "reviewer": "development_agent_proxy",
            "success": success,
            "rationale": rationale.strip(),
            "paper_table_eligible": False,
        }
        outcome_path = candidate_manifest.parent / "proxy_review.json"
        _write_new(outcome_path, outcome)
        reviewed.append(
            {
                **item,
                "success": success,
                "review": outcome_path.relative_to(root).as_posix(),
                "review_sha256": _file_sha256(outcome_path),
            }
        )
    rates: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    for component in sorted({item["component"] for item in reviewed}):
        component_rows = [item for item in reviewed if item["component"] == component]
        rates[component] = {
            item["condition"]: float(item["success"]) for item in component_rows
        }
        by_condition = {item["condition"]: item for item in component_rows}
        effect = None
        if "complete" in by_condition and "no_rag" in by_condition:
            effect = float(by_condition["complete"]["success"]) - float(
                by_condition["no_rag"]["success"]
            )
        comparisons.append(
            {
                "component": component,
                "reference_condition": "complete",
                "module_off_condition": "no_rag",
                "proxy_absolute_success_difference": effect,
            }
        )
    summary = {
        "schema_version": 1,
        "kind": "table3_live_provider_proxy_review_v1",
        "status": "completed_with_development_agent_proxy_review",
        "reviewer": "development_agent_proxy",
        "items": reviewed,
        "proxy_success_rates": rates,
        "comparisons": comparisons,
        "runtime": generation["runtime"],
        "paper_table_eligible": False,
        "claim_scope": (
            "live-provider matched micro-ablation with development-agent proxy labels; "
            "not human-reviewed paper Table 3"
        ),
    }
    _write_new(root / "review_summary.json", summary)
    return summary
