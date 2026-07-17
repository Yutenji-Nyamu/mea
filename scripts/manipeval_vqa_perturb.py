#!/usr/bin/env python3
"""Run a 1/3/5 cached-montage image-proxy VQA perturbation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.execution_vqa import analyze_execution_montage
from mea.protocol import AGILE_BUDGETS, write_json_atomic
from mea.providers import OpenAICompatibleProvider, available_model_profiles, resolve_model_profile
from mea.validation import aggregate_vqa_cases
from mea.vqa_perturbations import (
    VQAPerturbationError,
    build_proxy_images,
    file_sha256,
    validate_perturbation_suite,
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise VQAPerturbationError(f"artifact is missing or escapes repo: {value}")
    return path


def _run_id(value: str | None) -> str:
    if value is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        value = f"validation_vqa_proxy_{stamp}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"validation_[A-Za-z0-9_]+", value):
        raise VQAPerturbationError("run_id must begin with validation_")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=AGILE_BUDGETS, default=1)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--model-profile", choices=available_model_profiles(), default="economy"
    )
    parser.add_argument("--vision-model")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        suite = validate_perturbation_suite(
            json.loads(args.suite.read_text(encoding="utf-8"))
        )
        run_id = _run_id(args.run_id)
    except (OSError, json.JSONDecodeError, VQAPerturbationError) as exc:
        raise SystemExit(str(exc)) from exc
    if not os.getenv("UIUI_API_KEY"):
        raise SystemExit("UIUI_API_KEY is required and is never written to artifacts")
    run_dir = (root / "mea/validation_runs" / run_id).resolve()
    if not run_dir.is_relative_to(root / "mea/validation_runs"):
        raise SystemExit("validation run path escapes validation_runs")
    if run_dir.exists():
        raise SystemExit(f"validation run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    models = resolve_model_profile(
        args.model_profile,
        {"vision": args.vision_model},
    )
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=models["vision"],
        vision_model=models["vision"],
        timeout=180.0,
    )
    rows: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    for case in suite["cases"][: args.budget]:
        source_path = _inside(root, case["source_artifact"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        artifacts = source.get("artifacts") or {}
        montage_path = _inside(root, str(artifacts.get("montage") or ""))
        query = source.get("query")
        selection = source.get("selection")
        numeric = source.get("numeric_tool_results")
        if not isinstance(query, dict) or not isinstance(selection, dict):
            raise SystemExit(f"source VQA artifact is incomplete: {source_path}")
        if case["phenomenon_id"] not in query.get("phenomenon_ids", []):
            raise SystemExit(f"source query lacks phenomenon for {case['id']}")
        case_dir = run_dir / "cases" / case["id"]
        records = build_proxy_images(
            montage_path,
            case_dir / "images",
            seed=int(case["transform_seed"]),
        )
        query_hash = _canonical_hash(query)
        numeric_hash = _canonical_hash(numeric)
        for record in records:
            record.update(
                {
                    "case_id": case["id"],
                    "source_artifact": str(source_path.relative_to(root)),
                    "source_artifact_sha256": file_sha256(source_path),
                    "query_sha256": query_hash,
                    "numeric_evidence_sha256": numeric_hash,
                }
            )
            transforms.append(record)
            result_path = case_dir / f"{record['condition']}.json"
            try:
                result = analyze_execution_montage(
                    provider=provider,
                    model=models["vision"],
                    montage_path=record["derived_image"],
                    selection=selection,
                    numeric_tool_results=numeric,
                    query=query,
                    destination=result_path,
                )
                phenomenon = next(
                    item
                    for item in result["observation"]["phenomena"]
                    if item["id"] == case["phenomenon_id"]
                )
                observed = phenomenon["observed"]
                confidence = float(phenomenon["confidence"])
                score = (
                    confidence
                    if observed is True
                    else 1.0 - confidence
                    if observed is False
                    else 0.5
                )
                error = None
            except Exception as exc:
                observed = None
                confidence = 0.0
                score = 0.5
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "id": f"{case['id']}:{record['condition']}",
                    "target": "vqa",
                    "artifact": (
                        str(result_path.relative_to(root)) if result_path.is_file() else None
                    ),
                    "artifact_sha256": (
                        file_sha256(result_path) if result_path.is_file() else None
                    ),
                    "phenomenon_id": case["phenomenon_id"],
                    "perturbation": record["condition"],
                    "gold_observed": case["gold_observed"],
                    "label_source": case["label_source"],
                    "schema_valid": error is None,
                    "error": error,
                    "predicted_observed": observed,
                    "confidence": confidence,
                    "positive_score": score,
                    "covered": isinstance(observed, bool),
                    "correct_strict": observed == case["gold_observed"],
                }
            )
    by_perturbation = {
        condition: aggregate_vqa_cases(
            [row for row in rows if row["perturbation"] == condition]
        )
        for condition in sorted({row["perturbation"] for row in rows})
    }
    summary = {
        "schema_version": 1,
        "protocol": "cached_montage_image_proxy_v1",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "budget_source_clips": args.budget,
        "vision_calls_requested": args.budget * 4,
        "model_profile": args.model_profile,
        "model": models["vision"],
        "proxy_metrics": aggregate_vqa_cases(rows),
        "by_perturbation": by_perturbation,
        "human_metrics": None,
        "paper_table_eligible": False,
        "unavailable_reason": "simulator_proxy_labels_and_image_proxy_perturbations",
        "cases": rows,
        "transforms": transforms,
        "credentials_recorded": False,
    }
    write_json_atomic(run_dir / "validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
