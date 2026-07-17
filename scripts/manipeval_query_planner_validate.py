#!/usr/bin/env python3
"""Call the Global Plan Agent on 1/3/5/20 development-proxy queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.planner import GlobalQueryRouter, build_act_catalog
from mea.protocol import write_json_atomic
from mea.providers import (
    OpenAICompatibleProvider,
    available_model_profiles,
    resolve_model_profile,
)
from mea.query_dataset import QueryDatasetError
from mea.query_planner_validation import (
    aggregate_live_query_cases,
    score_live_query_case,
    validate_capability_snapshot,
    validate_live_query_budget,
)


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id(value: str | None) -> str:
    if value is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        value = f"query_validation_{stamp}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"query_validation_[A-Za-z0-9_]+", value):
        raise QueryDatasetError(
            "run_id must begin with query_validation_ and contain safe characters"
        )
    return value


def _report(summary: dict) -> str:
    metrics = summary["metrics"]
    return "\n".join(
        [
            "# Live Global Planner Development-Proxy Validation",
            "",
            f"- run: `{summary['run_id']}`",
            f"- dataset: `{summary['dataset_id']}`",
            f"- dataset SHA-256: `{summary['dataset_sha256']}`",
            f"- budget: `{summary['budget']}`",
            f"- provider called: `{str(summary['provider_called']).lower()}`",
            f"- annotation source: `{summary['annotation_source']}`",
            f"- paper Table 6 eligible: `{str(summary['paper_table_eligible']).lower()}`",
            "",
            "## Metrics",
            "",
            f"- schema-valid rate: `{metrics['schema_valid_rate']}`",
            f"- capability-decision accuracy: `{metrics['capability_decision_accuracy']}`",
            f"- paper-category counts: `{json.dumps(metrics['paper_category_counts'], sort_keys=True)}`",
            f"- task accuracy (evaluable only): `{metrics['task_accuracy']}`",
            f"- aspect micro precision: `{metrics['aspect_micro_precision']}`",
            f"- aspect micro recall: `{metrics['aspect_micro_recall']}`",
            f"- aspect micro F1: `{metrics['aspect_micro_f1']}`",
            f"- aspect exact-set accuracy: `{metrics['aspect_exact_set_accuracy']}`",
            f"- first-aspect accuracy: `{metrics['first_aspect_accuracy']}`",
            f"- task-qualified unsupported-gap coverage: `{metrics['task_qualified_gap_coverage']}`",
            "",
            "## Scope",
            "",
            "Codex acted only as a development annotation proxy. These labels are "
            "not human gold, do not estimate human-agent agreement, and must be "
            "replaced by independent human-majority labels before a paper Table 6 claim.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=[1, 3, 5, 20], default=1)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--model-profile", choices=available_model_profiles(), default="economy"
    )
    parser.add_argument("--planner-model")
    parser.add_argument("--vision-model")
    parser.add_argument("--base-url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    try:
        dataset_path = args.dataset.expanduser().resolve()
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        selected = validate_live_query_budget(dataset, args.budget)
        run_id = _run_id(args.run_id)
    except (OSError, json.JSONDecodeError, QueryDatasetError) as exc:
        raise SystemExit(str(exc)) from exc
    catalog = build_act_catalog(repo_root)
    try:
        validate_capability_snapshot(list(dataset["cases"]), catalog)
    except QueryDatasetError as exc:
        raise SystemExit(str(exc)) from exc
    runs_root = (repo_root / "mea/validation_runs").resolve()
    run_dir = (runs_root / run_id).resolve()
    if not run_dir.is_relative_to(runs_root):
        raise SystemExit("query validation run directory escapes validation_runs")
    if run_dir.exists():
        raise SystemExit(f"query validation run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    models = resolve_model_profile(
        args.model_profile,
        {"planner": args.planner_model, "vision": args.vision_model},
    )
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=models["planner"],
        vision_model=models["vision"],
        timeout=180.0,
    )
    write_json_atomic(run_dir / "global_act_catalog.json", catalog)
    scored_cases: list[dict] = []
    for case in selected:
        case_dir = run_dir / "cases" / case["id"]
        case_dir.mkdir(parents=True)
        router = GlobalQueryRouter(
            provider,
            model=models["planner"],
            catalog=catalog,
        )
        result = None
        error = None
        try:
            result = router.route(case["query"], history_context=[])
        except Exception as exc:  # failure is scored instead of aborting the run
            error = f"{type(exc).__name__}: {exc}"
        (case_dir / "prompt.md").write_text(router.last_prompt or "", encoding="utf-8")
        for index, response in enumerate(router.last_responses, start=1):
            (case_dir / f"response_{index}.txt").write_text(
                str(response), encoding="utf-8"
            )
        trace = {
            "schema_version": 1,
            "case_id": case["id"],
            "query": case["query"],
            "result": result,
            "error": error,
            "provider_metadata": dict(getattr(provider, "last_metadata", {})),
        }
        write_json_atomic(case_dir / "route_trace.json", trace)
        scored = score_live_query_case(case, result, error=error)
        write_json_atomic(case_dir / "score.json", scored)
        scored_cases.append(scored)

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_head": _git_head(repo_root),
        "dataset_id": dataset.get("dataset_id"),
        "dataset_source": str(dataset_path),
        "dataset_sha256": _file_sha256(dataset_path),
        "catalog_sha256": catalog["catalog_sha256"],
        "budget": args.budget,
        "provider_called": True,
        "model": models["planner"],
        "annotation_source": "development_agent_proxy",
        "human_reviewer_count": 0,
        "paper_table_eligible": False,
        "paper_ineligible_reason": "development_agent_proxy_is_not_human_gold",
        "metrics": aggregate_live_query_cases(scored_cases),
        "cases": scored_cases,
    }
    write_json_atomic(run_dir / "validation_summary.json", summary)
    (run_dir / "validation_report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
