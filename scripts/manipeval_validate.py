#!/usr/bin/env python3
"""Score a tiny cached Planner/VQA validation suite with budget 1/3/5."""

from __future__ import annotations

import argparse
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

from mea.protocol import AGILE_BUDGETS, write_json_atomic
from mea.validation import ValidationError, score_cached_suite, validate_suite


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


def _run_id(value: str | None) -> str:
    if value is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        value = f"validation_{stamp}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"validation_[A-Za-z0-9_]+", value):
        raise ValidationError(
            "run_id must begin with validation_ and contain safe characters"
        )
    return value


def _render_report(result: dict) -> str:
    planner = result["planner"]["metrics"]
    vqa = result["vqa"]["metrics"]
    return "\n".join(
        [
            "# MEA Cached Mini Validation",
            "",
            f"- suite: `{result['suite_id']}`",
            f"- budget: `{result['budget']}`",
            f"- target: `{result['target']}`",
            "- provider called: `false`",
            "",
            "## Planner",
            "",
            f"- selected cases: `{planner['selected_case_count']}`",
            f"- model case count: `{planner['case_count']}`",
            f"- excluded deterministic plans: `{planner['excluded_non_model_count']}`",
            f"- schema-valid rate: `{planner['schema_valid_rate']}`",
            f"- template micro precision: `{planner['template_micro_precision']}`",
            f"- template micro recall: `{planner['template_micro_recall']}`",
            f"- template micro F1: `{planner['template_micro_f1']}`",
            f"- exact-set accuracy: `{planner['template_exact_set_accuracy']}`",
            "",
            "## VQA",
            "",
            f"- case count: `{vqa['case_count']}`",
            f"- schema-valid rate: `{vqa['schema_valid_rate']}`",
            f"- strict accuracy: `{vqa['accuracy_strict']}`",
            f"- coverage: `{vqa['coverage']}`",
            f"- precision: `{vqa['precision']['value']}`",
            f"- precision unavailable reason: `{vqa['precision']['unavailable_reason']}`",
            f"- AUROC: `{vqa['auroc']['value']}`",
            f"- AUROC unavailable reason: `{vqa['auroc']['unavailable_reason']}`",
            f"- label sources: `{json.dumps(vqa['label_source_counts'], sort_keys=True)}`",
            f"- simulator-proxy only: `{str(vqa['proxy_only']).lower()}`",
            "",
            "## Scope",
            "",
            "This cached 1/3/5 runner validates metric and artifact plumbing. "
            "It does not replace the paper's human-annotated Planner/VQA datasets.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=AGILE_BUDGETS, default=1)
    parser.add_argument("--target", choices=["planner", "vqa", "both"], default="both")
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    try:
        suite = validate_suite(
            json.loads(args.suite.expanduser().read_text(encoding="utf-8"))
        )
        result = score_cached_suite(
            repo_root,
            suite,
            budget=args.budget,
            target=args.target,
        )
        run_id = _run_id(args.run_id)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(str(exc)) from exc
    runs_root = (repo_root / "mea/validation_runs").resolve()
    if not runs_root.is_relative_to(repo_root):
        raise SystemExit("validation_runs resolves outside repo_root")
    run_dir = (runs_root / run_id).resolve()
    if not run_dir.is_relative_to(runs_root):
        raise SystemExit("validation run directory escapes validation_runs")
    if run_dir.exists():
        raise SystemExit(f"validation run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    result.update(
        {
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "git_head": _git_head(repo_root),
            "suite_source": str(args.suite.expanduser().resolve()),
        }
    )
    write_json_atomic(run_dir / "validation_summary.json", result)
    (run_dir / "validation_report.md").write_text(
        _render_report(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
