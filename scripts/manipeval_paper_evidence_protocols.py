#!/usr/bin/env python3
"""Preregister and audit paper-evidence pilots without starting costly calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.live_paper_protocols import (
    LivePaperProtocolError,
    build_click_bell_efficiency_preregistration,
    build_ranking_preregistration,
    build_table3_codegen_preregistration,
    evaluate_click_bell_efficiency,
    evaluate_exact_seed_ranking,
    evaluate_table3_codegen,
    validate_proxy_gold_manifest,
)
from mea.prospective_error_ledger import (
    ProspectiveLedgerError,
    ProspectiveOperationLedger,
    initialize_ledger,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LivePaperProtocolError(f"input must be a JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _checkpoint(identifier: str, sha256: str) -> dict:
    return {"checkpoint_id": identifier, "artifact_sha256": sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    efficiency_pre = sub.add_parser("efficiency-preregister")
    efficiency_pre.add_argument("--study-id", required=True)
    efficiency_pre.add_argument("--mode", choices=("smoke_3act", "toy_5to7act"), required=True)
    efficiency_pre.add_argument("--checkpoint-id", required=True)
    efficiency_pre.add_argument("--checkpoint-sha256", required=True)
    efficiency_pre.add_argument("--seed", type=int, required=True)
    efficiency_pre.add_argument("--created-at-utc", required=True)
    efficiency_pre.add_argument("--output", type=Path, required=True)

    efficiency_eval = sub.add_parser("efficiency-finalize")
    efficiency_eval.add_argument("--preregistration", type=Path, required=True)
    efficiency_eval.add_argument("--fixed-result", type=Path, required=True)
    efficiency_eval.add_argument("--adaptive-result", type=Path, required=True)
    efficiency_eval.add_argument("--output", type=Path, required=True)

    ranking_pre = sub.add_parser("ranking-preregister")
    ranking_pre.add_argument("--study-id", required=True)
    ranking_pre.add_argument("--act-checkpoint-id", required=True)
    ranking_pre.add_argument("--act-checkpoint-sha256", required=True)
    ranking_pre.add_argument("--dp3-checkpoint-id", required=True)
    ranking_pre.add_argument("--dp3-checkpoint-sha256", required=True)
    ranking_pre.add_argument("--seed", type=int, action="append", required=True)
    ranking_pre.add_argument("--created-at-utc", required=True)
    ranking_pre.add_argument("--reference-source-ref", required=True)
    ranking_pre.add_argument("--act-reference-score", type=float, required=True)
    ranking_pre.add_argument("--dp3-reference-score", type=float, required=True)
    ranking_pre.add_argument("--output", type=Path, required=True)

    ranking_eval = sub.add_parser("ranking-finalize")
    ranking_eval.add_argument("--preregistration", type=Path, required=True)
    ranking_eval.add_argument("--runs", type=Path, required=True)
    ranking_eval.add_argument("--output", type=Path, required=True)

    table3_pre = sub.add_parser("table3-preregister")
    table3_pre.add_argument("--study-id", required=True)
    table3_pre.add_argument("--created-at-utc", required=True)
    table3_pre.add_argument("--output", type=Path, required=True)

    table3_eval = sub.add_parser("table3-finalize")
    table3_eval.add_argument("--preregistration", type=Path, required=True)
    table3_eval.add_argument("--runs", type=Path, required=True)
    table3_eval.add_argument("--output", type=Path, required=True)

    proxy = sub.add_parser("proxy-validate")
    proxy.add_argument("--manifest", type=Path, required=True)
    proxy.add_argument("--output", type=Path, required=True)

    ledger_init = sub.add_parser("ledger-init")
    ledger_init.add_argument("--directory", type=Path, required=True)
    ledger_init.add_argument("--study-id", required=True)

    ledger_record = sub.add_parser("ledger-record")
    ledger_record.add_argument("--directory", type=Path, required=True)
    ledger_record.add_argument("--operation-id", required=True)
    ledger_record.add_argument("--run-id", required=True)
    ledger_record.add_argument("--category", required=True)
    ledger_record.add_argument("--status", choices=("started", "completed", "error"), required=True)
    ledger_record.add_argument("--evidence-ref")
    ledger_record.add_argument("--error-class")

    ledger_summary = sub.add_parser("ledger-summary")
    ledger_summary.add_argument("--directory", type=Path, required=True)
    ledger_summary.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "efficiency-preregister":
            output = build_click_bell_efficiency_preregistration(
                study_id=args.study_id,
                mode=args.mode,
                checkpoint=_checkpoint(args.checkpoint_id, args.checkpoint_sha256),
                seed=args.seed,
                created_at_utc=args.created_at_utc,
            )
            _write(args.output, output)
        elif args.command == "efficiency-finalize":
            output = evaluate_click_bell_efficiency(
                _read(args.preregistration),
                _read(args.fixed_result),
                _read(args.adaptive_result),
            )
            _write(args.output, output)
        elif args.command == "ranking-preregister":
            output = build_ranking_preregistration(
                study_id=args.study_id,
                act_checkpoint=_checkpoint(args.act_checkpoint_id, args.act_checkpoint_sha256),
                dp3_checkpoint=_checkpoint(args.dp3_checkpoint_id, args.dp3_checkpoint_sha256),
                seeds=args.seed,
                created_at_utc=args.created_at_utc,
                reference_source_ref=args.reference_source_ref,
                reference_scores={
                    "act": args.act_reference_score,
                    "dp3": args.dp3_reference_score,
                },
            )
            _write(args.output, output)
        elif args.command == "ranking-finalize":
            output = evaluate_exact_seed_ranking(
                _read(args.preregistration), _read(args.runs)
            )
            _write(args.output, output)
        elif args.command == "table3-preregister":
            output = build_table3_codegen_preregistration(
                study_id=args.study_id, created_at_utc=args.created_at_utc
            )
            _write(args.output, output)
        elif args.command == "table3-finalize":
            output = evaluate_table3_codegen(
                _read(args.preregistration), _read(args.runs)
            )
            _write(args.output, output)
        elif args.command == "proxy-validate":
            output = validate_proxy_gold_manifest(REPO_ROOT, _read(args.manifest))
            _write(args.output, output)
        elif args.command == "ledger-init":
            output = initialize_ledger(args.directory, study_id=args.study_id)
        elif args.command == "ledger-record":
            output = ProspectiveOperationLedger(args.directory).append(
                operation_id=args.operation_id,
                run_id=args.run_id,
                category=args.category,
                status=args.status,
                evidence_ref=args.evidence_ref,
                error_class=args.error_class,
            )
        else:
            output = ProspectiveOperationLedger(args.directory).summarize()
            _write(args.output, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (
        OSError,
        json.JSONDecodeError,
        LivePaperProtocolError,
        ProspectiveLedgerError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
