"""Multi-rater agreement and paired VQA-control validation.

This is an offline annotation importer and deterministic aggregator.  Rater
types are preserved instead of being silently upgraded: a development agent is
never counted as human gold, and synthetic fixture raters remain synthetic.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Any, Mapping


PROTOCOL = "independent_vqa_validity_v1"
RATER_KINDS = {"human", "development_agent", "synthetic_fixture_rater"}
RATER_ROLES = {"primary_annotator", "senior_tiebreaker", "proxy"}
CONTROL_POLARITIES = {"positive_control", "negative_control"}
EVIDENCE_SOURCES = {"synthetic_fixture", "cached_artifact", "live_annotation"}
PAPER_CONDITIONS = (
    "clean",
    "scene_clutter",
    "background_texture",
    "lighting",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class IndependentValidityError(ValueError):
    """Raised when annotations or VQA controls violate the protocol."""


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(text):
        raise IndependentValidityError(
            f"{field} must be a non-empty identifier"
        )
    return text


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise IndependentValidityError(f"{field} must be non-empty text")
    return value.strip()


def _binary(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise IndependentValidityError(f"{field} must be boolean")
    return value


def _condition(value: Any, *, field: str) -> str:
    if value not in PAPER_CONDITIONS:
        raise IndependentValidityError(
            f"{field} must be one of {list(PAPER_CONDITIONS)}"
        )
    return str(value)


def _finite_probability(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise IndependentValidityError(
            f"{field} must be null or a finite number in [0, 1]"
        )
    return float(value)


def _majority(labels: list[bool]) -> bool | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == negatives:
        return None
    return positives > negatives


def validate_independent_validity_study(value: Any) -> dict[str, Any]:
    """Validate and normalize one multi-rater/VQA-control study."""

    if not isinstance(value, Mapping):
        raise IndependentValidityError("study must be an object")
    if value.get("schema_version") != 1 or value.get("protocol") != PROTOCOL:
        raise IndependentValidityError(
            f"study must use schema_version=1 and protocol={PROTOCOL}"
        )
    study_id = _identifier(value.get("study_id"), field="study_id")
    evidence_source = value.get("evidence_source")
    if evidence_source not in EVIDENCE_SOURCES:
        raise IndependentValidityError(
            f"evidence_source must be one of {sorted(EVIDENCE_SOURCES)}"
        )
    fixed_threshold = _finite_probability(
        value.get("fixed_threshold", 0.5), field="fixed_threshold"
    )
    assert fixed_threshold is not None

    raw_items = value.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise IndependentValidityError("items must be a non-empty list")
    items: list[dict[str, str]] = []
    item_ids: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise IndependentValidityError(f"items[{index}] must be an object")
        item_id = _identifier(raw.get("item_id"), field=f"items[{index}].item_id")
        if item_id in item_ids:
            raise IndependentValidityError(f"duplicate item_id: {item_id}")
        item_ids.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "phenomenon_id": _identifier(
                    raw.get("phenomenon_id"),
                    field=f"items[{index}].phenomenon_id",
                ),
                "source_id": _identifier(
                    raw.get("source_id"), field=f"items[{index}].source_id"
                ),
                "condition": _condition(
                    raw.get("condition"), field=f"items[{index}].condition"
                ),
            }
        )

    raw_raters = value.get("raters")
    if not isinstance(raw_raters, list) or not raw_raters:
        raise IndependentValidityError("raters must be a non-empty list")
    raters: list[dict[str, str]] = []
    rater_specs: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_raters):
        if not isinstance(raw, Mapping):
            raise IndependentValidityError(f"raters[{index}] must be an object")
        rater_id = _identifier(
            raw.get("rater_id"), field=f"raters[{index}].rater_id"
        )
        if rater_id in rater_specs:
            raise IndependentValidityError(f"duplicate rater_id: {rater_id}")
        kind = raw.get("kind")
        if kind not in RATER_KINDS:
            raise IndependentValidityError(
                f"raters[{index}].kind must be one of {sorted(RATER_KINDS)}"
            )
        role = raw.get("role")
        if role not in RATER_ROLES:
            raise IndependentValidityError(
                f"raters[{index}].role must be one of {sorted(RATER_ROLES)}"
            )
        if kind == "human" and role not in {
            "primary_annotator",
            "senior_tiebreaker",
        }:
            raise IndependentValidityError(
                "human raters must be primary_annotator or senior_tiebreaker"
            )
        if kind != "human" and role != "proxy":
            raise IndependentValidityError(
                "non-human raters must use role=proxy"
            )
        rater_specs[rater_id] = {"kind": kind, "role": role}
        raters.append({"rater_id": rater_id, "kind": kind, "role": role})

    raw_annotations = value.get("annotations")
    if not isinstance(raw_annotations, list) or not raw_annotations:
        raise IndependentValidityError(
            "annotations must be a non-empty long-form list"
        )
    annotations: list[dict[str, Any]] = []
    annotation_pairs: set[tuple[str, str]] = set()
    item_annotation_counts = Counter()
    rater_annotation_counts = Counter()
    for index, raw in enumerate(raw_annotations):
        if not isinstance(raw, Mapping):
            raise IndependentValidityError(
                f"annotations[{index}] must be an object"
            )
        item_id = _identifier(
            raw.get("item_id"), field=f"annotations[{index}].item_id"
        )
        rater_id = _identifier(
            raw.get("rater_id"), field=f"annotations[{index}].rater_id"
        )
        if item_id not in item_ids:
            raise IndependentValidityError(
                f"annotation references unknown item: {item_id}"
            )
        if rater_id not in rater_specs:
            raise IndependentValidityError(
                f"annotation references unknown rater: {rater_id}"
            )
        pair = (item_id, rater_id)
        if pair in annotation_pairs:
            raise IndependentValidityError(
                f"duplicate annotation for item/rater: {pair}"
            )
        annotation_pairs.add(pair)
        item_annotation_counts[item_id] += 1
        rater_annotation_counts[rater_id] += 1
        annotations.append(
            {
                "item_id": item_id,
                "rater_id": rater_id,
                "rater_kind": rater_specs[rater_id]["kind"],
                "rater_role": rater_specs[rater_id]["role"],
                "observed": _binary(
                    raw.get("observed"),
                    field=f"annotations[{index}].observed",
                ),
            }
        )
    missing_items = sorted(item_ids - set(item_annotation_counts))
    if missing_items:
        raise IndependentValidityError(
            f"items missing annotations: {missing_items}"
        )
    missing_raters = sorted(set(rater_specs) - set(rater_annotation_counts))
    if missing_raters:
        raise IndependentValidityError(
            f"raters missing annotations: {missing_raters}"
        )

    raw_controls = value.get("vqa_controls")
    if not isinstance(raw_controls, list) or not raw_controls:
        raise IndependentValidityError("vqa_controls must be a non-empty list")
    controls: list[dict[str, Any]] = []
    control_ids: set[str] = set()
    control_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_controls):
        if not isinstance(raw, Mapping):
            raise IndependentValidityError(
                f"vqa_controls[{index}] must be an object"
            )
        control_id = _identifier(
            raw.get("control_id"), field=f"vqa_controls[{index}].control_id"
        )
        if control_id in control_ids:
            raise IndependentValidityError(
                f"duplicate VQA control_id: {control_id}"
            )
        control_ids.add(control_id)
        item_id = _identifier(
            raw.get("item_id"), field=f"vqa_controls[{index}].item_id"
        )
        if item_id not in item_ids:
            raise IndependentValidityError(
                f"VQA control references unknown item: {item_id}"
            )
        polarity = raw.get("polarity")
        if polarity not in CONTROL_POLARITIES:
            raise IndependentValidityError(
                f"vqa_controls[{index}].polarity must be one of "
                f"{sorted(CONTROL_POLARITIES)}"
            )
        pair = (item_id, polarity)
        if pair in control_pairs:
            raise IndependentValidityError(
                f"duplicate {polarity} for item {item_id}"
            )
        control_pairs.add(pair)
        positive_score = _finite_probability(
            raw.get("positive_score"),
            field=f"vqa_controls[{index}].positive_score",
        )
        if positive_score is None:
            raise IndependentValidityError(
                f"vqa_controls[{index}].positive_score must not be null"
            )
        predicted_observed = _binary(
            raw.get("predicted_observed"),
            field=f"vqa_controls[{index}].predicted_observed",
        )
        if predicted_observed != (positive_score >= fixed_threshold):
            raise IndependentValidityError(
                f"vqa_controls[{index}] prediction disagrees with fixed_threshold"
            )
        controls.append(
            {
                "control_id": control_id,
                "item_id": item_id,
                "polarity": polarity,
                "transformation": _text(
                    raw.get("transformation"),
                    field=f"vqa_controls[{index}].transformation",
                ),
                "predicted_observed": predicted_observed,
                "positive_score": positive_score,
                "confidence": _finite_probability(
                    raw.get("confidence"),
                    field=f"vqa_controls[{index}].confidence",
                ),
            }
        )
    expected_pairs = {
        (item_id, polarity)
        for item_id in item_ids
        for polarity in CONTROL_POLARITIES
    }
    if control_pairs != expected_pairs:
        missing = sorted(expected_pairs - control_pairs)
        extra = sorted(control_pairs - expected_pairs)
        raise IndependentValidityError(
            "every item requires one positive and one negative VQA control; "
            f"missing={missing}, extra={extra}"
        )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "study_id": study_id,
        "evidence_source": evidence_source,
        "fixed_threshold": fixed_threshold,
        "items": items,
        "raters": raters,
        "annotations": annotations,
        "vqa_controls": controls,
    }


def _agreement_rows(
    study: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raters = {
        item["rater_id"]: {"kind": item["kind"], "role": item["role"]}
        for item in study["raters"]
    }
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in study["annotations"]:
        by_item[annotation["item_id"]].append(annotation)
    rows: list[dict[str, Any]] = []
    total_pairs = 0
    agreeing_pairs = 0
    human_total_pairs = 0
    human_agreeing_pairs = 0
    for item in study["items"]:
        item_id = item["item_id"]
        annotations = sorted(
            by_item[item_id], key=lambda value: value["rater_id"]
        )
        labels = [value["observed"] for value in annotations]
        pairs = list(combinations(labels, 2))
        pair_agreements = sum(left == right for left, right in pairs)
        total_pairs += len(pairs)
        agreeing_pairs += pair_agreements
        primary_human_annotations = [
            value
            for value in annotations
            if raters[value["rater_id"]]
            == {"kind": "human", "role": "primary_annotator"}
        ]
        senior_human_annotations = [
            value
            for value in annotations
            if raters[value["rater_id"]]
            == {"kind": "human", "role": "senior_tiebreaker"}
        ]
        human_pairs = list(
            combinations(
                [value["observed"] for value in primary_human_annotations], 2
            )
        )
        human_total_pairs += len(human_pairs)
        human_agreeing_pairs += sum(
            left == right for left, right in human_pairs
        )

        primary_human_labels = [
            value["observed"] for value in primary_human_annotations
        ]
        senior_human_labels = [
            value["observed"] for value in senior_human_annotations
        ]
        synthetic_labels = [
            value["observed"]
            for value in annotations
            if raters[value["rater_id"]]["kind"] == "synthetic_fixture_rater"
        ]
        development_labels = [
            value["observed"]
            for value in annotations
            if raters[value["rater_id"]]["kind"] == "development_agent"
        ]
        if len(primary_human_labels) >= 2:
            reference = _majority(primary_human_labels)
            if reference is not None:
                reference_source = "declared_human_majority"
            elif senior_human_labels:
                reference = _majority(senior_human_labels)
                reference_source = (
                    "declared_human_senior_tiebreak"
                    if reference is not None
                    else "declared_human_senior_tie_unscorable"
                )
            else:
                reference_source = "declared_human_tie_without_senior_unscorable"
        elif primary_human_labels:
            reference = primary_human_labels[0]
            reference_source = "single_human_not_consensus"
        elif synthetic_labels:
            reference = _majority(synthetic_labels)
            reference_source = (
                "synthetic_fixture_consensus_not_human_gold"
                if reference is not None
                else "synthetic_fixture_tie_unscorable"
            )
        else:
            reference = _majority(development_labels)
            reference_source = (
                "development_agent_proxy_not_human_gold"
                if reference is not None
                else "development_agent_tie_unscorable"
            )
        rows.append(
            {
                **item,
                "annotation_count": len(annotations),
                "positive_votes": sum(labels),
                "negative_votes": len(labels) - sum(labels),
                "unanimous": len(set(labels)) == 1,
                "pair_count": len(pairs),
                "agreeing_pair_count": pair_agreements,
                "reference_observed": reference,
                "reference_source": reference_source,
                "declared_primary_human_annotation_count": len(
                    primary_human_labels
                ),
                "declared_senior_human_annotation_count": len(
                    senior_human_labels
                ),
                "synthetic_annotation_count": len(synthetic_labels),
                "development_agent_annotation_count": len(development_labels),
            }
        )
    metrics = {
        "pairwise_percent_agreement": (
            agreeing_pairs / total_pairs if total_pairs else None
        ),
        "agreeing_pairs": agreeing_pairs,
        "total_pairs": total_pairs,
        "unanimous_item_rate": sum(row["unanimous"] for row in rows)
        / len(rows),
        "unanimous_item_count": sum(row["unanimous"] for row in rows),
        "item_count": len(rows),
        "disagreement_item_ids": [
            row["item_id"] for row in rows if not row["unanimous"]
        ],
        "declared_primary_human_pairwise_percent_agreement": (
            human_agreeing_pairs / human_total_pairs
            if human_total_pairs
            else None
        ),
        "declared_primary_human_agreeing_pairs": human_agreeing_pairs,
        "declared_primary_human_total_pairs": human_total_pairs,
    }
    return rows, metrics


def _auroc(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scorable = [row for row in rows if row["expected_observed"] is not None]
    positives = [
        float(row["positive_score"])
        for row in scorable
        if row["expected_observed"] is True
    ]
    negatives = [
        float(row["positive_score"])
        for row in scorable
        if row["expected_observed"] is False
    ]
    if not positives or not negatives:
        return {
            "value": None,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "unavailable_reason": "requires_both_positive_and_negative_references",
        }
    credit = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return {
        "value": credit / (len(positives) * len(negatives)),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "unavailable_reason": None,
    }


def summarize_independent_validity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate multi-rater agreement and paired positive/negative controls."""

    study = validate_independent_validity_study(value)
    references, agreement = _agreement_rows(study)
    reference_by_item = {
        row["item_id"]: row["reference_observed"] for row in references
    }
    condition_by_item = {
        row["item_id"]: row["condition"] for row in references
    }
    control_rows: list[dict[str, Any]] = []
    by_polarity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for control in study["vqa_controls"]:
        reference = reference_by_item[control["item_id"]]
        expected = (
            None
            if reference is None
            else reference
            if control["polarity"] == "positive_control"
            else not reference
        )
        correct = (
            None
            if expected is None
            else control["predicted_observed"] == expected
        )
        row = {
            **control,
            "condition": condition_by_item[control["item_id"]],
            "reference_observed": reference,
            "expected_observed": expected,
            "correct": correct,
        }
        control_rows.append(row)
        by_polarity[control["polarity"]].append(row)

    def accuracy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        scorable = [row for row in rows if row["correct"] is not None]
        correct = sum(bool(row["correct"]) for row in scorable)
        return {
            "value": correct / len(scorable) if scorable else None,
            "correct": correct,
            "scorable": len(scorable),
            "unscorable": len(rows) - len(scorable),
        }

    controls_by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in control_rows:
        controls_by_item[row["item_id"]][row["polarity"]] = row
    contrasts: list[dict[str, Any]] = []
    for item in study["items"]:
        pair = controls_by_item[item["item_id"]]
        positive = pair["positive_control"]
        negative = pair["negative_control"]
        contrasts.append(
            {
                "item_id": item["item_id"],
                "prediction_changed": (
                    positive["predicted_observed"]
                    != negative["predicted_observed"]
                ),
                "both_controls_correct": (
                    None
                    if positive["correct"] is None
                    or negative["correct"] is None
                    else bool(positive["correct"] and negative["correct"])
                ),
            }
        )
    scorable_contrasts = [
        row for row in contrasts if row["both_controls_correct"] is not None
    ]
    rater_kind_counts = Counter(item["kind"] for item in study["raters"])
    rater_role_counts = Counter(item["role"] for item in study["raters"])
    annotation_kind_counts = Counter(
        item["rater_kind"] for item in study["annotations"]
    )
    human_reference_sources = {
        "declared_human_majority",
        "declared_human_senior_tiebreak",
    }
    declared_human_consensus_items = sum(
        row["reference_source"] in human_reference_sources
        for row in references
    )
    all_items_declared_human_consensus = (
        declared_human_consensus_items == len(references)
    )
    development_count = annotation_kind_counts["development_agent"]
    synthetic = study["evidence_source"] == "synthetic_fixture"
    by_condition = {
        condition: {
            "fixed_threshold_accuracy": accuracy(
                [row for row in control_rows if row["condition"] == condition]
            ),
            "auroc": _auroc(
                [row for row in control_rows if row["condition"] == condition]
            ),
            "item_count": sum(
                row["condition"] == condition for row in references
            ),
        }
        for condition in PAPER_CONDITIONS
    }
    paper_conditions_covered = all(
        by_condition[condition]["item_count"] > 0
        for condition in PAPER_CONDITIONS
    )
    four_primary_annotations_per_item = all(
        row["declared_primary_human_annotation_count"] == 4
        for row in references
    )
    paper_annotation_target_met = (
        rater_role_counts["primary_annotator"] == 4
        and rater_role_counts["senior_tiebreaker"] >= 1
        and four_primary_annotations_per_item
        and all_items_declared_human_consensus
        and paper_conditions_covered
        and not synthetic
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "study_id": study["study_id"],
        "mode": "offline_annotation_and_cached_prediction_audit",
        "evidence_source": study["evidence_source"],
        "calls_started_by_summary": {
            "provider": 0,
            "simulator": 0,
            "act": 0,
        },
        "rater_count": len(study["raters"]),
        "rater_kind_counts": dict(sorted(rater_kind_counts.items())),
        "rater_role_counts": dict(sorted(rater_role_counts.items())),
        "annotation_count": len(study["annotations"]),
        "annotation_kind_counts": dict(sorted(annotation_kind_counts.items())),
        "agreement": agreement,
        "item_references": references,
        "declared_human_consensus_item_count": declared_human_consensus_items,
        "all_items_declared_human_consensus": all_items_declared_human_consensus,
        "human_gold_status": {
            "available_by_declared_manifest": (
                all_items_declared_human_consensus
                and rater_kind_counts["human"] >= 2
                and not synthetic
            ),
            "independently_verified_by_this_aggregator": False,
            "development_agent_is_human_gold": False,
            "synthetic_fixture_is_human_gold": False,
        },
        "development_agent_annotation_count": development_count,
        "vqa_control_evaluation": {
            "fixed_threshold": study["fixed_threshold"],
            "overall": accuracy(control_rows),
            "auroc": _auroc(control_rows),
            "by_polarity": {
                polarity: accuracy(by_polarity[polarity])
                for polarity in sorted(CONTROL_POLARITIES)
            },
            "by_paper_condition": by_condition,
            "paired_contrasts": contrasts,
            "prediction_change_rate": sum(
                row["prediction_changed"] for row in contrasts
            )
            / len(contrasts),
            "both_controls_correct_rate": (
                sum(bool(row["both_controls_correct"]) for row in scorable_contrasts)
                / len(scorable_contrasts)
                if scorable_contrasts
                else None
            ),
            "rows": control_rows,
        },
        "paper_reference_configuration": {
            "primary_robotics_annotators": 4,
            "aggregation": "majority_vote",
            "tie_break": "senior_annotator",
            "vqa_conditions": list(PAPER_CONDITIONS),
            "vqa_metrics": ["fixed_threshold_accuracy", "auroc"],
            "sources": {
                "vqa_perturbations_and_metrics": "paper_appendix_A.2.4",
                "four_annotators_and_tie_break": "paper_appendix_A.4.3",
            },
        },
        "paper_reference_configuration_met": paper_annotation_target_met,
        "paper_reference_unmet": [
            reason
            for condition, reason in (
                (
                    rater_role_counts["primary_annotator"] == 4,
                    "requires_four_declared_primary_human_annotators",
                ),
                (
                    rater_role_counts["senior_tiebreaker"] >= 1,
                    "requires_declared_senior_tiebreaker",
                ),
                (
                    four_primary_annotations_per_item,
                    "requires_four_primary_annotations_per_item",
                ),
                (
                    all_items_declared_human_consensus,
                    "requires_majority_or_senior_tiebreak_reference_for_every_item",
                ),
                (
                    paper_conditions_covered,
                    "requires_clean_clutter_background_texture_lighting_coverage",
                ),
                (
                    not synthetic,
                    "synthetic_fixture_cannot_meet_paper_annotation_target",
                ),
            )
            if not condition
        ],
        "paper_table_eligible": False,
        "empirical_validity_claim_eligible": False,
        "empirical_validity_claim_ineligible_reason": (
            "synthetic_fixture"
            if synthetic
            else "rater_identity_and_source_provenance_not_authenticated_here"
        ),
        "limitations": [
            (
                "All annotations and predictions are synthetic functional fixtures."
                if synthetic
                else "Rater identities are declared by the imported manifest, not authenticated by this aggregator."
            ),
            "Development-agent annotations are proxy labels and never human gold.",
            "Agreement measures consistency, not correctness.",
            "Positive/negative controls test response sensitivity only within the imported cases.",
            "Fixed-threshold accuracy and AUROC here do not become paper evidence unless the four-condition and annotation targets are met with real imported records.",
        ],
    }


def build_synthetic_validity_demonstration() -> dict[str, Any]:
    """Build a no-human, no-runtime fixture with one explicit disagreement."""

    items = [
        {
            "item_id": "bell_pressed",
            "phenomenon_id": "bell_visibly_pressed",
            "source_id": "synthetic_clip_a",
            "condition": "clean",
        },
        {
            "item_id": "target_selected",
            "phenomenon_id": "bell_target_selected",
            "source_id": "synthetic_clip_b",
            "condition": "scene_clutter",
        },
        {
            "item_id": "return_motion",
            "phenomenon_id": "bell_return_visible",
            "source_id": "synthetic_clip_c",
            "condition": "background_texture",
        },
        {
            "item_id": "lighting_visibility",
            "phenomenon_id": "bell_visible_under_lighting",
            "source_id": "synthetic_clip_d",
            "condition": "lighting",
        },
    ]
    reference = {
        "bell_pressed": True,
        "target_selected": False,
        "return_motion": True,
        "lighting_visibility": True,
    }
    raters = [
        {
            "rater_id": "synthetic_rater_a",
            "kind": "synthetic_fixture_rater",
            "role": "proxy",
        },
        {
            "rater_id": "synthetic_rater_b",
            "kind": "synthetic_fixture_rater",
            "role": "proxy",
        },
        {
            "rater_id": "development_agent",
            "kind": "development_agent",
            "role": "proxy",
        },
    ]
    annotations = []
    for item in items:
        item_id = item["item_id"]
        for rater in raters:
            observed = reference[item_id]
            if item_id == "target_selected" and rater["rater_id"] == "development_agent":
                observed = not observed
            annotations.append(
                {
                    "item_id": item_id,
                    "rater_id": rater["rater_id"],
                    "observed": observed,
                }
            )
    controls = []
    for item in items:
        item_id = item["item_id"]
        for polarity in sorted(CONTROL_POLARITIES):
            expected = (
                reference[item_id]
                if polarity == "positive_control"
                else not reference[item_id]
            )
            predicted = expected
            if item_id == "return_motion" and polarity == "negative_control":
                predicted = reference[item_id]
            controls.append(
                {
                    "control_id": f"{item_id}_{polarity}",
                    "item_id": item_id,
                    "polarity": polarity,
                    "transformation": (
                        "synthetic_semantics_preserving_control"
                        if polarity == "positive_control"
                        else "synthetic_semantics_inverting_control"
                    ),
                    "predicted_observed": predicted,
                    "positive_score": 0.8 if predicted else 0.2,
                    "confidence": 0.8,
                }
            )
    study = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "study_id": "synthetic_multirater_vqa_control_smoke",
        "evidence_source": "synthetic_fixture",
        "fixed_threshold": 0.5,
        "items": items,
        "raters": raters,
        "annotations": annotations,
        "vqa_controls": controls,
    }
    return {
        "study": validate_independent_validity_study(study),
        "summary": summarize_independent_validity(study),
    }


__all__ = [
    "CONTROL_POLARITIES",
    "EVIDENCE_SOURCES",
    "IndependentValidityError",
    "PAPER_CONDITIONS",
    "PROTOCOL",
    "RATER_KINDS",
    "RATER_ROLES",
    "build_synthetic_validity_demonstration",
    "summarize_independent_validity",
    "validate_independent_validity_study",
]
