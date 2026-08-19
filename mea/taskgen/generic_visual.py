"""First-frame visual diagnosis for generic RoboTwin TaskGen candidates."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from mea.providers.json_response import (
    ProviderJSONError,
    extract_json_response,
)

from .preservation_facts import describe_preservation_fact


class GenericVisualDiagnosisError(ValueError):
    """Raised when a generic TaskGen visual observation is invalid."""


_RESPONSE_KEYS = {
    "schema_version",
    "render_usable",
    "key_task_actors_visible",
    "requested_change_assessment",
    "visual_physical_plausibility",
    "unexpected_changes",
    "diagnosis",
    "repair_instructions",
    "confidence",
}
_CHANGE_ASSESSMENTS = {
    "consistent",
    "contradicted",
    "not_visually_decidable",
}
_PLAUSIBILITY = {"plausible", "implausible", "uncertain"}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenericVisualDiagnosisError(f"{field} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise GenericVisualDiagnosisError(f"{field} must be a string list")
    result = [_text(item, f"{field}[]") for item in value]
    if len(result) != len(set(result)):
        raise GenericVisualDiagnosisError(f"{field} must not contain duplicates")
    return result


def validate_generic_visual_response(
    value: Mapping[str, Any],
    *,
    scene_change_passed: bool,
) -> dict[str, Any]:
    """Validate model output and derive acceptance without trusting a self-score."""

    if not isinstance(value, Mapping) or set(value) != _RESPONSE_KEYS:
        raise GenericVisualDiagnosisError(
            "generic visual response fields must be exactly "
            f"{sorted(_RESPONSE_KEYS)}"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != 1:
        raise GenericVisualDiagnosisError(
            "generic visual response schema_version must be 1"
        )
    for field in ("render_usable", "key_task_actors_visible"):
        if not isinstance(result.get(field), bool):
            raise GenericVisualDiagnosisError(f"{field} must be bool")
    assessment = result.get("requested_change_assessment")
    if assessment not in _CHANGE_ASSESSMENTS:
        raise GenericVisualDiagnosisError(
            "requested_change_assessment must be one of "
            f"{sorted(_CHANGE_ASSESSMENTS)}"
        )
    plausibility = result.get("visual_physical_plausibility")
    if plausibility not in _PLAUSIBILITY:
        raise GenericVisualDiagnosisError(
            "visual_physical_plausibility must be one of "
            f"{sorted(_PLAUSIBILITY)}"
        )
    result["unexpected_changes"] = _text_list(
        result.get("unexpected_changes"), "unexpected_changes"
    )
    result["repair_instructions"] = _text_list(
        result.get("repair_instructions"), "repair_instructions"
    )
    result["diagnosis"] = _text(result.get("diagnosis"), "diagnosis")
    confidence = result.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise GenericVisualDiagnosisError("confidence must be in [0, 1]")
    result["confidence"] = float(confidence)
    if not isinstance(scene_change_passed, bool):
        raise GenericVisualDiagnosisError("scene_change_passed must be bool")
    change_supported = assessment == "consistent" or (
        assessment == "not_visually_decidable" and scene_change_passed
    )
    result["scene_change_authority_available"] = scene_change_passed
    result["passed"] = bool(
        result["render_usable"]
        and result["key_task_actors_visible"]
        and change_supported
        and plausibility == "plausible"
        and not result["unexpected_changes"]
        and result["confidence"] >= 0.5
    )
    return result


def build_scene_comparison(
    official_image: str | Path,
    generated_image: str | Path,
    output_path: str | Path,
) -> Path:
    """Create one labeled official/generated image for a single VLM request."""

    official_path = Path(official_image).expanduser().resolve()
    generated_path = Path(generated_image).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    for label, path in (
        ("official_image", official_path),
        ("generated_image", generated_path),
    ):
        if not path.is_file():
            raise GenericVisualDiagnosisError(f"{label} is unavailable: {path}")
    with Image.open(official_path) as source:
        official = source.convert("RGB")
    with Image.open(generated_path) as source:
        generated = source.convert("RGB")
    max_height = max(official.height, generated.height)

    def fit_height(image: Image.Image) -> Image.Image:
        if image.height == max_height:
            return image
        width = max(1, round(image.width * max_height / image.height))
        return image.resize((width, max_height), Image.Resampling.LANCZOS)

    official = fit_height(official)
    generated = fit_height(generated)
    label_height = 36
    gap = 8
    canvas = Image.new(
        "RGB",
        (
            official.width + generated.width + gap,
            max_height + label_height,
        ),
        "white",
    )
    canvas.paste(official, (0, label_height))
    canvas.paste(generated, (official.width + gap, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "OFFICIAL SAME-SEED SCENE", fill="black")
    draw.text(
        (official.width + gap + 8, 10),
        "QUERY-DERIVED GENERATED SCENE",
        fill="black",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _need_description(candidate: Mapping[str, Any], field: str) -> str:
    need = candidate.get(field)
    if need is None:
        return "No change requested; preserve the official implementation."
    if isinstance(need, Mapping):
        return _text(need.get("description"), f"candidate.{field}.description")
    return _text(need, f"candidate.{field}")


def build_generic_visual_prompt(candidate: Mapping[str, Any]) -> str:
    """Describe only visual judgments that one initial frame can support."""

    concern = _text(candidate.get("semantic_concern"), "semantic_concern")
    scene_need = _need_description(candidate, "scene_need")
    checker_need = _need_description(candidate, "checker_need")
    evaluation_intent = candidate.get("evaluation_intent")
    preserved_conditions = (
        evaluation_intent.get("preserved_conditions") or []
        if isinstance(evaluation_intent, Mapping)
        else []
    )
    preserved_context = (
        "\n".join(
            "- " + describe_preservation_fact(item)
            if isinstance(item, Mapping)
            else "- unverified legacy condition: " + str(item)
            for item in preserved_conditions
        )
        if preserved_conditions
        else "- None declared."
    )
    example = {
        "schema_version": 1,
        "render_usable": True,
        "key_task_actors_visible": True,
        "requested_change_assessment": "consistent",
        "visual_physical_plausibility": "plausible",
        "unexpected_changes": [],
        "diagnosis": "The generated scene is visible and physically plausible.",
        "repair_instructions": [],
        "confidence": 0.8,
    }
    return f"""You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
{concern}

REQUESTED SCENE NEED:
{scene_need}

CHECKER NEED (context only; RGB cannot validate success logic):
{checker_need}

DECLARED CONDITIONS TO PRESERVE:
{preserved_context}

Judge only visible facts: render usability, whether key task actors are visible,
whether the requested visible change is consistent or contradicted, obvious
physical implausibility, and visible unintended changes. Report every
visible preservation violation of a declared condition in unexpected_changes. Use
not_visually_decidable for mass, friction, identity, exact coordinates,
contacts, predicates, or other facts that RGB cannot establish. Do not infer
checker correctness or task success from the initial frame.
requested_change_assessment must be exactly one of: consistent, contradicted,
not_visually_decidable. Never substitute synonyms such as inconsistent.
visual_physical_plausibility must be exactly one of: plausible, implausible,
uncertain. Never substitute synonyms such as realistic or good.

Return strict JSON with exactly these fields:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


def diagnose_generic_scene_render(
    provider: Any,
    candidate: Mapping[str, Any],
    *,
    official_image: str | Path,
    generated_image: str | Path,
    output_dir: str | Path,
    model: str,
    scene_change_passed: bool,
) -> dict[str, Any]:
    """Run and persist one VLM diagnosis for a generated scene attempt."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    comparison = build_scene_comparison(
        official_image,
        generated_image,
        destination / "official_vs_generated.png",
    )
    prompt = build_generic_visual_prompt(candidate)
    (destination / "vision_prompt.md").write_text(prompt, encoding="utf-8")
    response = provider.vision(
        prompt,
        comparison,
        model=_text(model, "model"),
        max_tokens=700,
        temperature=0.0,
    )
    (destination / "vision_response.txt").write_text(
        str(response) + "\n", encoding="utf-8"
    )
    try:
        parsed = extract_json_response(str(response))
    except ProviderJSONError as exc:
        raise GenericVisualDiagnosisError(str(exc)) from exc
    result = validate_generic_visual_response(
        parsed,
        scene_change_passed=scene_change_passed,
    )
    result["model_requested"] = model
    result["provider_metadata"] = deepcopy(
        dict(getattr(provider, "last_metadata", {}) or {})
    )
    result["comparison_image"] = "official_vs_generated.png"
    (destination / "vision.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "GenericVisualDiagnosisError",
    "build_generic_visual_prompt",
    "build_scene_comparison",
    "diagnose_generic_scene_render",
    "validate_generic_visual_response",
]
