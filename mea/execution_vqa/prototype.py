"""Bounded Execution VQA over rollout video and simulator evidence.

The numeric Tool results are authoritative.  Vision can add appearance-level
observations or report an evidence conflict, but this module never rewrites a
simulator result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mea.taskgen import extract_json_response

from .query import (
    LEGACY_PHENOMENON_IDS,
    QUESTION_CATALOG,
    build_execution_vqa_query,
    validate_execution_vqa_query,
)


class ExecutionVQAError(RuntimeError):
    """Raised when video evidence or a Vision response violates its contract."""


PHENOMENON_IDS = set(QUESTION_CATALOG)
CONSISTENCY_VALUES = {"consistent", "conflict", "uncertain"}


def _tool_results_by_name(
    tool_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    def add_semantic_aliases(
        result: dict[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        for value in list(result.values()):
            details = value.get("details")
            if not isinstance(details, Mapping):
                continue
            if {
                "pickup_detected",
                "contact_detected",
                "pickup_physics_step",
                "contact_physics_step",
            }.issubset(details):
                result.setdefault("pickup_to_first_contact_time", value)
        return result

    if tool_results is None:
        return {}
    if isinstance(tool_results, Mapping):
        if "tool" in tool_results:
            name = str(tool_results.get("tool") or "")
            return add_semantic_aliases({name: tool_results} if name else {})
        result: dict[str, Mapping[str, Any]] = {}
        for name, value in tool_results.items():
            if isinstance(value, Mapping):
                result[str(name)] = value
        return add_semantic_aliases(result)
    result = {}
    for value in tool_results:
        if not isinstance(value, Mapping):
            continue
        name = str(value.get("tool") or "")
        if name:
            result[name] = value
    return add_semantic_aliases(result)


def read_contact_events(path: str | Path | None) -> list[dict[str, Any]]:
    """Read the sparse recorder timeline without requiring a TrajectoryView."""

    if path is None:
        return []
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return []
    events = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExecutionVQAError(
                f"Invalid JSON in {source.name} line {line_number}"
            ) from exc
        if isinstance(value, dict):
            events.append(value)
    return events


def read_semantic_timeline(path: str | Path | None) -> dict[int, int]:
    """Return the recorded physics-step to policy-frame mapping."""

    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {}
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - RoboTwin depends on NumPy
        raise ExecutionVQAError("Reading semantic timeline requires NumPy") from exc
    with np.load(source) as archive:
        if "physics_step" not in archive or "policy_step" not in archive:
            raise ExecutionVQAError(
                f"semantic timeline lacks physics_step/policy_step: {source}"
            )
        physics = archive["physics_step"].reshape(-1)
        policy = archive["policy_step"].reshape(-1)
        if len(physics) != len(policy):
            raise ExecutionVQAError("semantic timeline arrays have different lengths")
        return {
            int(physics_step): max(int(policy_step), 0)
            for physics_step, policy_step in zip(physics, policy)
        }


def _evidence_frame(
    result: Mapping[str, Any] | None,
    which: str,
    *,
    physics_to_policy: Mapping[int, int] | None = None,
    evidence_position: str = "first",
) -> int | None:
    if not result:
        return None
    evidence = result.get("evidence")
    if isinstance(evidence, list) and evidence and isinstance(evidence[0], Mapping):
        item = evidence[0]
        keys = (
            ("video_frame_before", "policy_step")
            if which == "before"
            else ("video_frame_after", "policy_step")
        )
        for key in keys:
            value = item.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                frame = int(value)
                return (
                    frame
                    if which == "before" or key == "video_frame_after"
                    else frame + 1
                )
    evidence_steps = result.get("evidence_steps")
    if (
        isinstance(evidence_steps, list)
        and evidence_steps
        and physics_to_policy
    ):
        ordered = sorted(
            int(value)
            for value in evidence_steps
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        if ordered:
            physics_step = ordered[-1] if evidence_position == "last" else ordered[0]
            policy_step = physics_to_policy.get(physics_step)
            if policy_step is not None:
                return max(int(policy_step) + (which == "after"), 0)
    return None


def _first_event_frame(events: Iterable[Mapping[str, Any]]) -> int | None:
    candidates = []
    for event in events:
        if event.get("type") != "contact_interval":
            continue
        if event.get("physical_contact") is not True:
            continue
        actors = event.get("actors")
        if not isinstance(actors, list):
            continue
        normalized = [str(actor).casefold() for actor in actors]
        has_hammer = any("hammer" in actor for actor in normalized)
        has_block = any(
            actor == "box" or "block" in actor for actor in normalized
        )
        if not (has_hammer and has_block):
            continue
        for key in (
            "first_physical_policy_step",
            "start_policy_step",
            "policy_step",
        ):
            value = event.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                candidates.append(int(value))
                break
    return min(candidates) if candidates else None


def select_keyframes(
    *,
    frame_count: int,
    tool_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    physics_to_policy: Mapping[int, int] | None = None,
    max_frames: int = 8,
    min_frames: int = 4,
) -> list[dict[str, Any]]:
    """Select initial/event-adjacent/final frames using numeric evidence.

    ``video_frame_before`` and ``video_frame_after`` in Tool evidence are used
    directly.  Contact events are only a fallback.  When an event is absent,
    uniformly spaced context frames fill the sheet to four frames, so missing
    pickup/contact remains visually inspectable instead of silently producing
    an almost empty request.
    """

    if frame_count <= 0:
        raise ExecutionVQAError("frame_count must be positive")
    if max_frames < 4 or max_frames > 8:
        raise ExecutionVQAError("max_frames must be between 4 and 8")
    min_frames = max(1, min(int(min_frames), max_frames, frame_count))
    by_name = _tool_results_by_name(tool_results)

    duration = by_name.get("pickup_to_first_contact_time")
    duration_details = (
        duration.get("details", {}) if isinstance(duration, Mapping) else {}
    )
    pickup = by_name.get("first_hammer_pickup_step")
    if pickup is None and duration_details.get("pickup_detected") is True:
        pickup = duration
    contact = by_name.get("first_contact_step")
    if contact is None:
        contact = by_name.get("hammer_block_contact_ever")
    if contact is None and duration_details.get("contact_detected") is True:
        contact = duration

    candidates: list[tuple[str, int, str]] = [
        ("initial", 0, "video_boundary"),
    ]
    pickup_before = _evidence_frame(
        pickup, "before", physics_to_policy=physics_to_policy
    )
    pickup_after = _evidence_frame(
        pickup, "after", physics_to_policy=physics_to_policy
    )
    if pickup_before is not None:
        candidates.append(("pickup_before", pickup_before, "tool_evidence"))
    if pickup_after is not None:
        candidates.append(("pickup_after", pickup_after, "tool_evidence"))

    contact_position = "last" if contact is duration else "first"
    contact_before = _evidence_frame(
        contact,
        "before",
        physics_to_policy=physics_to_policy,
        evidence_position=contact_position,
    )
    contact_after = _evidence_frame(
        contact,
        "after",
        physics_to_policy=physics_to_policy,
        evidence_position=contact_position,
    )
    if contact_before is None:
        event_frame = _first_event_frame(events or [])
        if event_frame is not None:
            # Rollout videos store the pre-action observation for policy step
            # k.  A contact during action k is therefore bracketed by frames
            # k and k+1, matching the Trusted Tool evidence convention.
            contact_before = event_frame
            contact_after = event_frame + 1
    if contact_before is not None:
        candidates.append(("contact_before", contact_before, "contact_timeline"))
    if contact_after is not None:
        candidates.append(("contact_after", contact_after, "contact_timeline"))
    candidates.append(("final", frame_count - 1, "video_boundary"))

    # Preserve semantic labels while de-duplicating clamped frame indices.
    selected: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for frame_id, index, source in candidates:
        index = max(0, min(int(index), frame_count - 1))
        if index in used_indices:
            continue
        used_indices.add(index)
        selected.append(
            {"frame_id": frame_id, "frame_index": index, "source": source}
        )

    # Add deterministic context frames only when real event evidence is sparse.
    if len(selected) < min_frames:
        denominator = max(min_frames - 1, 1)
        for position in range(1, min_frames - 1):
            index = round(position * (frame_count - 1) / denominator)
            if index in used_indices:
                continue
            used_indices.add(index)
            selected.append(
                {
                    "frame_id": f"context_{position}",
                    "frame_index": index,
                    "source": "uniform_fallback",
                }
            )

    selected.sort(key=lambda item: int(item["frame_index"]))
    if len(selected) > max_frames:
        selected = selected[: max_frames - 1] + [selected[-1]]
    return selected


def _read_video_frames(
    video_path: Path,
    indices: Sequence[int] | None = None,
) -> tuple[int, float, dict[int, Any]]:
    """Read selected video frames with OpenCV, imported lazily for tests/docs."""

    try:
        import cv2
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on RoboTwin runtime
        raise ExecutionVQAError(
            "Execution VQA requires opencv-python and Pillow at runtime"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExecutionVQAError(f"Cannot open rollout video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise ExecutionVQAError(f"Rollout video has no frames: {video_path}")
        frames = {}
        for index in indices or []:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise ExecutionVQAError(
                    f"Cannot decode rollout video frame {index}: {video_path}"
                )
            frames[int(index)] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return frame_count, fps, frames
    finally:
        capture.release()


def _make_montage(
    *,
    reference_scene: Path | None,
    selected: Sequence[Mapping[str, Any]],
    frames: Mapping[int, Any],
    destination: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise ExecutionVQAError("Execution VQA requires Pillow") from exc

    tiles: list[tuple[str, Any]] = []
    if reference_scene is not None:
        if not reference_scene.is_file():
            raise FileNotFoundError(reference_scene)
        with Image.open(reference_scene) as source:
            tiles.append(("reference_scene", source.convert("RGB").copy()))
    for item in selected:
        index = int(item["frame_index"])
        if index not in frames:
            raise ExecutionVQAError(f"Decoded frame {index} is missing")
        tiles.append((f"{item['frame_id']} | frame {index}", frames[index].copy()))

    tile_width, tile_height, label_height = 320, 240, 28
    columns = 3 if len(tiles) > 4 else 2
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for tile_index, (label, image) in enumerate(tiles):
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (tile_index % columns) * tile_width
        y = (tile_index // columns) * (tile_height + label_height)
        paste_x = x + (tile_width - image.width) // 2
        paste_y = y + label_height + (tile_height - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
        draw.text((x + 6, y + 7), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def build_execution_montage(
    *,
    video_path: str | Path,
    destination: str | Path,
    tool_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    events_path: str | Path | None = None,
    reference_scene: str | Path | None = None,
    semantic_trace_path: str | Path | None = None,
    max_frames: int = 8,
) -> dict[str, Any]:
    """Decode selected frames and write one reference-plus-rollout PNG sheet."""

    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    frame_count, fps, _ = _read_video_frames(video)
    events = read_contact_events(events_path)
    physics_to_policy = read_semantic_timeline(semantic_trace_path)
    selected = select_keyframes(
        frame_count=frame_count,
        tool_results=tool_results,
        events=events,
        physics_to_policy=physics_to_policy,
        max_frames=max_frames,
    )
    _, _, frames = _read_video_frames(
        video, [int(item["frame_index"]) for item in selected]
    )
    output = Path(destination).expanduser().resolve()
    reference = (
        Path(reference_scene).expanduser().resolve() if reference_scene else None
    )
    _make_montage(
        reference_scene=reference,
        selected=selected,
        frames=frames,
        destination=output,
    )
    return {
        "video_path": str(video),
        "frame_count": frame_count,
        "fps": fps,
        "selected_frames": selected,
        "reference_scene": str(reference) if reference else None,
        "montage_path": str(output),
    }


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ExecutionVQAError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionVQAError(f"{name} must be numeric") from exc
    if not 0.0 <= result <= 1.0:
        raise ExecutionVQAError(f"{name} must be between 0 and 1")
    return result


def _frame_ids(value: Any, *, allowed: set[str], name: str) -> list[str]:
    if not isinstance(value, list):
        raise ExecutionVQAError(f"{name} must be a list")
    result = []
    for item in value:
        frame_id = str(item)
        if frame_id not in allowed:
            raise ExecutionVQAError(f"{name} references unknown frame_id={frame_id}")
        if frame_id not in result:
            result.append(frame_id)
    return result


def validate_execution_vqa_response(
    value: dict[str, Any],
    *,
    allowed_frame_ids: Sequence[str],
    expected_phenomenon_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the fixed Vision schema and derive ``evidence_conflict``."""

    if not isinstance(value, dict):
        raise ExecutionVQAError("Execution VQA response must be a JSON object")
    expected_keys = {
        "phenomena",
        "confidence",
        "frame_ids",
        "numeric_consistency",
        "conflicts",
    }
    if set(value) != expected_keys:
        raise ExecutionVQAError(
            "Execution VQA response keys must be exactly "
            + ", ".join(sorted(expected_keys))
        )
    allowed = {str(item) for item in allowed_frame_ids}
    expected_ids = (
        tuple(LEGACY_PHENOMENON_IDS)
        if expected_phenomenon_ids is None
        else tuple(expected_phenomenon_ids)
    )
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ExecutionVQAError(
            "expected_phenomenon_ids must be non-empty and unique"
        )
    if any(item not in PHENOMENON_IDS for item in expected_ids):
        raise ExecutionVQAError("expected_phenomenon_ids contains unknown ids")
    phenomena_value = value.get("phenomena")
    if not isinstance(phenomena_value, list) or not phenomena_value:
        raise ExecutionVQAError("phenomena must be a non-empty list")
    phenomena = []
    seen_phenomena: set[str] = set()
    for index, item in enumerate(phenomena_value):
        if not isinstance(item, dict):
            raise ExecutionVQAError(f"phenomena[{index}] must be an object")
        expected_phenomenon_keys = {
            "id",
            "observed",
            "description",
            "confidence",
            "frame_ids",
        }
        if set(item) != expected_phenomenon_keys:
            raise ExecutionVQAError(
                f"phenomena[{index}] has invalid keys"
            )
        phenomenon_id = str(item.get("id"))
        if phenomenon_id not in expected_ids:
            raise ExecutionVQAError(
                f"phenomena[{index}].id is not allowlisted: {phenomenon_id}"
            )
        if phenomenon_id in seen_phenomena:
            raise ExecutionVQAError(
                f"phenomena contains duplicate id: {phenomenon_id}"
            )
        seen_phenomena.add(phenomenon_id)
        observed = item.get("observed")
        if observed is not None and not isinstance(observed, bool):
            raise ExecutionVQAError(
                f"phenomena[{index}].observed must be boolean or null"
            )
        phenomena.append(
            {
                "id": phenomenon_id,
                "observed": observed,
                "description": str(item.get("description") or "").strip(),
                "confidence": _number(
                    item.get("confidence"), name=f"phenomena[{index}].confidence"
                ),
                "frame_ids": _frame_ids(
                    item.get("frame_ids"),
                    allowed=allowed,
                    name=f"phenomena[{index}].frame_ids",
                ),
            }
        )

    missing_phenomena = sorted(set(expected_ids) - seen_phenomena)
    if missing_phenomena:
        raise ExecutionVQAError(
            "phenomena must contain every allowlisted id exactly once; missing: "
            + ", ".join(missing_phenomena)
        )

    consistency = str(value.get("numeric_consistency") or "").strip().lower()
    if consistency not in CONSISTENCY_VALUES:
        raise ExecutionVQAError(
            "numeric_consistency must be consistent, conflict, or uncertain"
        )
    conflicts_value = value.get("conflicts")
    if not isinstance(conflicts_value, list):
        raise ExecutionVQAError("conflicts must be a list")
    conflicts = []
    for index, item in enumerate(conflicts_value):
        if not isinstance(item, dict) or set(item) != {
            "phenomenon",
            "description",
            "frame_ids",
        }:
            raise ExecutionVQAError(f"conflicts[{index}] has invalid schema")
        conflicts.append(
            {
                "phenomenon": str(item.get("phenomenon") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "frame_ids": _frame_ids(
                    item.get("frame_ids"),
                    allowed=allowed,
                    name=f"conflicts[{index}].frame_ids",
                ),
            }
        )
    if consistency == "conflict" and not conflicts:
        raise ExecutionVQAError(
            "numeric_consistency=conflict requires at least one conflict"
        )
    if consistency == "consistent" and conflicts:
        raise ExecutionVQAError(
            "numeric_consistency=consistent requires conflicts=[]"
        )
    return {
        "phenomena": phenomena,
        "confidence": _number(value.get("confidence"), name="confidence"),
        "frame_ids": _frame_ids(
            value.get("frame_ids"), allowed=allowed, name="frame_ids"
        ),
        "numeric_consistency": consistency,
        "conflicts": conflicts,
        "evidence_conflict": consistency == "conflict" or bool(conflicts),
    }


def _numeric_pickup_observed(tool_results: Any) -> bool | None:
    by_name = _tool_results_by_name(tool_results)
    first = by_name.get("first_hammer_pickup_step")
    if first is not None:
        return first.get("value") is not None
    height = by_name.get("hammer_pickup_height")
    if height is not None and isinstance(height.get("passed"), bool):
        return bool(height["passed"])
    duration = by_name.get("pickup_to_first_contact_time")
    if duration is not None:
        detected = duration.get("details", {}).get("pickup_detected")
        if isinstance(detected, bool):
            return detected
    return None


def _apply_numeric_guard(
    observation: dict[str, Any], numeric_tool_results: Any
) -> dict[str, Any]:
    """Turn a known visual/numeric mismatch into an auditable conflict.

    This deterministic guard covers only the pickup fact for which the current
    telemetry has a precise threshold.  Color has no numeric oracle and visible
    displacement is deliberately not converted into an exact distance claim.
    """

    pickup = _numeric_pickup_observed(numeric_tool_results)
    if pickup is None:
        return observation
    phenomenon = next(
        (
            item
            for item in observation["phenomena"]
            if item["id"] == "hammer_visibly_lifted"
        ),
        None,
    )
    if phenomenon is None or phenomenon["observed"] is None:
        return observation
    if bool(phenomenon["observed"]) == pickup:
        return observation

    conflict = {
        "phenomenon": "hammer_visibly_lifted",
        "description": (
            "Deterministic numeric guard found pickup="
            f"{str(pickup).lower()} while Vision reported observed="
            f"{str(bool(phenomenon['observed'])).lower()}."
        ),
        "frame_ids": list(phenomenon["frame_ids"]),
    }
    if conflict not in observation["conflicts"]:
        observation["conflicts"].append(conflict)
    observation["numeric_consistency"] = "conflict"
    observation["evidence_conflict"] = True
    return observation


def _vision_prompt(
    *,
    selection: Mapping[str, Any],
    numeric_tool_results: Any,
    query: Mapping[str, Any],
) -> str:
    frames = selection.get("selected_frames", [])
    frame_ids = [str(item["frame_id"]) for item in frames]
    questions = query.get("questions", [])
    phenomenon_ids = query.get("phenomenon_ids", [])
    return f"""You are the Execution VQA observer for an already completed RoboTwin rollout.

The image is a labeled sheet containing an optional reference scene and selected
rollout frames. The reference tile is comparison context only; all phenomena must
describe the labeled rollout frame ids. The simulator-derived numeric Tool results
below are authoritative. Do not overwrite or recalculate them. Report an apparent
disagreement only as a conflict for the Feedback/Plan Agent.

SELECTED FRAME IDS:
{json.dumps(frame_ids, ensure_ascii=False)}

NUMERIC TOOL RESULTS:
{json.dumps(numeric_tool_results, ensure_ascii=False, indent=2)}

AUDITED VISUAL QUERY CONTRACT:
{json.dumps(questions, ensure_ascii=False, indent=2)}

Check only the allowlisted phenomena in that query contract. Exact distance,
contact, impulse, success, and every field marked simulator-authoritative remain
simulator judgments. Never turn ToolSpec free text into an additional question.

Return JSON only, with exactly this schema:
{{
  "phenomena": [
    {{
      "id": "one id from the audited query contract",
      "observed": true,
      "description": "short observation",
      "confidence": 0.0,
      "frame_ids": ["initial"]
    }}
  ],
  "confidence": 0.0,
  "frame_ids": ["initial"],
  "numeric_consistency": "consistent | conflict | uncertain",
  "conflicts": [
    {{
      "phenomenon": "hammer_visibly_lifted",
      "description": "visual and numeric evidence disagree",
      "frame_ids": ["pickup_after"]
    }}
  ]
}}
Use conflicts=[] when no conflict is visible. Never invent frame ids.
Use observed=null when the selected visual evidence is insufficient.
Return exactly one phenomena item for each requested id, in this exact order:
{json.dumps(phenomenon_ids, ensure_ascii=False)}
"""


def analyze_execution_montage(
    *,
    provider: Any,
    model: str,
    montage_path: str | Path,
    selection: Mapping[str, Any],
    numeric_tool_results: Any,
    query: Mapping[str, Any] | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Run one Vision call and preserve numeric evidence unchanged in the result."""

    frames = selection.get("selected_frames")
    if not isinstance(frames, list) or not frames:
        raise ExecutionVQAError("selection.selected_frames is required")
    frame_ids = [str(item["frame_id"]) for item in frames]
    resolved_query = validate_execution_vqa_query(
        dict(query) if query is not None else build_execution_vqa_query()
    )
    prompt = _vision_prompt(
        selection=selection,
        numeric_tool_results=numeric_tool_results,
        query=resolved_query,
    )
    response = provider.vision(
        prompt,
        Path(montage_path).expanduser().resolve(),
        model=model,
        max_tokens=1024,
        temperature=0.0,
    )
    try:
        parsed = extract_json_response(response)
    except Exception as exc:
        raise ExecutionVQAError("Vision provider did not return valid JSON") from exc
    observation = validate_execution_vqa_response(
        parsed,
        allowed_frame_ids=frame_ids,
        expected_phenomenon_ids=resolved_query["phenomenon_ids"],
    )
    observation = _apply_numeric_guard(observation, numeric_tool_results)
    result = {
        "schema_version": 1,
        "model_requested": model,
        "selection": dict(selection),
        "query": resolved_query,
        # Simulator evidence is copied verbatim and never merged with Vision fields.
        "numeric_tool_results": numeric_tool_results,
        "observation": observation,
        "evidence_conflict": bool(observation["evidence_conflict"]),
        "provider_metadata": dict(getattr(provider, "last_metadata", {})),
    }
    if destination is not None:
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        prompt_path = output.with_name(f"{output.stem}_prompt.md")
        response_path = output.with_name(f"{output.stem}_response.txt")
        prompt_path.write_text(prompt, encoding="utf-8")
        response_path.write_text(response + "\n", encoding="utf-8")
        result["artifacts"] = {
            "result": str(output),
            "prompt": str(prompt_path),
            "response": str(response_path),
        }
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def run_execution_vqa(
    *,
    provider: Any,
    model: str,
    video_path: str | Path,
    output_dir: str | Path,
    numeric_tool_results: Any,
    events_path: str | Path | None = None,
    reference_scene: str | Path | None = None,
    semantic_trace_path: str | Path | None = None,
    max_frames: int = 8,
    query: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build evidence, call Vision once, and persist a complete audit bundle."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selection = build_execution_montage(
        video_path=video_path,
        destination=destination / "execution_montage.png",
        tool_results=numeric_tool_results,
        events_path=events_path,
        reference_scene=reference_scene,
        semantic_trace_path=semantic_trace_path,
        max_frames=max_frames,
    )
    (destination / "keyframe_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_query = validate_execution_vqa_query(
        dict(query) if query is not None else build_execution_vqa_query()
    )
    query_path = destination / "execution_vqa_query.json"
    query_path.write_text(
        json.dumps(resolved_query, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = analyze_execution_montage(
        provider=provider,
        model=model,
        montage_path=selection["montage_path"],
        selection=selection,
        numeric_tool_results=numeric_tool_results,
        query=resolved_query,
        destination=destination / "execution_vqa.json",
    )
    result.setdefault("artifacts", {}).update(
        {
            "montage": str(destination / "execution_montage.png"),
            "selection": str(destination / "keyframe_selection.json"),
            "query": str(query_path),
        }
    )
    (destination / "execution_vqa.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
