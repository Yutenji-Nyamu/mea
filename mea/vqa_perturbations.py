"""Deterministic image-proxy perturbations for cached Execution VQA montages."""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageEnhance


class VQAPerturbationError(ValueError):
    pass


PERTURBATIONS = (
    "clean",
    "scene_clutter_image_proxy",
    "background_texture_image_proxy",
    "lighting_image_proxy",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_perturbation_suite(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise VQAPerturbationError("perturbation suite schema_version must be 1")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise VQAPerturbationError("perturbation suite must contain five source clips")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise VQAPerturbationError("perturbation case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise VQAPerturbationError("perturbation case id is missing or duplicate")
        seen.add(case_id)
        if not isinstance(case.get("phenomenon_id"), str):
            raise VQAPerturbationError(f"{case_id} phenomenon_id is invalid")
        if not isinstance(case.get("gold_observed"), bool):
            raise VQAPerturbationError(f"{case_id} gold_observed must be boolean")
        if case.get("label_source") != "simulator_proxy":
            raise VQAPerturbationError(
                f"{case_id} must not present cached proxy labels as human"
            )
    return dict(value)


def build_proxy_images(
    source: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    source_path = Path(source).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise VQAPerturbationError(f"source montage does not exist: {source_path}")
    if destination.exists():
        raise VQAPerturbationError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)
    source_hash = file_sha256(source_path)
    source_image = Image.open(source_path).convert("RGB")
    records: list[dict[str, Any]] = []
    for index, name in enumerate(PERTURBATIONS):
        output = destination / f"{name}.png"
        parameters: dict[str, Any]
        if name == "clean":
            shutil.copyfile(source_path, output)
            parameters = {"operation": "byte_exact_copy"}
        elif name == "scene_clutter_image_proxy":
            image = source_image.copy()
            draw = ImageDraw.Draw(image, "RGBA")
            rng = random.Random(int(seed) + index * 1009)
            count = 12
            for _ in range(count):
                width = rng.randint(max(8, image.width // 40), max(12, image.width // 16))
                height = rng.randint(max(8, image.height // 40), max(12, image.height // 16))
                x = rng.randint(0, max(image.width - width, 0))
                y = rng.randint(0, max(image.height - height, 0))
                color = (rng.randrange(256), rng.randrange(256), rng.randrange(256), 150)
                draw.rectangle((x, y, x + width, y + height), fill=color)
            image.save(output)
            parameters = {"seed": int(seed), "shape_count": count, "alpha": 150}
        elif name == "background_texture_image_proxy":
            image = source_image.copy().convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            cell = max(12, min(image.size) // 24)
            for y in range(0, image.height, cell):
                for x in range(0, image.width, cell):
                    if (x // cell + y // cell) % 2:
                        draw.rectangle(
                            (x, y, x + cell, y + cell), fill=(40, 120, 210, 42)
                        )
            Image.alpha_composite(image, overlay).convert("RGB").save(output)
            parameters = {"checker_cell_px": cell, "alpha": 42}
        else:
            factor = 0.55 if int(seed) % 2 else 1.45
            ImageEnhance.Brightness(source_image).enhance(factor).save(output)
            parameters = {"brightness_factor": factor}
        derived_hash = file_sha256(output)
        if name == "clean" and derived_hash != source_hash:
            raise VQAPerturbationError("clean condition is not a byte-exact copy")
        records.append(
            {
                "condition": name,
                "source_montage": str(source_path),
                "source_sha256": source_hash,
                "derived_image": str(output),
                "derived_sha256": derived_hash,
                "parameters": parameters,
                "paper_condition_equivalent": name == "clean",
                "condition_scope": (
                    "cached_clean" if name == "clean" else "image_proxy_not_simulator_perturbation"
                ),
            }
        )
    return records
