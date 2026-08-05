"""Validated artifact publication and bundle-manifest writing."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from .evidence_projection import EvidenceReportError


MAX_PUBLIC_ARTIFACT_BYTES = 5_000_000
PLAN_ARTIFACTS = (
    "query_sufficiency_contract.json",
    "global_query_route.json",
    "open_task_resolution.json",
)


class EvidencePublisher:
    """Own all writes into one compact evidence bundle."""

    def __init__(
        self,
        *,
        root: Path,
        bundle_root: Path,
        publish: bool,
    ) -> None:
        self.root = root
        self.bundle_root = bundle_root
        self.publish = bool(publish)
        self.published_files: list[str] = []
        self.copied_destinations: set[Path] = set()
        self.asset_dir = bundle_root / "assets"
        self.code_dir = bundle_root / "code"
        self.data_dir = bundle_root / "data"
        self.semantic_dir = bundle_root / "artifacts"

    def prepare(self) -> None:
        self.bundle_root.mkdir(parents=True, exist_ok=True)
        previous_manifest_path = (
            self.bundle_root / "evidence_bundle_manifest.json"
        )
        if previous_manifest_path.is_file():
            previous = self.read_json(previous_manifest_path, required=True)
            old_files = previous.get("files")
            if not isinstance(old_files, list):
                raise EvidenceReportError(
                    "previous evidence manifest has no files list"
                )
            old_path_root = (
                self.bundle_root
                if previous.get("path_basis") == "bundle_relative"
                else self.root
            )
            old_paths: list[Path] = []
            for raw in old_files:
                if not isinstance(raw, str) or not raw:
                    raise EvidenceReportError(
                        "previous evidence manifest has invalid path"
                    )
                old_path = old_path_root / raw
                old_resolved = old_path.resolve()
                if self.bundle_root not in old_resolved.parents:
                    raise EvidenceReportError(
                        "previous evidence manifest points outside its bundle"
                    )
                if old_path.is_symlink() or old_path.absolute() != old_resolved:
                    raise EvidenceReportError(
                        "refusing to clear a symlinked old artifact"
                    )
                old_paths.append(old_resolved)
            current_files = {
                path.resolve()
                for path in self.bundle_root.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            if self.publish and current_files != set(old_paths):
                raise EvidenceReportError(
                    "previous manifest did not account for every published file"
                )
            for old_path in old_paths:
                if old_path.is_file():
                    old_path.unlink()
        elif self.publish and any(self.bundle_root.iterdir()):
            raise EvidenceReportError(
                "publish destination must be fresh or contain its prior "
                "evidence_bundle_manifest.json"
            )

    @staticmethod
    def read_json(path: Path, *, required: bool = False) -> dict[str, Any]:
        if not path.is_file():
            if required:
                raise EvidenceReportError(
                    f"required JSON artifact is missing: {path}"
                )
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceReportError(
                f"invalid JSON artifact {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceReportError(
                f"JSON artifact must contain an object: {path}"
            )
        return value

    def publish_copy(
        self,
        source: Path | str | None,
        destination: Path,
        *,
        allowed_suffixes: frozenset[str] | None = None,
        max_bytes: int = MAX_PUBLIC_ARTIFACT_BYTES,
        skip_oversize: bool = False,
    ) -> Path | None:
        if source is None or (isinstance(source, str) and not source.strip()):
            return None
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = self.root / source_path
        if not source_path.exists():
            return None
        source_resolved = source_path.resolve()
        if self.root not in source_resolved.parents:
            raise EvidenceReportError("artifact source is outside repo root")
        if (
            not source_resolved.is_file()
            or source_path.is_symlink()
            or source_path.absolute() != source_resolved
        ):
            raise EvidenceReportError(
                "artifact source must be a regular non-symlink file"
            )
        if (
            allowed_suffixes
            and source_resolved.suffix.lower() not in allowed_suffixes
        ):
            raise EvidenceReportError(
                f"artifact has invalid role suffix: {source_resolved.name}"
            )
        destination_resolved = destination.resolve()
        if self.bundle_root not in destination_resolved.parents:
            raise EvidenceReportError("artifact destination is outside bundle")
        if (
            destination.is_symlink()
            or destination.absolute() != destination_resolved
        ):
            raise EvidenceReportError(
                "artifact destination must be a fresh non-symlink path"
            )
        if destination_resolved in self.copied_destinations:
            raise EvidenceReportError("duplicate artifact destination")
        if destination_resolved.exists():
            raise EvidenceReportError("artifact destination already exists")
        if source_resolved.stat().st_size > max_bytes:
            if skip_oversize:
                return None
            raise EvidenceReportError(
                f"artifact exceeds {max_bytes} byte limit: "
                f"{source_resolved.name}"
            )
        destination_resolved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_resolved, destination_resolved)
        self.copied_destinations.add(destination_resolved)
        relative = destination_resolved.relative_to(
            self.bundle_root
        ).as_posix()
        self.published_files.append(relative)
        return destination_resolved

    def publish_first(
        self,
        sources: tuple[Path, ...],
        destination: Path,
    ) -> Path | None:
        source = next((item for item in sources if item.is_file()), None)
        return self.publish_copy(source, destination)

    def publish_json(
        self,
        value: Mapping[str, Any],
        destination: Path,
    ) -> Path:
        destination_resolved = destination.resolve()
        if self.bundle_root not in destination_resolved.parents:
            raise EvidenceReportError("artifact destination is outside bundle")
        if (
            destination.is_symlink()
            or destination.absolute() != destination_resolved
        ):
            raise EvidenceReportError(
                "artifact destination must be a fresh non-symlink path"
            )
        if destination_resolved.exists():
            raise EvidenceReportError("artifact destination already exists")
        destination_resolved.parent.mkdir(parents=True, exist_ok=True)
        destination_resolved.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.published_files.append(
            destination_resolved.relative_to(self.bundle_root).as_posix()
        )
        return destination_resolved

    def finish(
        self,
        *,
        report_path: Path,
        report_lines: list[str],
        run_summary_path: Path,
        manifest: Mapping[str, Any],
        evaluation: Path,
        round_count: int,
        max_video_bytes: int,
        include_repair_id: str | None,
    ) -> dict[str, Any]:
        report_path.write_text(
            "\n".join(report_lines).rstrip() + "\n",
            encoding="utf-8",
        )
        self.published_files.append(
            report_path.relative_to(self.bundle_root).as_posix()
        )
        artifact_inventory = []
        for relative in sorted(set(self.published_files)):
            artifact_path = self.bundle_root / relative
            artifact_inventory.append(
                {
                    "path": relative,
                    "bytes": artifact_path.stat().st_size,
                    "sha256": hashlib.sha256(
                        artifact_path.read_bytes()
                    ).hexdigest(),
                }
            )
        bundle_manifest = {
            "schema_version": 3,
            "path_basis": "bundle_relative",
            "evaluation_id": manifest.get("evaluation_id"),
            "source_evaluation": str(
                evaluation.relative_to(self.root)
            ).replace("\\", "/"),
            "source_server_path": (
                str(evaluation.resolve()) if self.publish else None
            ),
            "report": report_path.relative_to(self.bundle_root).as_posix(),
            "summary": run_summary_path.relative_to(
                self.bundle_root
            ).as_posix(),
            "publish_mode": self.publish,
            "files": sorted(set(self.published_files)),
            "artifacts": artifact_inventory,
            "round_count": round_count,
            "video_size_limit_bytes": int(max_video_bytes),
            "included_repair_id": include_repair_id,
        }
        manifest_path = self.bundle_root / "evidence_bundle_manifest.json"
        manifest_path.write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        bundle_manifest["files"] = sorted(
            set(
                [
                    *bundle_manifest["files"],
                    manifest_path.relative_to(self.bundle_root).as_posix(),
                ]
            )
        )
        manifest_path.write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return bundle_manifest


def publish_plan_artifacts(
    publisher: EvidencePublisher,
    *,
    evaluation: Path,
    round_count: int,
) -> None:
    """Publish Query interpretation and per-round Plan Agent traces."""

    publisher.publish_copy(
        evaluation / "request.json",
        publisher.semantic_dir / "query/request.json",
    )
    for relative in PLAN_ARTIFACTS:
        publisher.publish_copy(
            evaluation / "plan" / relative,
            publisher.semantic_dir / "plan" / relative,
        )
    publisher.publish_first(
        (
            evaluation / "plan/query_interpretation.json",
            evaluation / "plan/free_concern.json",
        ),
        publisher.semantic_dir / "plan/query_interpretation.json",
    )
    publisher.publish_first(
        (
            evaluation / "plan/query_interpretation_prompt.md",
            evaluation / "plan/free_concern_prompt.md",
        ),
        publisher.semantic_dir / "plan/query_interpretation_prompt.md",
    )
    for response_index in range(1, 5):
        publisher.publish_first(
            (
                evaluation
                / f"plan/query_interpretation_response_{response_index}.txt",
                evaluation
                / f"plan/free_concern_response_{response_index}.txt",
            ),
            publisher.semantic_dir
            / f"plan/query_interpretation_response_{response_index}.txt",
        )
    publisher.publish_first(
        (
            evaluation / "plan/query_interpretation_response.txt",
            evaluation / "plan/free_concern_response.txt",
        ),
        publisher.semantic_dir / "plan/query_interpretation_response.txt",
    )
    for step_index in range(1, round_count + 1):
        destination_root = (
            publisher.semantic_dir
            / f"plan/plan_agent_steps/after_round_{step_index:02d}"
        )
        for name in (
            "prompt.md",
            "semantic_proposal_bundle.json",
            "bound_semantic_step.json",
        ):
            publisher.publish_first(
                (
                    evaluation
                    / f"plan/plan_agent_steps/after_round_{step_index:02d}"
                    / name,
                    evaluation
                    / f"plan/claim_first_steps/after_round_{step_index:02d}"
                    / name,
                ),
                destination_root / name,
            )
        for response_index in range(1, 5):
            publisher.publish_first(
                (
                    evaluation
                    / f"plan/plan_agent_steps/after_round_{step_index:02d}"
                    / f"response_{response_index}.txt",
                    evaluation
                    / f"plan/claim_first_steps/after_round_{step_index:02d}"
                    / f"response_{response_index}.txt",
                ),
                destination_root / f"response_{response_index}.txt",
            )
    publisher.publish_first(
        (
            evaluation / "plan/plan_agent_session/query_answer.json",
            evaluation / "plan/claim_first_runtime/query_answer.json",
        ),
        publisher.semantic_dir / "answer/query_answer.json",
    )


__all__ = [
    "EvidencePublisher",
    "MAX_PUBLIC_ARTIFACT_BYTES",
    "publish_plan_artifacts",
]
