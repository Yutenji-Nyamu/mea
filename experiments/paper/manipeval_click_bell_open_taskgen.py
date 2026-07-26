"""One open-resolution -> ClickBell provider TaskGen Gate0 protocol.

This runner deliberately consumes a previously generated FreeConcern and
open-task resolution. It does not add the concern to the planner catalog and
does not start ACT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.planner.open_task_resolver import (
    OpenTaskResolutionError,
    validate_free_concern,
)
from mea.providers import OpenAICompatibleProvider
from mea.taskgen.click_bell_distractor import (
    ClickBellDistractorTaskGenError,
    default_click_bell_distractor_proposal,
    materialize_click_bell_distractor_candidate,
    validate_click_bell_distractor_proposal,
)
from scripts.manipeval_taskgen import run_probe


class OpenClickBellGate0Error(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def proposal_from_open_resolution(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt a catalog-external concern only after nearest-task retrieval."""

    if not isinstance(value, Mapping):
        raise OpenClickBellGate0Error("open resolution must be an object")
    if value.get("decision") != "retrieve_and_adapt":
        raise OpenClickBellGate0Error(
            "open resolution did not authorize nearest-task adaptation"
        )
    selected = value.get("selected_base_task")
    if not isinstance(selected, Mapping) or selected.get("task_name") != "click_bell":
        raise OpenClickBellGate0Error(
            "open resolution did not select click_bell"
        )
    concern = validate_free_concern(value.get("free_concern", {}))
    semantic_text = " ".join(
        concern[field]
        for field in (
            "sub_aspect",
            "hypothesis",
            "requested_variation",
            "measurement_need",
        )
    ).lower()
    if not any(
        token in semantic_text
        for token in (
            "distractor",
            "look-alike",
            "lookalike",
            "similar object",
            "similar bell",
            "target selectivity",
            "相似",
            "干扰",
        )
    ):
        raise OpenClickBellGate0Error(
            "FreeConcern is outside the available ClickBell distractor dialect"
        )
    proposal = default_click_bell_distractor_proposal()
    identity = hashlib.sha256(
        json.dumps(
            concern, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:12]
    proposal.update(
        {
            "proposal_id": f"click_bell.open_concern.{identity}",
            "query": concern["source_query"],
            "intent": (
                "adapt_nearest_official_click_bell_for_"
                + concern["sub_aspect"]
            ),
        }
    )
    return validate_click_bell_distractor_proposal(proposal), concern


def run_gate0(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.expanduser().resolve()
    resolution_document = json.loads(
        args.open_resolution_json.read_text(encoding="utf-8")
    )
    resolution = resolution_document.get(
        "repaired_resolution",
        resolution_document,
    )
    proposal, concern = proposal_from_open_resolution(resolution)
    checkpoint = (
        repo_root
        / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt"
    )
    if not checkpoint.is_file():
        raise OpenClickBellGate0Error(
            "the selected base task has no ready click_bell ACT checkpoint"
        )
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=args.text_model,
        timeout=180.0,
        max_retries=1,
    )
    manifest = materialize_click_bell_distractor_candidate(
        repo_root=repo_root,
        run_id=args.run_id,
        proposal=proposal,
        provider=provider,
        model=args.text_model,
        max_regenerations=1,
    )
    run_dir = repo_root / "mea/generated_tasks" / args.run_id
    for child in ("generation", "validation", "evidence", "evaluation"):
        (run_dir / child).mkdir(exist_ok=True)
    (run_dir / "overlay.yml").write_text("{}\n", encoding="utf-8")
    _write_json(run_dir / "generation/free_concern.json", concern)
    _write_json(run_dir / "generation/open_task_resolution.json", resolution)
    if resolution is not resolution_document:
        _write_json(
            run_dir / "generation/open_task_resolution_source_replay.json",
            resolution_document,
        )
    _write_json(run_dir / "generation/adapted_proposal.json", proposal)
    render = run_probe(
        repo_root,
        run_dir,
        manifest,
        seed=args.seed,
        expert=False,
        scene_json=run_dir / "validation/render_probe.json",
        image=run_dir / "evidence/initial_head.png",
        log_path=run_dir / "validation/render_probe.log",
        raise_on_failure=False,
    )
    render_passed = bool(
        render.get("setup_success")
        and render.get("render_success")
        and render.get("rule_check", {}).get("passed")
        and render.get("returncode") == 0
    )
    expert: dict[str, Any] | None = None
    if render_passed:
        expert = run_probe(
            repo_root,
            run_dir,
            manifest,
            seed=args.seed,
            expert=True,
            scene_json=run_dir / "validation/expert_probe.json",
            image=run_dir / "evidence/expert_head.png",
            log_path=run_dir / "validation/expert_probe.log",
            telemetry_dir=run_dir / "evaluation/telemetry/expert",
            raise_on_failure=False,
            max_expert_attempts=1,
        )
    expert_passed = bool(
        expert
        and expert.get("returncode") == 0
        and expert.get("expert", {}).get("passed")
    )
    result = {
        "schema_version": 1,
        "status": "taskgen_gate0_passed" if expert_passed else "taskgen_gate0_failed",
        "query": concern["source_query"],
        "free_concern": concern,
        "open_task_resolution": resolution,
        "selected_base_task": "click_bell",
        "adaptation_mode": "provider_scene_checker_codegen",
        "candidate_manifest": manifest,
        "render_passed": render_passed,
        "expert_passed": expert_passed,
        "taskgen_gate0_eligible": expert_passed,
        "policy_performance_evidence_eligible": False,
        "act_rollouts_completed": 0,
    }
    _write_json(run_dir / "gate0_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--open-resolution-json", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--text-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--seed", type=int, default=100405)
    return parser.parse_args()


def main() -> None:
    try:
        result = run_gate0(parse_args())
    except (
        ClickBellDistractorTaskGenError,
        OpenClickBellGate0Error,
        OpenTaskResolutionError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"ClickBell open TaskGen Gate0 failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "taskgen_gate0_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
