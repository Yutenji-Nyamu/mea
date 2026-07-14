"""Generate, validate, render, and optionally evaluate one TaskGen variant."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.providers import OpenAICompatibleProvider
from mea.taskgen import (
    TaskGenPrototype,
    VisualReflectionError,
    execute_reflection_loop,
    inject_oversized_block_fixture,
    inject_wrong_color_fixture,
    repair_generated_method,
    validate_vision_observation,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_manifest(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def run_probe(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    seed: int,
    expert: bool,
    scene_json: Path | None = None,
    image: Path | None = None,
    log_path: Path | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    scene_json = scene_json or run_dir / "validation/scene.json"
    image = image or run_dir / "evidence/initial_head.png"
    log_path = log_path or run_dir / "validation/probe.log"
    command = [
        sys.executable,
        "-m",
        "mea.taskgen.probe",
        "--repo-root",
        str(repo_root),
        "--task-name",
        manifest["task_name"],
        "--task-module",
        manifest["task_module"],
        "--task-config",
        "demo_clean",
        "--ckpt-setting",
        "demo_clean",
        "--overlay",
        str(run_dir / "overlay.yml"),
        "--seed",
        str(seed),
        "--image",
        str(image),
        "--output",
        str(scene_json),
    ]
    if expert:
        command.append("--expert")

    returncode = run_command(
        command,
        cwd=repo_root,
        log_path=log_path,
    )
    scene = json.loads(scene_json.read_text(encoding="utf-8")) if scene_json.exists() else {}
    scene["returncode"] = returncode
    write_json(scene_json, scene)
    if raise_on_failure and returncode != 0:
        raise RuntimeError(f"setup/expert probe 失败，returncode={returncode}")
    return scene


def run_vision_check(
    provider: OpenAICompatibleProvider,
    run_dir: Path,
    spec: dict[str, Any],
    *,
    model: str,
    image_path: Path | None = None,
    prompt_path: Path | None = None,
    response_path: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    image_path = image_path or run_dir / "evidence/initial_head.png"
    prompt_path = prompt_path or run_dir / "validation/vision_prompt.md"
    response_path = response_path or run_dir / "validation/vision_response.txt"
    result_path = result_path or run_dir / "validation/vision.json"
    expected_half_size = 0.025 * float(spec["changes"]["block"]["scale"])
    prompt = f"""这是 RoboTwin beat_block_hammer 的初始场景首帧。
请检查被锤子敲击的方块是否符合下面的 VariantSpec，并检查场景是否有明显异常：
{json.dumps(spec, ensure_ascii=False, indent=2)}

官方 scale=1.0 的方块 half_size 是 (0.025, 0.025, 0.025) 米；本次预期
half_size 是 ({expected_half_size:.6f}, {expected_half_size:.6f}, {expected_half_size:.6f}) 米。
请结合方块与锤子的相对尺寸判断是否明显偏大或偏小。

只输出 JSON：
{{
  "aligned": true,
  "target_actor": "block",
  "observed_color": "blue",
  "unexpected_changes": [],
  "diagnosis": "场景与需求是否一致，以及不一致的具体原因",
  "suggestions": ["若不一致，给出只修改 load_actors() 的具体建议"],
  "confidence": 0.0
}}
"""
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    response = provider.vision(
        prompt,
        image_path,
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    response_path.write_text(response + "\n", encoding="utf-8")
    from mea.taskgen import extract_json_response

    result = validate_vision_observation(extract_json_response(response), spec)
    result["provider_metadata"] = dict(provider.last_metadata)
    write_json(result_path, result)
    return result


def run_visual_self_reflection(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    provider: OpenAICompatibleProvider,
    *,
    seed: int,
    text_model: str,
    vision_model: str,
    max_repairs: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = json.loads((run_dir / "variant_spec.json").read_text(encoding="utf-8"))
    reflection_dir = run_dir / "reflection"
    reflection_dir.mkdir(parents=True, exist_ok=True)

    def observe(attempt_index: int) -> dict[str, Any]:
        attempt_dir = reflection_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        scene_path = attempt_dir / "scene.json"
        image_path = attempt_dir / "render.png"
        scene = run_probe(
            repo_root,
            run_dir,
            manifest,
            seed=seed,
            expert=False,
            scene_json=scene_path,
            image=image_path,
            log_path=attempt_dir / "probe.log",
            raise_on_failure=False,
        )
        probe_passed = bool(
            scene.get("setup_success")
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and scene.get("returncode") == 0
        )
        if probe_passed:
            vision = run_vision_check(
                provider,
                run_dir,
                spec,
                model=vision_model,
                image_path=image_path,
                prompt_path=attempt_dir / "vision_prompt.md",
                response_path=attempt_dir / "vision_response.txt",
                result_path=attempt_dir / "vision.json",
            )
        else:
            error = scene.get("error") or {}
            vision = {
                "aligned": False,
                "target_actor": "block",
                "expected_color": "blue",
                "observed_color": "unavailable",
                "color_matches": False,
                "unexpected_changes": ["scene_probe_failed"],
                "diagnosis": (
                    f"Scene setup/render/rule probe failed: "
                    f"{error.get('type', 'unknown')}: {error.get('message', '')}"
                ),
                "suggestions": [
                    "Repair load_actors() so setup, render, hammer/block actor checks pass."
                ],
                "confidence": 1.0,
                "passed": False,
                "provider_metadata": {},
            }
            write_json(attempt_dir / "vision.json", vision)
        return {
            "passed": bool(probe_passed and vision.get("passed")),
            "probe_passed": probe_passed,
            "scene_path": str(scene_path.relative_to(run_dir)),
            "image_path": str(image_path.relative_to(run_dir)),
            "vision_path": str((attempt_dir / "vision.json").relative_to(run_dir)),
            "vision": vision,
        }

    def repair(repair_index: int, observation: dict[str, Any]) -> dict[str, Any]:
        update_manifest(
            run_dir,
            status=f"visual_reflection_repair_{repair_index}",
        )
        result = repair_generated_method(
            repo_root,
            run_dir,
            provider,
            model=text_model,
            spec=spec,
            observation=observation,
            repair_index=repair_index,
            protected_before=manifest["protected_hashes_before"],
        )
        update_manifest(
            run_dir,
            static_validation=result["static_validation"],
        )
        return result

    summary = execute_reflection_loop(
        max_repairs=max_repairs,
        observe=observe,
        repair=repair,
    )
    write_json(reflection_dir / "summary.json", summary)
    if not summary["passed"]:
        raise VisualReflectionError(
            f"Visual Self-Reflection 用尽 {max_repairs} 次 repair: {summary}"
        )

    final_attempt = reflection_dir / f"attempt_{summary['final_attempt']:02d}"
    shutil.copy2(final_attempt / "render.png", run_dir / "evidence/initial_head.png")
    shutil.copy2(final_attempt / "vision.json", run_dir / "validation/vision.json")
    if (final_attempt / "vision_prompt.md").is_file():
        shutil.copy2(
            final_attempt / "vision_prompt.md",
            run_dir / "validation/vision_prompt.md",
        )
    if (final_attempt / "vision_response.txt").is_file():
        shutil.copy2(
            final_attempt / "vision_response.txt",
            run_dir / "validation/vision_response.txt",
        )
    final_scene = json.loads((final_attempt / "scene.json").read_text(encoding="utf-8"))
    final_vision = json.loads((final_attempt / "vision.json").read_text(encoding="utf-8"))
    return summary, final_scene, final_vision


def newest_eval_dir(repo_root: Path, before: set[Path]) -> Path | None:
    root = repo_root / "eval_result/beat_block_hammer/ACT/demo_clean/demo_clean"
    after = {path for path in root.glob("*") if path.is_dir()} if root.exists() else set()
    created = after - before
    candidates = created or after
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_act(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    seed: int,
    gpu: int,
) -> dict[str, Any]:
    eval_root = repo_root / "eval_result/beat_block_hammer/ACT/demo_clean/demo_clean"
    before = {path for path in eval_root.glob("*") if path.is_dir()} if eval_root.exists() else set()
    command = [
        "bash",
        "policy/ACT/eval_mea.sh",
        "beat_block_hammer",
        "demo_clean",
        "demo_clean",
        "50",
        "0",
        str(gpu),
        "1",
        manifest["task_module"],
        str(run_dir / "overlay.yml"),
        str(seed),
    ]
    started = datetime.now().astimezone().isoformat()
    returncode = run_command(
        command,
        cwd=repo_root,
        log_path=run_dir / "evaluation/act.log",
    )
    source_dir = newest_eval_dir(repo_root, before)
    copied = []
    if source_dir:
        for name in ("episode0.mp4", "_result.txt"):
            source = source_dir / name
            if source.is_file():
                destination = run_dir / "evaluation" / name
                shutil.copy2(source, destination)
                copied.append(str(destination.relative_to(repo_root)))

    result = {
        "command": command,
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(),
        "returncode": returncode,
        "source_eval_dir": str(source_dir) if source_dir else None,
        "copied_artifacts": copied,
        "passed": returncode == 0 and (run_dir / "evaluation/episode0.mp4").is_file(),
    }
    write_json(run_dir / "evaluation/act.json", result)
    if not result["passed"]:
        raise RuntimeError(f"ACT 1-episode 未通过: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume-run",
        help="Resume an existing run_id without calling the text-generation stages again.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--mode", choices=["reuse", "force_codegen"], default="force_codegen")
    parser.add_argument("--text-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--vision-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--vision-check", action="store_true")
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum number of CodeGen repairs after failed visual observations.",
    )
    parser.add_argument(
        "--reflection-fixture",
        choices=["wrong_color", "oversized_block"],
        help="Test-only injected visual mismatch used to exercise the repair loop.",
    )
    parser.add_argument("--run-act", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    provider = None
    if not args.resume_run or args.vision_check:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=args.text_model,
            vision_model=args.vision_model,
            timeout=180.0,
        )
    if args.resume_run:
        if args.run_id:
            raise SystemExit("--resume-run 与 --run-id 不能同时使用")
        run_dir = repo_root / "mea/generated_tasks" / args.resume_run
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"run manifest 不存在: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if not args.request:
            raise SystemExit("新 TaskGen run 必须提供 --request")
        prototype = TaskGenPrototype(repo_root, provider, model=args.text_model)
        manifest = prototype.generate(
            args.request,
            task_name=args.task_name,
            mode=args.mode,
            run_id=args.run_id,
        )
        run_dir = repo_root / "mea/generated_tasks" / manifest["run_id"]

    try:
        if args.reflection_fixture:
            if args.resume_run:
                raise RuntimeError("reflection fixture 只允许用于新的 TaskGen run")
            if not args.vision_check:
                raise RuntimeError("reflection fixture 必须与 --vision-check 一起使用")
            spec = json.loads((run_dir / "variant_spec.json").read_text(encoding="utf-8"))
            fixture_function = {
                "wrong_color": inject_wrong_color_fixture,
                "oversized_block": inject_oversized_block_fixture,
            }[args.reflection_fixture]
            fixture = fixture_function(
                repo_root, run_dir, spec, manifest["protected_hashes_before"]
            )
            update_manifest(run_dir, reflection_fixture=fixture)

        scene = None
        if args.vision_check:
            if provider is None:
                raise RuntimeError("vision check 缺少 provider")
            reflection, reflected_scene, vision = run_visual_self_reflection(
                repo_root,
                run_dir,
                manifest,
                provider,
                seed=args.seed,
                text_model=args.text_model,
                vision_model=args.vision_model,
                max_repairs=args.max_reflections,
            )
            update_manifest(
                run_dir,
                status="vision_passed",
                visual_self_reflection=reflection,
                vision_validation=vision,
            )
            scene = reflected_scene

        if args.expert or args.run_act:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=True,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif args.probe and not args.vision_check:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=False,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif scene is not None:
            write_json(run_dir / "validation/scene.json", scene)
            update_manifest(run_dir, scene_validation=scene)

        if args.run_act:
            act = run_act(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                gpu=args.gpu,
            )
            update_manifest(run_dir, status="completed", act_evaluation=act)
        else:
            update_manifest(run_dir, status="completed_without_act")
    except Exception as exc:
        update_manifest(
            run_dir,
            status="failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    print(json.dumps(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
