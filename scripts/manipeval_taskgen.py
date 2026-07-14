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
from mea.taskgen import TaskGenPrototype, extract_json_response


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
) -> dict[str, Any]:
    scene_json = run_dir / "validation/scene.json"
    image = run_dir / "evidence/initial_head.png"
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
        log_path=run_dir / "validation/probe.log",
    )
    scene = json.loads(scene_json.read_text(encoding="utf-8")) if scene_json.exists() else {}
    scene["returncode"] = returncode
    write_json(scene_json, scene)
    if returncode != 0:
        raise RuntimeError(f"setup/expert probe 失败，returncode={returncode}")
    return scene


def run_vision_check(
    provider: OpenAICompatibleProvider,
    run_dir: Path,
    spec: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    prompt = f"""这是 RoboTwin beat_block_hammer 的初始场景首帧。
请检查被锤子敲击的方块是否符合下面的 VariantSpec，并检查场景是否有明显异常：
{json.dumps(spec, ensure_ascii=False, indent=2)}

只输出 JSON：
{{
  "aligned": true,
  "target_actor": "block",
  "observed_color": "blue",
  "unexpected_changes": [],
  "confidence": 0.0
}}
"""
    (run_dir / "validation/vision_prompt.md").write_text(prompt, encoding="utf-8")
    response = provider.vision(
        prompt,
        run_dir / "evidence/initial_head.png",
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    (run_dir / "validation/vision_response.txt").write_text(
        response + "\n", encoding="utf-8"
    )
    result = extract_json_response(response)
    result["provider_metadata"] = dict(provider.last_metadata)
    result["passed"] = bool(result.get("aligned")) and str(
        result.get("observed_color", "")
    ).lower() in {"blue", "蓝色", "蓝"}
    write_json(run_dir / "validation/vision.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Visual Self-Check 未通过: {result}")
    return result


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
        )
        run_dir = repo_root / "mea/generated_tasks" / manifest["run_id"]

    try:
        if args.probe or args.expert or args.vision_check or args.run_act:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=args.expert or args.run_act,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)

        if args.vision_check:
            if provider is None:
                raise RuntimeError("vision check 缺少 provider")
            spec = json.loads((run_dir / "variant_spec.json").read_text(encoding="utf-8"))
            vision = run_vision_check(
                provider,
                run_dir,
                spec,
                model=args.vision_model,
            )
            update_manifest(run_dir, status="vision_passed", vision_validation=vision)

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
