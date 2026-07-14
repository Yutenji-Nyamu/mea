"""Run small text and vision checks against the UIUI OpenAI-compatible endpoint."""

import argparse
import json
from pathlib import Path

from mea.providers import OpenAICompatibleProvider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vision", "both"], default="text")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--text-model", default="gpt-4o-mini")
    parser.add_argument("--vision-model", default="gpt-4o")
    return parser.parse_args()


def report(kind: str, content: str, metadata: dict):
    print(f"[{kind}] content={content}")
    print(f"[{kind}] metadata={json.dumps(metadata, ensure_ascii=False)}")


def main():
    args = parse_args()
    if args.mode in {"vision", "both"} and args.image is None:
        raise SystemExit("--image is required for vision mode")

    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=args.text_model,
        vision_model=args.vision_model,
    )

    if args.mode in {"text", "both"}:
        content = provider.text(
            "这是连通性测试。请只回复 UIUI_TEXT_OK，不要添加其他内容。"
        )
        report("text", content, provider.last_metadata)

    if args.mode in {"vision", "both"}:
        content = provider.vision(
            "请观察图像中的桌面场景：指出方块的主要颜色，并用一句话说明。",
            args.image,
        )
        report("vision", content, provider.last_metadata)


if __name__ == "__main__":
    main()
