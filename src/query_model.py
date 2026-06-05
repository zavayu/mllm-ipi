"""Query a local vision-language model from the command line."""

from __future__ import annotations

import argparse
import sys

from src.model_clients import (
    DEFAULT_QWEN_MODEL_FAMILY,
    DEFAULT_QWEN_MODEL_ID,
    SUPPORTED_QWEN_MODEL_FAMILIES,
    QwenVisionModelClient,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query a local Qwen VL model.")
    parser.add_argument("--image", required=True, help="Path to the image to inspect.")
    parser.add_argument(
        "--instruction",
        required=True,
        help="Instruction or question to send with the image.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_QWEN_MODEL_ID,
        help=f"Hugging Face model id. Defaults to {DEFAULT_QWEN_MODEL_ID}.",
    )
    parser.add_argument(
        "--model-family",
        choices=SUPPORTED_QWEN_MODEL_FAMILIES,
        default=DEFAULT_QWEN_MODEL_FAMILY,
        help=(
            "Qwen model family. Use 'auto' to infer from --model-id. "
            f"Defaults to {DEFAULT_QWEN_MODEL_FAMILY}."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU inference when CUDA is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = QwenVisionModelClient(
        model_id=args.model_id,
        model_family=args.model_family,
        max_new_tokens=args.max_new_tokens,
        require_gpu=not args.allow_cpu,
    )
    try:
        print(client.query(args.image, args.instruction))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
