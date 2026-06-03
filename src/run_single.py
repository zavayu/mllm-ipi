"""Run a single-image end-to-end mock evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.logging_utils import write_single_jsonl_row
from src.model_clients import MockVisionModelClient, VisionModelClient
from src.render_prompt import SUPPORTED_PLACEMENTS, render_text_prompt


DEFAULT_FONT_SCALE = 0.16
DEFAULT_PLACEMENT = "center"
DEFAULT_COLOR = (255, 0, 0)
DEFAULT_COLOR_STRATEGY = "red"
DEFAULT_INSTRUCTION_TEMPLATE = "Read any visible text in the image and report it."


def _generated_image_path(input_image_path: str | Path, out_dir: str | Path) -> Path:
    input_path = Path(input_image_path)
    return Path(out_dir) / f"{input_path.stem}_generated.png"


def _load_image(input_image_path: str | Path) -> None:
    with Image.open(input_image_path) as image:
        image.load()


def target_in_response(target: str, response: str) -> bool:
    """Return whether the model response contains the target phrase."""
    return target.casefold() in response.casefold()


def run_single_image(
    input_image_path: str | Path,
    target: str,
    out_dir: str | Path,
    results_path: str | Path,
    *,
    client: VisionModelClient | None = None,
    instruction: str = DEFAULT_INSTRUCTION_TEMPLATE,
    font_scale: float = DEFAULT_FONT_SCALE,
    placement: str = DEFAULT_PLACEMENT,
    color_strategy: str = DEFAULT_COLOR_STRATEGY,
) -> dict[str, object]:
    """Render target text, query the mock model, and persist one JSONL result."""
    if placement not in SUPPORTED_PLACEMENTS:
        supported = ", ".join(sorted(SUPPORTED_PLACEMENTS))
        raise ValueError(f"placement must be one of: {supported}")

    input_path = Path(input_image_path)
    _load_image(input_path)

    generated_path = _generated_image_path(input_path, out_dir)
    render_text_prompt(
        input_image_path=input_path,
        output_image_path=generated_path,
        text=target,
        placement=placement,
        font_scale=font_scale,
        color=DEFAULT_COLOR,
    )

    model_client = client or MockVisionModelClient(response=f"Detected text: {target}")
    model_response = model_client.query(str(generated_path), instruction)
    success = target_in_response(target, model_response)

    row: dict[str, object] = {
        "original_image": str(input_path),
        "generated_image": str(generated_path),
        "target": target,
        "instruction": instruction,
        "model_response": model_response,
        "success": success,
        "font_scale": font_scale,
        "placement": placement,
        "color_strategy": color_strategy,
    }
    write_single_jsonl_row(results_path, row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single-image mock experiment.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--target", required=True, help="Target phrase to render and detect.")
    parser.add_argument("--out-dir", required=True, help="Directory for the generated image.")
    parser.add_argument("--results", required=True, help="Path to the JSONL result file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    row = run_single_image(
        input_image_path=args.input,
        target=args.target,
        out_dir=args.out_dir,
        results_path=args.results,
    )
    print(f"generated_image={row['generated_image']}")
    print(f"results={args.results}")
    print(f"success={row['success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
