"""Run a single-image end-to-end mock evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.logging_utils import write_single_jsonl_row
from src.model_clients import MockVisionModelClient, VisionModelClient
from src.region_ranker import compute_and_rank_regions
from src.render_prompt import (
    DEFAULT_BRIGHTNESS_OFFSET,
    SUPPORTED_PLACEMENTS,
    render_text_prompt,
    render_text_prompt_in_top_ranked_mask,
)
from src.segment_regions import generate_candidate_masks

DEFAULT_FONT_SCALE = 0.16
DEFAULT_PLACEMENT = "center"
DEFAULT_COLOR = (255, 0, 0)
DEFAULT_COLOR_STRATEGY = "red"
MASK_COLOR_STRATEGY = "top_ranked_mask_average"
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
    brightness_offset: int = DEFAULT_BRIGHTNESS_OFFSET,
) -> dict[str, object]:
    """Render target text, query the mock model, and persist one JSONL result."""
    if placement not in SUPPORTED_PLACEMENTS:
        supported = ", ".join(sorted(SUPPORTED_PLACEMENTS))
        raise ValueError(f"placement must be one of: {supported}")

    input_path = Path(input_image_path)
    _load_image(input_path)

    generated_path = _generated_image_path(input_path, out_dir)
    if color_strategy == MASK_COLOR_STRATEGY:
        with Image.open(input_path) as image:
            image_array = np.asarray(image.convert("RGB"))
        masks = generate_candidate_masks(image_array)
        ranked_regions = compute_and_rank_regions(image_array, masks)
        render_metadata = render_text_prompt_in_top_ranked_mask(
            input_image_path=input_path,
            output_image_path=generated_path,
            text=target,
            ranked_regions=ranked_regions,
            font_scale=font_scale,
            brightness_offset=brightness_offset,
        )
    else:
        render_metadata = render_text_prompt(
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
        "success": success,
        "original_image": str(input_path),
        "generated_image": str(generated_path),
        "target": target,
        "instruction": instruction,
        "model_response": model_response,
        "font_scale": font_scale,
        "placement": placement,
        "color_strategy": color_strategy,
        "selected_mask_id": render_metadata.selected_mask_id,
        "average_rgb": (
            list(render_metadata.average_rgb)
            if render_metadata.average_rgb is not None
            else None
        ),
        "brightness_offset": render_metadata.brightness_offset,
        "final_rgb": (
            list(render_metadata.final_rgb)
            if render_metadata.final_rgb is not None
            else None
        ),
        "render_text_bbox": list(render_metadata.text_bbox),
    }
    write_single_jsonl_row(results_path, row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single-image mock experiment.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument(
        "--target", required=True, help="Target phrase to render and detect."
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory for the generated image."
    )
    parser.add_argument(
        "--results", required=True, help="Path to the JSONL result file."
    )
    parser.add_argument(
        "--color-strategy",
        default=DEFAULT_COLOR_STRATEGY,
        choices=[DEFAULT_COLOR_STRATEGY, MASK_COLOR_STRATEGY],
        help=(
            "Color strategy. Use top_ranked_mask_average to render into the "
            "top-ranked mask with region-average coloring."
        ),
    )
    parser.add_argument(
        "--brightness-offset",
        type=int,
        default=DEFAULT_BRIGHTNESS_OFFSET,
        help=f"RGB brightness offset for mask rendering. Defaults to {DEFAULT_BRIGHTNESS_OFFSET}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    row = run_single_image(
        input_image_path=args.input,
        target=args.target,
        out_dir=args.out_dir,
        results_path=args.results,
        color_strategy=args.color_strategy,
        brightness_offset=args.brightness_offset,
    )
    print(f"generated_image={row['generated_image']}")
    print(f"results={args.results}")
    print(f"success={row['success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
