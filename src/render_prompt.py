"""Render visible prompt text onto images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont


SUPPORTED_PLACEMENTS = {"center", "top_right", "bottom_middle"}


class RenderMetadata(NamedTuple):
    """Metadata describing rendered text."""

    output_path: str
    placement: str
    font_size: int
    text_bbox: tuple[int, int, int, int]


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_names = (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
    )
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _parse_color(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must use R,G,B format")

    try:
        color = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color values must be integers") from exc

    if any(channel < 0 or channel > 255 for channel in color):
        raise argparse.ArgumentTypeError("color values must be between 0 and 255")
    return color


def _text_position(
    image_size: tuple[int, int],
    text_size: tuple[int, int],
    placement: str,
) -> tuple[int, int]:
    image_width, image_height = image_size
    text_width, text_height = text_size
    margin = max(4, round(min(image_size) * 0.05))

    if placement == "center":
        return ((image_width - text_width) // 2, (image_height - text_height) // 2)
    if placement == "top_right":
        return (image_width - text_width - margin, margin)
    if placement == "bottom_middle":
        return ((image_width - text_width) // 2, image_height - text_height - margin)

    raise ValueError(f"unsupported placement: {placement}")


def render_text_prompt(
    input_image_path: str | Path,
    output_image_path: str | Path,
    text: str,
    placement: str,
    font_scale: float,
    color: tuple[int, int, int] = (255, 0, 0),
) -> RenderMetadata:
    """Render visible text onto an image and save the result."""
    if placement not in SUPPORTED_PLACEMENTS:
        supported = ", ".join(sorted(SUPPORTED_PLACEMENTS))
        raise ValueError(f"placement must be one of: {supported}")
    if font_scale <= 0:
        raise ValueError("font_scale must be greater than 0")
    if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("color must be an RGB tuple with values between 0 and 255")

    input_path = Path(input_image_path)
    output_path = Path(output_image_path)

    with Image.open(input_path) as image:
        rendered = image.convert("RGB")

    font_size = max(1, round(min(rendered.size) * font_scale))
    font = _load_font(font_size)
    draw = ImageDraw.Draw(rendered)

    raw_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = raw_bbox[2] - raw_bbox[0]
    text_height = raw_bbox[3] - raw_bbox[1]
    target_x, target_y = _text_position(rendered.size, (text_width, text_height), placement)
    draw_position = (target_x - raw_bbox[0], target_y - raw_bbox[1])

    draw.text(draw_position, text, fill=color, font=font)
    text_bbox = draw.textbbox(draw_position, text, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    return RenderMetadata(
        output_path=str(output_path),
        placement=placement,
        font_size=font_size,
        text_bbox=tuple(int(value) for value in text_bbox),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render visible prompt text onto an image.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Path for the rendered output image.")
    parser.add_argument("--text", required=True, help="Text to render onto the image.")
    parser.add_argument(
        "--placement",
        required=True,
        choices=sorted(SUPPORTED_PLACEMENTS),
        help="Where to place the text.",
    )
    parser.add_argument(
        "--font-scale",
        required=True,
        type=float,
        help="Font size as a fraction of the smaller image dimension.",
    )
    parser.add_argument(
        "--color",
        default="255,0,0",
        type=_parse_color,
        help="Text color as R,G,B. Defaults to 255,0,0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata = render_text_prompt(
        input_image_path=args.input,
        output_image_path=args.output,
        text=args.text,
        placement=args.placement,
        font_scale=args.font_scale,
        color=args.color,
    )
    print(f"output_path={metadata.output_path}")
    print(f"placement={metadata.placement}")
    print(f"font_size={metadata.font_size}")
    print(f"text_bbox={metadata.text_bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
