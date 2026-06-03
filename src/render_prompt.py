"""Render visible prompt text onto images."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.region_ranker import RegionFeatures

SUPPORTED_PLACEMENTS = {"center", "top_right", "bottom_middle"}
DEFAULT_BRIGHTNESS_OFFSET = 35
DEFAULT_MULTI_MASK_METADATA_SUFFIX = ".chunks.json"
DEFAULT_MULTI_MASK_VISUALIZATION_SUFFIX = "_chunks.png"


@dataclass(frozen=True)
class ChunkRenderMetadata:
    """Metadata describing one prompt chunk rendered into one mask."""

    chunk_index: int
    mask_id: int
    rank: int | None
    text: str
    region_bbox: tuple[int, int, int, int]
    average_rgb: tuple[int, int, int]
    brightness_offset: int
    final_rgb: tuple[int, int, int]
    font_size: int
    text_bbox: tuple[int, int, int, int]
    line_count: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable chunk metadata."""
        return {
            "chunk_index": self.chunk_index,
            "mask_id": self.mask_id,
            "rank": self.rank,
            "text": self.text,
            "region_bbox": list(self.region_bbox),
            "average_rgb": list(self.average_rgb),
            "brightness_offset": self.brightness_offset,
            "final_rgb": list(self.final_rgb),
            "font_size": self.font_size,
            "text_bbox": list(self.text_bbox),
            "line_count": self.line_count,
        }


@dataclass(frozen=True)
class RenderMetadata:
    """Metadata describing rendered text."""

    output_path: str
    placement: str
    font_size: int
    text_bbox: tuple[int, int, int, int]
    rendered_text: str
    line_count: int
    selected_mask_id: int | None = None
    average_rgb: tuple[int, int, int] | None = None
    brightness_offset: int | None = None
    final_rgb: tuple[int, int, int] | None = None
    region_bbox: tuple[int, int, int, int] | None = None
    chunks: tuple[ChunkRenderMetadata, ...] = ()
    chunk_metadata_path: str | None = None
    visualization_path: str | None = None
    adaptive_fallback_used: bool | None = None


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


def _text_position_in_rect(
    rect: tuple[int, int, int, int],
    text_size: tuple[int, int],
) -> tuple[int, int]:
    x, y, width, height = rect
    text_width, text_height = text_size
    return (x + (width - text_width) // 2, y + (height - text_height) // 2)


def _split_long_word(word: str, max_characters_per_line: int) -> list[str]:
    return [
        word[index : index + max_characters_per_line]
        for index in range(0, len(word), max_characters_per_line)
    ]


def _wrap_text_by_character_count(
    text: str,
    max_characters_per_line: int | None,
) -> str:
    if max_characters_per_line is None:
        return text
    if max_characters_per_line <= 0:
        raise ValueError("max_characters_per_line must be greater than 0")

    wrapped_lines: list[str] = []
    for source_line in text.splitlines() or [""]:
        words = source_line.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = ""
        for word in words:
            chunks = _split_long_word(word, max_characters_per_line)
            for chunk in chunks:
                if not current_line:
                    current_line = chunk
                elif len(current_line) + 1 + len(chunk) <= max_characters_per_line:
                    current_line = f"{current_line} {chunk}"
                else:
                    wrapped_lines.append(current_line)
                    current_line = chunk
        if current_line:
            wrapped_lines.append(current_line)

    return "\n".join(wrapped_lines)


def _split_text_into_chunks(text: str, chunk_count: int) -> list[str]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be greater than 0")

    words = text.split()
    if not words:
        return [""]

    chunk_count = min(chunk_count, len(words))
    base_size, remainder = divmod(len(words), chunk_count)
    chunks: list[str] = []
    cursor = 0
    for index in range(chunk_count):
        size = base_size + (1 if index < remainder else 0)
        chunk_words = words[cursor : cursor + size]
        chunks.append(" ".join(chunk_words))
        cursor += size
    return chunks


def _reading_order_regions(
    ranked_regions: list[RegionFeatures | dict[str, Any]],
) -> list[RegionFeatures]:
    regions = [_coerce_region(region) for region in ranked_regions]
    return sorted(
        regions,
        key=lambda region: (
            region.bbox[1],
            region.bbox[0],
            region.rank if region.rank is not None else len(regions) + 1,
        ),
    )


def _clamp_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(0, min(255, channel)) for channel in color)


def _apply_brightness_offset(
    average_rgb: tuple[int, int, int],
    brightness_offset: int,
) -> tuple[int, int, int]:
    return _clamp_rgb(tuple(channel + brightness_offset for channel in average_rgb))


def _rect_from_region_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = bbox
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(image_width, int(x + width))
    bottom = min(image_height, int(y + height))
    if left >= right or top >= bottom:
        raise ValueError("selected mask bbox does not overlap the image")
    return (left, top, right - left, bottom - top)


def _average_rgb_for_rect(
    image: Image.Image,
    rect: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    x, y, width, height = rect
    pixels = np.asarray(image, dtype=np.uint8)[y : y + height, x : x + width]
    if pixels.size == 0:
        raise ValueError("selected region contains no pixels")
    return tuple(int(round(value)) for value in pixels.mean(axis=(0, 1)))


def _coerce_region(data: RegionFeatures | dict[str, Any]) -> RegionFeatures:
    if isinstance(data, RegionFeatures):
        return data
    required = {
        "mask_id",
        "area",
        "bbox",
        "average_rgb",
        "rgb_variance",
        "center",
        "location",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"ranked mask metadata is missing keys: {missing}")
    return RegionFeatures(
        mask_id=int(data["mask_id"]),
        area=int(data["area"]),
        bbox=tuple(int(value) for value in data["bbox"]),
        average_rgb=tuple(float(value) for value in data["average_rgb"]),
        rgb_variance=float(data["rgb_variance"]),
        center=tuple(float(value) for value in data["center"]),
        location=str(data["location"]),
        score=float(data.get("score", 0.0)),
        rank=int(data["rank"]) if data.get("rank") is not None else None,
    )


def _load_ranked_regions(path: str | Path) -> list[RegionFeatures]:
    metadata_path = Path(path)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ranked mask metadata must contain a list of regions")
    return [_coerce_region(region) for region in data]


def _select_top_ranked_region(
    ranked_regions: list[RegionFeatures | dict[str, Any]],
) -> RegionFeatures:
    regions = [_coerce_region(region) for region in ranked_regions]
    if not regions:
        raise ValueError("ranked_regions must contain at least one region")
    return min(
        regions,
        key=lambda region: region.rank if region.rank is not None else len(regions) + 1,
    )


def _render_text_on_image(
    rendered: Image.Image,
    text: str,
    font_scale: float,
    color: tuple[int, int, int],
    max_characters_per_line: int | None,
    placement: str | None = None,
    target_rect: tuple[int, int, int, int] | None = None,
) -> tuple[int, tuple[int, int, int, int], str, int]:
    font_size = max(1, round(min(rendered.size) * font_scale))
    font = _load_font(font_size)
    draw = ImageDraw.Draw(rendered)
    line_spacing = max(1, round(font_size * 0.2))
    rendered_text = _wrap_text_by_character_count(text, max_characters_per_line)
    line_count = rendered_text.count("\n") + 1

    raw_bbox = draw.multiline_textbbox(
        (0, 0), rendered_text, font=font, spacing=line_spacing
    )
    text_width = raw_bbox[2] - raw_bbox[0]
    text_height = raw_bbox[3] - raw_bbox[1]
    if target_rect is not None:
        target_x, target_y = _text_position_in_rect(
            target_rect, (text_width, text_height)
        )
    elif placement is not None:
        target_x, target_y = _text_position(
            rendered.size, (text_width, text_height), placement
        )
    else:
        raise ValueError("placement or target_rect is required")
    draw_position = (target_x - raw_bbox[0], target_y - raw_bbox[1])

    draw.multiline_text(
        draw_position,
        rendered_text,
        fill=color,
        font=font,
        spacing=line_spacing,
    )
    text_bbox = draw.multiline_textbbox(
        draw_position, rendered_text, font=font, spacing=line_spacing
    )
    return (
        font_size,
        tuple(int(value) for value in text_bbox),
        rendered_text,
        line_count,
    )


def _measure_multiline_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    spacing: int,
) -> tuple[int, int, tuple[int, int, int, int]]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox


def _split_word_to_fit_width(
    word: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for character in word:
        candidate = f"{current}{character}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if current and width > max_width:
            chunks.append(current)
            current = character
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [word]


def _wrap_text_to_pixel_width(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> str:
    lines: list[str] = []
    for source_line in text.splitlines() or [""]:
        words = source_line.split()
        if not words:
            lines.append("")
            continue

        current_line = ""
        for word in words:
            word_parts = _split_word_to_fit_width(word, font, max_width, draw)
            for part in word_parts:
                candidate = part if not current_line else f"{current_line} {part}"
                candidate_width = draw.textbbox((0, 0), candidate, font=font)[2]
                if not current_line or candidate_width <= max_width:
                    current_line = candidate
                else:
                    lines.append(current_line)
                    current_line = part
        if current_line:
            lines.append(current_line)
    return "\n".join(lines)


def _fit_text_to_rect(
    rendered: Image.Image,
    text: str,
    font_scale: float,
    rect: tuple[int, int, int, int],
    min_font_scale: float | None = None,
    require_fit: bool = False,
) -> tuple[int, tuple[int, int], str, int, tuple[int, int, int, int]]:
    draw = ImageDraw.Draw(rendered)
    rect_width, rect_height = rect[2], rect[3]
    max_font_size = max(1, round(min(rendered.size) * font_scale))
    min_font_size = (
        max(1, round(min(rendered.size) * min_font_scale))
        if min_font_scale is not None
        else 1
    )
    if min_font_size > max_font_size:
        raise ValueError("min_font_scale must be less than or equal to font_scale")

    best: tuple[int, tuple[int, int], str, int, tuple[int, int, int, int]] | None = None
    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = _load_font(font_size)
        spacing = max(1, round(font_size * 0.2))
        wrapped_text = _wrap_text_to_pixel_width(text, font, rect_width, draw)
        text_width, text_height, raw_bbox = _measure_multiline_text(
            draw, wrapped_text, font, spacing
        )
        line_count = wrapped_text.count("\n") + 1
        best = (
            font_size,
            (text_width, text_height),
            wrapped_text,
            line_count,
            raw_bbox,
        )
        if text_width <= rect_width and text_height <= rect_height:
            return best

    if best is None or require_fit:
        raise ValueError("unable to fit text into region")
    return best


def _render_fitted_text_in_rect(
    rendered: Image.Image,
    text: str,
    font_scale: float,
    color: tuple[int, int, int],
    rect: tuple[int, int, int, int],
    min_font_scale: float | None = None,
    require_fit: bool = False,
) -> tuple[int, tuple[int, int, int, int], str, int]:
    font_size, text_size, wrapped_text, line_count, raw_bbox = _fit_text_to_rect(
        rendered, text, font_scale, rect, min_font_scale, require_fit
    )
    font = _load_font(font_size)
    spacing = max(1, round(font_size * 0.2))
    target_x, target_y = _text_position_in_rect(rect, text_size)
    draw_position = (target_x - raw_bbox[0], target_y - raw_bbox[1])
    draw = ImageDraw.Draw(rendered)
    draw.multiline_text(
        draw_position,
        wrapped_text,
        fill=color,
        font=font,
        spacing=spacing,
    )
    text_bbox = draw.multiline_textbbox(
        draw_position, wrapped_text, font=font, spacing=spacing
    )
    return (
        font_size,
        tuple(int(value) for value in text_bbox),
        wrapped_text,
        line_count,
    )


def render_text_prompt(
    input_image_path: str | Path,
    output_image_path: str | Path,
    text: str,
    placement: str,
    font_scale: float,
    color: tuple[int, int, int] = (255, 0, 0),
    max_characters_per_line: int | None = None,
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

    font_size, text_bbox, rendered_text, line_count = _render_text_on_image(
        rendered=rendered,
        text=text,
        font_scale=font_scale,
        color=color,
        max_characters_per_line=max_characters_per_line,
        placement=placement,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    return RenderMetadata(
        output_path=str(output_path),
        placement=placement,
        font_size=font_size,
        text_bbox=tuple(int(value) for value in text_bbox),
        rendered_text=rendered_text,
        line_count=line_count,
    )


def render_text_prompt_in_top_ranked_mask(
    input_image_path: str | Path,
    output_image_path: str | Path,
    text: str,
    ranked_regions: list[RegionFeatures | dict[str, Any]],
    font_scale: float,
    brightness_offset: int = DEFAULT_BRIGHTNESS_OFFSET,
    max_characters_per_line: int | None = None,
) -> RenderMetadata:
    """Render text centered in the top-ranked mask rectangle using region color."""
    if font_scale <= 0:
        raise ValueError("font_scale must be greater than 0")

    input_path = Path(input_image_path)
    output_path = Path(output_image_path)
    selected_region = _select_top_ranked_region(ranked_regions)

    with Image.open(input_path) as image:
        rendered = image.convert("RGB")

    region_rect = _rect_from_region_bbox(selected_region.bbox, rendered.size)
    average_rgb = _average_rgb_for_rect(rendered, region_rect)
    final_rgb = _apply_brightness_offset(average_rgb, brightness_offset)
    font_size, text_bbox, rendered_text, line_count = _render_text_on_image(
        rendered=rendered,
        text=text,
        font_scale=font_scale,
        color=final_rgb,
        max_characters_per_line=max_characters_per_line,
        target_rect=region_rect,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    return RenderMetadata(
        output_path=str(output_path),
        placement="top_ranked_mask",
        font_size=font_size,
        text_bbox=text_bbox,
        rendered_text=rendered_text,
        line_count=line_count,
        selected_mask_id=selected_region.mask_id,
        average_rgb=average_rgb,
        brightness_offset=brightness_offset,
        final_rgb=final_rgb,
        region_bbox=region_rect,
    )


def _metadata_path_for_output(output_image_path: str | Path) -> Path:
    output_path = Path(output_image_path)
    return output_path.with_suffix(DEFAULT_MULTI_MASK_METADATA_SUFFIX)


def _visualization_path_for_output(output_image_path: str | Path) -> Path:
    output_path = Path(output_image_path)
    return output_path.with_name(
        f"{output_path.stem}{DEFAULT_MULTI_MASK_VISUALIZATION_SUFFIX}"
    )


def _write_chunk_metadata(
    path: str | Path,
    chunks: tuple[ChunkRenderMetadata, ...],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2),
        encoding="utf-8",
    )
    return output_path


def save_multi_mask_visualization(
    input_image_path: str | Path,
    chunks: tuple[ChunkRenderMetadata, ...],
    output_path: str | Path,
) -> Path:
    """Save an overlay showing which mask received each rendered chunk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_image_path) as image:
        base = image.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    palette = [
        (255, 60, 60),
        (40, 170, 255),
        (70, 210, 120),
        (255, 190, 50),
        (190, 110, 255),
        (255, 110, 170),
    ]

    for chunk in chunks:
        x, y, width, height = chunk.region_bbox
        color = palette[chunk.chunk_index % len(palette)]
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            fill=(*color, 54),
            outline=(*color, 255),
            width=3,
        )
        label = f"{chunk.chunk_index + 1}: mask {chunk.mask_id}"
        label_bbox = draw.textbbox((x + 4, y + 4), label, font=font)
        padding = 3
        draw.rectangle(
            (
                label_bbox[0] - padding,
                label_bbox[1] - padding,
                label_bbox[2] + padding,
                label_bbox[3] + padding,
            ),
            fill=(255, 255, 255, 230),
        )
        draw.text((x + 4, y + 4), label, fill=(*color, 255), font=font)

    visualized = Image.alpha_composite(base, overlay).convert("RGB")
    visualized.save(output)
    return output


def _union_text_bbox(
    chunks: tuple[ChunkRenderMetadata, ...],
) -> tuple[int, int, int, int]:
    left = min(chunk.text_bbox[0] for chunk in chunks)
    top = min(chunk.text_bbox[1] for chunk in chunks)
    right = max(chunk.text_bbox[2] for chunk in chunks)
    bottom = max(chunk.text_bbox[3] for chunk in chunks)
    return (left, top, right, bottom)


def render_text_prompt_across_ranked_masks(
    input_image_path: str | Path,
    output_image_path: str | Path,
    text: str,
    ranked_regions: list[RegionFeatures | dict[str, Any]],
    font_scale: float,
    brightness_offset: int = DEFAULT_BRIGHTNESS_OFFSET,
    metadata_output_path: str | Path | None = None,
    visualization_output_path: str | Path | None = None,
    write_auxiliary_outputs: bool = True,
) -> RenderMetadata:
    """Split text and render chunks into ranked masks in reading order."""
    if font_scale <= 0:
        raise ValueError("font_scale must be greater than 0")

    ordered_regions = _reading_order_regions(ranked_regions)
    if not ordered_regions:
        raise ValueError("ranked_regions must contain at least one region")

    chunks_text = _split_text_into_chunks(text, len(ordered_regions))
    selected_regions = ordered_regions[: len(chunks_text)]
    input_path = Path(input_image_path)
    output_path = Path(output_image_path)

    with Image.open(input_path) as image:
        rendered = image.convert("RGB")

    chunk_metadata: list[ChunkRenderMetadata] = []
    for chunk_index, (chunk_text, region) in enumerate(
        zip(chunks_text, selected_regions)
    ):
        region_rect = _rect_from_region_bbox(region.bbox, rendered.size)
        average_rgb = _average_rgb_for_rect(rendered, region_rect)
        final_rgb = _apply_brightness_offset(average_rgb, brightness_offset)
        font_size, text_bbox, rendered_chunk, line_count = _render_fitted_text_in_rect(
            rendered=rendered,
            text=chunk_text,
            font_scale=font_scale,
            color=final_rgb,
            rect=region_rect,
        )
        chunk_metadata.append(
            ChunkRenderMetadata(
                chunk_index=chunk_index,
                mask_id=region.mask_id,
                rank=region.rank,
                text=rendered_chunk,
                region_bbox=region_rect,
                average_rgb=average_rgb,
                brightness_offset=brightness_offset,
                final_rgb=final_rgb,
                font_size=font_size,
                text_bbox=text_bbox,
                line_count=line_count,
            )
        )

    chunks = tuple(chunk_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    metadata_path = None
    visualization_path = None
    if write_auxiliary_outputs:
        metadata_path = (
            Path(metadata_output_path)
            if metadata_output_path is not None
            else _metadata_path_for_output(output_path)
        )
        visualization_path = (
            Path(visualization_output_path)
            if visualization_output_path is not None
            else _visualization_path_for_output(output_path)
        )
        _write_chunk_metadata(metadata_path, chunks)
        save_multi_mask_visualization(input_path, chunks, visualization_path)

    return RenderMetadata(
        output_path=str(output_path),
        placement="multi_mask",
        font_size=max(chunk.font_size for chunk in chunks),
        text_bbox=_union_text_bbox(chunks),
        rendered_text=" ".join(chunk.text.replace("\n", " ") for chunk in chunks),
        line_count=sum(chunk.line_count for chunk in chunks),
        chunks=chunks,
        chunk_metadata_path=str(metadata_path) if metadata_path is not None else None,
        visualization_path=(
            str(visualization_path) if visualization_path is not None else None
        ),
    )


def render_text_prompt_adaptive_mask_average(
    input_image_path: str | Path,
    output_image_path: str | Path,
    text: str,
    ranked_regions: list[RegionFeatures | dict[str, Any]],
    font_scale: float,
    min_font_scale: float = 0.02,
    brightness_offset: int = DEFAULT_BRIGHTNESS_OFFSET,
    write_auxiliary_outputs: bool = False,
) -> RenderMetadata:
    """Try a fitted top-ranked mask first, then fall back to multi-mask rendering."""
    if font_scale <= 0:
        raise ValueError("font_scale must be greater than 0")
    if min_font_scale <= 0:
        raise ValueError("min_font_scale must be greater than 0")
    if min_font_scale > font_scale:
        raise ValueError("min_font_scale must be less than or equal to font_scale")

    input_path = Path(input_image_path)
    output_path = Path(output_image_path)
    selected_region = _select_top_ranked_region(ranked_regions)

    with Image.open(input_path) as image:
        rendered = image.convert("RGB")

    region_rect = _rect_from_region_bbox(selected_region.bbox, rendered.size)
    average_rgb = _average_rgb_for_rect(rendered, region_rect)
    final_rgb = _apply_brightness_offset(average_rgb, brightness_offset)

    try:
        font_size, text_bbox, rendered_text, line_count = _render_fitted_text_in_rect(
            rendered=rendered,
            text=text,
            font_scale=font_scale,
            color=final_rgb,
            rect=region_rect,
            min_font_scale=min_font_scale,
            require_fit=True,
        )
    except ValueError:
        metadata = render_text_prompt_across_ranked_masks(
            input_image_path=input_path,
            output_image_path=output_path,
            text=text,
            ranked_regions=ranked_regions,
            font_scale=font_scale,
            brightness_offset=brightness_offset,
            write_auxiliary_outputs=write_auxiliary_outputs,
        )
        return RenderMetadata(
            output_path=metadata.output_path,
            placement=metadata.placement,
            font_size=metadata.font_size,
            text_bbox=metadata.text_bbox,
            rendered_text=metadata.rendered_text,
            line_count=metadata.line_count,
            selected_mask_id=metadata.selected_mask_id,
            average_rgb=metadata.average_rgb,
            brightness_offset=metadata.brightness_offset,
            final_rgb=metadata.final_rgb,
            region_bbox=metadata.region_bbox,
            chunks=metadata.chunks,
            chunk_metadata_path=metadata.chunk_metadata_path,
            visualization_path=metadata.visualization_path,
            adaptive_fallback_used=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    return RenderMetadata(
        output_path=str(output_path),
        placement="top_ranked_mask",
        font_size=font_size,
        text_bbox=text_bbox,
        rendered_text=rendered_text,
        line_count=line_count,
        selected_mask_id=selected_region.mask_id,
        average_rgb=average_rgb,
        brightness_offset=brightness_offset,
        final_rgb=final_rgb,
        region_bbox=region_rect,
        adaptive_fallback_used=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render visible prompt text onto an image."
    )
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output", required=True, help="Path for the rendered output image."
    )
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
    parser.add_argument(
        "--max-characters-per-line",
        type=int,
        help="Wrap rendered text after this many characters per line.",
    )
    parser.add_argument(
        "--ranked-masks",
        help=(
            "Path to ranked_masks.json. When provided, render into the top-ranked "
            "mask rectangle using region-average color."
        ),
    )
    parser.add_argument(
        "--brightness-offset",
        type=int,
        default=DEFAULT_BRIGHTNESS_OFFSET,
        help=f"RGB brightness offset for mask rendering. Defaults to {DEFAULT_BRIGHTNESS_OFFSET}.",
    )
    parser.add_argument(
        "--multi-mask",
        action="store_true",
        help=(
            "Split the prompt across ranked masks in reading order. Requires "
            "--ranked-masks."
        ),
    )
    parser.add_argument(
        "--chunk-metadata-output",
        help="Path for multi-mask chunk metadata JSON.",
    )
    parser.add_argument(
        "--chunk-visualization-output",
        help="Path for multi-mask chunk assignment visualization.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.multi_mask:
        if not args.ranked_masks:
            parser.error("--multi-mask requires --ranked-masks")
        metadata = render_text_prompt_across_ranked_masks(
            input_image_path=args.input,
            output_image_path=args.output,
            text=args.text,
            ranked_regions=_load_ranked_regions(args.ranked_masks),
            font_scale=args.font_scale,
            brightness_offset=args.brightness_offset,
            metadata_output_path=args.chunk_metadata_output,
            visualization_output_path=args.chunk_visualization_output,
        )
    elif args.ranked_masks:
        metadata = render_text_prompt_in_top_ranked_mask(
            input_image_path=args.input,
            output_image_path=args.output,
            text=args.text,
            ranked_regions=_load_ranked_regions(args.ranked_masks),
            font_scale=args.font_scale,
            brightness_offset=args.brightness_offset,
            max_characters_per_line=args.max_characters_per_line,
        )
    else:
        metadata = render_text_prompt(
            input_image_path=args.input,
            output_image_path=args.output,
            text=args.text,
            placement=args.placement,
            font_scale=args.font_scale,
            color=args.color,
            max_characters_per_line=args.max_characters_per_line,
        )
    print(f"output_path={metadata.output_path}")
    print(f"placement={metadata.placement}")
    print(f"font_size={metadata.font_size}")
    print(f"text_bbox={metadata.text_bbox}")
    print(f"line_count={metadata.line_count}")
    if metadata.selected_mask_id is not None:
        print(f"selected_mask_id={metadata.selected_mask_id}")
        print(f"average_rgb={metadata.average_rgb}")
        print(f"brightness_offset={metadata.brightness_offset}")
        print(f"final_rgb={metadata.final_rgb}")
    if metadata.chunks:
        print(f"chunk_count={len(metadata.chunks)}")
        print(f"chunk_metadata_path={metadata.chunk_metadata_path}")
        print(f"visualization_path={metadata.visualization_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
