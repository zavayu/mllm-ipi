"""Generate and rank candidate image masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.region_ranker import RegionFeatures, compute_and_rank_regions


def _load_rgb_image(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _extract_sam_mask(candidate: dict[str, Any]) -> np.ndarray:
    if "segmentation" not in candidate:
        raise ValueError("SAM mask candidate is missing 'segmentation'")
    return np.asarray(candidate["segmentation"], dtype=bool)


def _generate_sam_masks(
    image: np.ndarray,
    checkpoint: str | Path,
    model_type: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
) -> list[np.ndarray]:
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment_anything is not installed; install SAM or omit --sam-checkpoint "
            "to use the deterministic fallback generator"
        ) from exc

    if model_type not in sam_model_registry:
        supported = ", ".join(sorted(sam_model_registry))
        raise ValueError(f"unsupported SAM model type '{model_type}'; choose: {supported}")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    candidates = generator.generate(image)
    return [_extract_sam_mask(candidate) for candidate in candidates]


def _grid_fallback_masks(image: np.ndarray) -> list[np.ndarray]:
    """Create deterministic candidates when SAM is unavailable in local dev."""
    height, width = image.shape[:2]
    masks: list[np.ndarray] = []

    for row in range(3):
        for column in range(3):
            y0 = round(row * height / 3)
            y1 = round((row + 1) * height / 3)
            x0 = round(column * width / 3)
            x1 = round((column + 1) * width / 3)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            masks.append(mask)

    for y0, y1, x0, x1 in (
        (0, height // 2, width // 2, width),
        (height // 2, height, width // 4, 3 * width // 4),
    ):
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        masks.append(mask)

    return masks


def generate_candidate_masks(
    image: np.ndarray,
    sam_checkpoint: str | Path | None = None,
    sam_model_type: str = "vit_b",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> list[np.ndarray]:
    """Generate candidate masks with SAM when configured, otherwise fallback masks."""
    checkpoint = sam_checkpoint or os.getenv("SAM_CHECKPOINT")
    if checkpoint:
        return _generate_sam_masks(
            image=image,
            checkpoint=checkpoint,
            model_type=sam_model_type,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
    return _grid_fallback_masks(image)


def _draw_rank_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    padding = 3
    background = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )
    draw.rectangle(background, fill=(255, 255, 255))
    draw.text(xy, text, fill=color, font=font)


def save_ranked_mask_visualization(
    image: np.ndarray,
    masks: list[np.ndarray],
    ranked_regions: list[RegionFeatures],
    output_path: str | Path,
) -> Path:
    """Save an overlay showing ranked candidate masks."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = [
        (255, 60, 60),
        (40, 170, 255),
        (70, 210, 120),
        (255, 190, 50),
        (190, 110, 255),
    ]

    for region in ranked_regions:
        if region.rank is None:
            continue
        color = palette[(region.rank - 1) % len(palette)]
        mask = np.asarray(masks[region.mask_id], dtype=bool)
        mask_layer = Image.new("RGBA", base.size, (*color, 0))
        alpha = Image.fromarray((mask.astype(np.uint8) * 72), mode="L")
        mask_layer.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, mask_layer)

        x, y, width, height = region.bbox
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(*color, 255), width=3)
        _draw_rank_label(draw, (x + 4, y + 4), f"#{region.rank}", color)

    visualized = Image.alpha_composite(base, overlay).convert("RGB")
    visualized.save(output)
    return output


def generate_and_rank_masks(
    image_path: str | Path,
    out_dir: str | Path,
    sam_checkpoint: str | Path | None = None,
    sam_model_type: str = "vit_b",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> list[RegionFeatures]:
    """Generate masks, rank them, and save metadata plus visualization."""
    image = _load_rgb_image(image_path)
    masks = generate_candidate_masks(
        image=image,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    ranked_regions = compute_and_rank_regions(image, masks)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "ranked_masks.json"
    visualization_path = output_dir / "ranked_masks.png"

    metadata_path.write_text(
        json.dumps([region.to_dict() for region in ranked_regions], indent=2),
        encoding="utf-8",
    )
    save_ranked_mask_visualization(image, masks, ranked_regions, visualization_path)
    return ranked_regions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate candidate masks and rank regions for prompt placement."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--out-dir", required=True, help="Directory for mask outputs.")
    parser.add_argument(
        "--sam-checkpoint",
        help="Path to a SAM checkpoint. Defaults to SAM_CHECKPOINT when set.",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_b",
        help="SAM model type, for example vit_b, vit_l, or vit_h.",
    )
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ranked_regions = generate_and_rank_masks(
        image_path=args.image,
        out_dir=args.out_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    print(f"mask_count={len(ranked_regions)}")
    print(f"metadata_path={Path(args.out_dir) / 'ranked_masks.json'}")
    print(f"visualization_path={Path(args.out_dir) / 'ranked_masks.png'}")
    if ranked_regions:
        best = ranked_regions[0]
        print(f"best_mask_id={best.mask_id}")
        print(f"best_location={best.location}")
        print(f"best_bbox={best.bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
