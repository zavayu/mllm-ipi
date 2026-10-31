"""Rank image regions for prompt placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Iterable

import numpy as np

PREFERRED_LOCATIONS = {"top_right", "bottom_middle"}


@dataclass(frozen=True)
class RegionFeatures:
    """Computed features and score for one candidate mask."""

    mask_id: int
    area: int
    bbox: tuple[int, int, int, int]
    average_rgb: tuple[float, float, float]
    rgb_variance: float
    center: tuple[float, float]
    location: str
    score: float = 0.0
    rank: int | None = None
    usable_bbox: tuple[int, int, int, int] | None = None
    mask_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable region metadata."""
        return asdict(self)


def _validate_image_and_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be an RGB array, got shape {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(
            "mask dimensions must match image height and width: "
            f"image={image.shape[:2]}, mask={mask.shape}"
        )
    return mask.astype(bool)


def classify_location(
    center: tuple[float, float],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> str:
    """Classify a mask center into a coarse image location."""
    height, width = image_shape[:2]
    x, y = center

    if y < height / 3:
        vertical = "top"
    elif y < 2 * height / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    if x < width / 3:
        horizontal = "left"
    elif x < 2 * width / 3:
        horizontal = "middle"
    else:
        horizontal = "right"

    if vertical == "middle" and horizontal == "middle":
        return "center"
    return f"{vertical}_{horizontal}"


def largest_interior_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest all-true axis-aligned rectangle, using a row histogram in O(HW)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask must be a nonempty 2D region")
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    cropped = mask[y0:y1, x0:x1]
    if cropped.all():
        return (x0, y0, x1 - x0, y1 - y0)
    heights = np.zeros(cropped.shape[1], dtype=int)
    best_area, best = 0, None
    for y, row in enumerate(cropped):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > height:
                start, previous_height = stack.pop()
                area = previous_height * (x - start)
                if area > best_area:
                    best_area = area
                    best = (
                        x0 + start,
                        y0 + y - previous_height + 1,
                        x - start,
                        previous_height,
                    )
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best


def compute_region_features(
    image: np.ndarray,
    mask: np.ndarray,
    mask_id: int = 0,
) -> RegionFeatures:
    """Compute area, bbox, color statistics, and location for a mask."""
    boolean_mask = _validate_image_and_mask(image, mask)
    ys, xs = np.nonzero(boolean_mask)
    area = int(boolean_mask.sum())
    if area == 0:
        raise ValueError("mask must contain at least one pixel")

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    center = (float(xs.mean()), float(ys.mean()))

    pixels = image[boolean_mask].astype(np.float64)
    average_rgb = tuple(float(value) for value in pixels.mean(axis=0))
    rgb_variance = float(pixels.var(axis=0).mean())

    return RegionFeatures(
        mask_id=mask_id,
        area=area,
        bbox=bbox,
        average_rgb=average_rgb,
        rgb_variance=rgb_variance,
        center=center,
        location=classify_location(center, image.shape),
        usable_bbox=largest_interior_rectangle(boolean_mask),
        mask_sha256=sha256(boolean_mask.tobytes()).hexdigest(),
    )


def _score_region(
    region: RegionFeatures,
    max_area: int,
    max_variance: float,
    location_bonus: float,
) -> float:
    area_score = region.area / max_area if max_area else 0.0
    variance_score = 1.0 - (region.rgb_variance / max_variance if max_variance else 0.0)
    preferred_score = location_bonus if region.location in PREFERRED_LOCATIONS else 0.0
    return float(area_score + variance_score + preferred_score)


def rank_regions(
    regions: Iterable[RegionFeatures],
    location_bonus: float = 0.35,
) -> list[RegionFeatures]:
    """Rank masks by area, low color variance, and preferred location."""
    region_list = list(regions)
    if not region_list:
        return []

    max_area = max(region.area for region in region_list)
    max_variance = max(region.rgb_variance for region in region_list)

    scored = [
        replace(
            region, score=_score_region(region, max_area, max_variance, location_bonus)
        )
        for region in region_list
    ]
    scored.sort(
        key=lambda region: (
            region.score,
            region.area,
            -region.rgb_variance,
            region.location in PREFERRED_LOCATIONS,
        ),
        reverse=True,
    )

    return [replace(region, rank=index + 1) for index, region in enumerate(scored)]


def compute_and_rank_regions(
    image: np.ndarray,
    masks: Iterable[np.ndarray],
    location_bonus: float = 0.35,
) -> list[RegionFeatures]:
    """Compute features for candidate masks and return them ranked."""
    features = [
        compute_region_features(image=image, mask=mask, mask_id=index)
        for index, mask in enumerate(masks)
        if np.asarray(mask).astype(bool).any()
    ]
    return rank_regions(features, location_bonus=location_bonus)
