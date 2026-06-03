"""Rank image regions for prompt placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        RegionFeatures(
            mask_id=region.mask_id,
            area=region.area,
            bbox=region.bbox,
            average_rgb=region.average_rgb,
            rgb_variance=region.rgb_variance,
            center=region.center,
            location=region.location,
            score=_score_region(region, max_area, max_variance, location_bonus),
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

    return [
        RegionFeatures(
            mask_id=region.mask_id,
            area=region.area,
            bbox=region.bbox,
            average_rgb=region.average_rgb,
            rgb_variance=region.rgb_variance,
            center=region.center,
            location=region.location,
            score=region.score,
            rank=index + 1,
        )
        for index, region in enumerate(scored)
    ]


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
