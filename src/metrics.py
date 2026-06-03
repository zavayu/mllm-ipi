"""Image distortion metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def _validate_same_shape(original: np.ndarray, modified: np.ndarray) -> None:
    if original.shape != modified.shape:
        raise ValueError(
            "images must have the same dimensions and channels: "
            f"original={original.shape}, modified={modified.shape}"
        )


def compute_mse(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute mean squared error between two image arrays."""
    _validate_same_shape(original, modified)
    diff = original.astype(np.float64) - modified.astype(np.float64)
    return float(np.mean(diff * diff))


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute structural similarity between two image arrays."""
    _validate_same_shape(original, modified)
    height, width = original.shape[:2]
    min_side = min(height, width)
    if min_side < 3:
        raise ValueError("images must be at least 3x3 pixels to compute SSIM")

    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1

    return float(
        structural_similarity(
            original,
            modified,
            channel_axis=2,
            data_range=255,
            win_size=win_size,
        )
    )


def compute_image_metrics(
    original_path: str | Path, modified_path: str | Path
) -> dict[str, float]:
    """Compute image distortion metrics for two image files."""
    original = load_image(original_path)
    modified = load_image(modified_path)
    return {
        "mse": compute_mse(original, modified),
        "ssim": compute_ssim(original, modified),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute image distortion metrics.")
    parser.add_argument("--original", required=True, help="Path to the original image.")
    parser.add_argument("--modified", required=True, help="Path to the modified image.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metrics = compute_image_metrics(args.original, args.modified)
    print(f"mse={metrics['mse']}")
    print(f"ssim={metrics['ssim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
