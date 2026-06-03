import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from src.metrics import compute_image_metrics, compute_mse, compute_ssim


def _save_rgb(path, array):
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def test_compute_mse_for_small_arrays():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    modified = np.full((4, 4, 3), 10, dtype=np.uint8)

    assert compute_mse(original, modified) == 100.0


def test_compute_ssim_is_one_for_identical_images():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 100
    image[:, :, 1] = 120
    image[:, :, 2] = 140

    assert compute_ssim(image, image) == pytest.approx(1.0)


def test_compute_image_metrics_from_generated_images(tmp_path):
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    modified = original.copy()
    modified[0, 0] = [30, 30, 30]

    original_path = tmp_path / "original.jpg"
    modified_path = tmp_path / "modified.png"
    _save_rgb(original_path, original)
    _save_rgb(modified_path, modified)

    metrics = compute_image_metrics(original_path, modified_path)

    assert metrics["mse"] > 0
    assert 0 <= metrics["ssim"] < 1


def test_metrics_reject_mismatched_shapes():
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    modified = np.zeros((7, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="same dimensions"):
        compute_mse(original, modified)


def test_metrics_cli_outputs_mse_and_ssim(tmp_path):
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    modified = original.copy()
    modified[2:4, 2:4] = [255, 255, 255]

    original_path = tmp_path / "original.jpg"
    modified_path = tmp_path / "modified.png"
    _save_rgb(original_path, original)
    _save_rgb(modified_path, modified)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.metrics",
            "--original",
            str(original_path),
            "--modified",
            str(modified_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mse=" in result.stdout
    assert "ssim=" in result.stdout
