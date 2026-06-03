import json
import subprocess
import sys

import numpy as np
from PIL import Image

from src.region_ranker import compute_and_rank_regions, compute_region_features


def test_compute_region_features_extracts_mask_statistics():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[0:2, 4:6] = (10, 20, 30)
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 4:6] = True

    features = compute_region_features(image, mask, mask_id=7)

    assert features.mask_id == 7
    assert features.area == 4
    assert features.bbox == (4, 0, 2, 2)
    assert features.average_rgb == (10.0, 20.0, 30.0)
    assert features.rgb_variance == 0.0
    assert features.center == (4.5, 0.5)
    assert features.location == "top_right"


def test_rank_regions_prefers_large_uniform_preferred_locations():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, :] = (128, 128, 128)
    image[4:8, 4:8] = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)

    small_noisy_center = np.zeros((12, 12), dtype=bool)
    small_noisy_center[4:8, 4:8] = True
    top_right = np.zeros((12, 12), dtype=bool)
    top_right[0:4, 8:12] = True
    bottom_middle = np.zeros((12, 12), dtype=bool)
    bottom_middle[8:12, 4:8] = True

    ranked = compute_and_rank_regions(
        image,
        [small_noisy_center, top_right, bottom_middle],
    )

    assert [region.rank for region in ranked] == [1, 2, 3]
    assert ranked[0].location in {"top_right", "bottom_middle"}
    assert ranked[-1].location == "center"
    assert ranked[0].score > ranked[-1].score


def test_segment_regions_cli_saves_metadata_and_visualization(tmp_path):
    input_path = tmp_path / "input.jpg"
    out_dir = tmp_path / "masks"
    Image.new("RGB", (90, 60), (200, 200, 200)).save(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.segment_regions",
            "--image",
            str(input_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata_path = out_dir / "ranked_masks.json"
    visualization_path = out_dir / "ranked_masks.png"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path.exists()
    assert visualization_path.exists()
    assert "mask_count=" in result.stdout
    assert len(metadata) > 0
    assert metadata[0]["rank"] == 1
    assert {"area", "bbox", "average_rgb", "rgb_variance", "center", "location"}.issubset(
        metadata[0]
    )
