import json

import numpy as np
from PIL import Image
import pytest

from src.benchmark_data import (
    image_features,
    duplicate_groups,
    make_manifest,
    load_manifest,
    prepare_coco,
)
from src.provenance import fingerprint


@pytest.fixture
def images(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(470)
    entries = []
    for i in range(5):
        path = root / f"{i:012d}.jpg"
        Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)).save(path)
        entries.append({"id": path.stem, "filename": path.name, **image_features(path)})
    return root, entries


def test_transitive_near_duplicates_stay_together():
    entries = [
        {"id": str(i), "pixel_sha256": str(i), "dhash": f"{h:016x}"}
        for i, h in enumerate((0, 1, 3))
    ]
    assert len(duplicate_groups(entries, threshold=1)) == 1


def test_splits_are_deterministic_disjoint_and_verified(tmp_path, images):
    root, entries = images
    path = tmp_path / "manifest.json"
    first = make_manifest(entries, root, path, (2, 1, 1), 470, {"purpose": "test"})
    assert (
        make_manifest(
            list(reversed(entries)), root, path, (2, 1, 1), 470, {"purpose": "test"}
        )
        == first
    )
    seen = set()
    for split, count in first["split_sizes"].items():
        _, paths = load_manifest(path, split)
        assert len(paths) == count
        assert seen.isdisjoint(paths)
        seen.update(paths)
    chosen = first["images"][0]
    (root / chosen["filename"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        load_manifest(path, chosen["split"])


def test_frozen_manifest_cannot_be_overwritten(tmp_path, images):
    root, entries = images
    path = tmp_path / "manifest.json"
    make_manifest(entries, root, path, (2, 1, 1), 470, {})
    with pytest.raises(ValueError, match="already frozen"):
        make_manifest(entries, root, path, (2, 1, 1), 471, {})


def test_duplicate_images_are_not_independent_samples(tmp_path, images):
    root, entries = images
    duplicate = {**entries[0], "id": "duplicate", "filename": "duplicate.jpg"}
    manifest = make_manifest(
        entries + [duplicate], root, tmp_path / "manifest.json", (2, 1, 1), 470, {}
    )
    assert manifest["unique_group_count"] == 5
    assert len(manifest["duplicate_groups"]) == 1
    assert len({entry["pixel_sha256"] for entry in manifest["images"]}) == 4


def test_manifest_tampering_and_cross_split_duplicates_are_rejected(tmp_path, images):
    root, entries = images
    path = tmp_path / "manifest.json"
    manifest = make_manifest(entries, root, path, (2, 1, 1), 470, {})
    manifest["images"][2]["group_id"] = manifest["images"][0]["group_id"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        load_manifest(path, "development")
    manifest.pop("manifest_id")
    manifest["manifest_id"] = fingerprint(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="crosses split"):
        load_manifest(path, "development")


def test_coco_preparation_uses_cached_images_without_network(
    tmp_path, images, monkeypatch
):
    root, entries = images
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": int(e["id"]),
                        "file_name": e["filename"],
                        "width": 60,
                        "height": 40,
                        "license": 1,
                    }
                    for e in entries
                ]
            }
        )
    )

    def cached_only(url, path):
        assert path.exists()

    monkeypatch.setattr("src.benchmark_data.download", cached_only)
    manifest = prepare_coco(tmp_path / "manifest.json", root, metadata, sizes=(2, 1, 1))
    assert manifest["source"]["dataset"] == "COCO 2017 test images"
    assert manifest["split_sizes"] == {"development": 2, "validation": 1, "test": 1}


def test_experiment_runner_consumes_only_frozen_split(tmp_path, images):
    from src.config import (
        DatasetConfig,
        ExperimentConfig,
        TargetConfig,
        RenderingConfig,
        ModelConfig,
        OutputConfig,
    )
    from src.run_experiment import run_experiment

    root, entries = images
    path = tmp_path / "manifest.json"
    manifest = make_manifest(entries, root, path, (2, 1, 1), 470, {})
    config = ExperimentConfig(
        dataset=DatasetConfig(root, manifest=path, split="development"),
        target=TargetConfig("BANANA", "Output BANANA"),
        rendering=RenderingConfig(("center",), (0.1,)),
        model=ModelConfig("mock", response="BANANA"),
        output=OutputConfig(tmp_path / "generated", tmp_path / "results.jsonl"),
    )
    rows = run_experiment(config)
    assert len(rows) == 2
    expected = {
        entry["sha256"]
        for entry in manifest["images"]
        if entry["split"] == "development"
    }
    assert {row["image_sha256"] for row in rows} == expected
    provenance = json.loads(open(rows[0]["manifest_path"], encoding="utf-8").read())
    assert provenance["dataset_manifest"]["manifest_id"] == manifest["manifest_id"]
