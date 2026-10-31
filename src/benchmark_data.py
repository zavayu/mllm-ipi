"""Prepare a reproducible COCO subset without downloading the full image archive."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import requests

from src.provenance import file_hash, fingerprint, write_json_atomic

METADATA_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/image_info_test2017.zip"
IMAGE_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org/test2017/"
SPLITS = ("development", "validation", "test")
DEFAULT_SIZES = (30, 100, 100)


def image_features(path: Path) -> dict:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = gray[:, 1:] > gray[:, :-1]
        dhash = sum(int(bit) << i for i, bit in enumerate(bits.flat))
        return {
            "sha256": file_hash(path),
            "pixel_sha256": sha256(
                f"{width}x{height}|".encode() + image.tobytes()
            ).hexdigest(),
            "dhash": f"{dhash:016x}",
            "width": width,
            "height": height,
        }


def duplicate_groups(images: list[dict], threshold: int = 4) -> list[list[dict]]:
    """Connected components prevent a duplicate chain from crossing split boundaries."""
    if not 0 <= threshold <= 64:
        raise ValueError("duplicate threshold must be between 0 and 64")
    parent = list(range(len(images)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, image in enumerate(images):
        for j in range(i):
            other = images[j]
            if (
                image["pixel_sha256"] == other["pixel_sha256"]
                or (int(image["dhash"], 16) ^ int(other["dhash"], 16)).bit_count()
                <= threshold
            ):
                parent[find(i)] = find(j)
    groups = {}
    for i, image in enumerate(images):
        groups.setdefault(find(i), []).append(image)
    return sorted(
        [sorted(group, key=lambda image: image["id"]) for group in groups.values()],
        key=lambda group: group[0]["id"],
    )


def make_manifest(
    images: list[dict],
    image_root: Path,
    manifest_path: Path,
    sizes: tuple[int, int, int],
    seed: int,
    source: dict,
    threshold: int = 4,
) -> dict:
    if len(sizes) != 3 or any(size < 0 for size in sizes) or sum(sizes) == 0:
        raise ValueError(
            "split sizes must be nonnegative and select at least one image"
        )
    groups = duplicate_groups(images, threshold)
    if len({image["id"] for image in images}) != len(images):
        raise ValueError("image IDs must be unique")
    representatives = []
    for group in groups:
        ordered = sorted(group, key=lambda image: image["id"])
        group_id = fingerprint(sorted(image["pixel_sha256"] for image in ordered))
        representatives.append({**ordered[0], "group_id": group_id})
    representatives.sort(key=lambda image: fingerprint([seed, image["id"]]))
    if len(representatives) < sum(sizes):
        raise ValueError(
            f"need {sum(sizes)} distinct groups, found {len(representatives)}; increase candidate count"
        )
    import os

    selected = []
    cursor = 0
    for split, size in zip(SPLITS, sizes):
        selected.extend(
            {**image, "split": split}
            for image in representatives[cursor : cursor + size]
        )
        cursor += size
    manifest = {
        "schema_version": 1,
        "source": source,
        "seed": seed,
        "image_root": Path(
            os.path.relpath(image_root.resolve(), manifest_path.parent.resolve())
        ).as_posix(),
        "split_sizes": dict(zip(SPLITS, sizes)),
        "duplicate_threshold": threshold,
        "duplicate_method": "RGB pixel equality or 64-bit dHash Hamming distance, transitive grouping",
        "candidate_count": len(images),
        "unique_group_count": len(groups),
        "duplicate_groups": [
            [image["id"] for image in group] for group in groups if len(group) > 1
        ],
        "images": selected,
    }
    manifest = {"manifest_id": fingerprint(manifest), **manifest}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
    ):
        raise ValueError(
            "manifest is already frozen; use a new output path for a different dataset"
        )
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_manifest(
    path: Path, split: str, verify: bool = True
) -> tuple[dict, list[Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = manifest.pop("manifest_id", None)
    if fingerprint(manifest) != identity:
        raise ValueError("dataset manifest content does not match its identity")
    manifest["manifest_id"] = identity
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    root = (path.parent / manifest["image_root"]).resolve()
    assignments = {}
    entries = manifest["images"]
    if len({entry["id"] for entry in entries}) != len(entries) or len(
        {entry["filename"] for entry in entries}
    ) != len(entries):
        raise ValueError("manifest contains duplicate image IDs or filenames")
    for entry in entries:
        for key in ("group_id", "pixel_sha256"):
            value = entry[key]
            previous = assignments.setdefault((key, value), entry["split"])
            if previous != entry["split"]:
                raise ValueError("duplicate image group crosses split boundaries")
    selected = [entry for entry in entries if entry["split"] == split]
    if len(selected) != manifest["split_sizes"][split]:
        raise ValueError("split count does not match manifest")
    paths = []
    for entry in selected:
        image = (root / entry["filename"]).resolve()
        if not image.is_relative_to(root):
            raise ValueError("image path escapes dataset root")
        if verify and file_hash(image) != entry["sha256"]:
            raise ValueError(f"dataset image changed: {image.name}")
        paths.append(image)
    return manifest, paths


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    # requests verifies TLS; no credential, proxy, or certificate overrides.
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    temporary.replace(path)


def prepare_coco(
    manifest_path: Path,
    image_root: Path,
    metadata_path: Path,
    sizes=DEFAULT_SIZES,
    seed=470,
    candidate_count=None,
    workers=4,
) -> dict:
    download(METADATA_URL, metadata_path)
    if zipfile.is_zipfile(metadata_path):
        with zipfile.ZipFile(metadata_path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith("image_info_test2017.json")
            ]
            if len(candidates) != 1:
                raise ValueError("expected official COCO test2017 image-info metadata")
            metadata = json.loads(archive.read(candidates[0]))
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    count = candidate_count or sum(sizes) + max(32, sum(sizes) // 5)
    if count < sum(sizes):
        raise ValueError("candidate count must cover all requested splits")
    records = sorted(
        metadata["images"], key=lambda image: fingerprint([seed, image["id"]])
    )[:count]
    if len(records) < sum(sizes):
        raise ValueError("not enough source images for the requested splits")

    def fetch(record):
        filename = record["file_name"]
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            raise ValueError("COCO metadata filename must be a plain JPEG filename")
        image = image_root / filename
        url = IMAGE_BASE_URL + filename
        download(url, image)
        features = image_features(image)
        if (features["width"], features["height"]) != (
            record["width"],
            record["height"],
        ):
            raise ValueError(f"COCO image dimensions mismatch: {filename}")
        return {
            "id": f"coco-test2017-{record['id']:012d}",
            "coco_id": record["id"],
            "filename": filename,
            "url": url,
            "license_id": record.get("license"),
            **features,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        images = list(executor.map(fetch, records))
    return make_manifest(
        images,
        image_root,
        manifest_path,
        sizes,
        seed,
        {
            "dataset": "COCO 2017 test images",
            "metadata_url": METADATA_URL,
            "metadata_sha256": file_hash(metadata_path),
            "licenses": metadata.get("licenses", []),
            "purpose": "benchmark",
        },
    )


def prepare_local(manifest_path: Path, image_root: Path, seed=470) -> dict:
    images = [
        {"id": path.name, "filename": path.name, **image_features(path)}
        for path in sorted(image_root.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    count = len(duplicate_groups(images))
    return make_manifest(
        images,
        image_root,
        manifest_path,
        (count, 0, 0),
        seed,
        {
            "dataset": "Existing local images; origin unverified",
            "purpose": "runtime_pilot",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["coco", "local"])
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/coco2017.json")
    )
    parser.add_argument(
        "--image-dir", type=Path, default=Path("data/benchmark/coco2017/images")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/benchmark/coco2017/image_info_test2017.zip"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=DEFAULT_SIZES,
        metavar=("DEV", "VAL", "TEST"),
    )
    parser.add_argument("--seed", type=int, default=470)
    parser.add_argument("--candidate-count", type=int)
    args = parser.parse_args()
    if args.mode == "coco":
        manifest = prepare_coco(
            args.manifest,
            args.image_dir,
            args.metadata,
            tuple(args.sizes),
            args.seed,
            args.candidate_count,
        )
    else:
        manifest = prepare_local(args.manifest, args.image_dir, args.seed)
    print(
        json.dumps(
            {
                "manifest_id": manifest["manifest_id"],
                "split_sizes": manifest["split_sizes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
