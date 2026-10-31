"""Content-addressed experiment manifests; no model downloads or hidden services."""

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess

from src.config import ExperimentConfig
from src.render_prompt import font_provenance
from src.segment_regions import resolve_segmentation


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_manifest(
    config: ExperimentConfig, images: list[Path], client: object
) -> dict:
    root = Path(__file__).resolve().parent.parent
    sources = {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted((root / "src").rglob("*.py"))
    }
    dependencies = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "qwen-vl-utils",
        "Pillow",
        "numpy",
        "scikit-image",
        "segment-anything",
        "PyYAML",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    settings = asdict(config)
    settings.pop("output")  # Moving output files must not change scientific identity.
    model_provenance = client.provenance() if hasattr(client, "provenance") else {}
    mask_strategy = (
        config.rendering.color_strategy != "static"
        and config.evaluation.condition != "clean"
    )
    backend, checkpoint = (
        resolve_segmentation(
            config.rendering.mask_backend, config.rendering.sam_checkpoint
        )
        if mask_strategy
        else ("none", None)
    )
    manifest = {
        "schema_version": 2,
        "settings": settings,
        "images": [
            {
                "path": path.relative_to(config.dataset.image_dir).as_posix(),
                "sha256": file_hash(path),
            }
            for path in images
        ],
        "model": {
            "client_class": f"{type(client).__module__}.{type(client).__qualname__}",
            **model_provenance,
        },
        "segmentation": {
            "backend": backend,
            "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
            "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
        },
        "font": font_provenance(),
        "code_revision": commit,
        "source_hashes": sources,
        "dependencies": dependencies,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    # Normalize Paths and tuples before computing identity or returning to callers.
    return json.loads(json.dumps(manifest, default=str))


def write_manifest(results_path: Path, manifest: dict) -> tuple[str, Path]:
    experiment_id = fingerprint(manifest)
    path = results_path.parent / "manifests" / f"{experiment_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_id": experiment_id, **manifest}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"manifest contents do not match identity: {path}")
    else:
        write_json_atomic(path, payload)
    return experiment_id, path
