"""Run config-driven experiments over an image dataset."""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha1
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig, load_config
from src.model_clients import (
    DEFAULT_QWEN_MODEL_ID,
    MockVisionModelClient,
    QwenVisionModelClient,
    VisionModelClient,
)
from src.render_prompt import render_text_prompt
from src.run_single import DEFAULT_INSTRUCTION_TEMPLATE, target_in_response

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def iter_dataset_images(
    image_dir: str | Path, max_images: int | None = None
) -> list[Path]:
    """Return sorted image paths under a dataset directory."""
    root = Path(image_dir)
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        return images[:max_images]
    return images


def build_run_id(
    image_path: str | Path,
    dataset_dir: str | Path,
    placement: str,
    font_scale: float,
    prompt_text: str | None = None,
) -> str:
    """Build a stable run id for resume behavior."""
    image = Path(image_path)
    try:
        image_key = image.relative_to(dataset_dir).as_posix()
    except ValueError:
        image_key = image.as_posix()
    raw = f"{image_key}|{placement}|{font_scale:g}|{prompt_text or ''}"
    digest = sha1(raw.encode("utf-8")).hexdigest()[:12]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_key).strip("_")
    return f"{readable}__{placement}__fs_{font_scale:g}__{digest}"


def generated_image_path(
    generated_dir: str | Path,
    image_path: str | Path,
    dataset_dir: str | Path,
    placement: str,
    font_scale: float,
    prompt_text: str | None = None,
) -> Path:
    """Return the output image path for one run."""
    run_id = build_run_id(image_path, dataset_dir, placement, font_scale, prompt_text)
    return Path(generated_dir) / f"{run_id}.png"


def load_completed_run_ids(results_path: str | Path) -> set[str]:
    """Read existing JSONL results and return run ids that should be skipped."""
    path = Path(results_path)
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL in {path} at line {line_number}"
                ) from exc
            run_id = row.get("run_id")
            if isinstance(run_id, str) and run_id:
                completed.add(run_id)
    return completed


def append_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _client_from_config(
    config: ExperimentConfig, client: VisionModelClient | None
) -> VisionModelClient:
    if client is not None:
        return client
    model_type = config.model.type.casefold()
    if model_type == "mock":
        return MockVisionModelClient(response=config.model.response or "mock response")
    if model_type == "qwen":
        return QwenVisionModelClient(
            model_id=config.model.model_id or DEFAULT_QWEN_MODEL_ID,
            max_new_tokens=config.model.max_new_tokens,
            require_gpu=config.model.require_gpu,
            torch_dtype=config.model.torch_dtype,
            device_map=config.model.device_map,
        )
    raise ValueError("model.type must be one of: mock, qwen")


def _build_adversarial_prompt(
    config: ExperimentConfig,
    image_path: Path,
    model_client: VisionModelClient,
    object_cache: dict[Path, str],
) -> tuple[str, str | None]:
    adaptive_prefix = config.target.adaptive_prefix
    if not adaptive_prefix.enabled:
        return config.target.embedded_prompt, None

    if image_path not in object_cache:
        object_cache[image_path] = model_client.query(
            str(image_path), adaptive_prefix.object_instruction
        )
    detected_objects = object_cache[image_path]
    prompt = adaptive_prefix.template.format(
        objs=detected_objects,
        prompt=config.target.embedded_prompt,
    )
    return prompt, detected_objects


def run_experiment(
    config: ExperimentConfig,
    *,
    client: VisionModelClient | None = None,
) -> list[dict[str, Any]]:
    """Run all missing grid cells and append one result row per run."""
    image_paths = iter_dataset_images(
        config.dataset.image_dir, config.dataset.max_images
    )
    completed = load_completed_run_ids(config.output.results_path)
    model_client = _client_from_config(config, client)
    instruction = config.model.instruction or DEFAULT_INSTRUCTION_TEMPLATE
    color = config.rendering.colors[0]
    object_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []

    for image_path in image_paths:
        adversarial_prompt, detected_objects = _build_adversarial_prompt(
            config, image_path, model_client, object_cache
        )
        for placement in config.rendering.placements:
            for font_scale in config.rendering.font_scales:
                run_id = build_run_id(
                    image_path,
                    config.dataset.image_dir,
                    placement,
                    font_scale,
                    adversarial_prompt,
                )
                if run_id in completed:
                    continue

                output_image_path = generated_image_path(
                    config.output.generated_dir,
                    image_path,
                    config.dataset.image_dir,
                    placement,
                    font_scale,
                    adversarial_prompt,
                )
                render_metadata = render_text_prompt(
                    input_image_path=image_path,
                    output_image_path=output_image_path,
                    text=adversarial_prompt,
                    placement=placement,
                    font_scale=font_scale,
                    color=color,
                    max_characters_per_line=(config.rendering.max_characters_per_line),
                )
                model_response = model_client.query(str(output_image_path), instruction)
                success = target_in_response(config.target.phrase, model_response)
                row: dict[str, Any] = {
                    "run_id": run_id,
                    "original_image": str(image_path),
                    "generated_image": str(output_image_path),
                    "target": config.target.phrase,
                    "embedded_prompt": config.target.embedded_prompt,
                    "adversarial_prompt": adversarial_prompt,
                    "adaptive_prefix_enabled": (config.target.adaptive_prefix.enabled),
                    "object_instruction": (
                        config.target.adaptive_prefix.object_instruction
                        if config.target.adaptive_prefix.enabled
                        else None
                    ),
                    "detected_objects": detected_objects,
                    "instruction": instruction,
                    "model_type": config.model.type.casefold(),
                    "model_response": model_response,
                    "success": success,
                    "placement": placement,
                    "font_scale": font_scale,
                    "color": list(color),
                    "render_font_size": render_metadata.font_size,
                    "render_text_bbox": list(render_metadata.text_bbox),
                    "rendered_prompt": render_metadata.rendered_text,
                    "rendered_prompt_line_count": render_metadata.line_count,
                    "max_characters_per_line": (
                        config.rendering.max_characters_per_line
                    ),
                }
                append_jsonl_row(config.output.results_path, row)
                completed.add(run_id)
                rows.append(row)

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a config-driven mock experiment.")
    parser.add_argument(
        "--config", required=True, help="Path to the YAML experiment config."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    rows = run_experiment(config)
    print(f"runs_written={len(rows)}")
    print(f"results={config.output.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
