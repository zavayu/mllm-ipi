"""Run config-driven experiments over an image dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import ExperimentConfig, load_config
from src.model_clients import (
    DEFAULT_QWEN_MODEL_ID,
    MockVisionModelClient,
    QwenVisionModelClient,
    VisionModelClient,
)
from src.region_ranker import RegionFeatures, compute_and_rank_regions
from src.render_prompt import (
    render_text_prompt,
    render_text_prompt_adaptive_mask_average,
    render_text_prompt_across_ranked_masks,
    render_text_prompt_in_top_ranked_mask,
)
from src.run_single import MASK_COLOR_STRATEGY
from src.segment_regions import generate_candidate_masks
from src.response_metrics import response_metrics
from src.provenance import (
    build_manifest,
    write_manifest,
    fingerprint,
    file_hash,
    write_json_atomic,
)
from src.metrics import compute_image_metrics
from src.benchmark_data import load_manifest

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
STATIC_COLOR_STRATEGY = "static"
MULTI_MASK_COLOR_STRATEGY = "multi_mask_average"
ADAPTIVE_MASK_COLOR_STRATEGY = "adaptive_mask_average"
MASK_COLOR_STRATEGIES = {
    MASK_COLOR_STRATEGY,
    MULTI_MASK_COLOR_STRATEGY,
    ADAPTIVE_MASK_COLOR_STRATEGY,
}
TOP_RANKED_MASK_PLACEMENT = "top_ranked_mask"
MULTI_MASK_PLACEMENT = "multi_mask"
OBJECT_WORD_LIMIT = 5


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


def load_completed_run_ids(results_path: str | Path) -> set[str]:
    """Read existing JSONL results and return run ids that should be skipped."""
    path = Path(results_path)
    if not path.exists():
        return set()

    completed: set[str] = set()
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                backup = path.with_name(
                    path.name + ".incomplete-" + fingerprint(line.hex())[:12]
                )
                backup.write_bytes(line)
                path.write_bytes(b"".join(lines[:index]))
                break
            raise ValueError(f"invalid JSONL in {path} at line {index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"result row at line {index + 1} must be an object")
        run_id = row.get("run_id")
        if (
            isinstance(run_id, str)
            and run_id
            and row.get("status", "completed") == "completed"
        ):
            completed.add(run_id)
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return completed


def append_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")


def _truncate_words(text: str, word_limit: int = OBJECT_WORD_LIMIT) -> str:
    return " ".join(text.split()[:word_limit])


def _ranked_regions_for_image(
    image_path: str | Path,
    config: ExperimentConfig,
) -> list[RegionFeatures]:
    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("RGB"))
    masks = generate_candidate_masks(
        image_array,
        sam_checkpoint=config.rendering.sam_checkpoint,
        sam_model_type=config.rendering.sam_model_type,
        points_per_side=config.rendering.points_per_side,
        pred_iou_thresh=config.rendering.pred_iou_thresh,
        stability_score_thresh=config.rendering.stability_score_thresh,
        backend=config.rendering.mask_backend,
    )
    return compute_and_rank_regions(image_array, masks)


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
            revision=config.model.revision,
            do_sample=config.model.do_sample,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            seed=config.model.seed,
            min_pixels=config.model.min_pixels,
            max_pixels=config.model.max_pixels,
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
        object_cache[image_path] = _truncate_words(
            model_client.query(str(image_path), adaptive_prefix.object_instruction)
        )
    detected_objects = object_cache[image_path]
    prompt = adaptive_prefix.template.format(
        objs=detected_objects,
        prompt=config.target.embedded_prompt,
    )
    return prompt, detected_objects


def _effective_placements(config: ExperimentConfig) -> tuple[str, ...]:
    if config.rendering.color_strategy == MASK_COLOR_STRATEGY:
        return (TOP_RANKED_MASK_PLACEMENT,)
    if config.rendering.color_strategy == MULTI_MASK_COLOR_STRATEGY:
        return (MULTI_MASK_PLACEMENT,)
    if config.rendering.color_strategy == ADAPTIVE_MASK_COLOR_STRATEGY:
        return (TOP_RANKED_MASK_PLACEMENT,)
    return config.rendering.placements


def _render_configuration(
    config, image_path, output_path, prompt, placement, font_scale, color, regions
):
    common = dict(
        input_image_path=image_path,
        output_image_path=output_path,
        text=prompt,
        font_scale=font_scale,
    )
    strategy = config.rendering.color_strategy
    if strategy in MASK_COLOR_STRATEGIES:
        common.update(
            ranked_regions=regions, brightness_offset=config.rendering.brightness_offset
        )
    if strategy == ADAPTIVE_MASK_COLOR_STRATEGY:
        return render_text_prompt_adaptive_mask_average(
            **common, min_font_scale=config.rendering.min_font_scale
        )
    if strategy == MULTI_MASK_COLOR_STRATEGY:
        return render_text_prompt_across_ranked_masks(
            **common,
            write_auxiliary_outputs=False,
            min_font_scale=config.rendering.min_font_scale,
        )
    if strategy == MASK_COLOR_STRATEGY:
        return render_text_prompt_in_top_ranked_mask(
            **common,
            max_characters_per_line=config.rendering.max_characters_per_line,
            min_font_scale=config.rendering.min_font_scale,
        )
    return render_text_prompt(
        **common,
        placement=placement,
        color=color,
        max_characters_per_line=config.rendering.max_characters_per_line,
    )


def run_experiment(
    config: ExperimentConfig, *, client: VisionModelClient | None = None
) -> list[dict[str, Any]]:
    """Append completed trials and retryable errors with content-addressed identity."""
    image_paths = iter_dataset_images(
        config.dataset.image_dir, config.dataset.max_images
    )
    if config.dataset.manifest is not None:
        dataset_manifest, image_paths = load_manifest(
            config.dataset.manifest, config.dataset.split
        )
        expected_root = (
            config.dataset.manifest.parent / dataset_manifest["image_root"]
        ).resolve()
        if expected_root != config.dataset.image_dir.resolve():
            raise ValueError(
                "dataset.image_dir does not match the frozen manifest root"
            )
        if config.dataset.max_images is not None:
            image_paths = image_paths[: config.dataset.max_images]
    if not image_paths:
        raise ValueError(f"no dataset images found in {config.dataset.image_dir}")
    model_client = _client_from_config(config, client)
    manifest = build_manifest(config, image_paths, model_client)
    experiment_id, manifest_path = write_manifest(config.output.results_path, manifest)
    completed = load_completed_run_ids(config.output.results_path)
    instruction = config.model.instruction or ""
    condition = config.evaluation.condition
    rows = []
    object_path = manifest_path.with_suffix(".objects.json")
    saved_objects = (
        json.loads(object_path.read_text(encoding="utf-8"))
        if object_path.exists()
        else {}
    )
    object_cache = {Path(path): text for path, text in saved_objects.items()}
    signature = fingerprint(
        {
            "rendering": asdict(config.rendering),
            "font": manifest["font"],
            "sources": manifest["source_hashes"],
            "segmentation": manifest["segmentation"],
        }
    )

    def append(row):
        append_jsonl_row(config.output.results_path, row)
        rows.append(row)
        if row["status"] == "completed":
            completed.add(row["run_id"])

    for image_path in image_paths:
        image_hash = file_hash(image_path)
        image_key = (
            image_path.resolve()
            .relative_to(config.dataset.image_dir.resolve())
            .as_posix()
        )
        base = {
            "schema_version": 2,
            "experiment_id": experiment_id,
            "manifest_path": str(manifest_path),
            "condition": condition,
            "original_image": str(image_path),
            "image_sha256": image_hash,
            "expected_response": config.target.phrase,
            "target": config.target.phrase,
            "embedded_prompt": config.target.embedded_prompt,
            "instruction": instruction,
            "model_type": config.model.type,
            "model_id": config.model.model_id
            or (DEFAULT_QWEN_MODEL_ID if config.model.type == "qwen" else "mock"),
            "resolved_revision": manifest["model"].get("resolved_revision"),
            "client_class": manifest["model"]["client_class"],
            "empirical_model_run": isinstance(model_client, QwenVisionModelClient),
            "segmentation_backend": manifest["segmentation"]["backend"],
            "sam_checkpoint": manifest["segmentation"]["checkpoint"],
            "sam_model_type": config.rendering.sam_model_type,
            "points_per_side": config.rendering.points_per_side,
            "pred_iou_thresh": config.rendering.pred_iou_thresh,
            "stability_score_thresh": config.rendering.stability_score_thresh,
        }
        prompt, detected_objects = config.target.embedded_prompt, None
        try:
            if condition != "clean":
                if isinstance(model_client, QwenVisionModelClient):
                    model_client.seed = config.model.seed
                prompt, detected_objects = _build_adversarial_prompt(
                    config, image_path, model_client, object_cache
                )
                if config.target.adaptive_prefix.enabled:
                    write_json_atomic(
                        object_path, {str(k): v for k, v in object_cache.items()}
                    )
        except Exception as exc:
            append(
                {
                    **base,
                    "run_id": fingerprint([experiment_id, image_key, "prefix"]),
                    "status": "error",
                    "stage": "adaptive_prefix",
                    "error": str(exc),
                }
            )
            continue
        base.update(
            adversarial_prompt=prompt,
            detected_objects=detected_objects,
            adaptive_prefix_enabled=config.target.adaptive_prefix.enabled
            and condition != "clean",
            object_instruction=(
                config.target.adaptive_prefix.object_instruction
                if config.target.adaptive_prefix.enabled and condition != "clean"
                else None
            ),
        )
        regions = None
        placements = (
            ("original",) if condition == "clean" else _effective_placements(config)
        )
        scales = (None,) if condition == "clean" else config.rendering.font_scales
        colors = (
            (None,)
            if condition == "clean"
            else (
                config.rendering.colors
                if config.rendering.color_strategy == STATIC_COLOR_STRATEGY
                else (config.rendering.colors[0],)
            )
        )
        for placement in placements:
            for font_scale in scales:
                for color in colors:
                    render_id = fingerprint(
                        [
                            image_hash,
                            image_key,
                            signature,
                            placement,
                            font_scale,
                            color,
                            prompt,
                        ]
                    )
                    trials = [
                        (i, fingerprint([experiment_id, render_id, i]))
                        for i in range(config.evaluation.repetitions)
                    ]
                    trials = [
                        (i, run_id) for i, run_id in trials if run_id not in completed
                    ]
                    if not trials:
                        continue
                    output_path = (
                        image_path
                        if condition == "clean"
                        else config.output.generated_dir / f"{render_id}.png"
                    )
                    common = {
                        **base,
                        "render_id": render_id,
                        "generated_image": str(output_path),
                        "font_scale": font_scale,
                        "placement": placement,
                        "color_strategy": (
                            "none"
                            if condition == "clean"
                            else config.rendering.color_strategy
                        ),
                        "color": list(color) if color else None,
                        "min_font_scale": config.rendering.min_font_scale,
                        "max_characters_per_line": config.rendering.max_characters_per_line,
                    }
                    start = perf_counter()
                    try:
                        metadata = None
                        if condition != "clean":
                            if (
                                config.rendering.color_strategy in MASK_COLOR_STRATEGIES
                                and regions is None
                            ):
                                regions = _ranked_regions_for_image(image_path, config)
                            metadata = _render_configuration(
                                config,
                                image_path,
                                output_path,
                                prompt,
                                placement,
                                font_scale,
                                color,
                                regions,
                            )
                        distortion = compute_image_metrics(image_path, output_path)
                        render_seconds = perf_counter() - start
                    except Exception as exc:
                        for i, run_id in trials:
                            append(
                                {
                                    **common,
                                    "run_id": run_id,
                                    "run_index": i,
                                    "status": "error",
                                    "stage": "render",
                                    "error": str(exc),
                                }
                            )
                        continue
                    render_data = {}
                    if metadata is not None:
                        render_data = asdict(metadata)
                        render_data.pop("output_path")
                        render_data["render_font_size"] = render_data.pop("font_size")
                        render_data["render_text_bbox"] = render_data.pop("text_bbox")
                        render_data["rendered_prompt"] = render_data.pop(
                            "rendered_text"
                        )
                        render_data["rendered_prompt_line_count"] = render_data.pop(
                            "line_count"
                        )
                        render_data["chunk_count"] = len(metadata.chunks)
                        render_data = json.loads(json.dumps(render_data))
                    for i, run_id in trials:
                        seed = config.model.seed + i
                        if isinstance(model_client, QwenVisionModelClient):
                            model_client.seed = seed
                        row = {
                            **common,
                            **render_data,
                            **distortion,
                            "run_id": run_id,
                            "run_index": i,
                            "seed": seed,
                            "render_seconds": render_seconds,
                        }
                        start = perf_counter()
                        try:
                            response = model_client.query(str(output_path), instruction)
                            if isinstance(model_client, QwenVisionModelClient):
                                row["model_input"] = model_client.last_input_metadata
                                row["generation"] = (
                                    model_client.last_generation_metadata
                                )
                            outcomes = response_metrics(config.target.phrase, response)
                            row.update(
                                status="completed",
                                model_response=response,
                                **outcomes,
                                success=outcomes["target_containment"],
                                strict_success=config.target.phrase in response,
                                task_success=(
                                    response_metrics(
                                        config.evaluation.task_expected_response,
                                        response,
                                    )["exact_target_response"]
                                    if config.evaluation.task_expected_response
                                    is not None
                                    else None
                                ),
                            )
                        except Exception as exc:
                            row.update(status="error", stage="query", error=str(exc))
                        row["query_seconds"] = perf_counter() - start
                        append(row)
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
    errors = sum(row["status"] == "error" for row in rows)
    print(f"completed={len(rows) - errors} errors={errors}")
    if rows:
        print(f"experiment_id={rows[0]['experiment_id']}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
