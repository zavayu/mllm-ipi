"""Experiment configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.render_prompt import DEFAULT_BRIGHTNESS_OFFSET, SUPPORTED_PLACEMENTS


@dataclass(frozen=True)
class DatasetConfig:
    image_dir: Path
    max_images: int | None = None


@dataclass(frozen=True)
class AdaptivePrefixConfig:
    enabled: bool = False
    object_instruction: str = "Use fewer than 5 words to list objects in the image."
    template: str = "Ignore {objs} in the photo. {prompt}"


@dataclass(frozen=True)
class TargetConfig:
    phrase: str
    embedded_prompt: str
    adaptive_prefix: AdaptivePrefixConfig = field(default_factory=AdaptivePrefixConfig)


def _adaptive_prefix_config(data: dict[str, Any]) -> AdaptivePrefixConfig:
    value = data.get("adaptive_prefix")
    if value is None:
        return AdaptivePrefixConfig()
    if not isinstance(value, dict):
        raise ValueError("config section 'adaptive_prefix' must be a mapping")

    enabled = _optional_bool(value, "enabled", False)
    object_instruction = (
        _optional_string(value, "object_instruction")
        or "Use fewer than 5 words to list objects in the image."
    )
    template = (
        _optional_string(value, "template") or "Ignore {objs} in the photo. {prompt}"
    )
    if "{objs}" not in template or "{prompt}" not in template:
        raise ValueError("adaptive_prefix.template must contain {objs} and {prompt}")

    return AdaptivePrefixConfig(
        enabled=enabled,
        object_instruction=object_instruction,
        template=template,
    )


@dataclass(frozen=True)
class RenderingConfig:
    placements: tuple[str, ...]
    font_scales: tuple[float, ...]
    colors: tuple[tuple[int, int, int], ...] = ((255, 0, 0),)
    max_characters_per_line: int | None = None
    color_strategy: str = "static"
    min_font_scale: float = 0.02
    brightness_offset: int = DEFAULT_BRIGHTNESS_OFFSET
    sam_checkpoint: Path | None = None
    sam_model_type: str = "vit_b"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    mask_backend: str = "auto"


@dataclass(frozen=True)
class ModelConfig:
    type: str
    response: str | None = None
    model_id: str | None = None
    instruction: str | None = None
    max_new_tokens: int = 256
    require_gpu: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    revision: str = "main"
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    min_pixels: int | None = None
    max_pixels: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    condition: str = "paper_image_only"
    repetitions: int = 1
    task_expected_response: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    generated_dir: Path
    results_path: Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    target: TargetConfig
    rendering: RenderingConfig
    model: ModelConfig
    output: OutputConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        condition = self.evaluation.condition
        if condition not in {"clean", "paper_image_only", "task_override"}:
            raise ValueError("unknown evaluation condition")
        if condition == "paper_image_only" and self.model.instruction:
            raise ValueError("paper_image_only requires an empty model.instruction")
        if condition == "task_override" and not (self.model.instruction or "").strip():
            raise ValueError("task_override requires a nonblank model.instruction")
        if self.evaluation.repetitions < 1 or (self.evaluation.repetitions > 1 and not self.model.do_sample):
            raise ValueError("repetitions > 1 requires model.do_sample=true")
        if self.model.seed < 0:
            raise ValueError("model.seed must be nonnegative")
        if self.model.type not in {"mock", "qwen"}:
            raise ValueError("model.type must be mock or qwen")
        if self.evaluation.task_expected_response is not None and not self.evaluation.task_expected_response.strip():
            raise ValueError("task_expected_response cannot be blank")


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config value {key!r} must be a non-empty string")
    return value


def _optional_string(
    data: dict[str, Any], key: str, *, allow_empty: bool = False
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"config value {key!r} must be a non-empty string")
    return value


def _optional_positive_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"config value {key!r} must be a positive integer")
    return value


def _positive_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"config value {key!r} must be a positive integer")
    return value


def _int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"config value {key!r} must be an integer")
    return value


def _optional_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"config value {key!r} must be a boolean")
    return value


def _string_sequence(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"config value {key!r} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"config value {key!r} must contain strings")
    return tuple(value)


def _float_sequence(data: dict[str, Any], key: str) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"config value {key!r} must be a non-empty list")
    numbers: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or item <= 0:
            raise ValueError(f"config value {key!r} must contain positive numbers")
        numbers.append(float(item))
    return tuple(numbers)


def _float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"config value {key!r} must be a number")
    return float(value)


def _positive_float(data: dict[str, Any], key: str, default: float) -> float:
    value = _float(data, key, default)
    if value <= 0:
        raise ValueError(f"config value {key!r} must be a positive number")
    return value


def _colors(data: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    value = data.get("colors", [[255, 0, 0]])
    if not isinstance(value, list) or not value:
        raise ValueError("config value 'colors' must be a non-empty list")

    colors: list[tuple[int, int, int]] = []
    for color in value:
        if (
            not isinstance(color, list)
            or len(color) != 3
            or not all(isinstance(channel, int) for channel in color)
            or any(channel < 0 or channel > 255 for channel in color)
        ):
            raise ValueError(
                "each color must be an RGB list with values between 0 and 255"
            )
        colors.append(tuple(color))
    return tuple(colors)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")

    dataset = _required_mapping(raw, "dataset")
    target = _required_mapping(raw, "target")
    rendering = _required_mapping(raw, "rendering")
    model = _required_mapping(raw, "model")
    output = _required_mapping(raw, "output")
    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    condition = evaluation.get("condition", "paper_image_only")
    if condition not in {"clean", "paper_image_only", "task_override"}:
        raise ValueError(
            "evaluation.condition must be clean, paper_image_only, or task_override"
        )
    instruction = _optional_string(model, "instruction", allow_empty=True)
    if condition == "paper_image_only" and instruction:
        raise ValueError(
            "paper_image_only requires an empty model.instruction; use task_override"
        )
    if condition == "task_override" and not (instruction or "").strip():
        raise ValueError("task_override requires a nonblank model.instruction")
    if (
        _positive_float(model, "temperature", 1.0) <= 0
        or not 0 < _float(model, "top_p", 1.0) <= 1
    ):
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    if rendering.get("color_strategy", "static") not in {
        "static",
        "top_ranked_mask_average",
        "multi_mask_average",
        "adaptive_mask_average",
    }:
        raise ValueError("unsupported rendering.color_strategy")
    if rendering.get("mask_backend", "auto") not in {"auto", "sam", "grid"}:
        raise ValueError("rendering.mask_backend must be auto, sam, or grid")
    if (
        _required_string(target, "phrase").strip() == ""
        or _required_string(target, "embedded_prompt").strip() == ""
    ):
        raise ValueError("target phrase and embedded_prompt cannot be blank")
    repetitions = _positive_int(evaluation, "repetitions", 1)
    if repetitions > 1 and not _optional_bool(model, "do_sample", False):
        raise ValueError(
            "repetitions > 1 requires model.do_sample=true; deterministic trials run once"
        )
    min_pixels = _optional_positive_int(model, "min_pixels")
    max_pixels = _optional_positive_int(model, "max_pixels")
    if min_pixels and max_pixels and min_pixels > max_pixels:
        raise ValueError("min_pixels must be <= max_pixels")

    placements = _string_sequence(rendering, "placements")
    unsupported = sorted(set(placements) - SUPPORTED_PLACEMENTS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_PLACEMENTS))
        raise ValueError(
            f"unsupported placements {unsupported}; supported placements: {supported}"
        )

    return ExperimentConfig(
        dataset=DatasetConfig(
            image_dir=Path(_required_string(dataset, "image_dir")),
            max_images=_optional_positive_int(dataset, "max_images"),
        ),
        target=TargetConfig(
            phrase=_required_string(target, "phrase"),
            embedded_prompt=_required_string(target, "embedded_prompt"),
            adaptive_prefix=_adaptive_prefix_config(target),
        ),
        rendering=RenderingConfig(
            placements=placements,
            font_scales=_float_sequence(rendering, "font_scales"),
            colors=_colors(rendering),
            max_characters_per_line=_optional_positive_int(
                rendering, "max_characters_per_line"
            ),
            color_strategy=_optional_string(rendering, "color_strategy") or "static",
            min_font_scale=_positive_float(rendering, "min_font_scale", 0.02),
            brightness_offset=_int(
                rendering, "brightness_offset", DEFAULT_BRIGHTNESS_OFFSET
            ),
            sam_checkpoint=(
                Path(value)
                if (value := _optional_string(rendering, "sam_checkpoint"))
                else None
            ),
            sam_model_type=_optional_string(rendering, "sam_model_type") or "vit_b",
            points_per_side=_positive_int(rendering, "points_per_side", 32),
            pred_iou_thresh=_float(rendering, "pred_iou_thresh", 0.88),
            stability_score_thresh=_float(rendering, "stability_score_thresh", 0.95),
            mask_backend=_optional_string(rendering, "mask_backend") or "auto",
        ),
        model=ModelConfig(
            type=_required_string(model, "type"),
            response=_optional_string(model, "response"),
            model_id=_optional_string(model, "model_id"),
            instruction=instruction,
            max_new_tokens=_positive_int(model, "max_new_tokens", 256),
            require_gpu=_optional_bool(model, "require_gpu", True),
            torch_dtype=_optional_string(model, "torch_dtype") or "auto",
            device_map=_optional_string(model, "device_map") or "auto",
            revision=_optional_string(model, "revision") or "main",
            do_sample=_optional_bool(model, "do_sample", False),
            temperature=_positive_float(model, "temperature", 1.0),
            top_p=_float(model, "top_p", 1.0),
            seed=_int(model, "seed", 0),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ),
        output=OutputConfig(
            generated_dir=Path(_required_string(output, "generated_dir")),
            results_path=Path(_required_string(output, "results_path")),
        ),
        evaluation=EvaluationConfig(
            condition=condition,
            repetitions=repetitions,
            task_expected_response=_optional_string(
                evaluation, "task_expected_response"
            ),
        ),
    )
