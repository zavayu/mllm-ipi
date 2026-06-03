"""Experiment configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.render_prompt import SUPPORTED_PLACEMENTS


@dataclass(frozen=True)
class DatasetConfig:
    image_dir: Path
    max_images: int | None = None


@dataclass(frozen=True)
class TargetConfig:
    phrase: str
    embedded_prompt: str


@dataclass(frozen=True)
class RenderingConfig:
    placements: tuple[str, ...]
    font_scales: tuple[float, ...]
    colors: tuple[tuple[int, int, int], ...] = ((255, 0, 0),)


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


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
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
        ),
        rendering=RenderingConfig(
            placements=placements,
            font_scales=_float_sequence(rendering, "font_scales"),
            colors=_colors(rendering),
        ),
        model=ModelConfig(
            type=_required_string(model, "type"),
            response=_optional_string(model, "response"),
            model_id=_optional_string(model, "model_id"),
            instruction=_optional_string(model, "instruction"),
            max_new_tokens=_positive_int(model, "max_new_tokens", 256),
            require_gpu=_optional_bool(model, "require_gpu", True),
            torch_dtype=_optional_string(model, "torch_dtype") or "auto",
            device_map=_optional_string(model, "device_map") or "auto",
        ),
        output=OutputConfig(
            generated_dir=Path(_required_string(output, "generated_dir")),
            results_path=Path(_required_string(output, "results_path")),
        ),
    )
