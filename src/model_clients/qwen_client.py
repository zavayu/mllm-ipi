"""Local Qwen VL clients backed by Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_QWEN_MODEL_FAMILY = "auto"
QWEN25_VL_FAMILY = "qwen2.5-vl"
QWEN3_VL_FAMILY = "qwen3-vl"
SUPPORTED_QWEN_MODEL_FAMILIES = (
    DEFAULT_QWEN_MODEL_FAMILY,
    QWEN25_VL_FAMILY,
    QWEN3_VL_FAMILY,
)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "QwenVisionModelClient requires PyTorch. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc
    return torch


def _import_qwen_libraries() -> tuple[Any, Any, Any]:
    """Import Qwen2.5-VL dependencies lazily."""
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "QwenVisionModelClient with model_family='qwen2.5-vl' requires "
            "`transformers` and `qwen-vl-utils`. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc
    return AutoProcessor, Qwen2_5_VLForConditionalGeneration, process_vision_info


def _import_qwen3_libraries() -> tuple[Any, Any]:
    """Import Qwen3-VL dependencies lazily."""
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "QwenVisionModelClient with model_family='qwen3-vl' requires a "
            "Transformers version that includes Qwen3VLForConditionalGeneration. "
            "Upgrade `transformers` in the active environment."
        ) from exc
    return AutoProcessor, Qwen3VLForConditionalGeneration


def infer_qwen_model_family(model_id: str) -> str:
    """Infer the Qwen VL family from a common Hugging Face model id."""
    normalized = model_id.casefold()
    if "qwen3-vl" in normalized or "qwen3vl" in normalized:
        return QWEN3_VL_FAMILY
    if "qwen2.5-vl" in normalized or "qwen2_5_vl" in normalized:
        return QWEN25_VL_FAMILY
    return QWEN25_VL_FAMILY


def resolve_qwen_model_family(model_family: str, model_id: str) -> str:
    """Resolve `auto` and validate an explicit Qwen model family."""
    normalized = model_family.casefold()
    if normalized == DEFAULT_QWEN_MODEL_FAMILY:
        return infer_qwen_model_family(model_id)
    if normalized not in SUPPORTED_QWEN_MODEL_FAMILIES:
        supported = ", ".join(SUPPORTED_QWEN_MODEL_FAMILIES)
        raise ValueError(
            f"unsupported Qwen model family '{model_family}'. Supported values: "
            f"{supported}."
        )
    return normalized


@dataclass
class QwenVisionModelClient:
    """VisionModelClient implementation for local Qwen VL inference."""

    model_id: str = DEFAULT_QWEN_MODEL_ID
    model_family: str = DEFAULT_QWEN_MODEL_FAMILY
    max_new_tokens: int = 256
    require_gpu: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _process_vision_info: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _resolved_model_family: str | None = field(default=None, init=False, repr=False)

    def query(self, image_path: str, instruction: str) -> str:
        """Return the Qwen VL response for an image and instruction."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")

        self._ensure_loaded()
        messages = self._build_messages(path, instruction)
        if self._resolved_model_family == QWEN3_VL_FAMILY:
            return self._query_qwen3(messages)
        return self._query_qwen25(messages)

    def _query_qwen25(self, messages: list[dict[str, Any]]) -> str:
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return self._generate_and_decode(inputs)

    def _query_qwen3(self, messages: list[dict[str, Any]]) -> str:
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return self._generate_and_decode(inputs)

    def _generate_and_decode(self, inputs: Any) -> str:
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._input_device())

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return responses[0].strip()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        self._resolved_model_family = resolve_qwen_model_family(
            self.model_family, self.model_id
        )
        self._torch = _import_torch()
        if self.require_gpu and not self._torch.cuda.is_available():
            raise RuntimeError(
                "QwenVisionModelClient requires a CUDA/ROCm PyTorch GPU backend "
                "for local inference, but PyTorch reports that no GPU backend is "
                "available. Install a CUDA-enabled PyTorch build for NVIDIA GPUs "
                "or a ROCm-enabled PyTorch build for AMD GPUs, then verify that "
                "`torch.cuda.is_available()` returns True. For CPU testing, "
                "construct the client with require_gpu=False."
            )

        self._processor, model_class = self._load_hugging_face_classes()
        try:
            self._model = model_class.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            if hasattr(self._model, "eval"):
                self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"failed to load {self._resolved_model_family} model "
                f"'{self.model_id}'. Ensure the model id is valid, Hugging Face "
                "model files are available, and the machine has enough GPU memory."
            ) from exc

    def _load_hugging_face_classes(self) -> tuple[Any, Any]:
        if self._resolved_model_family == QWEN3_VL_FAMILY:
            AutoProcessor, model_class = _import_qwen3_libraries()
            return AutoProcessor.from_pretrained(self.model_id), model_class

        AutoProcessor, model_class, self._process_vision_info = _import_qwen_libraries()
        return AutoProcessor.from_pretrained(self.model_id), model_class

    def _input_device(self) -> str:
        if self._torch is not None and self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _build_messages(image_path: Path, instruction: str) -> list[dict[str, Any]]:
        content = [{"type": "image", "image": str(image_path)}]
        if instruction:
            content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]
