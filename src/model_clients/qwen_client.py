"""Local Qwen2.5-VL client backed by Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


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
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "QwenVisionModelClient requires `transformers` and "
            "`qwen-vl-utils`. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return AutoProcessor, Qwen2_5_VLForConditionalGeneration, process_vision_info


@dataclass
class QwenVisionModelClient:
    """VisionModelClient implementation for local Qwen2.5-VL inference."""

    model_id: str = DEFAULT_QWEN_MODEL_ID
    max_new_tokens: int = 256
    require_gpu: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _process_vision_info: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def query(self, image_path: str, instruction: str) -> str:
        """Return the Qwen2.5-VL response for an image and instruction."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        if not instruction.strip():
            raise ValueError("instruction must be a non-empty string")

        self._ensure_loaded()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(path)},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
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
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._input_device())

        generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
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

        (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            self._process_vision_info,
        ) = _import_qwen_libraries()
        try:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            if hasattr(self._model, "eval"):
                self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Qwen2.5-VL model '{self.model_id}'. Ensure the "
                "model id is valid, Hugging Face model files are available, and "
                "the machine has enough GPU memory."
            ) from exc

    def _input_device(self) -> str:
        if self._torch is not None and self._torch.cuda.is_available():
            return "cuda"
        return "cpu"
