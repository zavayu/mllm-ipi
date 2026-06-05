"""Streamlit demo for the image prompt-injection pipeline."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.metrics import compute_image_metrics
from src.model_clients import (
    DEFAULT_QWEN_MODEL_FAMILY,
    DEFAULT_QWEN_MODEL_ID,
    MockVisionModelClient,
    QwenVisionModelClient,
    SUPPORTED_QWEN_MODEL_FAMILIES,
)
from src.region_ranker import RegionFeatures, compute_and_rank_regions
from src.render_prompt import (
    DEFAULT_BRIGHTNESS_OFFSET,
    SUPPORTED_PLACEMENTS,
    RenderMetadata,
    render_text_prompt,
    render_text_prompt_across_ranked_masks,
    render_text_prompt_adaptive_mask_average,
    render_text_prompt_in_top_ranked_mask,
)
from src.run_single import (
    DEFAULT_COLOR,
    DEFAULT_FONT_SCALE,
    DEFAULT_INSTRUCTION_TEMPLATE,
    DEFAULT_PLACEMENT,
    target_in_response,
)
from src.segment_regions import generate_candidate_masks, save_ranked_mask_visualization


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw"
DEMO_OUTPUT_DIR = PROJECT_ROOT / "results" / "streamlit_demo"

RENDERING_STRATEGIES = {
    "Fixed red text": "fixed_red",
    "Top-ranked mask average": "top_ranked_mask",
    "Multi-mask average": "multi_mask",
    "Adaptive mask average": "adaptive_mask",
}


def _available_images() -> list[Path]:
    if not RAW_IMAGE_DIR.exists():
        return []
    return sorted(
        path
        for path in RAW_IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _save_uploaded_image(uploaded_file: object, output_dir: Path) -> Path:
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12]
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    output_path = output_dir / f"uploaded_{digest}{suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(uploaded_file) as image:
        image.convert("RGB").save(output_path)
    return output_path


def _run_id(*values: object) -> str:
    joined = "|".join(str(value) for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _load_rgb_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _rank_masks(
    image_path: Path,
    output_dir: Path,
    *,
    use_sam: bool,
) -> tuple[list[RegionFeatures], Path | None]:
    image = _load_rgb_array(image_path)
    sam_checkpoint = os.getenv("SAM_CHECKPOINT") if use_sam else None
    masks = generate_candidate_masks(image, sam_checkpoint=sam_checkpoint)
    ranked_regions = compute_and_rank_regions(image, masks)
    if not ranked_regions:
        return ranked_regions, None

    visualization_path = output_dir / "ranked_masks.png"
    save_ranked_mask_visualization(image, masks, ranked_regions, visualization_path)
    return ranked_regions, visualization_path


def _render_image(
    *,
    image_path: Path,
    output_path: Path,
    target: str,
    strategy: str,
    placement: str,
    font_scale: float,
    brightness_offset: int,
    ranked_regions: list[RegionFeatures],
) -> RenderMetadata:
    if strategy == "top_ranked_mask":
        return render_text_prompt_in_top_ranked_mask(
            input_image_path=image_path,
            output_image_path=output_path,
            text=target,
            ranked_regions=ranked_regions,
            font_scale=font_scale,
            brightness_offset=brightness_offset,
        )
    if strategy == "multi_mask":
        return render_text_prompt_across_ranked_masks(
            input_image_path=image_path,
            output_image_path=output_path,
            text=target,
            ranked_regions=ranked_regions,
            font_scale=font_scale,
            brightness_offset=brightness_offset,
        )
    if strategy == "adaptive_mask":
        return render_text_prompt_adaptive_mask_average(
            input_image_path=image_path,
            output_image_path=output_path,
            text=target,
            ranked_regions=ranked_regions,
            font_scale=font_scale,
            brightness_offset=brightness_offset,
            write_auxiliary_outputs=True,
        )
    return render_text_prompt(
        input_image_path=image_path,
        output_image_path=output_path,
        text=target,
        placement=placement,
        font_scale=font_scale,
        color=DEFAULT_COLOR,
    )


@st.cache_resource(show_spinner=False)
def _qwen_client(
    model_id: str,
    model_family: str,
    max_new_tokens: int,
    allow_cpu: bool,
) -> QwenVisionModelClient:
    return QwenVisionModelClient(
        model_id=model_id,
        model_family=model_family,
        max_new_tokens=max_new_tokens,
        require_gpu=not allow_cpu,
    )


def _query_model(
    *,
    backend: str,
    image_path: Path,
    instruction: str,
    target: str,
    model_id: str,
    model_family: str,
    max_new_tokens: int,
    allow_cpu: bool,
) -> str:
    if backend == "Mock":
        return MockVisionModelClient(response=f"Detected text: {target}").query(
            str(image_path), instruction
        )
    client = _qwen_client(model_id, model_family, max_new_tokens, allow_cpu)
    return client.query(str(image_path), instruction)


def _display_metadata(metadata: RenderMetadata) -> None:
    rows = {
        "placement": metadata.placement,
        "font_size": metadata.font_size,
        "text_bbox": metadata.text_bbox,
        "selected_mask_id": metadata.selected_mask_id,
        "average_rgb": metadata.average_rgb,
        "final_rgb": metadata.final_rgb,
        "brightness_offset": metadata.brightness_offset,
        "adaptive_fallback_used": metadata.adaptive_fallback_used,
        "chunk_count": len(metadata.chunks),
    }
    st.json({key: value for key, value in rows.items() if value is not None})


def main() -> None:
    st.set_page_config(page_title="IPI Qwen Pipeline Demo", layout="wide")
    st.title("IPI Qwen Pipeline Demo")

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_images = _available_images()

    with st.sidebar:
        st.header("Image")
        uploaded_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )
        selected_sample = None
        if sample_images:
            selected_sample = st.selectbox(
                "Or select sample",
                sample_images,
                format_func=lambda path: path.name,
            )

        st.header("Render")
        target = st.text_input("Target phrase", value="BANANA")
        strategy_label = st.selectbox(
            "Rendering strategy", list(RENDERING_STRATEGIES.keys())
        )
        font_scale = st.slider("Font scale", 0.01, 0.40, DEFAULT_FONT_SCALE, 0.01)
        placement_options = sorted(SUPPORTED_PLACEMENTS)
        placement = st.selectbox(
            "Placement",
            placement_options,
            index=placement_options.index(DEFAULT_PLACEMENT),
        )
        brightness_offset = st.slider(
            "Brightness offset", -128, 128, DEFAULT_BRIGHTNESS_OFFSET, 1
        )
        use_sam = st.checkbox("Use SAM", value=False)

        st.header("Model")
        backend = st.selectbox("Model backend", ["Mock", "Qwen"])
        instruction = st.text_area(
            "Instruction", value=DEFAULT_INSTRUCTION_TEMPLATE, height=90
        )
        model_id = st.text_input("Qwen model id", value=DEFAULT_QWEN_MODEL_ID)
        model_family = st.selectbox(
            "Qwen model family",
            SUPPORTED_QWEN_MODEL_FAMILIES,
            index=SUPPORTED_QWEN_MODEL_FAMILIES.index(DEFAULT_QWEN_MODEL_FAMILY),
        )
        max_new_tokens = st.number_input("Max new tokens", 1, 1024, 256, 1)
        allow_cpu = st.checkbox("Allow CPU Qwen inference", value=False)

        run_clicked = st.button("Run pipeline", type="primary")

    if uploaded_file is not None:
        image_path = _save_uploaded_image(uploaded_file, DEMO_OUTPUT_DIR / "uploads")
    elif selected_sample is not None:
        image_path = selected_sample
    else:
        st.info("Upload an image or add one to data/raw to begin.")
        return

    st.subheader("Original Image")
    st.image(str(image_path), use_container_width=True)

    if not run_clicked:
        return

    if not target.strip():
        st.error("Target phrase cannot be empty.")
        return

    strategy = RENDERING_STRATEGIES[strategy_label]
    run_hash = _run_id(
        image_path,
        target,
        strategy,
        placement,
        font_scale,
        brightness_offset,
        use_sam,
        backend,
    )
    run_dir = DEMO_OUTPUT_DIR / run_hash
    generated_path = run_dir / f"{image_path.stem}_modified.png"

    needs_masks = strategy != "fixed_red" or use_sam
    ranked_regions: list[RegionFeatures] = []
    mask_visualization_path: Path | None = None

    try:
        with st.spinner("Generating masks and rendering image..."):
            if needs_masks:
                ranked_regions, mask_visualization_path = _rank_masks(
                    image_path, run_dir, use_sam=use_sam
                )
                if strategy != "fixed_red" and not ranked_regions:
                    st.error("No candidate masks were generated for the selected strategy.")
                    return
                if use_sam and not os.getenv("SAM_CHECKPOINT"):
                    st.warning(
                        "SAM_CHECKPOINT is not set, so the deterministic fallback masks were used."
                    )

            metadata = _render_image(
                image_path=image_path,
                output_path=generated_path,
                target=target.strip(),
                strategy=strategy,
                placement=placement,
                font_scale=font_scale,
                brightness_offset=brightness_offset,
                ranked_regions=ranked_regions,
            )

        with st.spinner("Querying model and computing metrics..."):
            model_response = _query_model(
                backend=backend,
                image_path=generated_path,
                instruction=instruction,
                target=target.strip(),
                model_id=model_id,
                model_family=model_family,
                max_new_tokens=int(max_new_tokens),
                allow_cpu=allow_cpu,
            )
            success = target_in_response(target.strip(), model_response)
            metrics = compute_image_metrics(image_path, generated_path)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return

    st.subheader("Modified Image")
    st.image(str(generated_path), use_container_width=True)

    if use_sam and mask_visualization_path is not None:
        st.subheader("Selected SAM Masks")
        st.image(str(mask_visualization_path), use_container_width=True)

    if metadata.visualization_path:
        st.subheader("Rendered Mask Chunks")
        st.image(metadata.visualization_path, use_container_width=True)

    st.subheader("Qwen Response" if backend == "Qwen" else "Model Response")
    st.write(model_response)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Success", "true" if success else "false")
    metric_columns[1].metric("MSE", f"{metrics['mse']:.4f}")
    metric_columns[2].metric("SSIM", f"{metrics['ssim']:.4f}")

    st.subheader("Run Metadata")
    _display_metadata(metadata)


if __name__ == "__main__":
    main()
