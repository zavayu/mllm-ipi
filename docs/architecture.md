# Architecture

## Pipeline

1. Load raw image
2. Render embedded prompt into image
3. Query local vision-language model
4. Log model response
5. Evaluate attack success
6. Compute stealth metrics
7. Later: use SAM for mask-based placement
8. Later: build Streamlit UI

## Core Modules

- `src/render_prompt.py`: image text rendering
- `src/model_clients/`: local Qwen and mock model clients
- `src/run_experiment.py`: experiment runner
- `src/evaluate.py`: ASR computation
- `src/metrics.py`: MSE, SSIM, OCR detection
- `src/segment_regions.py`: SAM-based region detection
- `src/ui/`: Streamlit demo