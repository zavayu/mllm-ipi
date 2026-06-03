# MLLM-IPI

MLLM-IPI (Multimodal Large Language Model Image-based Prompt Injection) is a local, reproducible research pipeline for studying image-based prompt injection in open-source vision-language models.

The project investigates whether adversarial instructions embedded directly into images can influence a multimodal model's response. It is inspired by research on visually embedded prompt injection, but the goal here is defensive: build a controlled framework for measuring model robustness, comparing attack conditions, and evaluating practical mitigations.

The initial model focus is the Qwen-VL family, especially Qwen2.5-VL through Hugging Face Transformers. The pipeline is designed so model access is wrapped behind a small client interface, making it possible to run fast mock experiments during development and local Qwen inference when hardware is available.

## Research Goal

The central question is:

> Can a local open-source vision-language model be influenced by text embedded inside an image, and how do visual choices affect attack success and detectability?

The project studies harmless target strings such as `BANANA`, `TEST123`, or `HELLO_WORLD`. It should remain a model safety and robustness evaluation, not a tool for real-world misuse.

## What The Pipeline Does

At a high level, the intended experiment loop is:

```text
Raw images
  -> render embedded prompt text
  -> query a local vision-language model
  -> log responses as JSONL
  -> evaluate attack success rate
  -> compare results across rendering conditions
```

Current implemented pieces include:

- Python package and CLI module scaffold.
- Pillow-based visible text rendering.
- Config-driven dataset and parameter sweeps.
- Mock model client for reproducible tests.
- Qwen2.5-VL client scaffold for local inference.
- JSONL result logging.
- Basic attack success evaluation and summary plotting.

Later phases are expected to add richer stealth metrics, defense testing, SAM-based region selection, multi-mask prompt placement, and an interactive UI. SAM and Streamlit are not part of the current implementation.

## Experiment Dimensions

The project is meant to compare attack success across controlled visual and prompt variables:

- Prompt wording and target phrase.
- Font size.
- Text placement.
- Text color and contrast.
- Background-aware rendering strategies.
- Image region characteristics.
- Defensive preprocessing such as OCR detection, blur, downsampling, and compression.

The main metric is Attack Success Rate, where a trial succeeds if the model response contains or matches the target string. Secondary metrics may include visual distortion, OCR detectability, response categories, runtime, and GPU memory usage.

## Project Structure

```text
configs/      Experiment configuration files
data/         Local image inputs and generated images
docs/         Project notes and design docs
results/      JSONL results and evaluation outputs
src/          Python package and CLI modules
tests/        Pytest suite
```

## Current Status

The repository has moved beyond the bare CLI scaffold into an early experiment pipeline. Mock-model runs are the safest path for reproducible development. Qwen local inference support is scaffolded, but depends on the correct local PyTorch, Transformers, Qwen utilities, model files, and hardware setup.

The current work should still be treated as an early research codebase. The near-term priority is keeping the mock path reliable while gradually expanding rendering strategies and model evaluation.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Show the base CLI help:

```bash
python -m src.main --help
```

Render visible prompt text onto one image:

```bash
python -m src.render_prompt --input data/raw/example.png --output data/generated/example.png --text "BANANA" --placement center --font-scale 0.2
```

Run a single mock evaluation:

```bash
python -m src.run_single --input data/raw/example.png --target BANANA --out-dir data/generated --results results/single.jsonl
```

Run a config-driven mock experiment:

```bash
python -m src.run_experiment --config configs/baseline_mock.yaml
```

Evaluate JSONL results:

```bash
python -m src.evaluate --results results/baseline_mock.jsonl --out-dir results/eval
```

## Tests

```bash
.\.venv\Scripts\pytest.exe
```

If `pytest` is available on your shell path, this is equivalent:

```bash
pytest
```
