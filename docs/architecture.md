# Architecture

The batch runner is the canonical research path:

1. Load and validate a YAML configuration and enumerate images.
2. Record a content-addressed manifest covering data, model, code, font, and runtime.
3. Query original images for clean conditions, or construct and render instructions.
4. Select fully contained mask rectangles when mask placement is requested.
5. Query the local model using explicit decoding settings.
6. Append completed outcomes or retryable errors to JSONL.
7. Summarize one experiment using a named response metric.

`src/config.py` defines the experiment contract. `src/provenance.py` records
identity and manifests. `src/response_metrics.py` defines response matching.
`src/region_ranker.py`, `src/segment_regions.py`, and `src/render_prompt.py` handle
geometry and rendering. `src/model_clients/` performs local or mock inference.
`src/run_experiment.py` coordinates trials; `src/evaluate.py` generates summaries.

`app.py` is a Streamlit diagnostic interface. `src/run_single.py` is a small
single-image helper. Neither replaces the batch runner's manifest-based workflow.
`src/validate_milestone1.py` generates reviewable diagnostic artifacts.

See [evaluation.md](evaluation.md) for exact semantics and compatibility notes.
