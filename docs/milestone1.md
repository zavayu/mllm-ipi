# Milestone 1: trustworthy evaluation and rendering

Implementation is ready for review. The benchmark, parameter study, optimizer,
and UI redesign remain later milestones.

## What changed

| Area | Result |
| --- | --- |
| Evaluation | Explicit exact-response and containment metrics; separate rendered instructions, expected responses, and external instructions |
| Conditions | Clean image, paper-style image-only, and explicit task-override runs |
| Provenance | Content-addressed manifests covering images, requested and resolved model settings, processor, font, segmentation, source, and runtime |
| Resume | Completed trials are skipped; failed queries are retried; interrupted final records are preserved before recovery |
| Decoding | One deterministic trial by default; sampled repetitions require explicit settings and record per-trial seeds |
| Geometry | Rendering uses a fully contained interior rectangle, with original mask hashes and exportable mask arrays |
| Fitting | Complete text fits or fails explicitly; overlapping multi-mask regions are excluded; fallback preserves the minimum size |
| Diagnostics | A repeatable validation command produces a gallery, manifests, result logs, and a JSON report |

The full contract and backward-compatibility notes are in [evaluation.md](evaluation.md).
Historical result files were not rewritten. Legacy containment fields remain
available, but new summaries default to exact output.

## Validation evidence

- **93 automated tests passed**, compared with 59 before this milestone.
- Ruff checks and `git diff --check` passed.
- Synthetic rendering checks passed for irregular-mask containment, complete prompt
  preservation, multi-mask fallback, and minimum font size.
- A brute-force comparison on small randomized masks verified the interior-rectangle
  algorithm. Regression tests cover overlapping masks and impossible fits.
- Clean, image-only, and task-override mock trials completed and resumed without
  duplicate model queries.
- Streamlit's application test completed a mock pipeline run with no application
  exceptions and displayed separate exact-response and containment metrics.
- A real, offline Qwen2.5-VL-3B clean/modified pair completed on CPU using revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`. The clean image did not produce the
  target; the rendered image produced exactly `BANANA`. Both rows record actual
  processor grid dimensions, input token count, and pixel tensor shape.

The real-model check uses one synthetic 112x112 image and an eight-token output
budget. It validates loading, preprocessing, generation, logging, and response
matching. It does **not** estimate attack success rate, establish human
imperceptibility, or demonstrate task override. The clean response is truncated
by the small token budget.

This environment has PyTorch `2.12.0+cpu` and no GPU backend. No SAM checkpoint is
configured. Real SAM segmentation and accelerated inference therefore remain
unvalidated here. Synthetic masks exercise the geometry independently; requiring
SAM now produces an explicit error instead of silently substituting grid masks.

## Reproduce the checks

From the repository root with dependencies installed:

```powershell
python -m pytest -q --basetemp .pytest_cache/milestone1-tests
python -m ruff check src app.py tests
python -m src.validate_milestone1 --out-dir results/milestone1/review
```

For the cached-weight CPU smoke, explicitly prevent network downloads:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m src.validate_milestone1 --out-dir results/milestone1/review --qwen-smoke --allow-cpu --revision 66285546d2b821cf421d4f5eb2576359d3770cd3
```

The output directory contains `render_gallery.png`, `report.json`, modified images,
and JSONL results with manifests. Repeating the command skips completed trials.
For a shared results file, pass the selected row's experiment ID to the evaluator:

```powershell
python -m src.evaluate --results results/milestone1/review/qwen_results.jsonl --out-dir results/milestone1/review/summary --experiment-id EXPERIMENT_ID
```

## Paper alignment and remaining differences

Reference: [Nagaraja et al., Image-based Prompt Injection](https://arxiv.org/abs/2603.03637).

The paper's image-only condition is retained as a distinct protocol. Its placement
description motivates the interior rectangle, approximately 10% font reductions,
and spatially ordered multi-mask fallback. The existing area/variance/location
score remains a documented implementation choice rather than an experimentally
validated ranking rule. Integer pixel rounding and the current greedy word
allocation are explicit implementation details.

This milestone does not reproduce the full coloring study or the paper's
commercial-model results. It does not establish equivalence between font-scale
units, human visibility, or success-scoring rules across implementations. Those
questions require the benchmark and controlled comparisons in Milestones 2–3.

## Review focus

Inspect the gallery for complete readable text and intended placement, review the
metric definitions, and inspect a manifest beside its result rows. The expected
next milestone is a fixed dataset split with clean and visible-text baselines.
