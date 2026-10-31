# Project milestones

1. **Trustworthy evaluation and rendering:** explicit response metrics and experiment
   conditions, provenance, correct resume behavior, and audited text placement.
2. **Reproducible benchmark and baselines:** fixed development/validation/test splits,
   clean-image and visible-text baselines, and measured compute cost.
3. **Controlled experiments:** isolate placement, readability, contrast, prompting,
   and layout effects; select settings on validation data and freeze test evaluation.
4. **Automatic generation and verification:** bounded candidate search that improves
   over a fixed configuration, with explicit failure and query-budget reporting.
5. **End-to-end UI:** generate, inspect, verify, and download images through Streamlit,
   retaining a manual research mode.
6. **Generalization:** evaluate frozen choices on another model and unseen prompts;
   document success, visual changes, runtime, query cost, and failure cases.

Milestone 1 implementation and evidence are documented in [milestone1.md](milestone1.md).
Subsequent milestones are not implemented by this milestone.
