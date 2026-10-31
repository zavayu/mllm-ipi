# Initial baseline design

The initial protocol is pinned in `configs/milestone2_baseline.yaml`. The intended
data source is COCO 2017 test images, split into project development (30), validation
(100), and test (100) sets before model evaluation. Selection is deterministic and
detected duplicates are grouped before choosing representatives. The baseline
runner opens only the development split.

Each image receives one clean query and one query with a fixed, conspicuous
repetition prompt rendered in magenta. Both queries are image-only. Model revision,
image-processing limits, prompt wording, decoding, output budget, and font-fitting
behavior are fixed. Adaptive prefixes, mask ranking, subtle coloring, and parameter
sweeps are excluded from this baseline.

The primary outcome is exact target output after trimming surrounding whitespace.
Containment is diagnostic. Analysis pairs responses by image, never counts repeated
queries as independent images, and retains token-limited responses at the declared
budget. Failed attempts and missing pairs are reported separately. A run with
missing pairs is incomplete and can be resumed.

Reports include image-level Wilson intervals, capped-response counts, actual font
sizes, end-to-end query time, and resource measurements. Runtime extrapolations are
planning estimates. Visible-text success establishes a prerequisite for subsequent
subtle rendering experiments; it does not demonstrate invisibility or override of
an explicit legitimate task.

The local seven-image runtime pilot is documented separately and is not pooled
with future COCO results. See [milestone2.md](milestone2.md) for actual status and
[the benchmark instructions](../benchmarks/README.md) for reproduction.
