"""Summarize paired, single-trial baselines without pooling experiments or images."""

from math import sqrt
from pathlib import Path
from statistics import median

from src.evaluate import read_jsonl_results
from src.provenance import write_json_atomic


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def summarize_condition(
    path: Path, experiment_id: str, expected: set[str]
) -> tuple[dict, dict]:
    rows = [
        row
        for row in read_jsonl_results(path)
        if row.get("experiment_id") == experiment_id
    ]
    completed = {row["run_id"]: row for row in rows if row.get("status") == "completed"}
    by_image = {}
    for row in completed.values():
        image = row["image_sha256"]
        if image not in expected:
            raise ValueError("results contain images outside the frozen cohort")
        if image in by_image:
            raise ValueError("baseline requires exactly one completed trial per image")
        by_image[image] = row
    exact = sum(row["exact_target_response"] for row in by_image.values())
    times = [row["query_seconds"] for row in by_image.values()]
    tokens = sum(
        row.get("generation", {}).get("output_tokens", 0) for row in by_image.values()
    )
    summary = {
        "experiment_id": experiment_id,
        "expected_images": len(expected),
        "completed_images": len(by_image),
        "missing_images": len(expected - by_image.keys()),
        "failed_attempts": sum(row.get("status") == "error" for row in rows),
        "exact_responses": exact,
        "exact_rate": exact / len(by_image) if by_image else None,
        "exact_wilson_95": wilson_interval(exact, len(by_image)),
        "target_mentions": sum(row["target_containment"] for row in by_image.values()),
        "token_limit_responses": sum(
            row.get("generation", {}).get("hit_token_limit", False)
            for row in by_image.values()
        ),
        "median_query_seconds": median(times) if times else None,
        "total_query_seconds": sum(times),
        "output_tokens": tokens,
        "end_to_end_output_tokens_per_second": (
            tokens / sum(times) if sum(times) else None
        ),
        "empirical": bool(by_image)
        and all(row.get("empirical_model_run") for row in by_image.values()),
        "actual_font_sizes": sorted(
            {
                row["render_font_size"]
                for row in by_image.values()
                if "render_font_size" in row
            }
        ),
    }
    return summary, by_image


def write_report(
    run_dir: Path, protocol: dict, experiment_ids: dict, resources: dict
) -> dict:
    expected = {image["sha256"] for image in protocol["images"]}
    summaries, paired = {}, {}
    for condition, identity in experiment_ids.items():
        summaries[condition], paired[condition] = summarize_condition(
            run_dir / f"{condition}.jsonl", identity, expected
        )
    clean, visible = paired["clean"], paired["paper_image_only"]
    shared = sorted(clean.keys() & visible.keys())
    improved = sum(
        not clean[k]["exact_target_response"] and visible[k]["exact_target_response"]
        for k in shared
    )
    worsened = sum(
        clean[k]["exact_target_response"] and not visible[k]["exact_target_response"]
        for k in shared
    )
    pair_seconds = (
        sum(s["median_query_seconds"] for s in summaries.values())
        if all(s["median_query_seconds"] is not None for s in summaries.values())
        else None
    )
    report = {
        "protocol_id": protocol["protocol_id"],
        "dataset_manifest_id": protocol["dataset_manifest_id"],
        "scope": protocol["scope"],
        "dataset_source": protocol["dataset_source"],
        "complete": len(shared) == len(expected),
        "conditions": summaries,
        "empirical": all(summary["empirical"] for summary in summaries.values()),
        "paired": {
            "images": len(shared),
            "clean_failure_to_visible_success": improved,
            "clean_success_to_visible_failure": worsened,
            "exact_rate_difference": (
                (improved - worsened) / len(shared) if shared else None
            ),
        },
        "resources": resources,
        "runtime_projection": {
            "basis": "sum of condition medians; excludes model loading, rendering, and disk overhead",
            "development_30_pairs_seconds": (
                pair_seconds * 30 if pair_seconds is not None else None
            ),
            "all_230_pairs_seconds": (
                pair_seconds * 230 if pair_seconds is not None else None
            ),
        },
        "failure_examples": [
            {
                "image": Path(row["original_image"]).name,
                "response": row["model_response"],
                "generation": row.get("generation"),
                "font_size": row.get("render_font_size"),
            }
            for row in visible.values()
            if not row["exact_target_response"]
        ][:6],
        "limitations": [
            "Exact output is evaluated at the frozen output-token budget; capped responses remain in the denominator.",
            "Image-only redirection does not establish override of an explicit legitimate task.",
            "Wilson intervals describe this image sample under a binomial model, not population-wide guarantees.",
            "Perceptual hashing is a duplicate-screening heuristic; it cannot guarantee semantic independence.",
            "A runtime pilot on local images is not a COCO benchmark or a test-set result.",
        ],
    }
    write_json_atomic(run_dir / "summary.json", report)
    lines = [
        "# Baseline results",
        "",
        f"Scope: **{report['scope']}**. Dataset: {report['dataset_source']['dataset']}.",
        "",
        f"Protocol: `{report['protocol_id']}`. Empirical model run: **{report['empirical']}**.",
        "",
        "| Condition | Completed | Exact | Mentions | Token limit | Median seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, s in summaries.items():
        seconds = (
            f"{s['median_query_seconds']:.2f}"
            if s["median_query_seconds"] is not None
            else "unknown"
        )
        lines.append(
            f"| {condition} | {s['completed_images']}/{s['expected_images']} | {s['exact_responses']} | {s['target_mentions']} | {s['token_limit_responses']} | {seconds} |"
        )
    lines.extend(
        [
            "",
            f"Paired improvement: {improved} images changed from clean failure to visible-text exact response; {worsened} changed in the opposite direction.",
            "",
            "## Limits",
            "",
            *[f"- {text}" for text in report["limitations"]],
            "",
            "Full counts, uncertainty intervals, failure examples, and resource measurements are in `summary.json`.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
