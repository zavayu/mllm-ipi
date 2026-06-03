"""Evaluate JSONL experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_CSV_NAME = "summary.csv"
FONT_SCALE_PLOT_NAME = "asr_by_font_scale.png"


def read_jsonl_results(path: str | Path) -> list[dict[str, Any]]:
    """Read experiment result rows from a JSONL file."""
    results_path = Path(path)
    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL in {results_path} at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"result row in {results_path} at line {line_number} must be an object"
                )
            rows.append(row)
    return rows


def _success_value(row: dict[str, Any]) -> bool:
    value = row.get("success")
    if not isinstance(value, bool):
        raise ValueError("each result row must contain a boolean 'success' value")
    return value


def _color_strategy(row: dict[str, Any]) -> str:
    value = row.get("color_strategy")
    if value is not None:
        return str(value)

    color = row.get("color")
    if isinstance(color, list) and len(color) == 3:
        return f"rgb({color[0]},{color[1]},{color[2]})"
    return "unknown"


def _group_value(row: dict[str, Any], key: str) -> str:
    if key == "color_strategy":
        return _color_strategy(row)
    value = row.get(key)
    if value is None:
        return "unknown"
    return str(value)


def _summary_row(group: str, value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(1 for row in rows if _success_value(row))
    total = len(rows)
    asr = successes / total if total else 0.0
    return {
        "group": group,
        "value": value,
        "total": total,
        "successes": successes,
        "asr": asr,
    }


def compute_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute overall and grouped attack success rates."""
    if not rows:
        raise ValueError("results file contains no rows")

    summary = [_summary_row("overall", "all", rows)]
    for group in ("font_scale", "placement", "color_strategy"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_group_value(row, group)].append(row)
        for value in sorted(grouped, key=_sort_key):
            summary.append(_summary_row(group, value, grouped[value]))
    return summary


def _sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def write_summary_csv(path: str | Path, summary: list[dict[str, Any]]) -> None:
    """Write summary rows to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["group", "value", "total", "successes", "asr"]
        )
        writer.writeheader()
        writer.writerows(summary)


def plot_asr_by_font_scale(summary: list[dict[str, Any]], path: str | Path) -> None:
    """Generate a bar chart for ASR grouped by font scale."""
    font_rows = [row for row in summary if row["group"] == "font_scale"]
    labels = [str(row["value"]) for row in font_rows]
    values = [float(row["asr"]) for row in font_rows]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#3B82F6")
    ax.set_xlabel("Font scale")
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("ASR by font scale")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_results(results_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Evaluate a JSONL results file and write summary artifacts."""
    output_dir = Path(out_dir)
    rows = read_jsonl_results(results_path)
    summary = compute_summary(rows)

    summary_csv = output_dir / SUMMARY_CSV_NAME
    font_scale_plot = output_dir / FONT_SCALE_PLOT_NAME
    write_summary_csv(summary_csv, summary)
    plot_asr_by_font_scale(summary, font_scale_plot)
    return {"summary_csv": summary_csv, "font_scale_plot": font_scale_plot}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate JSONL experiment results.")
    parser.add_argument("--results", required=True, help="Path to JSONL results file.")
    parser.add_argument("--out-dir", required=True, help="Directory for summary outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = evaluate_results(args.results, args.out_dir)
    print(f"summary_csv={outputs['summary_csv']}")
    print(f"font_scale_plot={outputs['font_scale_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
