import csv
import json
import subprocess
import sys

import pytest

from src.evaluate import (
    compute_summary,
    evaluate_results,
    read_jsonl_results,
)


def _write_results(path):
    rows = [
        {
            "success": True,
            "font_scale": 0.1,
            "placement": "center",
            "color_strategy": "white",
        },
        {
            "success": False,
            "font_scale": 0.1,
            "placement": "center",
            "color_strategy": "white",
        },
        {
            "success": True,
            "font_scale": 0.2,
            "placement": "top_right",
            "color_strategy": "black",
        },
        {
            "success": True,
            "font_scale": 0.2,
            "placement": "top_right",
            "color_strategy": "white",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def test_read_jsonl_results_skips_blank_lines(tmp_path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        '{"success": true, "font_scale": 0.1}\n\n',
        encoding="utf-8",
    )

    rows = read_jsonl_results(results_path)

    assert rows == [{"success": True, "font_scale": 0.1}]


def test_read_jsonl_results_reports_bad_json_with_line_number(tmp_path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"success": true}\n{bad\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_jsonl_results(results_path)


def test_compute_summary_includes_overall_and_grouped_asr(tmp_path):
    rows = _write_results(tmp_path / "results.jsonl")

    summary = compute_summary(rows)
    by_key = {(row["group"], row["value"]): row for row in summary}

    assert by_key[("overall", "all")] == {
        "group": "overall",
        "value": "all",
        "total": 4,
        "successes": 3,
        "asr": 0.75,
    }
    assert by_key[("font_scale", "0.1")]["asr"] == 0.5
    assert by_key[("font_scale", "0.2")]["asr"] == 1.0
    assert by_key[("placement", "center")]["asr"] == 0.5
    assert by_key[("placement", "top_right")]["asr"] == 1.0
    assert by_key[("color_strategy", "black")]["asr"] == 1.0
    assert by_key[("color_strategy", "white")]["asr"] == pytest.approx(2 / 3)


def test_evaluate_results_creates_summary_csv_and_plot(tmp_path):
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path)
    out_dir = tmp_path / "summary"

    outputs = evaluate_results(results_path, out_dir)

    assert outputs["summary_csv"].exists()
    assert outputs["font_scale_plot"].exists()
    assert outputs["font_scale_plot"].stat().st_size > 0

    with outputs["summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0] == {
        "group": "overall",
        "value": "all",
        "total": "4",
        "successes": "3",
        "asr": "0.75",
    }
    assert {"font_scale", "placement", "color_strategy"} <= {
        row["group"] for row in rows
    }


def test_evaluate_cli_creates_artifacts(tmp_path):
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path)
    out_dir = tmp_path / "summary"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--results",
            str(results_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "summary_csv=" in result.stdout
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "asr_by_font_scale.png").exists()
