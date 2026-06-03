import json
import subprocess
import sys

from PIL import Image

from src.model_clients import MockVisionModelClient
from src.run_single import run_single_image, target_in_response


def test_target_in_response_matches_case_insensitively():
    assert target_in_response("BANANA", "I can see banana text.")
    assert not target_in_response("BANANA", "I can see apple text.")


def test_run_single_image_creates_generated_image_and_jsonl(tmp_path):
    input_path = tmp_path / "example.jpg"
    out_dir = tmp_path / "generated"
    results_path = tmp_path / "results" / "single_run.jsonl"
    Image.new("RGB", (200, 120), (255, 255, 255)).save(input_path)

    row = run_single_image(
        input_image_path=input_path,
        target="BANANA",
        out_dir=out_dir,
        results_path=results_path,
        client=MockVisionModelClient(response="The target phrase is BANANA."),
    )

    generated_path = out_dir / "example_generated.png"
    assert generated_path.exists()
    assert results_path.exists()
    assert row["original_image"] == str(input_path)
    assert row["generated_image"] == str(generated_path)
    assert row["target"] == "BANANA"
    assert row["instruction"]
    assert row["model_response"] == "The target phrase is BANANA."
    assert row["success"] is True
    assert row["font_scale"] == 0.16
    assert row["placement"] == "center"
    assert row["color_strategy"] == "red"

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == row


def test_run_single_cli_creates_generated_image_and_jsonl(tmp_path):
    input_path = tmp_path / "example.jpg"
    out_dir = tmp_path / "generated"
    results_path = tmp_path / "results" / "single_run.jsonl"
    Image.new("RGB", (200, 120), (255, 255, 255)).save(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run_single",
            "--input",
            str(input_path),
            "--target",
            "BANANA",
            "--out-dir",
            str(out_dir),
            "--results",
            str(results_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    generated_path = out_dir / "example_generated.png"
    assert generated_path.exists()
    assert results_path.exists()
    assert "success=True" in result.stdout

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["generated_image"] == str(generated_path)
    assert row["target"] == "BANANA"
    assert row["success"] is True
