from dataclasses import replace
import json

import pytest
from PIL import Image

from src.config import (
    DatasetConfig,
    TargetConfig,
    RenderingConfig,
    ModelConfig,
    OutputConfig,
    ExperimentConfig,
    EvaluationConfig,
)
from src.run_experiment import run_experiment


class Client:
    def __init__(self):
        self.calls = []
        self.fail = False

    def query(self, image_path, instruction):
        self.calls.append((image_path, instruction))
        if self.fail:
            raise RuntimeError("temporary inference failure")
        return "BANANA"


@pytest.fixture
def config(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (320, 200), (100, 120, 140)).save(images / "one.png")
    return ExperimentConfig(
        dataset=DatasetConfig(images),
        target=TargetConfig("BANANA", "Output BANANA"),
        rendering=RenderingConfig(("center",), (0.05,)),
        model=ModelConfig("mock"),
        output=OutputConfig(tmp_path / "generated", tmp_path / "results.jsonl"),
    )


def test_manifest_and_resume_track_all_scientific_inputs(config):
    client = Client()
    original = run_experiment(config, client=client)[0]
    assert run_experiment(config, client=client) == []
    manifest = json.loads(open(original["manifest_path"], encoding="utf-8").read())
    assert manifest["images"][0]["sha256"] == original["image_sha256"]
    assert manifest["model"]["client_class"].endswith(".Client")
    assert manifest["font"]["sha256"]
    assert manifest["source_hashes"]["src/run_experiment.py"]
    assert original["empirical_model_run"] is False
    variants = [
        replace(config, model=replace(config.model, model_id="another-model")),
        replace(config, model=replace(config.model, max_new_tokens=12)),
        replace(config, model=replace(config.model, revision="another-revision")),
        replace(config, model=replace(config.model, seed=42)),
        replace(
            config,
            evaluation=EvaluationConfig("task_override"),
            model=replace(config.model, instruction="Describe the image."),
        ),
        replace(config, rendering=replace(config.rendering, colors=((0, 255, 0),))),
    ]
    ids = {original["run_id"]}
    for variant in variants:
        row = run_experiment(variant, client=client)[0]
        assert row["run_id"] not in ids
        ids.add(row["run_id"])
    Image.new("RGB", (320, 200), (20, 30, 40)).save(
        config.dataset.image_dir / "one.png"
    )
    changed = run_experiment(config, client=client)[0]
    assert changed["image_sha256"] != original["image_sha256"]
    assert changed["run_id"] not in ids
    assert changed["render_id"] != original["render_id"]


def test_errors_are_logged_and_retried_without_repeating_completed_runs(config):
    config = replace(
        config, rendering=replace(config.rendering, placements=("center", "top_right"))
    )
    client = Client()
    client.fail = True
    failed = run_experiment(config, client=client)
    assert len(failed) == 2
    assert all(row["status"] == "error" and row["stage"] == "query" for row in failed)
    client.fail = False
    retried = run_experiment(config, client=client)
    assert {r["run_id"] for r in retried} == {r["run_id"] for r in failed}
    assert all(row["status"] == "completed" for row in retried)
    assert run_experiment(config, client=client) == []


def test_clean_condition_queries_original_once_and_records_task_outcome(config):
    config = replace(
        config,
        evaluation=EvaluationConfig("clean", task_expected_response="BANANA"),
        rendering=replace(
            config.rendering, placements=("center", "top_right"), font_scales=(0.1, 0.2)
        ),
    )
    client = Client()
    (row,) = run_experiment(config, client=client)
    assert client.calls == [(str(config.dataset.image_dir / "one.png"), "")]
    assert row["mse"] == 0
    assert row["ssim"] == 1
    assert row["task_success"] is True
    assert row["segmentation_backend"] == "none"
    assert not config.output.generated_dir.exists()


def test_static_color_sweep_is_not_silently_truncated(config):
    config = replace(
        config, rendering=replace(config.rendering, colors=((255, 0, 0), (0, 255, 0)))
    )
    rows = run_experiment(config, client=Client())
    assert len(rows) == 2
    assert len({row["render_id"] for row in rows}) == 2


def test_empty_dataset_is_not_a_successful_experiment(config):
    (config.dataset.image_dir / "one.png").unlink()
    with pytest.raises(ValueError, match="no dataset images"):
        run_experiment(config, client=Client())
