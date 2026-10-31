from src.validate_milestone1 import validate


def test_diagnostic_artifacts_and_conditions_are_reproducible(tmp_path):
    first = validate(tmp_path)
    assert all(first["render_checks"].values())
    assert all(value["new_trials"] == 1 for value in first["mock_conditions"].values())
    assert (tmp_path / "render_gallery.png").is_file()
    assert (tmp_path / "report.json").is_file()
    second = validate(tmp_path)
    assert all(value["new_trials"] == 0 for value in second["mock_conditions"].values())
    assert second["qwen_smoke"]["executed"] is False
