import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "image-based prompt injection" in result.stdout
