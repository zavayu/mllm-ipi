"""Utilities for writing experiment result logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_single_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    """Write exactly one JSON object as a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")
