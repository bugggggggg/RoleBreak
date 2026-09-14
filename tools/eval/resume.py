"""Skip the examples a previous run already finished."""

from __future__ import annotations

import json
from pathlib import Path


def recorded_examples(combined_path: str | Path) -> set[str]:
    """The examples a consolidated ``runs.jsonl`` already holds a row for."""
    path = Path(combined_path)
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {name for row in rows if (name := row.get("name")) is not None}
