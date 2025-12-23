"""Usage: load invoice template configurations from JSON files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "invoice_templates"


@lru_cache(maxsize=1)
def load_templates(template_dir: Path | None = None) -> list[dict[str, Any]]:
    target_dir = template_dir or TEMPLATE_DIR
    if not target_dir.exists():
        return []

    templates: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        _validate_template(data, source=path)
        data["_source_path"] = str(path)
        templates.append(data)
    return templates


def _validate_template(data: dict[str, Any], *, source: Path) -> None:
    required_keys = ("name", "match_rules", "fields", "non_table_fields", "table")
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Template {source} missing required key: {key}")
