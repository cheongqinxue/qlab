"""Load toy CoVe fixtures from ``eval/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def eval_root() -> Path:
    """Directory containing toy eval corpora."""
    return Path(__file__).resolve().parents[1] / "eval"


def load_toy_manifest() -> dict[str, Any]:
    """Return ``toy_mexican_american_war/manifest.json``."""
    path = eval_root() / "toy_mexican_american_war" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_toy_pipeline(query_id: str) -> dict[str, Any]:
    """Load committed pipeline fixture for a ``query_id``."""
    root = eval_root() / "toy_mexican_american_war"
    manifest = load_toy_manifest()
    if manifest.get("query_id") != query_id:
        msg = f"No toy pipeline for query_id={query_id!r}"
        raise KeyError(msg)
    return json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
