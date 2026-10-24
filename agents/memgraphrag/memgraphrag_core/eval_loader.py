"""Load committed toy eval fixtures."""

from __future__ import annotations

import json
from pathlib import Path


def toy_conflict_dir() -> Path:
    """Directory containing ``eval/toy_conflict``."""
    return Path(__file__).resolve().parents[1] / "eval" / "toy_conflict"


def load_toy_corpus() -> list[dict[str, str]]:
    """Return passage dicts from ``corpus.json``."""
    data = json.loads((toy_conflict_dir() / "corpus.json").read_text(encoding="utf-8"))
    return list(data["passages"])


def load_toy_extraction() -> dict[str, dict]:
    """Return passage id → extraction dict."""
    return json.loads((toy_conflict_dir() / "extraction.json").read_text(encoding="utf-8"))


def load_toy_conflicts() -> list[dict]:
    """Return conflict group list."""
    data = json.loads((toy_conflict_dir() / "conflicts.json").read_text(encoding="utf-8"))
    return list(data["groups"])


def load_toy_resolutions() -> dict[str, dict]:
    """Return target fact id → resolution plan."""
    return json.loads((toy_conflict_dir() / "resolutions.json").read_text(encoding="utf-8"))
