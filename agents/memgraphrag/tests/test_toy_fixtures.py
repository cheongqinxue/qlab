"""Sanity checks for committed eval JSON (no production loader yet)."""

from __future__ import annotations

import json
from pathlib import Path


class TestToyConflictFixtures:
    """``eval/toy_conflict`` files parse and align with agent tests."""

    @staticmethod
    def _eval_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "eval" / "toy_conflict"

    def test_corpus_json(self) -> None:
        data = json.loads((self._eval_dir() / "corpus.json").read_text(encoding="utf-8"))
        assert len(data["passages"]) >= 2

    def test_extraction_json(self) -> None:
        data = json.loads((self._eval_dir() / "extraction.json").read_text(encoding="utf-8"))
        assert "p1" in data and "p2" in data

    def test_conflicts_and_resolutions(self) -> None:
        conflicts = json.loads((self._eval_dir() / "conflicts.json").read_text(encoding="utf-8"))
        resolutions = json.loads((self._eval_dir() / "resolutions.json").read_text(encoding="utf-8"))
        target = conflicts["groups"][0]["target_id"]
        assert target in resolutions
