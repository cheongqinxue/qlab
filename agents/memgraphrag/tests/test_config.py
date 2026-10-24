"""Tests for ``MemGraphMemoryConfig``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memgraphrag_core.memory.config import MemGraphMemoryConfig

_REPO_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "default.json"


class TestMemGraphMemoryConfig:
    """Configuration loading and validation."""

    def test_defaults(self) -> None:
        cfg = MemGraphMemoryConfig()
        assert cfg.memory_data_dir == ".data/memory"
        assert cfg.ppr_damping == 0.5
        assert cfg.neo4j_enabled is True

    def test_from_repo_default_json(self) -> None:
        cfg = MemGraphMemoryConfig.from_file(_REPO_DEFAULT)
        assert cfg.retrieval_top_k == 5
        assert cfg.memory_data_dir == ".data/memory"

    def test_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text(
            json.dumps({"retrieval_top_k": 3, "ppr_damping": 0.4}),
            encoding="utf-8",
        )
        cfg = MemGraphMemoryConfig.from_file(path)
        assert cfg.retrieval_top_k == 3
        assert cfg.ppr_damping == 0.4

    def test_validation_rejects_bad_threshold(self) -> None:
        with pytest.raises(ValidationError):
            MemGraphMemoryConfig(retrieval_sim_threshold=1.5)
