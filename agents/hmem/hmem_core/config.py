"""JSON-backed configuration for H-MEM."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ScoreWeights(BaseModel):
    """Weights for semantic, temporal, and robustness terms in ranking."""

    semantic: float = 0.5
    temporal: float = 0.3
    robustness: float = 0.2


class HMemConfig(BaseModel):
    """Hybrid memory hyperparameters."""

    tree_levels: int = 3
    similarity_thresholds: list[float] = Field(default_factory=lambda: [0.85, 0.8, 0.75])
    time_window_seconds: list[int] = Field(default_factory=lambda: [86400, 604800, 2592000])
    retrieval_top_k: int = 5
    graph_max_hops: int = 2
    temporal_lambda: float = 0.5
    robustness_tau_seconds: float = 2_592_000.0
    robustness_eta: float = 0.1
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)
    memory_data_dir: str = ".data/hmem"

    @classmethod
    def from_file(cls, path: Path) -> HMemConfig:
        """Load config from JSON.

        Args:
            path: Path to ``default.json`` or override.

        Returns:
            Parsed configuration.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
