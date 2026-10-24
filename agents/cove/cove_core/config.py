"""Configuration for CoVe runtime."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class CoVeConfig(BaseModel):
    """Service-level CoVe settings."""

    verification_execution_mode: str = Field(
        default="factored",
        description="factored | two_step | joint (only factored wired in scaffold)",
    )
    runs_data_dir: str = Field(default=".data/runs")

    @classmethod
    def from_file(cls, path: str | Path) -> CoVeConfig:
        """Load JSON config."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
