"""Configuration for ``MemGraphGlobalMemory.build_or_restore``."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class MemGraphMemoryConfig(BaseModel):
    """Persisted memory + graph settings for the MemGraphRAG experiment.

    Attributes:
        memory_data_dir: Root for JSON snapshots (``memory.json``, openie cache).
        ontology_min_frequency: Schema stability threshold τ (paper Eq. 4 spirit).
        retrieval_sim_threshold: Minimum query–memory similarity for layer filtering.
        ppr_damping: Personalized PageRank damping for memory-guided retrieval.
        retrieval_top_k: Default passage count for ``afind`` / API retrieve.
        neo4j_enabled: When True, dual-write structured layers to Neo4j for Browser viz.
    """

    memory_data_dir: str = Field(default=".data/memory")
    ontology_min_frequency: int = Field(default=2, ge=1)
    retrieval_sim_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ppr_damping: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(default=5, ge=1)
    neo4j_enabled: bool = Field(default=True)

    @classmethod
    def from_file(cls, path: str | Path) -> MemGraphMemoryConfig:
        """Load JSON config (e.g. ``config/default.json`` merged with env overrides).

        Args:
            path: Path to a JSON object with optional MemGraphMemoryConfig fields.

        Returns:
            Parsed configuration instance.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
