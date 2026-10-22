"""Exported graph snapshot for PPR in Python."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PprSnapshot:
    """In-memory view of the entity graph and passage mention matrix."""

    entity_ids: list[str]
    entity_texts: list[str]
    entity_vectors: np.ndarray
    passage_ids: list[str]
    edges: list[tuple[int, int, float]]
    mentions: np.ndarray
