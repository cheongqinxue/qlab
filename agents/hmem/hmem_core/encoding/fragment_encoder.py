"""Fragment and event embeddings via shared ``BaseEncoder``.

All semantic similarity in tree merge and retrieval scoring must go through
``agents_utils.memory.base_encoder.BaseEncoder`` (sync or async) — never ad-hoc
embedding clients in ``hmem_core``.
"""

from __future__ import annotations

import os

from agents_utils.memory.base_encoder import BaseEncoder
from agents_utils.memory.hash_dense_encoder import HashDenseEncoder
from agents_utils.memory.sentence_transformers_encoder import SentenceTransformersEncoder


def build_fragment_encoder(*, override: BaseEncoder | None = None) -> BaseEncoder:
    """Construct the process-wide encoder for H-MEM.

    Args:
        override: Test double or fixture encoder (pytest).

    Returns:
        ``BaseEncoder`` implementation.
    """
    if override is not None:
        return override
    model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if os.environ.get("HMEM_AGENTS", "fixture").lower() == "fixture":
        return HashDenseEncoder()
    return SentenceTransformersEncoder(model)
