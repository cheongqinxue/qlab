"""NumPy helpers for dense vectors (hash fallback and similarity)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def dense_cosine_similarity_numpy(
    query: np.ndarray | Sequence[float],
    document: np.ndarray | Sequence[float],
) -> float:
    """Cosine similarity between dense vectors (NumPy).

    Args:
        query: Query vector.
        document: Document vector.

    Returns:
        Similarity in ``[-1, 1]`` (``0.0`` when either vector has zero norm).
    """
    q = np.asarray(query, dtype=np.float64)
    d = np.asarray(document, dtype=np.float64)
    if q.shape != d.shape:
        msg = f"Vector shape mismatch: {q.shape} vs {d.shape}"
        raise ValueError(msg)
    norm_q = float(np.linalg.norm(q))
    norm_d = float(np.linalg.norm(d))
    if norm_q == 0.0 or norm_d == 0.0:
        return 0.0
    return float(np.dot(q, d) / (norm_q * norm_d))


def hash_dense_vector(text: str, *, dim: int = 8) -> list[float]:
    """Deterministic L2-normalized dense vector from text (tests / fallback).

    Args:
        text: Input string (typically lowercased by callers).
        dim: Output dimension.

    Returns:
        Unit-norm vector as Python floats.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest[:dim], dtype=np.uint8).astype(np.float64)
    if raw.size < dim:
        raw = np.pad(raw, (0, dim - raw.size))
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        return raw.tolist()
    return (raw / norm).tolist()
