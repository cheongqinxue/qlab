"""Dense encoder backed by a fixed phrase → vector map (golden tests)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base_encoder import BaseEncoder, EncodedVector, EncoderKind


class FixedDenseEncoder(BaseEncoder):
    """Lookup precomputed unit vectors by normalized phrase key."""

    def __init__(
        self,
        vectors: dict[str, Sequence[float] | np.ndarray],
        *,
        vector_dimension: int | None = None,
    ) -> None:
        """Normalize and store vectors.

        Args:
            vectors: Keys are normalized to ``strip().lower()``.
            vector_dimension: Explicit dim; inferred from first vector when omitted.
        """
        normalized: dict[str, list[float]] = {}
        dim = vector_dimension
        for phrase, vec in vectors.items():
            arr = np.asarray(vec, dtype=np.float64)
            norm = float(np.linalg.norm(arr))
            unit = (arr / norm).tolist() if norm else arr.tolist()
            normalized[phrase.strip().lower()] = unit
            if dim is None and unit:
                dim = len(unit)
        if dim is None:
            dim = 8
        self._vectors = normalized
        self._vector_dimension = dim

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.DENSE

    @property
    def vector_dimension(self) -> int:
        return self._vector_dimension

    def encode_document(self, text: str) -> EncodedVector:
        """Return a fixed vector for ``text``."""
        return self._lookup(text)

    def encode_query(self, text: str) -> EncodedVector:
        """Return a fixed vector for ``text``."""
        return self._lookup(text)

    def _lookup(self, text: str) -> EncodedVector:
        key = text.strip().lower()
        if key not in self._vectors:
            msg = f"No fixed vector for phrase {key!r}"
            raise KeyError(msg)
        return EncodedVector(kind=EncoderKind.DENSE, dense=self._vectors[key])
