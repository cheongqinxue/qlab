"""Deterministic dense encoder for tests and offline demos."""

from __future__ import annotations

from .base_encoder import BaseEncoder, EncodedVector, EncoderKind
from .dense_numpy import hash_dense_vector


class HashDenseEncoder(BaseEncoder):
    """L2-normalized hash vectors for deterministic unit tests."""

    def __init__(self, *, dim: int = 8) -> None:
        """Configure hash output size."""
        self._dim = dim

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.DENSE

    @property
    def vector_dimension(self) -> int:
        return self._dim

    def encode_document(self, text: str) -> EncodedVector:
        """Encode document text."""
        return self._encode(text)

    def encode_query(self, text: str) -> EncodedVector:
        """Encode query text (identical to document for this backend)."""
        return self._encode(text)

    def _encode(self, text: str) -> EncodedVector:
        key = text.strip().lower()
        return EncodedVector(
            kind=EncoderKind.DENSE,
            dense=hash_dense_vector(key, dim=self._dim),
        )
