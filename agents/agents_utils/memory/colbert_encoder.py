"""Multi-vector encoder scaffold for ColBERT-style late interaction."""

from __future__ import annotations

from .base_encoder import BaseEncoder, EncodedVector, EncoderKind


class ColbertEncoder(BaseEncoder):
    """Late-interaction encoder producing token-level vectors.

    One ``MULTI_VECTOR`` option alongside dense encoders (sentence-transformers, etc.).
    Subclass and implement ``_encode_tokens_sync`` with your ColBERT model.
    """

    def __init__(self, *, vector_dimension: int) -> None:
        """Store per-token vector size.

        Args:
            vector_dimension: Dimension of each token vector.
        """
        self._vector_dimension = vector_dimension

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.MULTI_VECTOR

    @property
    def vector_dimension(self) -> int:
        return self._vector_dimension

    def encode_document(self, text: str) -> EncodedVector:
        """Encode a stored document."""
        token_vectors = self._encode_tokens_sync(text, is_query=False)
        return EncodedVector(kind=EncoderKind.MULTI_VECTOR, token_vectors=token_vectors)

    def encode_query(self, text: str) -> EncodedVector:
        """Encode a retrieval query."""
        token_vectors = self._encode_tokens_sync(text, is_query=True)
        return EncodedVector(kind=EncoderKind.MULTI_VECTOR, token_vectors=token_vectors)

    def _encode_tokens_sync(self, text: str, *, is_query: bool) -> list[list[float]]:
        """Encode text to token vectors (implement in a concrete model subclass).

        Args:
            text: Input text.
            is_query: True when encoding a retrieval query; False for stored documents.

        Returns:
            Matrix of shape ``(num_tokens, vector_dimension)``.

        Raises:
            NotImplementedError: Until a concrete ColBERT backend is provided.
        """
        msg = f"{type(self).__name__} must implement _encode_tokens_sync"
        raise NotImplementedError(msg)
