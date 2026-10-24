"""Dense encoder using sentence-transformers (bi-encoder / pooled embeddings)."""

from __future__ import annotations

from typing import Any

from .base_encoder import BaseEncoder, EncodedVector, EncoderKind


class SentenceTransformersEncoder(BaseEncoder):
    """Wrap a ``SentenceTransformer`` model as a dense ``BaseEncoder``."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Load a sentence-transformers model.

        Args:
            model_name_or_path: Hub id or local path (e.g. ``all-MiniLM-L6-v2``).
            device: Optional torch device string passed to ``SentenceTransformer``.
            encode_kwargs: Extra kwargs forwarded to ``SentenceTransformer.encode``
                (e.g. ``normalize_embeddings=True``).
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name_or_path, device=device)
        self._encode_kwargs = encode_kwargs or {"normalize_embeddings": True}
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            msg = f"Could not determine vector dimension for model {model_name_or_path!r}"
            raise ValueError(msg)
        self._vector_dimension = dimension

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.DENSE

    @property
    def vector_dimension(self) -> int:
        return self._vector_dimension

    def encode_document(self, text: str) -> EncodedVector:
        """Encode document-side text."""
        return EncodedVector(kind=EncoderKind.DENSE, dense=self._encode_dense_list(text))

    def encode_query(self, text: str) -> EncodedVector:
        """Encode query-side text (same bi-encoder path)."""
        return EncodedVector(kind=EncoderKind.DENSE, dense=self._encode_dense_list(text))

    def _encode_dense_list(self, text: str) -> list[float]:
        vector = self._model.encode(text, **self._encode_kwargs)
        return list(vector)
