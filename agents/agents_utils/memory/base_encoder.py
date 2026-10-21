"""Encoder backends — consistent API for dense and multi-vector memory encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from math import sqrt
from typing import Self

from pydantic import BaseModel, Field, model_validator


class EncoderKind(str, Enum):
    """How an encoder represents text for storage and retrieval."""

    DENSE = "dense"
    MULTI_VECTOR = "multi_vector"


class EncodedVector(BaseModel):
    """Encoder output: one dense vector or a token matrix (late interaction)."""

    kind: EncoderKind = Field(description="Dense bi-encoder vs. multi-vector (e.g. ColBERT).")
    dense: list[float] | None = Field(
        default=None,
        description="Single pooled vector (sentence-transformers, bi-encoders, etc.).",
    )
    token_vectors: list[list[float]] | None = Field(
        default=None,
        description="One vector per token for late-interaction encoders.",
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if self.kind == EncoderKind.DENSE:
            if not self.dense:
                msg = "dense kind requires non-empty ``dense`` vector"
                raise ValueError(msg)
            if self.token_vectors is not None:
                msg = "dense kind must not set ``token_vectors``"
                raise ValueError(msg)
        elif self.kind == EncoderKind.MULTI_VECTOR:
            if not self.token_vectors:
                msg = "multi_vector kind requires non-empty ``token_vectors``"
                raise ValueError(msg)
            if self.dense is not None:
                msg = "multi_vector kind must not set ``dense``"
                raise ValueError(msg)
        return self


def dense_cosine_similarity(query: Sequence[float], document: Sequence[float]) -> float:
    """Cosine similarity between two dense vectors.

    Args:
        query: Query vector from an encoder.
        document: Document vector from an encoder.

    Returns:
        Similarity in ``[-1, 1]`` (or ``0.0`` when either vector has zero norm).
    """
    if len(query) != len(document):
        msg = f"Vector length mismatch: {len(query)} vs {len(document)}"
        raise ValueError(msg)
    dot = sum(a * b for a, b in zip(query, document, strict=True))
    norm_q = sqrt(sum(x * x for x in query))
    norm_d = sqrt(sum(x * x for x in document))
    if norm_q == 0.0 or norm_d == 0.0:
        return 0.0
    return dot / (norm_q * norm_d)


def multi_vector_maxsim_score(
    query_tokens: Sequence[Sequence[float]],
    document_tokens: Sequence[Sequence[float]],
) -> float:
    """Late-interaction score (sum of max token dot products, ColBERT-style).

    Args:
        query_tokens: Query token vectors.
        document_tokens: Document token vectors.

    Returns:
        Unnormalized MaxSim score.
    """
    if not query_tokens or not document_tokens:
        return 0.0
    dim = len(query_tokens[0])
    total = 0.0
    for q_vec in query_tokens:
        if len(q_vec) != dim:
            msg = "All query token vectors must share the same dimension"
            raise ValueError(msg)
        best = float("-inf")
        for d_vec in document_tokens:
            if len(d_vec) != dim:
                msg = "Query and document token vectors must share the same dimension"
                raise ValueError(msg)
            dot = sum(a * b for a, b in zip(q_vec, d_vec, strict=True))
            if dot > best:
                best = dot
        if best > float("-inf"):
            total += best
    return total


class BaseEncoder(ABC):
    """Provider-agnostic text encoder for memory indexing and scoring.

    Implementations include dense encoders (e.g. sentence-transformers) and
    multi-vector encoders (e.g. ColBERT). Storage backends consume ``EncodedVector``.
    """

    @property
    @abstractmethod
    def kind(self) -> EncoderKind:
        """Return ``DENSE`` or ``MULTI_VECTOR`` for this encoder."""

    @property
    @abstractmethod
    def vector_dimension(self) -> int:
        """Vector dimension (per token for ``MULTI_VECTOR``)."""

    @abstractmethod
    async def aencode_document(self, text: str) -> EncodedVector:
        """Encode text stored as a memory item (passage / document side).

        Args:
            text: Raw text to encode.

        Returns:
            Encoded output matching ``kind``.
        """

    @abstractmethod
    async def aencode_query(self, text: str) -> EncodedVector:
        """Encode text used at retrieval time (query side).

        Args:
            text: Raw query text.

        Returns:
            Encoded output matching ``kind``.
        """

    async def ascore(self, query: EncodedVector, document: EncodedVector) -> float:
        """Score a query–document pair.

        Args:
            query: Output of ``aencode_query``.
            document: Output of ``aencode_document``.

        Returns:
            Similarity score (higher is better for ranking).
        """
        if query.kind != self.kind or document.kind != self.kind:
            msg = "Query and document kinds must match the encoder ``kind``"
            raise ValueError(msg)
        return self._score_pair(query, document)

    def _score_pair(self, query: EncodedVector, document: EncodedVector) -> float:
        if self.kind == EncoderKind.DENSE:
            assert query.dense is not None and document.dense is not None
            return dense_cosine_similarity(query.dense, document.dense)
        assert query.token_vectors is not None and document.token_vectors is not None
        return multi_vector_maxsim_score(query.token_vectors, document.token_vectors)
