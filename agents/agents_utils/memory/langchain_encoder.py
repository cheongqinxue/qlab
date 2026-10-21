"""Dense encoder adapter for LangChain ``Embeddings`` (optional integration path)."""

from __future__ import annotations

import asyncio

from langchain_core.embeddings import Embeddings

from .base_encoder import BaseEncoder, EncodedVector, EncoderKind


class LangchainEncoder(BaseEncoder):
    """Wrap LangChain ``Embeddings`` as a dense ``BaseEncoder``."""

    def __init__(self, embeddings: Embeddings, *, vector_dimension: int) -> None:
        """Configure the adapter.

        Args:
            embeddings: LangChain embedding model (OpenAI, HuggingFace, etc.).
            vector_dimension: Output dimension (must match the underlying model).
        """
        self._embeddings = embeddings
        self._vector_dimension = vector_dimension

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.DENSE

    @property
    def vector_dimension(self) -> int:
        return self._vector_dimension

    async def aencode_document(self, text: str) -> EncodedVector:
        vectors = await asyncio.to_thread(self._embeddings.embed_documents, [text])
        return EncodedVector(kind=EncoderKind.DENSE, dense=list(vectors[0]))

    async def aencode_query(self, text: str) -> EncodedVector:
        vector = await asyncio.to_thread(self._embeddings.embed_query, text)
        return EncodedVector(kind=EncoderKind.DENSE, dense=list(vector))
