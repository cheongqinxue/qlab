"""Client-side BM25 sparse vectors for Qdrant (FastEmbed ``Qdrant/bm25``)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from qdrant_client.http import models

if TYPE_CHECKING:
    from fastembed.sparse.sparse_embedding import SparseEmbedding


class Bm25Sparse:
    """Encode text to Qdrant sparse vectors with BM25-style weights."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """Load the FastEmbed BM25 model.

        Args:
            model_name: FastEmbed sparse model id (default matches Qdrant BM25).
        """
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name)

    @staticmethod
    def to_qdrant(sparse: SparseEmbedding) -> models.SparseVector:
        """Convert a FastEmbed sparse embedding to a Qdrant ``SparseVector``."""
        return models.SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )

    def embed_document(self, text: str) -> models.SparseVector:
        """BM25 sparse vector for a stored document/passage.

        Args:
            text: Document text.

        Returns:
            Sparse vector for upsert.
        """
        sparse = next(self._model.embed([text]))
        return self.to_qdrant(sparse)

    def embed_query(self, text: str) -> models.SparseVector:
        """BM25 sparse vector for a search query.

        Args:
            text: Query text.

        Returns:
            Sparse vector for prefetch search.
        """
        sparse = next(self._model.query_embed([text]))
        return self.to_qdrant(sparse)

    async def aembed_document(self, text: str) -> models.SparseVector:
        """Async wrapper around ``embed_document``."""
        return await asyncio.to_thread(self.embed_document, text)

    async def aembed_query(self, text: str) -> models.SparseVector:
        """Async wrapper around ``embed_query``."""
        return await asyncio.to_thread(self.embed_query, text)
