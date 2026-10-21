"""Qdrant-backed memory bank via LangChain's ``QdrantVectorStore`` (Qdrant server)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field, model_validator
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .base_encoder import BaseEncoder, EncoderKind
from .base_memory import BaseMemory, BaseMemoryItem
from .bm25_sparse import Bm25Sparse

logger = logging.getLogger(__name__)

_DEFAULT_QDRANT_URL = "http://localhost:6333"
RetrievalMode = Literal["dense", "bm25_colbert"]


class CollectionConfigMismatchError(Exception):
    """Existing Qdrant collection settings do not match the experiment config file."""


class QdrantMemoryConfig(BaseModel):
    """Construction settings for ``QdrantMemory`` (commit one per experiment)."""

    url: str = Field(default=_DEFAULT_QDRANT_URL, description="Qdrant server REST base URL.")
    api_key: str | None = Field(default=None, description="Optional Qdrant API key.")
    collection_name: str = Field(description="Target collection name.")
    retrieval_mode: RetrievalMode = Field(
        default="dense",
        description="``dense`` for single-vector search; ``bm25_colbert`` for sparse prefetch + ColBERT.",
    )
    vector_size: int = Field(gt=0, description="Dense or ColBERT per-token vector dimension.")
    distance: str = Field(default="COSINE", description="Qdrant ``Distance`` enum name.")
    vector_name: str = Field(
        default="",
        description="Dense mode: named vector slot; empty uses the default unnamed vector.",
    )
    colbert_vector_name: str = Field(
        default="colbert",
        description="BM25+ColBERT mode: multivector field name.",
    )
    sparse_vector_name: str = Field(
        default="bm25",
        description="BM25+ColBERT mode: sparse vector field name.",
    )
    prefetch_limit: int = Field(
        default=100,
        ge=1,
        description="BM25+ColBERT mode: sparse prefetch candidate count before ColBERT rerank.",
    )
    content_payload_key: str = Field(default=QdrantVectorStore.CONTENT_KEY)
    metadata_payload_key: str = Field(default=QdrantVectorStore.METADATA_KEY)
    default_k: int = Field(default=10, ge=1, description="Default ``afind`` result limit.")

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> QdrantMemoryConfig:
        if self.retrieval_mode == "bm25_colbert" and not self.colbert_vector_name:
            msg = "bm25_colbert mode requires non-empty ``colbert_vector_name``"
            raise ValueError(msg)
        if self.retrieval_mode == "bm25_colbert" and not self.sparse_vector_name:
            msg = "bm25_colbert mode requires non-empty ``sparse_vector_name``"
            raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> QdrantMemoryConfig:
        """Load config from a JSON file.

        Args:
            path: Path to a JSON object matching this model.

        Returns:
            Parsed config instance.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


def _encode_content(content: Any) -> str:
    """Serialize arbitrary content for Qdrant payload storage."""
    return json.dumps(content, default=str)


def _decode_content(raw: str) -> Any:
    """Deserialize content from a Qdrant payload string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _content_to_search_text(content: Any, encoded: str) -> str:
    """Text used for BM25 indexing and ColBERT encoding."""
    return content if isinstance(content, str) else encoded


def _metadata_filter(metadata: dict[str, Any] | None) -> models.Filter | None:
    """Build a Qdrant filter from flat metadata key/value pairs."""
    if not metadata:
        return None
    conditions = [
        models.FieldCondition(
            key=f"metadata.{key}",
            match=models.MatchValue(value=value),
        )
        for key, value in metadata.items()
    ]
    return models.Filter(must=conditions)


def _point_vector(
    store: QdrantVectorStore, vector: list[float]
) -> list[float] | dict[str, list[float]]:
    if store.vector_name:
        return {store.vector_name: vector}
    return vector


def _dense_vectors_config(
    config: QdrantMemoryConfig,
) -> models.VectorParams | dict[str, models.VectorParams]:
    distance = models.Distance[config.distance]
    params = models.VectorParams(size=config.vector_size, distance=distance)
    if config.vector_name:
        return {config.vector_name: params}
    return params


def _colbert_vectors_config(config: QdrantMemoryConfig) -> dict[str, models.VectorParams]:
    distance = models.Distance[config.distance]
    return {
        config.colbert_vector_name: models.VectorParams(
            size=config.vector_size,
            distance=distance,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        ),
    }


def _sparse_vectors_config(config: QdrantMemoryConfig) -> dict[str, models.SparseVectorParams]:
    return {
        config.sparse_vector_name: models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        ),
    }


def _create_client(config: QdrantMemoryConfig) -> QdrantClient:
    return QdrantClient(
        url=config.url,
        api_key=config.api_key,
        check_compatibility=False,
    )


def _actual_vector_params(
    vectors: models.VectorParams | dict[str, models.VectorParams] | None,
    vector_name: str,
) -> models.VectorParams:
    if vectors is None:
        msg = "Collection has no vector configuration"
        raise CollectionConfigMismatchError(msg)
    if vector_name:
        if not isinstance(vectors, dict):
            msg = f"Expected named vector {vector_name!r}, collection uses a single unnamed vector"
            raise CollectionConfigMismatchError(msg)
        if vector_name not in vectors:
            msg = f"Named vector {vector_name!r} not found in collection (keys: {list(vectors)})"
            raise CollectionConfigMismatchError(msg)
        return vectors[vector_name]
    if isinstance(vectors, dict):
        msg = f"Expected unnamed vector, collection has named vectors: {list(vectors)}"
        raise CollectionConfigMismatchError(msg)
    return vectors


def _validate_dense_collection(client: QdrantClient, config: QdrantMemoryConfig) -> None:
    info = client.get_collection(config.collection_name)
    params = info.config.params
    if params is None or params.vectors is None:
        msg = f"Collection {config.collection_name!r} has no vector params"
        raise CollectionConfigMismatchError(msg)
    if params.sparse_vectors:
        msg = f"Collection {config.collection_name!r} has sparse vectors but config mode is dense"
        raise CollectionConfigMismatchError(msg)

    expected_distance = models.Distance[config.distance]
    actual = _actual_vector_params(params.vectors, config.vector_name)
    mismatches: list[str] = []
    if actual.size != config.vector_size:
        mismatches.append(f"vector_size expected {config.vector_size}, got {actual.size}")
    if actual.distance != expected_distance:
        mismatches.append(
            f"distance expected {expected_distance.name}, got {actual.distance.name}"
        )
    if actual.multivector_config is not None:
        mismatches.append("expected dense vector, collection vector has multivector_config")
    if mismatches:
        detail = "; ".join(mismatches)
        msg = f"Collection {config.collection_name!r} exists but does not match config: {detail}"
        raise CollectionConfigMismatchError(msg)


def _validate_bm25_colbert_collection(client: QdrantClient, config: QdrantMemoryConfig) -> None:
    info = client.get_collection(config.collection_name)
    params = info.config.params
    if params is None or params.vectors is None:
        msg = f"Collection {config.collection_name!r} has no vector params"
        raise CollectionConfigMismatchError(msg)

    sparse = params.sparse_vectors or {}
    if config.sparse_vector_name not in sparse:
        msg = f"Missing sparse field {config.sparse_vector_name!r} (have: {list(sparse)})"
        raise CollectionConfigMismatchError(msg)
    if sparse[config.sparse_vector_name].modifier != models.Modifier.IDF:
        msg = f"Sparse field {config.sparse_vector_name!r} must use IDF modifier for BM25"
        raise CollectionConfigMismatchError(msg)

    if not isinstance(params.vectors, dict):
        msg = "bm25_colbert mode requires named ColBERT vector field"
        raise CollectionConfigMismatchError(msg)
    if config.colbert_vector_name not in params.vectors:
        msg = f"Missing ColBERT field {config.colbert_vector_name!r}"
        raise CollectionConfigMismatchError(msg)

    expected_distance = models.Distance[config.distance]
    colbert = params.vectors[config.colbert_vector_name]
    mismatches: list[str] = []
    if colbert.size != config.vector_size:
        mismatches.append(f"vector_size expected {config.vector_size}, got {colbert.size}")
    if colbert.distance != expected_distance:
        mismatches.append(
            f"distance expected {expected_distance.name}, got {colbert.distance.name}"
        )
    if colbert.multivector_config is None:
        mismatches.append("ColBERT field missing multivector_config")
    elif colbert.multivector_config.comparator != models.MultiVectorComparator.MAX_SIM:
        mismatches.append("ColBERT multivector comparator must be MAX_SIM")
    if mismatches:
        detail = "; ".join(mismatches)
        msg = f"Collection {config.collection_name!r} exists but does not match config: {detail}"
        raise CollectionConfigMismatchError(msg)


def _ensure_collection(client: QdrantClient, config: QdrantMemoryConfig) -> None:
    """Create the collection or verify it matches ``config``."""
    if not client.collection_exists(config.collection_name):
        if config.retrieval_mode == "dense":
            client.create_collection(
                collection_name=config.collection_name,
                vectors_config=_dense_vectors_config(config),
            )
        else:
            client.create_collection(
                collection_name=config.collection_name,
                vectors_config=_colbert_vectors_config(config),
                sparse_vectors_config=_sparse_vectors_config(config),
            )
        logger.info("Created collection %r on %s", config.collection_name, config.url)
        return

    if config.retrieval_mode == "dense":
        _validate_dense_collection(client, config)
    else:
        _validate_bm25_colbert_collection(client, config)


def _vector_store_from_config(
    client: QdrantClient,
    config: QdrantMemoryConfig,
) -> QdrantVectorStore:
    vector_name = (
        config.colbert_vector_name if config.retrieval_mode == "bm25_colbert" else config.vector_name
    )
    return QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
        embedding=None,
        vector_name=vector_name,
        content_payload_key=config.content_payload_key,
        metadata_payload_key=config.metadata_payload_key,
        distance=models.Distance[config.distance],
        validate_embeddings=False,
        validate_collection_config=False,
    )


class QdrantMemory(BaseMemory):
    """Memory bank stored in a Qdrant server collection through ``QdrantVectorStore``."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        config: QdrantMemoryConfig,
    ) -> None:
        """Wrap a LangChain Qdrant vector store bound to a server collection.

        Args:
            vector_store: Configured ``QdrantVectorStore``.
            config: Construction settings from the experiment config file.
        """
        self._store = vector_store
        self._config = config
        self._encoder: BaseEncoder | None = None
        self._bm25: Bm25Sparse | None = (
            Bm25Sparse() if config.retrieval_mode == "bm25_colbert" else None
        )

    @property
    def vector_store(self) -> QdrantVectorStore:
        """Underlying LangChain vector store."""
        return self._store

    @property
    def config(self) -> QdrantMemoryConfig:
        """Construction settings for this memory bank."""
        return self._config

    @classmethod
    def build_or_restore(
        cls,
        config: QdrantMemoryConfig,
        *,
        encoder: BaseEncoder | None = None,
    ) -> QdrantMemory:
        """Connect to Qdrant and ensure the experiment collection matches ``config``.

        Persistence is provided by the Qdrant server's mounted volume (per-experiment
        ``docker-compose.yml``). Existing data is used as-is when the collection already
        exists and matches; otherwise the collection is created empty.

        Args:
            config: Collection and client settings (typically ``QdrantMemoryConfig.from_file``).
            encoder: ``BaseEncoder`` for ``add`` / ``afind(query=...)``. Dense mode accepts
                dense encoders; ``bm25_colbert`` requires a ``MULTI_VECTOR`` encoder (ColBERT).

        Returns:
            Connected ``QdrantMemory`` instance.
        """
        if config.retrieval_mode == "dense":
            if encoder is not None and encoder.kind != EncoderKind.DENSE:
                msg = "dense mode requires a dense ``BaseEncoder``"
                raise ValueError(msg)
        elif encoder is None or encoder.kind != EncoderKind.MULTI_VECTOR:
            msg = "bm25_colbert mode requires a multi-vector ``BaseEncoder`` (e.g. ColBERT)"
            raise ValueError(msg)

        if encoder is not None and encoder.vector_dimension != config.vector_size:
            msg = (
                f"Encoder vector_dimension ({encoder.vector_dimension}) "
                f"does not match config vector_size ({config.vector_size})"
            )
            raise ValueError(msg)

        client = _create_client(config)
        _ensure_collection(client, config)
        store = _vector_store_from_config(client, config)
        memory = cls(store, config)
        memory._encoder = encoder
        return memory

    async def add(
        self,
        content: Any,
        *,
        vector: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        item_id = uuid.uuid4().hex
        encoded = _encode_content(content)
        payload_metadata = dict(metadata or {})
        payload = {
            self._config.content_payload_key: encoded,
            self._config.metadata_payload_key: payload_metadata,
        }

        if self._config.retrieval_mode == "bm25_colbert":
            return await self._add_bm25_colbert(
                item_id,
                content,
                encoded,
                payload,
                metadata=metadata,
            )

        if vector is not None:
            point = models.PointStruct(
                id=item_id,
                vector=_point_vector(self._store, list(vector)),
                payload=payload,
            )

            def _upsert() -> None:
                self._store.client.upsert(
                    collection_name=self._store.collection_name,
                    points=[point],
                )

            await asyncio.to_thread(_upsert)
            return item_id

        if self._encoder is not None:
            search_text = _content_to_search_text(content, encoded)
            encoded_vector = await self._encoder.aencode_document(search_text)
            assert encoded_vector.dense is not None
            return await self.add(content, vector=encoded_vector.dense, metadata=metadata)

        msg = "add requires ``vector=`` or a dense ``encoder`` on build_or_restore"
        raise ValueError(msg)

    async def _add_bm25_colbert(
        self,
        item_id: str,
        content: Any,
        encoded: str,
        payload: dict[str, Any],
        *,
        metadata: dict[str, Any] | None,
    ) -> str:
        assert self._encoder is not None and self._bm25 is not None
        search_text = _content_to_search_text(content, encoded)
        sparse = await self._bm25.aembed_document(search_text)
        colbert = await self._encoder.aencode_document(search_text)
        assert colbert.token_vectors is not None

        point = models.PointStruct(
            id=item_id,
            vector={
                self._config.sparse_vector_name: sparse,
                self._config.colbert_vector_name: colbert.token_vectors,
            },
            payload=payload,
        )

        def _upsert() -> None:
            self._store.client.upsert(
                collection_name=self._store.collection_name,
                points=[point],
            )

        await asyncio.to_thread(_upsert)
        return item_id

    async def afind(
        self,
        *,
        query: str | None = None,
        vector: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[BaseMemoryItem]:
        qdrant_filter = _metadata_filter(metadata)
        k = self._config.default_k

        if self._config.retrieval_mode == "bm25_colbert":
            return await self._afind_bm25_colbert(
                query=query,
                vector=vector,
                qdrant_filter=qdrant_filter,
                k=k,
            )

        if vector is not None:

            def _search_by_vector() -> list[tuple[Document, float]]:
                return self._store.similarity_search_with_score_by_vector(
                    list(vector),
                    k=k,
                    filter=qdrant_filter,
                )

            scored = await asyncio.to_thread(_search_by_vector)
            return [self._document_to_item(doc, score) for doc, score in scored]

        if query is not None:
            if self._encoder is not None:
                encoded_query = await self._encoder.aencode_query(query)
                assert encoded_query.dense is not None
                return await self.afind(
                    vector=encoded_query.dense,
                    metadata=metadata,
                )
            msg = "afind(query=...) requires a dense ``encoder`` on build_or_restore"
            raise ValueError(msg)

        if metadata is not None:
            return await self._scroll_items(qdrant_filter, k)

        msg = "afind requires at least one of query, vector, or metadata"
        raise ValueError(msg)

    async def _afind_bm25_colbert(
        self,
        *,
        query: str | None,
        vector: Sequence[float] | Sequence[Sequence[float]] | None,
        qdrant_filter: models.Filter | None,
        k: int,
    ) -> list[BaseMemoryItem]:
        assert self._encoder is not None and self._bm25 is not None

        if query is None and vector is None:
            if qdrant_filter is not None:
                return await self._scroll_items(qdrant_filter, k)
            msg = "bm25_colbert afind requires ``query=`` or metadata filter"
            raise ValueError(msg)

        if query is None:
            msg = "bm25_colbert afind with ``vector=`` still requires ``query=`` for BM25 prefetch"
            raise ValueError(msg)

        sparse_query = await self._bm25.aembed_query(query)
        if vector is not None and vector and isinstance(vector[0], (list, tuple)):
            colbert_matrix = [list(row) for row in vector]  # type: ignore[arg-type]
        else:
            colbert_query = await self._encoder.aencode_query(query)
            assert colbert_query.token_vectors is not None
            colbert_matrix = colbert_query.token_vectors

        def _query() -> list[BaseMemoryItem]:
            response = self._store.client.query_points(
                collection_name=self._store.collection_name,
                prefetch=models.Prefetch(
                    query=sparse_query,
                    using=self._config.sparse_vector_name,
                    limit=self._config.prefetch_limit,
                    filter=qdrant_filter,
                ),
                query=colbert_matrix,
                using=self._config.colbert_vector_name,
                query_filter=qdrant_filter,
                limit=k,
                with_payload=True,
            )
            return [self._scored_point_to_item(point) for point in response.points]

        return await asyncio.to_thread(_query)

    async def _scroll_items(
        self,
        qdrant_filter: models.Filter | None,
        k: int,
    ) -> list[BaseMemoryItem]:
        def _scroll() -> list[BaseMemoryItem]:
            items: list[BaseMemoryItem] = []
            offset = None
            while True:
                records, offset = self._store.client.scroll(
                    collection_name=self._store.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=k,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    doc = self._store._document_from_point(  # noqa: SLF001
                        record,
                        self._store.collection_name,
                        self._store.content_payload_key,
                        self._store.metadata_payload_key,
                    )
                    items.append(self._document_to_item(doc, 1.0))
                if offset is None:
                    break
            return items[:k]

        return await asyncio.to_thread(_scroll)

    async def adelete(self, item_id: str) -> bool:
        result = await self._store.adelete(ids=[item_id])
        return bool(result)

    async def aclear(self) -> None:
        def _clear() -> None:
            offset = None
            while True:
                records, next_offset = self._store.client.scroll(
                    collection_name=self._store.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                if records:
                    ids = [record.id for record in records]
                    self._store.client.delete(
                        collection_name=self._store.collection_name,
                        points_selector=models.PointIdsList(points=ids),
                    )
                if next_offset is None:
                    break
                offset = next_offset

        await asyncio.to_thread(_clear)

    def _scored_point_to_item(self, point: models.ScoredPoint) -> BaseMemoryItem:
        payload = point.payload or {}
        raw_content = payload.get(self._config.content_payload_key, "")
        return BaseMemoryItem(
            id=str(point.id),
            similarity=float(point.score),
            content=_decode_content(raw_content if isinstance(raw_content, str) else str(raw_content)),
        )

    def _document_to_item(self, document: Document, similarity: float) -> BaseMemoryItem:
        raw_id = document.metadata.get("_id", document.metadata.get("id"))
        item_id = str(raw_id) if raw_id is not None else ""
        return BaseMemoryItem(
            id=item_id,
            similarity=float(similarity),
            content=_decode_content(document.page_content),
        )
