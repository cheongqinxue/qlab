"""Tests for ``BaseEncoder`` sync/async and NumPy helpers."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from agents_utils.memory.base_encoder import EncodedVector, EncoderKind, encoded_to_dense_numpy
from agents_utils.memory.dense_numpy import dense_cosine_similarity_numpy, hash_dense_vector
from agents_utils.memory.fixed_dense_encoder import FixedDenseEncoder
from agents_utils.memory.hash_dense_encoder import HashDenseEncoder


class TestHashDenseEncoder:
    """Hash backend matches legacy test encoder behavior."""

    def test_sync_and_async_numpy_match(self) -> None:
        encoder = HashDenseEncoder(dim=8)
        sync_vec = encoder.encode_document_numpy("hello world")
        async_vec = asyncio.run(encoder.aencode_document_numpy("hello world"))
        np.testing.assert_allclose(sync_vec, async_vec)

    def test_hash_helper_unit_norm(self) -> None:
        dense = hash_dense_vector("test phrase", dim=8)
        norm = float(np.linalg.norm(np.asarray(dense)))
        assert norm == pytest.approx(1.0)


class TestFixedDenseEncoder:
    """Fixed map encoder."""

    def test_lookup_normalizes_key(self) -> None:
        raw = np.array([3.0, 4.0])
        encoder = FixedDenseEncoder({"Foo": raw})
        vec = encoder.encode_query_numpy("  foo ")
        assert vec.shape == (2,)
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0)


class TestEncodedToDenseNumpy:
    """Vector conversion guardrails."""

    def test_rejects_multi_vector(self) -> None:
        encoded = EncodedVector(
            kind=EncoderKind.MULTI_VECTOR,
            token_vectors=[[1.0, 0.0]],
        )
        with pytest.raises(TypeError):
            encoded_to_dense_numpy(encoded)
