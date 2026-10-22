"""Phrase embeddings for entity linking and synonymy."""

from __future__ import annotations

from dataclasses import dataclass, field

import hashlib

import numpy as np


@dataclass
class PhraseEncoder:
    """Encode phrases to L2-normalized vectors (HF model or fixed test vectors)."""

    _cache: dict[str, np.ndarray] = field(default_factory=dict)
    _model: object | None = None

    @classmethod
    def from_vectors(cls, vectors: dict[str, np.ndarray]) -> PhraseEncoder:
        """Build an encoder backed by a fixed phrase → vector map.

        Args:
            vectors: Keys are normalized phrases.

        Returns:
            Encoder instance.
        """
        normalized: dict[str, np.ndarray] = {}
        for phrase, vec in vectors.items():
            arr = np.asarray(vec, dtype=np.float64)
            norm = np.linalg.norm(arr)
            normalized[phrase.strip().lower()] = arr / norm if norm else arr
        return cls(_cache=normalized)

    @classmethod
    def from_env_model(cls, model_name: str) -> PhraseEncoder:
        """Load sentence-transformers model lazily on first encode.

        Args:
            model_name: HuggingFace model id.

        Returns:
            Encoder instance.
        """
        return cls(_model=model_name)

    def encode(self, phrase: str) -> np.ndarray:
        """Encode one phrase.

        Args:
            phrase: Raw text.

        Returns:
            1-D normalized embedding vector.
        """
        key = phrase.strip().lower()
        if key in self._cache:
            return self._cache[key]
        if self._model is None:
            vec = _hash_vector(key, dim=8)
            self._cache[key] = vec
            return vec
        if isinstance(self._model, str):
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model)
            self._model = model
        vec = np.asarray(
            self._model.encode(phrase, normalize_embeddings=True),
            dtype=np.float64,
        )
        self._cache[key] = vec
        return vec

    def encode_batch(self, phrases: list[str]) -> np.ndarray:
        """Encode multiple phrases.

        Args:
            phrases: Raw texts.

        Returns:
            2-D array ``(len(phrases), dim)``.
        """
        return np.stack([self.encode(p) for p in phrases], axis=0)


def _hash_vector(text: str, *, dim: int) -> np.ndarray:
    """Deterministic unit vector from text (test / fallback encoder)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vec = np.frombuffer(digest[:dim], dtype=np.uint8).astype(np.float64)
    if vec.size < dim:
        vec = np.pad(vec, (0, dim - vec.size))
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec
