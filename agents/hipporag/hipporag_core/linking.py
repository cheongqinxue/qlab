"""Link query NER phrases to graph entity nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents_utils.memory.base_encoder import BaseEncoder


@dataclass(frozen=True)
class EntityLink:
    """One query entity linked to a graph node."""

    query_entity: str
    entity_index: int
    entity_text: str
    similarity: float


def link_query_entities(
    query_entities: list[str],
    *,
    entity_texts: list[str],
    entity_vectors: np.ndarray,
    encoder: BaseEncoder,
) -> list[EntityLink]:
    """Argmax cosine link each query entity to one graph node.

    Args:
        query_entities: NER phrases from the question.
        entity_texts: Canonical text per indexed entity node.
        entity_vectors: ``(|N|, dim)`` matrix aligned with ``entity_texts``.
        encoder: Phrase encoder **M**.

    Returns:
        One link per query entity (skips empty strings).
    """
    if not entity_texts:
        return []
    matrix = np.asarray(entity_vectors, dtype=np.float64)
    links: list[EntityLink] = []
    for query in query_entities:
        q = query.strip()
        if not q:
            continue
        query_vec = encoder.encode_query_numpy(q)
        sims = matrix @ query_vec
        best = int(np.argmax(sims))
        links.append(
            EntityLink(
                query_entity=q,
                entity_index=best,
                entity_text=entity_texts[best],
                similarity=float(sims[best]),
            )
        )
    return links
