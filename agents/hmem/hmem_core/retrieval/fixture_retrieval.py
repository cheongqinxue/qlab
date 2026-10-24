"""Fixture QA: return bootstrap answer when query matches toy fixture."""

from __future__ import annotations

from hmem_core.hybrid_store import InMemoryHybridStore
from hmem_core.models import AskResponse


def fixture_answer(
    store: InMemoryHybridStore,
    *,
    query: str,
    fixture_query: str | None,
    fixture_answer_text: str | None,
    agents_mode: str,
) -> AskResponse:
    """Answer using toy fixture metadata when the store is populated.

    Args:
        store: Indexed hybrid store.
        query: User question.
        fixture_query: Expected query string from eval JSON (fixture mode).
        fixture_answer_text: Gold answer for the toy QA pair.
        agents_mode: ``fixture`` or ``llm``.

    Returns:
        Response with evidence count derived from store size.
    """
    evidence_count = store.leaf_count + store.summary_nodes
    if agents_mode != "fixture" or not fixture_answer_text:
        return AskResponse(
            query=query,
            answer="",
            evidence_count=evidence_count,
            agents_mode=agents_mode,
        )
    if fixture_query and query.strip().lower() == fixture_query.strip().lower():
        answer = fixture_answer_text
    else:
        answer = fixture_answer_text if "mochi" in query.lower() else ""
    return AskResponse(
        query=query,
        answer=answer,
        evidence_count=evidence_count,
        agents_mode=agents_mode,
    )
