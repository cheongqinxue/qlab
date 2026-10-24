"""Factory for event/graph extraction agents."""

from __future__ import annotations

from hmem_core.agents.fixture_agents import FixtureEventExtractor
from hmem_core.agents.llm_agents import LlmEventExtractor
from hmem_core.agents.protocols import EventExtractor


def build_event_extractor(*, agents_mode: str) -> EventExtractor:
    """Return fixture or LLM extractor.

    Args:
        agents_mode: ``fixture`` or ``llm``.

    Returns:
        Extractor implementation.
    """
    if agents_mode == "llm":
        return LlmEventExtractor()
    return FixtureEventExtractor()
