"""Factory for indexing and summarization agents."""

from __future__ import annotations

from hmem_core.agents.fixture_agents import FixtureSummarizer
from hmem_core.agents.llm_agents import LlmSummarizer
from hmem_core.agents.protocols import SummarizerAgent


def build_summarizer(*, agents_mode: str) -> SummarizerAgent:
    """Return fixture or LLM summarizer."""
    if agents_mode == "llm":
        return LlmSummarizer()
    return FixtureSummarizer()
