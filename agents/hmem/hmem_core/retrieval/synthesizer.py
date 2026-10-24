"""Answer synthesis from ranked evidence (paper §3.3 step 3).

Community reference: ``retrieval/synthesizer.py``.
"""

from __future__ import annotations

from hmem_core.agents.fixture_agents import FixtureSynthesizer
from hmem_core.agents.llm_agents import LlmSynthesizer
from hmem_core.agents.protocols import SynthesizerAgent


def build_synthesizer(*, agents_mode: str) -> SynthesizerAgent:
    """Fixture or LLM synthesizer factory."""
    if agents_mode == "llm":
        return LlmSynthesizer()
    return FixtureSynthesizer()


class AnswerSynthesizer:
    """Wrap synthesizer agent."""

    def __init__(self, agent: SynthesizerAgent) -> None:
        """Store agent."""
        self._agent = agent

    def answer(self, query: str, evidence_texts: list[str]) -> str:
        """Produce final string answer."""
        return self._agent.synthesize(query, evidence_texts)
