"""Extraction agent (A_ext) implementations."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from memgraphrag_core.agents.llm_client import create_chat_model
from memgraphrag_core.agents.llm_models import ExtractionResult
from memgraphrag_core.agents.prompts import EXTRACTION_HUMAN, EXTRACTION_SYSTEM


class LlmExtractionAgent:
    """Online extraction via OpenAI-compatible LLM (structured JSON)."""

    def __init__(self, model: str | None = None, *, runner: object | None = None) -> None:
        """Bind structured LLM runner (injectable for tests).

        Args:
            model: Chat model name; defaults to ``OPENAI_MODEL``.
            runner: Optional pre-built runnable with ``invoke(messages)``.
        """
        if runner is None:
            runner = create_chat_model(model).with_structured_output(ExtractionResult)
        self._runner = runner

    def extract(self, passage_id: str, text: str) -> dict:
        """Run A_ext on one passage.

        Args:
            passage_id: Chunk id.
            text: Passage text.

        Returns:
            Dict with ``schemas`` and ``facts`` lists for the orchestrator.
        """
        result: ExtractionResult = self._runner.invoke(
            [
                SystemMessage(content=EXTRACTION_SYSTEM),
                HumanMessage(
                    content=EXTRACTION_HUMAN.format(passage_id=passage_id, text=text),
                ),
            ]
        )
        payload = result.model_dump()
        for fact in payload.get("facts", []):
            if not fact.get("passage_ids"):
                fact["passage_ids"] = [passage_id]
        return payload


class FixtureExtractionAgent:
    """Replay committed extractions for toy corpora."""

    def __init__(self, by_passage_id: dict[str, dict]) -> None:
        """Bind fixture map passage_id → extraction dict."""
        self._by_passage_id = by_passage_id

    def extract(self, passage_id: str, text: str) -> dict:
        """Return fixture extraction or empty structure."""
        _ = text
        return self._by_passage_id.get(passage_id, {"schemas": [], "facts": []})
