"""LLM-backed agents (``HMEM_AGENTS=llm``)."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents_utils.llm.openai_chat import create_chat_model

from hmem_core.models import GraphEdge, MemoryEvent, MemoryFragment, MemoryScope, SubQuery

logger = logging.getLogger(__name__)


class LlmSummarizer:
    """Consolidate child memory texts via LLM."""

    def __init__(self) -> None:
        self._llm = create_chat_model()

    def summarize(self, child_texts: list[str]) -> str:
        """Return consolidated summary."""
        prompt = (
            "Summarize the following memory snippets into one concise paragraph "
            "preserving names, dates, and facts:\n"
            + "\n".join(f"- {text}" for text in child_texts)
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", str(response))
        return str(content).strip()


class LlmEventExtractor:
    """Extract events and relations via structured LLM output."""

    def __init__(self) -> None:
        self._llm = create_chat_model()

    def extract(self, fragment: MemoryFragment) -> tuple[list[MemoryEvent], list[GraphEdge]]:
        """Run OpenIE-style extraction."""
        system = SystemMessage(
            content=(
                "Return ONLY JSON with keys events (list of {text}) and "
                "relations (list of {head, relation, tail})."
            )
        )
        human = HumanMessage(content=fragment.text)
        raw = str(self._llm.invoke([system, human]).content)
        payload = _parse_json_object(raw)
        events = [
            MemoryEvent(
                event_id=f"ev_{fragment.id}_{index}",
                fragment_id=fragment.id,
                text=str(item.get("text", fragment.text)),
                timestamp=fragment.timestamp,
            )
            for index, item in enumerate(payload.get("events", [{"text": fragment.text}]))
        ]
        edges = [
            GraphEdge(
                head=str(item["head"]),
                relation=str(item.get("relation", "related_to")),
                tail=str(item["tail"]),
            )
            for item in payload.get("relations", [])
            if "head" in item and "tail" in item
        ]
        return events, edges


class LlmRetrievalPlanner:
    """Agentic sub-query decomposition."""

    def __init__(self) -> None:
        self._llm = create_chat_model()

    def plan(self, query: str) -> list[SubQuery]:
        """Decompose query into scoped sub-queries."""
        system = SystemMessage(
            content=(
                "Return ONLY JSON: {\"subqueries\": [{\"text\": str, \"scope\": "
                "\"SHORT\"|\"LONG\"|\"MIXED\"}]}"
            )
        )
        raw = str(self._llm.invoke([system, HumanMessage(content=query)]).content)
        payload = _parse_json_object(raw)
        subqueries: list[SubQuery] = []
        for item in payload.get("subqueries", [{"text": query, "scope": "MIXED"}]):
            scope_raw = str(item.get("scope", "MIXED")).upper()
            scope = MemoryScope(scope_raw if scope_raw in MemoryScope else MemoryScope.MIXED)
            subqueries.append(SubQuery(text=str(item.get("text", query)), scope=scope))
        return subqueries or [SubQuery(text=query, scope=MemoryScope.MIXED)]


class LlmSynthesizer:
    """Final answer generation from ranked evidence."""

    def __init__(self) -> None:
        self._llm = create_chat_model()

    def synthesize(self, query: str, evidence_texts: list[str]) -> str:
        """Synthesize answer using evidence only."""
        if not evidence_texts:
            return ""
        evidence_block = "\n".join(f"- {text}" for text in evidence_texts)
        prompt = (
            f"Question: {query}\n\nEvidence:\n{evidence_block}\n\n"
            "Answer concisely using only the evidence."
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return str(getattr(response, "content", response)).strip()


def _parse_json_object(raw: str) -> dict:
    """Extract JSON object from model output."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        logger.exception("Failed to parse LLM JSON")
        return {}
