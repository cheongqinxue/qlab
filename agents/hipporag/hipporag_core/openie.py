"""OpenIE extraction (LLM or fixed fixtures)."""

from __future__ import annotations

import json
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from hipporag_core.models import OpenIEExtraction, Triple
from hipporag_core.prompts import NER_PASSAGE_PROMPT, NER_QUERY_PROMPT, TRIPLE_PROMPT


class _NerResult(BaseModel):
    entities: list[str] = Field(default_factory=list)


class _TripleItem(BaseModel):
    subject: str
    predicate: str
    object: str


class _TripleResult(BaseModel):
    triples: list[_TripleItem] = Field(default_factory=list)


class OpenIEExtractor(Protocol):
    """Protocol for passage and query NER."""

    def extract_passage(self, passage_id: str, text: str) -> OpenIEExtraction: ...

    def extract_query_entities(self, query: str) -> list[str]: ...


class FixedOpenIEExtractor:
    """Deterministic OpenIE for tests and demos."""

    def __init__(self, by_passage: dict[str, OpenIEExtraction]) -> None:
        """Use pre-built extractions keyed by passage id."""
        self._by_passage = by_passage

    def extract_passage(self, passage_id: str, text: str) -> OpenIEExtraction:
        """Return fixture extraction or empty."""
        _ = text
        return self._by_passage.get(
            passage_id,
            OpenIEExtraction(passage_id=passage_id, entities=[], triples=[]),
        )

    def extract_query_entities(self, query: str) -> list[str]:
        """Heuristic NER for toy eval queries."""
        lowered = query.lower()
        entities: list[str] = []
        if "stanford" in lowered:
            entities.append("Stanford")
        if "professor" in lowered or "thomas" in lowered:
            entities.append("Professor Thomas")
        if "alzheimer" in lowered:
            entities.append("Alzheimer's disease")
        return entities


class LlmOpenIEExtractor:
    """Two-step OpenIE via structured LLM outputs (temp 0)."""

    def __init__(self, model: str) -> None:
        """Bind chat model name from env."""
        self._llm = ChatOpenAI(model=model, temperature=0)
        self._ner = self._llm.with_structured_output(_NerResult)
        self._triples = self._llm.with_structured_output(_TripleResult)

    def extract_passage(self, passage_id: str, text: str) -> OpenIEExtraction:
        """Run NER then triple extraction on one passage."""
        ner = self._ner.invoke(
            [
                SystemMessage(content=NER_PASSAGE_PROMPT),
                HumanMessage(content=text),
            ]
        )
        triple_msg = TRIPLE_PROMPT.format(
            passage=text,
            entities=json.dumps(ner.entities),
        )
        triple_out = self._triples.invoke(
            [
                SystemMessage(content=triple_msg),
                HumanMessage(content=text),
            ]
        )
        triples = [
            Triple(
                subject=item.subject,
                predicate=item.predicate,
                object=item.object,
                passage_id=passage_id,
            )
            for item in triple_out.triples
        ]
        return OpenIEExtraction(
            passage_id=passage_id,
            entities=list(ner.entities),
            triples=triples,
        )

    def extract_query_entities(self, query: str) -> list[str]:
        """Query NER step."""
        ner = self._ner.invoke(
            [
                SystemMessage(content=NER_QUERY_PROMPT),
                HumanMessage(content=query),
            ]
        )
        return list(ner.entities)


def openie_uses_fixture() -> bool:
    """True when OpenIE should use committed toy fixtures (no LLM cost)."""
    import os

    mode = os.environ.get("HIPPORAG_OPENIE", "fixture").strip().lower()
    if mode == "llm":
        return False
    if mode == "fixture":
        return True
    return os.environ.get("HIPPORAG_OPENIE_FIXTURE", "1") != "0"


def build_openie_extractor() -> OpenIEExtractor:
    """Factory: toy ``openie.json`` by default; LLM when ``HIPPORAG_OPENIE=llm``."""
    import os

    if openie_uses_fixture():
        from hipporag_core.eval_loader import load_toy_openie_fixtures, toy_multihop_dir

        return FixedOpenIEExtractor(load_toy_openie_fixtures(toy_multihop_dir()))
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return LlmOpenIEExtractor(model=model)
