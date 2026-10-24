"""QA route (retrieve + LLM reader)."""

from __future__ import annotations

import os

from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from server.app_state import AppState
from server.openie_factory import get_openie_extractor
from server.schemas import AskOut

router = APIRouter(tags=["ask"])


class AskIn(BaseModel):
    """Question request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/ask", response_model=AskOut)
async def ask(body: AskIn, request: Request) -> AskOut:
    """Retrieve supporting passages and generate an answer."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    extractor = get_openie_extractor()
    entities = extractor.extract_query_entities(body.query)
    hits = state.runtime.retriever.retrieve(
        query=body.query,
        query_entities=entities,
        top_k=body.top_k,
    )
    context = "\n\n".join(f"[{h.passage_id}] {h.text}" for h in hits)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    answer = llm.invoke(
        [
            SystemMessage(
                content="Answer using only the provided passages. Be concise.",
            ),
            HumanMessage(content=f"Passages:\n{context}\n\nQuestion: {body.query}"),
        ]
    )
    content = answer.content
    answer_text = content if isinstance(content, str) else str(content)
    return AskOut(query=body.query, answer=answer_text, hits=hits)
