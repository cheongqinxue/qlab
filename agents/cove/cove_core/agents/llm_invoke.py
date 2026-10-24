"""Shared LangChain invoke helpers with optional Langfuse callbacks."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agents_utils.observability.langfuse_tracing import (
    create_langfuse_callback_handler,
    langfuse_task_session,
)


def invoke_text(
    model: BaseChatModel,
    *,
    system: str,
    human: str,
    session_id: str,
) -> str:
    """Run a single chat turn and return trimmed text content."""
    with langfuse_task_session(task_id=session_id, task_preview=human):
        config = _langchain_config()
        response = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)],
            config=config,
        )
    return _message_text(response)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def langchain_invoke_config() -> dict[str, Any]:
    """Optional Langfuse callbacks for LangChain ``invoke``."""
    handler = create_langfuse_callback_handler()
    return {"callbacks": [handler]} if handler is not None else {}


def _langchain_config() -> dict[str, Any]:
    return langchain_invoke_config()
