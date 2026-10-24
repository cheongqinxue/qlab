"""OpenAI-compatible chat client shared by agent experiments."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def default_chat_model_name() -> str:
    """Resolve chat model from environment.

    Returns:
        Model name from ``OPENAI_MODEL``, default ``gpt-4o-mini``.
    """
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def create_chat_model(model: str | None = None) -> ChatOpenAI:
    """Create a temperature-0 chat model (uses ``OPENAI_API_KEY``).

    Args:
        model: Optional override; when omitted, uses :func:`default_chat_model_name`.

    Returns:
        Configured :class:`~langchain_openai.ChatOpenAI` instance.
    """
    return ChatOpenAI(model=model or default_chat_model_name(), temperature=0)
