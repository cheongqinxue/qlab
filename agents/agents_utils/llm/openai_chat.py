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


def resolve_openai_temperature() -> float | None:
    """Resolve chat temperature from ``OPENAI_TEMPERATURE``.

    Returns:
        ``0.0`` when the variable is unset (legacy default for deterministic harness tests).
        ``None`` when set to ``default``, ``none``, or empty — omits temperature so the
        provider model default applies (required for some models e.g. gpt-6-luna).
        Otherwise parses a float.
    """
    raw = os.environ.get("OPENAI_TEMPERATURE")
    if raw is None:
        return 0.0
    normalized = raw.strip().lower()
    if normalized in ("", "default", "none"):
        return None
    return float(raw)


def create_chat_model(
    model: str | None = None,
    *,
    temperature: float | None | object = ...,
) -> ChatOpenAI:
    """Create a chat model (uses ``OPENAI_API_KEY``).

    Args:
        model: Optional override; when omitted, uses :func:`default_chat_model_name`.
        temperature: Explicit override. When omitted, uses :func:`resolve_openai_temperature`.

    Returns:
        Configured :class:`~langchain_openai.ChatOpenAI` instance.
    """
    name = model or default_chat_model_name()
    if temperature is ...:
        temp = resolve_openai_temperature()
    else:
        temp = temperature  # type: ignore[assignment]
    if temp is None:
        return ChatOpenAI(model=name)
    return ChatOpenAI(model=name, temperature=temp)
