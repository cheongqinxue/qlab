"""Context window management for LangGraph agent message lists."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage

PressureHandler = Callable[["ContextPressureEvent"], Awaitable[None] | None]
TokenCounter = Callable[[Sequence[BaseMessage]], int]
"""Counts tokens in a message list; prefer ``llm_token_counter`` + ``get_num_tokens_from_messages``."""


@dataclass
class ContextPressureEvent:
    """Emitted when estimated tokens cross the configured threshold.

    Handlers may trim, summarize, or rewrite messages via ``manager`` (e.g. verbose
    tool results) before the next model call.
    """

    manager: Any
    token_count: int
    context_limit: int
    fill_ratio: float
    threshold_ratio: float
    last_added: BaseMessage | None


class BaseContextManager(ABC):
    """Interface for agent conversation context (including system message)."""

    @abstractmethod
    async def add(self, message: BaseMessage) -> None:
        """Append a message and run pressure handling when over threshold.

        Args:
            message: Any LangChain message except ``SystemMessage`` (set at construction).
        """

    @abstractmethod
    def get_context(self) -> list[BaseMessage]:
        """Return the full message list passed to the model (system message first)."""
