"""Default context manager with pressure callbacks for LangGraph agents."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage

from .base_context import BaseContextManager, ContextPressureEvent, PressureHandler, TokenCounter
from .token_counter import llm_token_counter


class AgentContextManager(BaseContextManager):
    """Mutable message buffer with context-limit pressure events.

    LangGraph nodes typically call ``add`` after each turn/tool result and pass
    ``get_context()`` into the chat model. The system message is fixed at construction
    and always included in ``get_context()``.

    Token counts use ``model.get_num_tokens_from_messages`` (not heuristics).
    """

    def __init__(
        self,
        *,
        model: BaseLanguageModel,
        system_message: str | SystemMessage,
        context_limit: int,
        threshold_ratio: float = 0.85,
        tools: Sequence[Any] | None = None,
        token_counter: TokenCounter | None = None,
        on_pressure: PressureHandler | None = None,
        max_pressure_rounds: int = 3,
    ) -> None:
        """Configure the context window and optional pressure handler.

        Args:
            model: LangChain chat model; used for accurate token counting via
                ``get_num_tokens_from_messages``.
            system_message: System prompt (always first in ``get_context()``).
            context_limit: Model context window size in tokens.
            threshold_ratio: Fire ``on_pressure`` when fill ratio reaches this value (0–1).
            tools: Tool schemas included in token counts when the model binds tools.
            token_counter: Override counting (defaults to ``llm_token_counter(model)``).
            on_pressure: Async or sync callback to trim/summarize/prune messages.
            max_pressure_rounds: Max handler invocations per ``add`` to avoid loops.
        """
        if context_limit <= 0:
            msg = "context_limit must be positive"
            raise ValueError(msg)
        if not 0.0 < threshold_ratio <= 1.0:
            msg = "threshold_ratio must be in (0, 1]"
            raise ValueError(msg)

        self._model = model
        self._tools = tools
        self._system = (
            system_message
            if isinstance(system_message, SystemMessage)
            else SystemMessage(content=system_message)
        )
        self._messages: list[BaseMessage] = []
        self._context_limit = context_limit
        self._threshold_ratio = threshold_ratio
        self._token_counter = token_counter or llm_token_counter(model, tools=tools)
        self._on_pressure = on_pressure
        self._max_pressure_rounds = max_pressure_rounds

    @property
    def model(self) -> BaseLanguageModel:
        """Chat model used for token counting."""
        return self._model

    @property
    def context_limit(self) -> int:
        """Configured context token limit."""
        return self._context_limit

    @property
    def threshold_ratio(self) -> float:
        """Fill ratio at which pressure handling runs."""
        return self._threshold_ratio

    @property
    def system_message(self) -> SystemMessage:
        """Current system message."""
        return self._system

    @property
    def conversation_messages(self) -> list[BaseMessage]:
        """Conversation tail (excluding system message)."""
        return list(self._messages)

    def token_count(self) -> int:
        """Token count for the full context via the chat model."""
        return self._token_counter(self.get_context())

    def fill_ratio(self) -> float:
        """Current fill ratio relative to ``context_limit``."""
        return self.token_count() / self._context_limit

    def get_context(self) -> list[BaseMessage]:
        """Return system message followed by conversation messages."""
        return [self._system, *self._messages]

    def set_system_message(self, system_message: str | SystemMessage) -> None:
        """Replace the system message.

        Args:
            system_message: New system prompt.
        """
        self._system = (
            system_message
            if isinstance(system_message, SystemMessage)
            else SystemMessage(content=system_message)
        )

    def set_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Replace conversation messages (system message unchanged).

        Args:
            messages: New tail messages; must not include ``SystemMessage``.
        """
        for message in messages:
            if isinstance(message, SystemMessage):
                msg = "Use set_system_message for system prompts"
                raise ValueError(msg)
        self._messages = list(messages)

    async def add(self, message: BaseMessage) -> None:
        """Append a message and invoke ``on_pressure`` when over threshold."""
        if isinstance(message, SystemMessage):
            msg = "Pass system_message to the constructor; do not add SystemMessage via add"
            raise ValueError(msg)
        self._messages.append(message)
        await self._run_pressure_handlers(last_added=message)

    async def _run_pressure_handlers(self, *, last_added: BaseMessage | None) -> None:
        if self._on_pressure is None:
            return
        for _ in range(self._max_pressure_rounds):
            count = self.token_count()
            ratio = count / self._context_limit
            if ratio < self._threshold_ratio:
                return
            event = ContextPressureEvent(
                manager=self,
                token_count=count,
                context_limit=self._context_limit,
                fill_ratio=ratio,
                threshold_ratio=self._threshold_ratio,
                last_added=last_added,
            )
            await self._invoke_handler(self._on_pressure, event)
            last_added = None

    @staticmethod
    async def _invoke_handler(handler: PressureHandler, event: ContextPressureEvent) -> None:
        result = handler(event)
        if inspect.isawaitable(result):
            await result
