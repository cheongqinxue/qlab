"""LangGraph state for task instances."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class TaskGraphState(TypedDict):
    """State for one streamed task instance."""

    messages: Annotated[list, add_messages]
    task_text: str
    retrieved: NotRequired[list[dict[str, Any]]]
    user_feedback: NotRequired[str]
    harness_complete: NotRequired[bool]
