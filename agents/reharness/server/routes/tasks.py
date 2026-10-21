"""Task submission and qualitative feedback endpoints."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from reharness_utils.message_content import extract_visible_assistant_text
from server.streaming import stream_graph_events

from agents_utils.observability.langfuse_tracing import (
    build_graph_invoke_config,
    flush_langfuse,
    langfuse_task_session,
)

from server.app_state import AppState

router = APIRouter(prefix="/tasks", tags=["tasks"])


class CreateTaskRequest(BaseModel):
    """Start a new streamed task instance."""

    task: str = Field(min_length=1, max_length=32_000)


class CreateTaskResponse(BaseModel):
    """Identifiers and status after starting a task."""

    task_id: str
    status: str
    assistant_message: str | None = None
    interrupt: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    """Qualitative feedback in response to the agent's final question."""

    feedback: str = Field(min_length=1, max_length=16_000)


class TaskStatusResponse(BaseModel):
    """Task status snapshot."""

    task_id: str
    status: str
    assistant_message: str | None = None
    harness_complete: bool = False


def _graph(request: Request) -> Any:
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    return state.graph


def _assistant_prompt(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = extract_visible_assistant_text(message.content)
            return text or None
    return None


def _status_from_snapshot(values: dict[str, Any]) -> str:
    if values.get("harness_complete"):
        return "complete"
    if values.get("__interrupt__"):
        return "awaiting_feedback"
    if values.get("user_feedback") and not values.get("harness_complete"):
        return "updating_harness"
    return "running"


@router.post("/stream")
async def create_task_stream(body: CreateTaskRequest, request: Request) -> StreamingResponse:
    """Stream NDJSON events until the agent requests qualitative feedback."""
    task_id = uuid.uuid4().hex
    graph = _graph(request)
    initial: dict[str, Any] = {
        "messages": [HumanMessage(content=body.task)],
        "task_text": body.task,
        "retrieved": [],
    }
    config = build_graph_invoke_config(task_id)

    async def event_generator():
        yield json.dumps({"type": "task", "task_id": task_id}) + "\n"
        try:
            with langfuse_task_session(task_id=task_id, task_preview=body.task):
                async for line in stream_graph_events(graph, initial, config=config):
                    yield line
        finally:
            flush_langfuse()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.post("/{task_id}/feedback/stream")
async def submit_feedback_stream(
    task_id: str,
    body: FeedbackRequest,
    request: Request,
) -> StreamingResponse:
    """Stream harness completion after qualitative feedback."""
    graph = _graph(request)
    config = build_graph_invoke_config(task_id)
    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    if not snapshot.interrupts:
        raise HTTPException(status_code=409, detail="Task is not awaiting feedback")

    task_preview = str(snapshot.values.get("task_text", ""))

    async def event_generator():
        try:
            with langfuse_task_session(task_id=task_id, task_preview=task_preview):
                async for line in stream_graph_events(
                    graph, Command(resume=body.feedback), config=config
                ):
                    yield line
        finally:
            flush_langfuse()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.post("", response_model=CreateTaskResponse)
async def create_task(body: CreateTaskRequest, request: Request) -> CreateTaskResponse:
    """Run retrieval and the task agent until qualitative feedback is requested."""
    task_id = uuid.uuid4().hex
    graph = _graph(request)
    initial: dict[str, Any] = {
        "messages": [HumanMessage(content=body.task)],
        "task_text": body.task,
        "retrieved": [],
    }
    with langfuse_task_session(task_id=task_id, task_preview=body.task):
        result = await graph.ainvoke(initial, config=build_graph_invoke_config(task_id))
    flush_langfuse()
    interrupt_payload = None
    interrupts = result.get("__interrupt__")
    if interrupts:
        first = interrupts[0]
        interrupt_payload = first.value if hasattr(first, "value") else first
    status = "awaiting_feedback" if interrupt_payload else _status_from_snapshot(result)
    return CreateTaskResponse(
        task_id=task_id,
        status=status,
        assistant_message=_assistant_prompt(result.get("messages", [])),
        interrupt=interrupt_payload if isinstance(interrupt_payload, dict) else None,
    )


@router.post("/{task_id}/feedback", response_model=TaskStatusResponse)
async def submit_feedback(
    task_id: str,
    body: FeedbackRequest,
    request: Request,
) -> TaskStatusResponse:
    """Resume after the agent's feedback question; runs harness revision + write."""
    graph = _graph(request)
    config = build_graph_invoke_config(task_id)
    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    if not snapshot.interrupts and snapshot.values.get("harness_complete"):
        raise HTTPException(status_code=409, detail="Task already complete")

    if not snapshot.interrupts:
        raise HTTPException(status_code=409, detail="Task is not awaiting feedback")

    task_preview = str(snapshot.values.get("task_text", ""))
    with langfuse_task_session(task_id=task_id, task_preview=task_preview):
        result = await graph.ainvoke(Command(resume=body.feedback), config=config)
    flush_langfuse()
    status = _status_from_snapshot(result)
    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        assistant_message=_assistant_prompt(result.get("messages", [])),
        harness_complete=bool(result.get("harness_complete")),
    )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str, request: Request) -> TaskStatusResponse:
    """Return checkpointed task status."""
    graph = _graph(request)
    config = build_graph_invoke_config(task_id)
    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    values = dict(snapshot.values)
    if snapshot.interrupts:
        values["__interrupt__"] = True
    return TaskStatusResponse(
        task_id=task_id,
        status=_status_from_snapshot(values),
        assistant_message=_assistant_prompt(values.get("messages", [])),
        harness_complete=bool(values.get("harness_complete")),
    )
