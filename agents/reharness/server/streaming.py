"""Map LangGraph ``astream_events`` to SSE payloads for the UI."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, ToolMessage

from reharness_utils.message_content import extract_reasoning_summary, extract_visible_assistant_text


def _json_line(event_type: str, payload: dict[str, Any]) -> str:
    return json.dumps({"type": event_type, **payload}, default=str) + "\n"


def _extract_ai_text(message: AIMessage) -> str:
    return extract_visible_assistant_text(message.content)


def _reasoning_from_ai(message: AIMessage) -> str | None:
    return extract_reasoning_summary(message.content)


def _stream_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        )
    return ""


async def stream_graph_events(
    graph: Any,
    input_payload: dict[str, Any] | Any,
    *,
    config: dict[str, Any],
) -> AsyncIterator[str]:
    """Stream newline-delimited JSON events from a graph run.

    Args:
        graph: Compiled LangGraph.
        input_payload: Initial state or ``Command`` resume payload.
        config: Runnable config (thread id, callbacks).

    Yields:
        JSON lines for the Streamlit / SSE client.
    """
    async for event in graph.astream_events(input_payload, config=config, version="v2"):
        kind = event.get("event")
        name = event.get("name")
        data = event.get("data") or {}

        if kind == "on_chain_start" and name == "retrieve":
            yield _json_line("node_start", {"node": "retrieve"})
        elif kind == "on_chain_end" and name == "retrieve":
            output = data.get("output") or {}
            retrieved = output.get("retrieved") or []
            yield _json_line(
                "retrieve",
                {"count": len(retrieved), "ids": [r.get("id") for r in retrieved]},
            )
        elif kind == "on_chat_model_start":
            yield _json_line("llm_start", {"node": name})
        elif kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk is not None:
                text = _stream_chunk_text(chunk)
                if text:
                    yield _json_line("llm_delta", {"text": text})
        elif kind == "on_chat_model_end":
            output = data.get("output")
            if isinstance(output, AIMessage):
                reasoning = _reasoning_from_ai(output)
                if reasoning:
                    yield _json_line("thinking", {"text": reasoning})
                text = _extract_ai_text(output)
                tool_calls = [
                    {"name": tc.get("name"), "args": tc.get("args")}
                    for tc in (output.tool_calls or [])
                ]
                yield _json_line(
                    "llm_message",
                    {"text": text, "tool_calls": tool_calls},
                )
        elif kind == "on_tool_start":
            yield _json_line(
                "tool_start",
                {"tool": name, "input": data.get("input")},
            )
        elif kind == "on_tool_end":
            output = data.get("output")
            if isinstance(output, ToolMessage):
                content = output.content if isinstance(output.content, str) else str(output.content)
            else:
                content = str(output)
            yield _json_line("tool_end", {"tool": name, "output": content})
        elif kind == "on_chain_start" and name == "await_feedback":
            yield _json_line("node_start", {"node": "await_feedback"})
        elif kind == "on_chain_start" and name == "harness":
            yield _json_line("node_start", {"node": "harness"})
        elif kind == "on_chain_end" and name == "harness":
            yield _json_line("harness_complete", {})

    snapshot = await graph.aget_state(config)
    if snapshot and snapshot.interrupts:
        interrupt = snapshot.interrupts[0]
        value = interrupt.value if hasattr(interrupt, "value") else interrupt
        yield _json_line("interrupt", {"payload": value})
    elif snapshot and snapshot.values.get("harness_complete"):
        yield _json_line("complete", {"status": "complete"})
