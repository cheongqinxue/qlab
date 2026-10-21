"""Build the LangGraph task + ReHarness postamble.

Cross-task learning is via Qdrant reflections (retrieve → system prompt), not a
mutable context manager that accumulates prior tasks in the LLM window. Each task
uses a fresh ``thread_id`` and initial ``messages``; the checkpointer only
supports interrupt/resume within that instance.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from harness.prompts import DECISION_SYSTEM_TEMPLATE
from harness.reharness import ReHarness
from reharness_utils.message_content import extract_visible_assistant_text

from .state import TaskGraphState


def _serialize_retrieved(items: list[Any]) -> list[dict[str, Any]]:
    return [
        {"id": item.id, "similarity": item.similarity, "content": item.content}
        for item in items
    ]


def build_task_graph(
    *,
    llm: BaseLanguageModel,
    reharness: ReHarness,
    tools: list[Any],
    checkpointer: BaseCheckpointSaver,
) -> Any:
    """Compile the task graph with retrieval, ReAct loop, feedback interrupt, harness.

    Args:
        llm: Chat model for the decision agent.
        reharness: Harness for retrieve + post-instance update.
        tools: LangChain tools (e.g. ``run_python``).
        checkpointer: Persisted checkpoints for interrupt/resume.

    Returns:
        Compiled ``StateGraph``.
    """
    model_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    async def retrieve_node(state: TaskGraphState) -> dict[str, Any]:
        if state.get("retrieved"):
            return {}
        task = state["task_text"]
        items = await reharness.retrieve(task)
        return {"retrieved": _serialize_retrieved(items)}

    def _system_content(state: TaskGraphState) -> str:
        from agents_utils.memory.base_memory import BaseMemoryItem

        raw = state.get("retrieved") or []
        items = [
            BaseMemoryItem(id=r["id"], similarity=r["similarity"], content=r["content"])
            for r in raw
        ]
        block = reharness.reflections_for_prompt(items)
        return DECISION_SYSTEM_TEMPLATE.format(reflections_block=block)

    async def agent_node(state: TaskGraphState) -> dict[str, Any]:
        # Instance-local ReAct history only — do not use AgentContextManager here;
        # past tasks must appear only as retrieved reflections, not full transcripts.
        system = SystemMessage(content=_system_content(state))
        response = await model_with_tools.ainvoke([system, *state["messages"]])
        return {"messages": [response]}

    def route_after_agent(
        state: TaskGraphState,
    ) -> Literal["tools", "await_feedback", "agent"]:
        messages = state["messages"]
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if isinstance(last, AIMessage) and not last.tool_calls:
            if state.get("user_feedback"):
                return "agent"
            return "await_feedback"
        return "agent"

    async def await_feedback_node(state: TaskGraphState) -> dict[str, Any]:
        last = state["messages"][-1]
        question = extract_visible_assistant_text(last.content) if isinstance(last, AIMessage) else ""
        feedback = interrupt(
            {
                "kind": "qualitative_feedback",
                "prompt": question,
            }
        )
        from langchain_core.messages import HumanMessage

        return {"messages": [HumanMessage(content=str(feedback))], "user_feedback": str(feedback)}

    async def harness_node(state: TaskGraphState) -> dict[str, Any]:
        from agents_utils.memory.base_memory import BaseMemoryItem

        feedback = state.get("user_feedback") or ""
        raw = state.get("retrieved") or []
        retrieved = [
            BaseMemoryItem(id=r["id"], similarity=r["similarity"], content=r["content"])
            for r in raw
        ]
        await reharness.after_instance(
            task=state["task_text"],
            messages=_conversation_without_system(state["messages"]),
            feedback=feedback,
            retrieved=retrieved,
        )
        return {"harness_complete": True}

    def after_feedback_route(state: TaskGraphState) -> Literal["harness", "end"]:
        if state.get("harness_complete"):
            return "end"
        if state.get("user_feedback"):
            return "harness"
        return "end"

    graph = StateGraph(TaskGraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("await_feedback", await_feedback_node)
    graph.add_node("harness", harness_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "await_feedback": "await_feedback",
            "agent": "agent",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "await_feedback",
        after_feedback_route,
        {"harness": "harness", "end": END},
    )
    graph.add_edge("harness", END)

    return graph.compile(checkpointer=checkpointer)


def _conversation_without_system(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if not isinstance(m, SystemMessage)]
