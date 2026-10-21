"""ReHarness orchestrator: retrieve, revise, forward write."""

from __future__ import annotations

import logging

from agents_utils.memory.base_memory import BaseMemory, BaseMemoryItem
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage

from .models import (
    MemoryAgentOutput,
    ReflectionPayload,
    RevisionAgentOutput,
    payload_from_memory_content,
)
from .prompts import (
    MEMORY_AGENT_PROMPT,
    REVISION_AGENT_PROMPT,
    format_reflections_block,
)
from .trajectory import messages_to_trajectory_text

logger = logging.getLogger(__name__)


class ReHarness:
    """Cross-task reflection harness backed by ``BaseMemory``."""

    def __init__(
        self,
        memory: BaseMemory,
        *,
        llm: BaseLanguageModel,
        top_k: int = 5,
    ) -> None:
        """Configure harness with memory bank and shared LLM.

        Args:
            memory: Vector memory for reflections.
            llm: Chat model for πM and πR.
            top_k: Retrieval count for new instances.
        """
        self._memory = memory
        self._llm = llm
        self._top_k = top_k
        self._revision_llm = llm.with_structured_output(RevisionAgentOutput)
        self._memory_llm = llm.with_structured_output(MemoryAgentOutput)

    @property
    def top_k(self) -> int:
        """Retrieval limit per instance."""
        return self._top_k

    async def retrieve(self, task: str) -> list[BaseMemoryItem]:
        """Forward read: top-k reflections for the current task.

        Args:
            task: Current user task text.

        Returns:
            Ranked memory items.
        """
        items = await self._memory.afind(query=task)
        return items[: self._top_k]

    def reflections_for_prompt(self, items: list[BaseMemoryItem]) -> str:
        """Build the decision-agent reflections block."""
        ranked: list[tuple[int, str]] = []
        for index, item in enumerate(items, start=1):
            payload = payload_from_memory_content(item.content)
            ranked.append((index, payload.reflection))
        return format_reflections_block(ranked)

    async def after_instance(
        self,
        *,
        task: str,
        messages: list[BaseMessage],
        feedback: str,
        retrieved: list[BaseMemoryItem],
        revision_demos: list[str] | None = None,
    ) -> None:
        """Backward revision on retrieved items, then forward write.

        Args:
            task: User task for this instance.
            messages: Full instance conversation (including feedback human message).
            feedback: Qualitative user feedback text.
            retrieved: Items retrieved at instance start.
            revision_demos: Optional revised reflection texts for πM demos.
        """
        trajectory = messages_to_trajectory_text(messages)
        demos = revision_demos or []

        for item in retrieved:
            revised = await self._revise_reflection(
                memory_id=item.id,
                item=item,
                current_task=task,
                current_trajectory=trajectory,
                current_feedback=feedback,
            )
            await self._memory.adelete(item.id)
            old = payload_from_memory_content(item.content)
            new_payload = ReflectionPayload(
                reflection=revised.revised_reflection,
                guidance=revised.guidance or old.guidance,
                source_task=old.source_task,
                trajectory_text=old.trajectory_text,
                user_feedback=old.user_feedback,
            )
            await self._memory.add(
                new_payload.model_dump_json(),
                metadata={"kind": "reflection"},
            )
            demos.append(new_payload.reflection)

        memory_out = await self._write_reflection(
            task=task,
            trajectory=trajectory,
            feedback=feedback,
            demonstrations=demos,
        )
        payload = ReflectionPayload(
            reflection=memory_out.reflection,
            guidance=None,
            source_task=task,
            trajectory_text=trajectory,
            user_feedback=feedback,
        )
        await self._memory.add(
            payload.model_dump_json(),
            metadata={"kind": "reflection"},
        )
        logger.info(
            "ReHarness updated bank after task (revised=%d, wrote=1)", len(retrieved)
        )

    async def _revise_reflection(
        self,
        *,
        memory_id: str,
        item: BaseMemoryItem,
        current_task: str,
        current_trajectory: str,
        current_feedback: str,
    ) -> RevisionAgentOutput:
        payload = payload_from_memory_content(item.content)

        prompt = REVISION_AGENT_PROMPT.format(
            memory_id=memory_id,
            reflection=payload.reflection,
            guidance=payload.guidance or "",
            source_trajectory=payload.trajectory_text,
            current_task=current_task,
            current_feedback=current_feedback,
            current_trajectory=current_trajectory,
        )
        result = await self._revision_llm.ainvoke([HumanMessage(content=prompt)])
        if not isinstance(result, RevisionAgentOutput):
            msg = f"Expected RevisionAgentOutput, got {type(result)!r}"
            raise TypeError(msg)
        return result

    async def _write_reflection(
        self,
        *,
        task: str,
        trajectory: str,
        feedback: str,
        demonstrations: list[str],
    ) -> MemoryAgentOutput:
        demo_block = "\n\n".join(
            f"[DEMONSTRATION {i}]\n{text}"
            for i, text in enumerate(demonstrations[:5], start=1)
        )
        prompt = MEMORY_AGENT_PROMPT.format(
            demonstrations=demo_block or "(none)",
            task=task,
            feedback=feedback,
            trajectory=trajectory,
        )
        result = await self._memory_llm.ainvoke([HumanMessage(content=prompt)])
        if not isinstance(result, MemoryAgentOutput):
            msg = f"Expected MemoryAgentOutput, got {type(result)!r}"
            raise TypeError(msg)
        return result
