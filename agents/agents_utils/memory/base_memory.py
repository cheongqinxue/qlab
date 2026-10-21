"""Memory bank backend — swappable storage (including vector-backed) implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Self


class BaseMemoryItem(BaseModel):
    """A single memory entry returned from search operations."""

    id: str = Field(description="Unique identifier for the stored item.")
    similarity: float = Field(
        description="Match score for this result (semantics are backend-defined)."
    )
    content: Any = Field(description="Stored payload in any serializable form.")


class BaseMemory(ABC):
    """Provider-agnostic memory bank for experiments.

    Implementations may use any persistence or vector index; this interface does not
    assume a particular database or embedding stack.
    """

    @abstractmethod
    async def add(
        self,
        content: Any,
        *,
        vector: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new memory item.

        Args:
            content: Primary payload for the item (any type the backend supports).
            vector: Optional embedding for similarity search (format is backend-defined).
            metadata: Optional structured fields stored with the item.

        Returns:
            Unique identifier for the stored item (used by ``adelete``).
        """

    @abstractmethod
    async def afind(
        self,
        *,
        query: str | None = None,
        vector: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[BaseMemoryItem]:
        """Retrieve items matching text, vector similarity, metadata filters, or a combination.

        Args:
            query: Optional text to match or embed for search (backend-defined).
            vector: Optional query embedding for similarity search.
            metadata: Optional filter criteria on stored metadata fields.

        Returns:
            Matching items with ids, similarity scores, and content.
        """

    @abstractmethod
    async def adelete(self, item_id: str) -> bool:
        """Remove a single item by id.

        Args:
            item_id: Identifier returned from ``add``.

        Returns:
            True if an item was removed, False if no item matched ``item_id``.
        """

    @abstractmethod
    async def aclear(self) -> None:
        """Remove all items from the memory store."""

    @classmethod
    @abstractmethod
    def build_or_restore(cls, config: Any, **kwargs: Any) -> Self:
        """Connect to a memory store using ``config`` (data lives in the backend volume).

        Args:
            config: Backend-specific construction settings (often loaded from a file).
            **kwargs: Backend-specific runtime dependencies (e.g. ``BaseEncoder`` instances).

        Returns:
            A ready-to-use memory instance.
        """
