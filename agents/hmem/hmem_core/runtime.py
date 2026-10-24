"""Runtime wiring for the H-MEM harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agents_utils.memory.base_encoder import BaseEncoder

from hmem_core.config import HMemConfig
from hmem_core.encoding.fragment_encoder import build_fragment_encoder
from hmem_core.hybrid_store import InMemoryHybridStore
from hmem_core.retrieval.engine import RetrievalEngine


@dataclass
class HMemRuntime:
    """Live hybrid memory state, encoder, and retrieval engine."""

    config: HMemConfig
    store: InMemoryHybridStore
    encoder: BaseEncoder
    agents_mode: str = "fixture"
    fixture_query: str | None = None
    fixture_answer: str | None = None
    engine: RetrievalEngine | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Build retrieval engine after dataclass init."""
        self.refresh_engine()

    def refresh_engine(self) -> None:
        """Rebuild retrieval engine after fixture QA or mode changes."""
        self.engine = RetrievalEngine(
            self.store,
            self.config,
            self.encoder,
            agents_mode=self.agents_mode,
            fixture_query=self.fixture_query,
            fixture_answer_text=self.fixture_answer,
        )


def build_runtime(
    config: HMemConfig,
    *,
    agents_mode: str = "fixture",
    encoder: BaseEncoder | None = None,
) -> HMemRuntime:
    """Create runtime from configuration.

    Args:
        config: Parsed JSON config.
        agents_mode: ``fixture`` or ``llm``.
        encoder: Optional test encoder override.

    Returns:
        Initialized runtime with empty hybrid store.
    """
    Path(config.memory_data_dir).mkdir(parents=True, exist_ok=True)
    store = InMemoryHybridStore(config=config)
    enc = build_fragment_encoder(override=encoder)
    return HMemRuntime(
        config=config,
        store=store,
        encoder=enc,
        agents_mode=agents_mode,
    )
