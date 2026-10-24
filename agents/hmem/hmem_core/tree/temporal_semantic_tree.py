"""Temporal–semantic tree for memory evolution (paper §3.2)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from agents_utils.memory.dense_numpy import dense_cosine_similarity_numpy

from hmem_core.config import HMemConfig
from hmem_core.models import MemoryEvent, TreeNode

SummarizeFn = Callable[[list[str]], str]


@dataclass
class TemporalSemanticTree:
    """Hierarchical tree: leaves = events; internal nodes = consolidated summaries."""

    config: HMemConfig
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    leaves: list[str] = field(default_factory=list)
    _embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    _children: dict[str, list[str]] = field(default_factory=dict)

    def insert_event(self, event: MemoryEvent, *, embedding: list[float]) -> str:
        """Insert a leaf memory event with its embedding vector."""
        node = TreeNode(
            node_id=event.event_id,
            level=0,
            text=event.text,
            start_time=event.timestamp,
            end_time=event.timestamp,
            fragment_id=event.fragment_id,
            is_summary=False,
        )
        self.nodes[node.node_id] = node
        self._embeddings[node.node_id] = np.asarray(embedding, dtype=np.float64)
        self._children[node.node_id] = []
        self.leaves.append(node.node_id)
        return node.node_id

    def consolidate(self, summarizer: SummarizeFn) -> int:
        """Bottom-up consolidation across configured tree levels.

        Args:
            summarizer: Produces parent summary text from child texts.

        Returns:
            Number of summary nodes after consolidation.
        """
        if not self.leaves:
            return 0
        levels = min(self.config.tree_levels, len(self.config.similarity_thresholds))
        for level in range(levels):
            self._consolidate_level(level, summarizer)
        return self.summary_count

    def consolidate_pending(self) -> int:
        """Alias for consolidate with default join summarizer."""
        return self.consolidate(lambda texts: " | ".join(texts))

    def nodes_at_level(self, level: int) -> list[TreeNode]:
        """Return nodes at a given level sorted by start time."""
        items = [node for node in self.nodes.values() if node.level == level]
        items.sort(key=lambda node: node.start_time)
        return items

    def get_embedding(self, node_id: str) -> np.ndarray | None:
        """Return stored embedding for a node."""
        return self._embeddings.get(node_id)

    def _consolidate_level(self, level: int, summarizer: SummarizeFn) -> None:
        """Merge semantically similar nodes within temporal windows at ``level``."""
        alpha = self._threshold_for_level(level)
        beta = timedelta(seconds=self._window_seconds_for_level(level))
        nodes = self.nodes_at_level(level)
        if len(nodes) < 2:
            return
        windows = self._partition_time_windows(nodes, beta)
        for window_nodes in windows:
            self._cluster_merge(window_nodes, level, alpha, summarizer)

    def _partition_time_windows(
        self,
        nodes: list[TreeNode],
        beta: timedelta,
    ) -> list[list[TreeNode]]:
        """Group nodes whose intervals fall in the same β-sized window."""
        if not nodes:
            return []
        windows: list[list[TreeNode]] = []
        current: list[TreeNode] = [nodes[0]]
        window_start = nodes[0].start_time
        for node in nodes[1:]:
            if node.start_time - window_start <= beta:
                current.append(node)
            else:
                windows.append(current)
                current = [node]
                window_start = node.start_time
        windows.append(current)
        return windows

    def _cluster_merge(
        self,
        nodes: list[TreeNode],
        level: int,
        alpha: float,
        summarizer: SummarizeFn,
    ) -> None:
        """Greedy merge adjacent similar nodes in one temporal window."""
        if len(nodes) < 2:
            return
        merged_any = True
        while merged_any and len(nodes) >= 2:
            merged_any = False
            index = 0
            while index < len(nodes) - 1:
                left = nodes[index]
                right = nodes[index + 1]
                if self._can_merge(left, right, alpha):
                    parent = self._merge_pair(left, right, level, summarizer)
                    nodes = nodes[:index] + [parent] + nodes[index + 2 :]
                    merged_any = True
                else:
                    index += 1

    def _can_merge(self, left: TreeNode, right: TreeNode, alpha: float) -> bool:
        left_emb = self._embeddings.get(left.node_id)
        right_emb = self._embeddings.get(right.node_id)
        if left_emb is None or right_emb is None:
            return False
        sim = dense_cosine_similarity_numpy(left_emb, right_emb)
        return sim >= alpha

    def _merge_pair(
        self,
        left: TreeNode,
        right: TreeNode,
        level: int,
        summarizer: SummarizeFn,
    ) -> TreeNode:
        """Create summary parent for two siblings."""
        summary_text = summarizer([left.text, right.text])
        parent_id = f"sum_{uuid.uuid4().hex[:12]}"
        parent = TreeNode(
            node_id=parent_id,
            level=level + 1,
            text=summary_text,
            start_time=min(left.start_time, right.start_time),
            end_time=max(left.end_time, right.end_time),
            fragment_id=left.fragment_id or right.fragment_id,
            is_summary=True,
            child_ids=[left.node_id, right.node_id],
            consolidation_count=max(left.consolidation_count, right.consolidation_count) + 1,
        )
        self.nodes[parent_id] = parent
        self._children[parent_id] = [left.node_id, right.node_id]
        left_emb = self._embeddings[left.node_id]
        right_emb = self._embeddings[right.node_id]
        self._embeddings[parent_id] = (left_emb + right_emb) / 2.0
        if left.node_id in self.leaves:
            self.leaves.remove(left.node_id)
        if right.node_id in self.leaves:
            self.leaves.remove(right.node_id)
        return parent

    def _threshold_for_level(self, level: int) -> float:
        thresholds = self.config.similarity_thresholds
        if level < len(thresholds):
            return thresholds[level]
        return thresholds[-1]

    def _window_seconds_for_level(self, level: int) -> int:
        windows = self.config.time_window_seconds
        if level < len(windows):
            return windows[level]
        return windows[-1]

    @property
    def summary_count(self) -> int:
        """Number of summary (non-leaf) nodes."""
        return sum(1 for node in self.nodes.values() if node.is_summary)
