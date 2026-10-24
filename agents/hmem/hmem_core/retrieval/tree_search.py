"""Tree-side bottom-up evidence collection (paper §3.3)."""

from __future__ import annotations

from hmem_core.models import Evidence, MemoryScope, SubQuery, TreeNode
from hmem_core.tree.temporal_semantic_tree import TemporalSemanticTree


class TreeSearcher:
    """Collect candidate evidence texts from the temporal tree."""

    def __init__(self, tree: TemporalSemanticTree) -> None:
        """Attach tree index."""
        self._tree = tree

    def collect(
        self,
        sub_query: SubQuery,
        *,
        allowed_fragment_ids: set[str] | None = None,
    ) -> list[Evidence]:
        """Gather evidence for one sub-query respecting SHORT/LONG/MIXED scope."""
        candidates = self._nodes_for_scope(sub_query.scope)
        items: list[Evidence] = []
        for node in candidates:
            if allowed_fragment_ids and node.fragment_id not in allowed_fragment_ids:
                continue
            items.append(self._to_evidence(node))
        return items

    def _nodes_for_scope(self, scope: MemoryScope) -> list[TreeNode]:
        if scope == MemoryScope.SHORT:
            return [self._tree.nodes[node_id] for node_id in self._tree.leaves]
        if scope == MemoryScope.LONG:
            return [node for node in self._tree.nodes.values() if node.is_summary]
        return list(self._tree.nodes.values())

    @staticmethod
    def _to_evidence(node: TreeNode) -> Evidence:
        return Evidence(
            text=node.text,
            fragment_id=node.fragment_id,
            event_id=node.node_id if not node.is_summary else None,
            start_time=node.start_time,
            end_time=node.end_time,
            consolidation_count=node.consolidation_count,
        )
