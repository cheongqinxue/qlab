"""Collapsed path tree for metric search previews."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_NUMERIC_SEGMENT = re.compile(r"^\d+$")
_INDEXED_SUFFIX = re.compile(r"^(?P<prefix>.*?)(?P<index>\d+)$")

DEFAULT_MAX_SUMMARY_NODES = 500


@dataclass
class _TrieNode:
    """Internal trie node keyed by path segment."""

    children: dict[str, _TrieNode] = field(default_factory=dict)
    key_count: int = 0


def build_search_summary_tree(
    keys: list[str],
    *,
    max_nodes: int = DEFAULT_MAX_SUMMARY_NODES,
) -> dict[str, Any]:
    """Build a collapsed segment tree covering all metric keys.

    Args:
        keys: Full metric keys matching the search scope.
        max_nodes: Cap on rendered nodes (remaining mass shown as ellipsis).

    Returns:
        JSON-serializable tree with ``label``, ``key_count``, ``children``,
        and optional ``ellipsis`` / ``collapsed`` flags.
    """
    root = _TrieNode()
    for key in keys:
        parts = [part for part in key.split("/") if part]
        if not parts:
            continue
        node = root
        node.key_count += 1
        for part in parts:
            node = node.children.setdefault(part, _TrieNode())
            node.key_count += 1

    budget = {"remaining": max_nodes}
    children = _render_children(root.children, budget)
    truncated = budget["remaining"] <= 0 and bool(root.children)
    return {
        "label": "matches",
        "key_count": len(keys),
        "children": children,
        "truncated": truncated,
        "max_nodes": max_nodes,
    }


def _render_children(
    children: dict[str, _TrieNode],
    budget: dict[str, int],
) -> list[dict[str, Any]]:
    if not children or budget["remaining"] <= 0:
        return []
    rows: list[dict[str, Any]] = []
    for label, node, key_count, collapsed in _collapse_children(children):
        if budget["remaining"] <= 0:
            break
        budget["remaining"] -= 1
        if collapsed:
            rows.append(
                {
                    "label": label,
                    "key_count": key_count,
                    "children": [],
                    "collapsed": True,
                }
            )
            continue
        child_payload = _render_children(node.children, budget)
        rows.append(
            {
                "label": label,
                "key_count": node.key_count,
                "children": child_payload,
            }
        )
    if budget["remaining"] <= 0 and len(rows) < len(_collapse_children(children)):
        hidden = sum(node.key_count for _, node, _, _ in _collapse_children(children)[len(rows) :])
        if hidden > 0:
            rows.append(
                {
                    "label": "…",
                    "key_count": hidden,
                    "children": [],
                    "ellipsis": True,
                }
            )
    return rows


def _collapse_children(
    children: dict[str, _TrieNode],
) -> list[tuple[str, _TrieNode, int, bool]]:
    """Return rows as label, node, key_count, is_collapsed_group."""
    if not children:
        return []

    numeric = [name for name in children if _NUMERIC_SEGMENT.fullmatch(name)]
    passthrough_names = sorted(name for name in children if name not in numeric)
    rows: list[tuple[str, _TrieNode, int, bool]] = []

    if len(numeric) >= 3:
        indices = sorted(int(name) for name in numeric)
        if indices[-1] - indices[0] + 1 == len(indices):
            total = sum(children[name].key_count for name in numeric)
            label = f"{{{indices[0]}..{indices[-1]}}} ({len(indices)} segments)"
            rows.append((label, children[numeric[0]], total, True))

    grouped: dict[str, list[tuple[int, str, _TrieNode]]] = {}
    loose: list[tuple[str, _TrieNode, int, bool]] = []
    for name in passthrough_names:
        match = _INDEXED_SUFFIX.match(name)
        if match is None:
            loose.append((name, children[name], children[name].key_count, False))
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        grouped.setdefault(prefix, []).append((index, name, children[name]))

    for prefix, entries in sorted(grouped.items()):
        if len(entries) >= 3:
            indices = sorted(entry[0] for entry in entries)
            if indices[-1] - indices[0] + 1 == len(indices):
                total = sum(entry[2].key_count for entry in entries)
                label = f"{prefix}{{{indices[0]}..{indices[-1]}}}"
                rows.append((label, entries[0][2], total, True))
                continue
        for _index, name, child in entries:
            loose.append((name, child, child.key_count, False))

    rows.extend(loose)
    rows.sort(key=lambda row: row[0])
    return rows
