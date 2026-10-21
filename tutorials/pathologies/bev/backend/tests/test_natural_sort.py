"""Tests for metric key natural sort (mirrors web/src/lib/naturalSort.js)."""

from __future__ import annotations

import re


def _token_sort_key(token: str) -> list[object]:
    parts = re.split(r"(\d+)", token)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def compare_path_tokens(a: str, b: str) -> int:
    tokens_a = a.split(".")
    tokens_b = b.split(".")
    max_len = max(len(tokens_a), len(tokens_b))
    for index in range(max_len):
        ka = _token_sort_key(tokens_a[index] if index < len(tokens_a) else "")
        kb = _token_sort_key(tokens_b[index] if index < len(tokens_b) else "")
        if ka < kb:
            return -1
        if ka > kb:
            return 1
    return 0


def compare_metric_key_paths(a: str, b: str) -> int:
    parts_a = [part for part in a.split("/") if part]
    parts_b = [part for part in b.split("/") if part]
    max_len = max(len(parts_a), len(parts_b))
    for index in range(max_len):
        seg_a = parts_a[index] if index < len(parts_a) else ""
        seg_b = parts_b[index] if index < len(parts_b) else ""
        cmp = compare_path_tokens(seg_a, seg_b)
        if cmp != 0:
            return cmp
    return 0


def sort_metric_key_paths(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda key: [_token_sort_key(part) for part in key.split("/")])


class TestNaturalSort:
    def test_layer_index_order(self) -> None:
        keys = [
            "train/loss/model.layers.10",
            "train/loss/model.layers.2",
            "train/loss/model.layers.1",
        ]
        ordered = sort_metric_key_paths(keys)
        assert ordered == [
            "train/loss/model.layers.1",
            "train/loss/model.layers.2",
            "train/loss/model.layers.10",
        ]

    def test_compare_path_tokens(self) -> None:
        assert compare_path_tokens("layers.1", "layers.10") < 0
        assert compare_path_tokens("layers.2", "layers.10") < 0

    def test_layer_suffix_lexicographic_trap(self) -> None:
        """Plain string sort puts ``layer.10`` before ``layer.2``; natural sort must not."""
        keys = [
            "train/update_rank/model.layers.1.mlp",
            "train/update_rank/model.layers.10.mlp",
            "train/update_rank/model.layers.2.mlp",
        ]
        ordered = sort_metric_key_paths(keys)
        assert ordered == [
            "train/update_rank/model.layers.1.mlp",
            "train/update_rank/model.layers.2.mlp",
            "train/update_rank/model.layers.10.mlp",
        ]
