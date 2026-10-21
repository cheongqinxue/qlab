"""Tests for metric key path tree (mirrors web/src/lib/metricKeyTree.js)."""

from __future__ import annotations


def build_metric_key_tree(keys: list[str]) -> list[dict]:
    sorted_keys = sorted(set(keys))
    root: dict = {"children": {}}

    for key in sorted_keys:
        parts = [part for part in key.split("/") if part]
        path_parts: list[str] = []
        cursor = root
        for index, segment in enumerate(parts):
            path_parts.append(segment)
            path = "/".join(path_parts)
            if segment not in cursor["children"]:
                cursor["children"][segment] = {
                    "segment": segment,
                    "path": path,
                    "key": None,
                    "children": {},
                }
            node = cursor["children"][segment]
            if index == len(parts) - 1:
                node["key"] = key
            cursor = node

    def to_array(map_node: dict) -> list[dict]:
        children = map_node["children"]
        return [
            {
                "segment": node["segment"],
                "path": node["path"],
                "key": node["key"],
                "children": to_array(node),
            }
            for _, node in sorted(children.items(), key=lambda item: item[0])
        ]

    return to_array(root)


def collect_branch_paths(nodes: list[dict]) -> list[str]:
    paths: list[str] = []

    def walk(items: list[dict]) -> None:
        for node in items:
            if node["children"]:
                paths.append(node["path"])
                walk(node["children"])

    walk(nodes)
    return paths


class TestBuildMetricKeyTree:
    def test_groups_by_prefix(self) -> None:
        tree = build_metric_key_tree(["train/loss", "train/acc", "val/loss"])
        assert [node["segment"] for node in tree] == ["train", "val"]
        assert tree[0]["children"][0]["key"] == "train/acc"

    def test_prefix_and_leaf_key(self) -> None:
        tree = build_metric_key_tree(["train", "train/loss"])
        train = tree[0]
        assert train["key"] == "train"
        assert train["children"][0]["key"] == "train/loss"


class TestCollectBranchPaths:
    def test_nested_paths(self) -> None:
        tree = build_metric_key_tree(["a/b/c", "a/b/d", "x/y"])
        assert collect_branch_paths(tree) == ["a", "a/b", "x"]
