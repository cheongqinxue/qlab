"""Tests for multipanel metric grouping (mirrors web/src/lib/metricTree.js)."""

from __future__ import annotations


def common_path_prefix(keys: list[str]) -> str:
    if not keys:
        return ""
    paths = [key.split("/") for key in keys]
    paths = [[part for part in path if part] for path in paths]
    min_len = min(len(path) for path in paths)
    prefix_parts: list[str] = []
    for index in range(min_len):
        segment = paths[0][index]
        if all(path[index] == segment for path in paths):
            prefix_parts.append(segment)
        else:
            break
    return "/".join(prefix_parts)


def build_group_node(keys: list[str]) -> dict:
    prefix = common_path_prefix(keys)
    prefix_len = len([part for part in prefix.split("/") if part])

    def relative_parts(key: str) -> list[str]:
        parts = [part for part in key.split("/") if part]
        return parts[prefix_len:]

    relatives = [relative_parts(key) for key in keys]
    if all(len(parts) <= 1 for parts in relatives):
        return {"title": prefix or keys[0], "chartKeys": keys, "children": []}

    by_head: dict[str, list[str]] = {}
    for key in keys:
        parts = relative_parts(key)
        head = parts[0] if parts else ""
        by_head.setdefault(head, []).append(key)

    children = []
    for head in sorted(by_head):
        bucket = by_head[head]
        child = build_group_node(bucket)
        if child["chartKeys"] and not child["children"]:
            children.append({**child, "title": head})
        elif child["title"] == common_path_prefix(bucket) and child["children"]:
            children.append({**child, "title": head})
        else:
            children.append(
                {"title": head, "chartKeys": child["chartKeys"], "children": child["children"]}
            )

    return {"title": prefix, "chartKeys": [], "children": children}


def build_multipanel_group_tree(keys: list[str]) -> dict:
    sorted_keys = sorted(set(keys))
    if not sorted_keys:
        return {"title": "", "chartKeys": [], "children": []}
    if len(sorted_keys) == 1:
        return {"title": sorted_keys[0], "chartKeys": sorted_keys, "children": []}
    return build_group_node(sorted_keys)


class TestBuildMultipanelGroupTree:
    def test_three_branches_under_common_t(self) -> None:
        tree = build_multipanel_group_tree(["t/a/0", "t/a/1", "t/b/3"])
        assert tree["title"] == "t"
        assert tree["chartKeys"] == []
        assert [child["title"] for child in tree["children"]] == ["a", "b"]
        assert tree["children"][0]["chartKeys"] == ["t/a/0", "t/a/1"]
        assert tree["children"][1]["chartKeys"] == ["t/b/3"]

    def test_shared_prefix_collapses_to_single_group(self) -> None:
        tree = build_multipanel_group_tree(["t/a/0", "t/a/1"])
        assert tree["title"] == "t/a"
        assert tree["chartKeys"] == ["t/a/0", "t/a/1"]
        assert tree["children"] == []


def extend_group_path_prefix(parent: str, group_title: str) -> str:
    if not group_title:
        return parent
    return f"{parent}/{group_title}" if parent else group_title


def leaf_chart_title(
    key: str,
    group_path_prefix: str,
    parent_path_prefix: str = "",
) -> str:
    def tail(path: str) -> str:
        segs = [part for part in path.split("/") if part]
        return segs[-1] if segs else path

    if not group_path_prefix:
        return tail(key)
    if key == group_path_prefix:
        if parent_path_prefix:
            parent_needle = f"{parent_path_prefix}/"
            if key.startswith(parent_needle):
                under_parent = key[len(parent_needle) :]
                if under_parent:
                    return under_parent
        return tail(key)
    needle = f"{group_path_prefix}/"
    if key.startswith(needle):
        return key[len(needle) :]
    return tail(key)


class TestLeafChartTitle:
    def test_strips_nested_group_path(self) -> None:
        assert leaf_chart_title("t/a/0", "t/a") == "0"
        assert leaf_chart_title("t/a/1", "t/a") == "1"

    def test_single_key_under_child_group_title(self) -> None:
        assert leaf_chart_title("t/b/3", "t/b") == "3"

    def test_metric_at_group_path_gets_label(self) -> None:
        assert leaf_chart_title("foo", "foo") == "foo"
        assert leaf_chart_title("foo/bar", "foo") == "bar"
        assert leaf_chart_title("train/a", "train/a") == "a"
        assert leaf_chart_title("train/a", "train/a", "train") == "a"
        assert leaf_chart_title("train/a/0", "train/a") == "0"
