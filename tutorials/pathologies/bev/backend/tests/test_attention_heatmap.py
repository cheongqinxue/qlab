"""Tests for attention heatmap parsing (mirrors web/src/lib/attentionHeatmap.js)."""

from __future__ import annotations

import re


DEFAULT_ATTENTION_HEATMAP_REGEX = r"train/attention_max_weight/.+/head\.\d+$"


def parse_attention_max_weight_key(key: str) -> dict[str, int | str] | None:
    match = re.match(r"^train/attention_max_weight/(.+)/head\.(\d+)$", key)
    if not match:
        return None
    module_path = match.group(1)
    head = int(match.group(2))
    layer_match = re.search(r"(?:^|\.)layer\.(\d+)(?:\.|$)", module_path)
    if layer_match is None:
        return None
    layer = int(layer_match.group(1))
    return {"layer": layer, "head": head, "modulePath": module_path}


def heatmap_cell_keys_matrix(layout: dict) -> list[list[str]]:
    layers = layout["layers"]
    heads = layout["heads"]
    key_by_cell = layout["keyByCell"]
    return [
        [key_by_cell.get(f"{layer}:{head}", "") for head in heads] for layer in layers
    ]


def build_attention_heatmap_layout(keys: list[str]) -> dict:
    key_by_cell: dict[str, str] = {}
    for key in keys:
        parsed = parse_attention_max_weight_key(key)
        if not parsed:
            continue
        key_by_cell[f"{parsed['layer']}:{parsed['head']}"] = key
    layers = sorted({int(cell.split(":")[0]) for cell in key_by_cell})
    heads = sorted({int(cell.split(":")[1]) for cell in key_by_cell})
    return {"layers": layers, "heads": heads, "keyByCell": key_by_cell}


def value_at_step(points: list[tuple[float, float]], step: float) -> float | None:
    if not points:
        return None
    best = None
    best_dist = float("inf")
    for x, y in points:
        if x == step:
            return y
        dist = abs(x - step)
        if dist < best_dist:
            best_dist = dist
            best = y
    return best


def test_parse_attention_key() -> None:
    key = "train/attention_max_weight/bert.encoder.layer.0.attention.self/head.1"
    parsed = parse_attention_max_weight_key(key)
    assert parsed == {
        "layer": 0,
        "head": 1,
        "modulePath": "bert.encoder.layer.0.attention.self",
    }


def test_parse_rejects_non_head_keys() -> None:
    assert parse_attention_max_weight_key("train/loss") is None
    assert parse_attention_max_weight_key("train/attention_max_weight/foo/_imbalance") is None


def test_build_layout_sorts_axes() -> None:
    keys = [
        "train/attention_max_weight/bert.encoder.layer.1.attention.self/head.0",
        "train/attention_max_weight/bert.encoder.layer.0.attention.self/head.1",
        "train/attention_max_weight/bert.encoder.layer.0.attention.self/head.0",
    ]
    layout = build_attention_heatmap_layout(keys)
    assert layout["layers"] == [0, 1]
    assert layout["heads"] == [0, 1]
    assert len(layout["keyByCell"]) == 3


def test_default_regex_matches_logged_keys() -> None:
    key = "train/attention_max_weight/bert.encoder.layer.2.attention.self/head.3"
    assert re.search(DEFAULT_ATTENTION_HEATMAP_REGEX, key)


HEATMAP_CELL_MIN_PX = 3
HEATMAP_CELL_MAX_PX = 30


def heatmap_square_cell_size_px(
    host_width_px: float,
    col_count: int,
    margin_left: int = 44,
    margin_right: int = 16,
) -> float:
    if col_count < 1:
        return HEATMAP_CELL_MIN_PX
    plot_budget = max(0.0, host_width_px - margin_left - margin_right)
    fit = plot_budget / col_count
    clamped = min(HEATMAP_CELL_MAX_PX, fit)
    return max(HEATMAP_CELL_MIN_PX, clamped)


def test_heatmap_cell_size_clamped() -> None:
    assert heatmap_square_cell_size_px(400, 4) == 30.0
    assert heatmap_square_cell_size_px(80, 20) == HEATMAP_CELL_MIN_PX
    assert heatmap_square_cell_size_px(200, 10) == 14.0


def test_value_at_step_exact_and_nearest() -> None:
    points = [(10.0, 0.5), (20.0, 0.8)]
    assert value_at_step(points, 20.0) == 0.8
    assert value_at_step(points, 19.0) == 0.8
    assert value_at_step(points, 5.0) == 0.5
