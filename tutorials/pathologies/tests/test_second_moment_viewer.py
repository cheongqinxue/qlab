"""Tests for second-moment viewer export helpers."""

import torch

from pathologies.analysis.second_moment_export import (
    build_param_tree,
    downsample_2d,
    moment_heatmap,
)


def test_downsample_2d_reduces_large_matrix() -> None:
    """Pooling should cap heatmap dimensions."""
    matrix = torch.ones(800, 600)
    out = downsample_2d(matrix, max_side=400)
    assert out.shape[0] <= 400
    assert out.shape[1] <= 400


def test_moment_heatmap_1d_bias() -> None:
    """1D parameters become a single-row heatmap."""
    payload = moment_heatmap(torch.ones(128), max_side=400)
    assert payload["rows"] == 1
    assert payload["cols"] == 128


def test_build_param_tree_groups_by_prefix() -> None:
    """Navigation tree should nest dotted parameter names."""
    tree = build_param_tree(
        [
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.1.attention.self.query.weight",
        ]
    )
    assert tree[0]["label"] == "bert"
    encoder = tree[0]["children"][0]
    assert encoder["label"] == "encoder"
