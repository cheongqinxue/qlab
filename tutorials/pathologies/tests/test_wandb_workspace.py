"""Tests for programmatic W&B workspace layout (no network)."""

from __future__ import annotations

import wandb_workspaces.reports.v2 as wr

from pathologies.instrumentation.wandb_workspace import (
    STEP_METRIC,
    _line_colors_for_layer_metrics,
    build_pathology_workspace,
)


def test_pathology_workspace_sections() -> None:
    """Workspace should define manual sections with expected panel counts."""
    workspace = build_pathology_workspace(entity="test-entity", project="pathologies")
    assert workspace.auto_generate_panels is False
    assert len(workspace.sections) == 8
    section_names = [section.name for section in workspace.sections]
    assert section_names[0] == "Core training"
    assert section_names[-1] == "Activations"
    total_panels = sum(len(section.panels) for section in workspace.sections)
    assert total_panels >= 15


def test_pathology_workspace_uses_train_prefix() -> None:
    """Panels should target HF Trainer metric keys under train/."""
    workspace = build_pathology_workspace(entity="test-entity")
    core = workspace.sections[0]
    loss_panel = core.panels[0]
    assert isinstance(loss_panel, wr.LinePlot)
    assert loss_panel.y == ["train/loss"]
    assert loss_panel.x == STEP_METRIC


def test_per_layer_grad_panels_use_distinct_colors() -> None:
    """Layer bands should assign different line colors so one run is not all-green."""
    workspace = build_pathology_workspace(entity="test-entity")
    grad_section = workspace.sections[1]
    layer_panels = [p for p in grad_section.panels if p.title and "Per-layer grad norm" in p.title]
    assert len(layer_panels) == 3
    first = layer_panels[0]
    assert first.line_colors is not None
    assert first.line_colors["train/grad_norm/layer.0"] != first.line_colors["train/grad_norm/layer.1"]


def test_line_color_helper_supports_run_wildcard_keys() -> None:
    """W&B series overrides may use ``*:metric`` keys for any run."""
    colors = _line_colors_for_layer_metrics(["train/grad_norm/layer.0"])
    assert colors["train/grad_norm/layer.0"] == colors["*:train/grad_norm/layer.0"]
