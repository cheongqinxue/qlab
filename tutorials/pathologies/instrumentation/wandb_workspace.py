"""Programmatic W&B workspace layout for pathology lab runs."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

if TYPE_CHECKING:
    from wandb_workspaces.workspaces.interface import Workspace

DEFAULT_PROJECT = "pathologies"
DEFAULT_WORKSPACE_NAME = "Pathology lab"
STEP_METRIC = "train/global_step"

# Distinct hues for per-layer series (single-run multi-metric panels default to one run color).
_LAYER_PALETTE = (
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
)


def _layer_color(layer_index: int) -> str:
    """Pick a stable color for ``layer.N`` metrics."""
    return _LAYER_PALETTE[layer_index % len(_LAYER_PALETTE)]


def _line_colors_for_layer_metrics(metric_keys: list[str]) -> dict[str, str]:
    """Build ``line_colors`` keys for per-layer train metrics (metric path and run wildcard)."""
    colors: dict[str, str] = {}
    for key in metric_keys:
        match = re.search(r"layer\.(\d+)$", key)
        layer_index = int(match.group(1)) if match else 0
        color = _layer_color(layer_index)
        colors[key] = color
        colors[f"*:{key}"] = color
    return colors


def _per_layer_bands(
    title_prefix: str,
    metric_stem: str,
    bands: tuple[tuple[int, int], ...] = ((0, 9), (10, 19), (20, 29)),
) -> list[wr.LinePlot]:
    """Line plots for ``train/{metric_stem}/layer.*`` split into layer bands with distinct colors."""
    panels: list[wr.LinePlot] = []
    for start, end in bands:
        metric_keys = [f"train/{metric_stem}/layer.{index}" for index in range(start, end + 1)]
        panels.append(
            wr.LinePlot(
                title=f"{title_prefix} (layers {start}–{end})",
                x=STEP_METRIC,
                y=metric_keys,
                line_colors=_line_colors_for_layer_metrics(metric_keys),
                legend_position="east",
                legend_template="{{metric}}",
            )
        )
    return panels


def _line(title: str, *metrics: str) -> wr.LinePlot:
    """Single panel comparing explicit metric keys across runs."""
    return wr.LinePlot(
        title=title,
        x=STEP_METRIC,
        y=[f"train/{metric}" for metric in metrics],
    )


def _regex(title: str, pattern: str) -> wr.LinePlot:
    """Single panel plotting all metrics whose keys match ``pattern`` (RE2)."""
    return wr.LinePlot(title=title, x=STEP_METRIC, metric_regex=pattern)


def build_pathology_workspace(
    *,
    entity: str,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_WORKSPACE_NAME,
) -> Workspace:
    """Build a manual W&B workspace with sections for pathology diagnostics.

    Metrics assume HuggingFace Trainer logging (``train/`` prefix) plus our
    callback keys from :mod:`pathologies.instrumentation.callbacks`.

    Args:
        entity: W&B entity (user or team).
        project: W&B project name.
        name: Workspace display name in the W&B UI.

    Returns:
        Unsaved :class:`~wandb_workspaces.workspaces.Workspace` instance.
    """
    return ws.Workspace(
        name=name,
        entity=entity,
        project=project,
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(
            group_by_prefix="last",
            ignore_outliers=True,
            max_runs=10,
        ),
        sections=[
            ws.Section(
                name="Core training",
                is_open=True,
                panels=[
                    _line("Loss", "loss"),
                    _line("Learning rate", "learning_rate"),
                ],
            ),
            ws.Section(
                name="Gradients",
                is_open=True,
                panels=[
                    _line("Global grad norm", "grad_norm", "grad_norm_max"),
                    *_per_layer_bands("Per-layer grad norm", "grad_norm"),
                    _line("Grad groups", "grad_norm/embedding", "grad_norm/encoder", "grad_norm/head"),
                    _line("Early vs late", "grad_norm/early_to_late"),
                ],
            ),
            ws.Section(
                name="Updates",
                is_open=True,
                panels=[
                    _line("Update / weight ratio", "update_to_weight_ratio"),
                    _line(
                        "Update ratio by block",
                        "update_ratio/embedding",
                        "update_ratio/encoder",
                        "update_ratio/head",
                    ),
                ],
            ),
            ws.Section(
                name="Attention",
                panels=[
                    _regex("Attention entropy", r"train/attention_entropy/.*"),
                    _regex("Attention max weight", r"train/attention_max_weight/.*"),
                    _regex("Head imbalance", r"train/attention_head_imbalance/.*"),
                ],
            ),
            ws.Section(
                name="Outputs",
                panels=[
                    _line("Logit entropy", "logit_entropy"),
                    _line("Token entropy (TRL SFT)", "entropy"),
                    _line("Embedding anisotropy", "embedding_anisotropy"),
                ],
            ),
            ws.Section(
                name="Rank",
                panels=[
                    _regex("Weight effective rank", r"train/effective_rank/.*"),
                    _regex("Update rank (ΔW)", r"train/update_rank/.*"),
                ],
            ),
            ws.Section(
                name="Adam",
                panels=_per_layer_bands("Adam second moment", "adam_second_moment"),
            ),
            ws.Section(
                name="Activations",
                panels=[
                    _regex("Activation RMS", r"train/activation_rms/.*"),
                ],
            ),
        ],
    )


def default_entity() -> str:
    """Resolve W&B entity from env or the logged-in default."""
    if entity := os.environ.get("WANDB_ENTITY"):
        return entity
    import wandb

    api = wandb.Api()
    if api.default_entity:
        return api.default_entity
    msg = "Set WANDB_ENTITY or run wandb login"
    raise RuntimeError(msg)


def save_pathology_workspace(
    *,
    entity: str | None = None,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_WORKSPACE_NAME,
    as_new_view: bool = False,
) -> Workspace:
    """Create or update the pathology lab workspace in W&B.

    Args:
        entity: W&B entity; inferred when omitted.
        project: W&B project name.
        name: Workspace display name.
        as_new_view: If True, call ``save_as_new_view`` instead of ``save``.

    Returns:
        Saved workspace (includes ``url`` when available).
    """
    resolved_entity = entity or default_entity()
    workspace = build_pathology_workspace(
        entity=resolved_entity,
        project=project,
        name=name,
    )
    if as_new_view:
        return workspace.save_as_new_view()
    return workspace.save()
