"""Synthetic large module-tree W&B runs for matrix-layout development."""

from __future__ import annotations

from pathlib import Path

import wandb

LARGE_RUN_ID = "bevlarge1"
LARGE_RUN_NAME = "large-transformer-sim"

# BERT-style block components (under bert.encoder.layer.{L}.)
LAYER_COMPONENTS = (
    "attention.self",
    "attention.output",
    "intermediate",
    "output",
)

# Parameter suffixes for rank metrics (full module path + suffix).
RANK_PARAM_SUFFIXES = (
    "query.weight",
    "key.weight",
    "value.weight",
    "dense.weight",
)

METRIC_FAMILIES = (
    "update_rank",
    "effective_rank",
    "activation_rms",
    "attention_max_weight",
)


def module_path(layer: int, component: str) -> str:
    """Torch-style module path segment under ``bert.encoder``."""
    return f"bert.encoder.layer.{layer}.{component}"


def iter_large_transformer_metric_keys(
    num_layers: int = 24,
    num_heads: int = 12,
    *,
    prefix: str = "train",
) -> list[str]:
    """Build metric keys for a simulated transformer logging hierarchy.

    Keys follow pathology conventions: ``{prefix}/{family}/{module.path}`` with
    optional ``/head.{H}`` for per-head attention metrics.

    Args:
        num_layers: Number of encoder layers (``layer.0`` … ``layer.{N-1}``).
        num_heads: Attention heads per layer for ``attention_max_weight``.
        prefix: Metric key prefix (typically ``train``).

    Returns:
        Sorted list of unique metric keys.
    """
    keys: set[str] = set()
    for layer in range(num_layers):
        for component in LAYER_COMPONENTS:
            base = module_path(layer, component)
            keys.add(f"{prefix}/activation_rms/{base}")
            for suffix in RANK_PARAM_SUFFIXES:
                keys.add(f"{prefix}/update_rank/{base}.{suffix}")
                keys.add(f"{prefix}/effective_rank/{base}.{suffix}")
            if component == "attention.self":
                for head in range(num_heads):
                    keys.add(f"{prefix}/attention_max_weight/{base}/head.{head}")
    keys.add(f"{prefix}/loss")
    return sorted(keys)


def metric_values_for_step(keys: list[str], step: int) -> dict[str, float]:
    """Deterministic scalar values for fixture runs (no randomness in CI)."""
    scale = 1.0 + step * 0.01
    out: dict[str, float] = {}
    for index, key in enumerate(keys):
        if key.endswith("/loss"):
            out[key] = 1.0 / scale
            continue
        # Spread values by key hash so cells differ visually.
        out[key] = ((index * 17 + step * 3) % 1000) / 1000.0 * scale
    return out


def bootstrap_large_transformer_scan_root(
    base: Path,
    *,
    num_layers: int = 24,
    num_heads: int = 12,
    steps: tuple[int, ...] = (0, 50, 100),
) -> Path:
    """Write an offline W&B run with a large module-tree metric catalog.

    Args:
        base: Writable directory; created if missing.
        num_layers: Encoder depth to simulate.
        num_heads: Heads per layer for attention metrics.
        steps: Training steps to log.

    Returns:
        Scan root path passed to ``bev serve --root``.
    """
    base.mkdir(parents=True, exist_ok=True)
    keys = iter_large_transformer_metric_keys(num_layers, num_heads)
    wandb_dir = base / "wandb"
    wandb.init(
        mode="offline",
        dir=str(wandb_dir),
        project="bev_large_sim",
        name=LARGE_RUN_NAME,
        id=LARGE_RUN_ID,
        reinit=True,
    )
    for step in steps:
        wandb.log(metric_values_for_step(keys, step), step=step)
    wandb.finish()
    return base


def fixture_summary(num_layers: int, num_heads: int) -> dict[str, int]:
    """Return expected counts for tests and docs."""
    keys = iter_large_transformer_metric_keys(num_layers, num_heads)
    per_layer_ranks = len(LAYER_COMPONENTS) * len(RANK_PARAM_SUFFIXES) * 2
    per_layer_rms = len(LAYER_COMPONENTS)
    per_layer_heads = num_heads
    return {
        "total_keys": len(keys),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "keys_per_layer_approx": per_layer_ranks + per_layer_rms + per_layer_heads,
        "update_rank_keys": sum(1 for k in keys if "/update_rank/" in k),
        "attention_head_keys": sum(1 for k in keys if "/attention_max_weight/" in k and "/head." in k),
    }
