"""Regenerate the committed Playwright scan root under ``fixtures/data/e2e_minimal``."""

from __future__ import annotations

from pathlib import Path

import wandb

E2E_RUN_ID = "beve2e1"
E2E_RUN_NAME = "e2e-minimal"

ATTENTION_HEAD_KEYS = (
    "train/attention_max_weight/bert.encoder.layer.0.attention.self/head.0",
    "train/attention_max_weight/bert.encoder.layer.0.attention.self/head.1",
    "train/attention_max_weight/bert.encoder.layer.1.attention.self/head.0",
    "train/attention_max_weight/bert.encoder.layer.1.attention.self/head.1",
)


def bootstrap_e2e_scan_root(base: Path) -> Path:
    """Write a small offline run under ``base`` and return the scan root.

    The scan root is the directory that contains a ``wandb/`` folder (see
    :func:`~bev.discovery.discover_wandb_runs`).

    Args:
        base: Writable directory; created if missing.

    Returns:
        Scan root path passed to ``bev serve --root``.
    """
    base.mkdir(parents=True, exist_ok=True)
    wandb_dir = base / "wandb"
    wandb.init(
        mode="offline",
        dir=str(wandb_dir),
        project="bev_e2e",
        name=E2E_RUN_NAME,
        id=E2E_RUN_ID,
        reinit=True,
    )
    wandb.log(
        {
            "train/loss": 1.0,
            "train/grad_norm/layer.0": 0.5,
            **{key: 0.26 for key in ATTENTION_HEAD_KEYS},
        },
        step=0,
    )
    wandb.log(
        {
            "train/loss": 0.5,
            "train/grad_norm/layer.0": 0.25,
            "train/grad_norm/layer.1": 0.1,
            **{key: 0.31 for key in ATTENTION_HEAD_KEYS},
        },
        step=20,
    )
    wandb.finish()
    return base
