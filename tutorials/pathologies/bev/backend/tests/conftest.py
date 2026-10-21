"""Pytest fixtures for the BEV package."""

from __future__ import annotations

from pathlib import Path

import pytest
import wandb


@pytest.fixture
def minimal_wandb_run(tmp_path: Path) -> Path:
    """Create a tiny offline W&B run and return the ``run-*.wandb`` path."""
    wandb_dir = tmp_path / "wandb"
    wandb.init(
        mode="offline",
        dir=str(wandb_dir),
        project="bev_test",
        name="minimal",
        id="bevminimal1",
        reinit=True,
    )
    wandb.log({"train/loss": 1.0, "train/grad_norm/layer.0": 0.5})
    wandb.log(
        {
            "train/loss": 0.5,
            "train/grad_norm/layer.0": 0.25,
            "train/grad_norm/layer.1": 0.1,
        }
    )
    wandb.finish()
    return next(wandb_dir.rglob("*.wandb"))
