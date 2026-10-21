#!/usr/bin/env python3
"""Write Llama 70B-scale offline W&B run for BEV stress testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from bev.fixtures.paths import LLAMA70B_UPDATE_RANK_SCAN_ROOT
from bev.fixtures.wandb_llama70b_scale import (
    LLAMA70B_DEFAULT_CHECKPOINT_STEPS,
    LLAMA70B_NUM_KV_HEADS,
    LLAMA70B_NUM_LAYERS,
    LLAMA70B_NUM_Q_HEADS,
    bootstrap_llama70b_scale_scan_root,
    llama70b_fixture_summary,
)


def main() -> None:
    """Bootstrap scan root and print summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_root",
        type=Path,
        nargs="?",
        default=LLAMA70B_UPDATE_RANK_SCAN_ROOT,
        help="Directory to write wandb offline run (default: fixtures/data/llama70b_update_rank_scan)",
    )
    parser.add_argument("--layers", type=int, default=LLAMA70B_NUM_LAYERS)
    parser.add_argument("--q-heads", type=int, default=LLAMA70B_NUM_Q_HEADS)
    parser.add_argument("--kv-heads", type=int, default=LLAMA70B_NUM_KV_HEADS)
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=len(LLAMA70B_DEFAULT_CHECKPOINT_STEPS),
        help="Number of checkpoint steps (0, 100, 200, ...)",
    )
    args = parser.parse_args()
    steps = tuple(step * 100 for step in range(args.checkpoints))
    bootstrap_llama70b_scale_scan_root(
        args.scan_root,
        num_layers=args.layers,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        steps=steps,
    )
    summary = llama70b_fixture_summary(
        args.layers,
        args.q_heads,
        args.kv_heads,
        num_checkpoints=len(steps),
    )
    print(f"Wrote scan root: {args.scan_root.resolve()}")
    print(
        f"update_rank keys per checkpoint: {summary['update_rank_keys']} "
        f"({summary['num_layers']} layers, "
        f"q_heads={summary['num_q_heads']}, kv_heads={summary['num_kv_heads']})"
    )
    print(
        f"checkpoints: {summary['checkpoints']} → "
        f"{summary['scalar_events']:,} scalar events in run file"
    )
    print(
        "Serve (from repo root):\n"
        f"  PYTHONPATH=tutorials/pathologies/bev/backend uv run python -m bev serve "
        f"--root {args.scan_root}"
    )


if __name__ == "__main__":
    main()
