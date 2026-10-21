#!/usr/bin/env python3
"""Write the large transformer offline W&B run for manual ``bev serve`` testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from bev.fixtures.wandb_large_module_tree import (
    bootstrap_large_transformer_scan_root,
    fixture_summary,
)


def main() -> None:
    """Bootstrap scan root and print summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_root",
        type=Path,
        help="Directory to write wandb offline run (bev serve --root)",
    )
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=12)
    args = parser.parse_args()
    bootstrap_large_transformer_scan_root(
        args.scan_root,
        num_layers=args.layers,
        num_heads=args.heads,
    )
    summary = fixture_summary(args.layers, args.heads)
    print(f"Wrote scan root: {args.scan_root.resolve()}")
    print(f"Metric keys: {summary['total_keys']} ({summary['num_layers']} layers, {summary['num_heads']} heads)")
    print(f"  update_rank: {summary['update_rank_keys']}, attention heads: {summary['attention_head_keys']}")
    print(f"Serve: uv run python -m bev serve --root {args.scan_root}")


if __name__ == "__main__":
    main()
