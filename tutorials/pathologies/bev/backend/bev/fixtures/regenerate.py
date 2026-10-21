"""CLI to refresh committed fixture data under ``fixtures/data/``."""

from __future__ import annotations

import argparse
from pathlib import Path

from bev.fixtures.paths import E2E_MINIMAL_NAME, LLAMA70B_UPDATE_RANK_NAME, scan_root
from bev.fixtures.wandb_e2e import bootstrap_e2e_scan_root
from bev.fixtures.wandb_llama70b_scale import bootstrap_llama70b_scale_scan_root


def main() -> None:
    """Regenerate one or more fixture scan roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        choices=(E2E_MINIMAL_NAME, LLAMA70B_UPDATE_RANK_NAME),
        help="Which committed dataset to rewrite",
    )
    parser.add_argument("--checkpoints", type=int, default=None, help="Llama only")
    args = parser.parse_args()
    target = scan_root(args.fixture)

    if args.fixture == E2E_MINIMAL_NAME:
        bootstrap_e2e_scan_root(target)
    else:
        checkpoints = args.checkpoints if args.checkpoints is not None else 1
        steps = tuple(step * 100 for step in range(checkpoints))
        bootstrap_llama70b_scale_scan_root(target, steps=steps)

    print(f"Wrote {target.resolve()}")


if __name__ == "__main__":
    main()
