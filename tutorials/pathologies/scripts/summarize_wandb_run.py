#!/usr/bin/env python3
"""Print a one-line W&B summary for a pathology run (metric column counts)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tutorials"))


def main() -> None:
    """Fetch run summary keys and print a compact report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    import wandb

    entity = args.entity or os.environ.get("WANDB_ENTITY", "")
    project = args.project or os.environ.get("WANDB_PROJECT", "pathologies")
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": args.run_name},
        order="-created_at",
        per_page=1,
    )
    run = next(iter(runs), None)
    if run is None:
        print(f"SUMMARY: {args.run_name} NOT_FOUND", file=sys.stderr)
        sys.exit(1)

    summary = dict(run.summary)
    train_keys = sorted(k for k in summary if k.startswith("train/"))
    custom = [
        k
        for k in train_keys
        if any(
            part in k
            for part in (
                "logit_entropy",
                "adam_second_moment",
                "adam_max_r",
                "adam_p99_r",
                "adam_low_moment_frac",
                "adam_max_g2_over_v",
                "adam_frac_r_above_threshold",
                "effective_rank",
                "update_rank",
                "activation_rms",
                "attention_entropy",
            )
        )
    ]
    state = run.state
    steps = summary.get("_step", summary.get("train/global_step", "?"))
    loss = summary.get("train/loss", summary.get("train_loss", "?"))
    logit_ent = summary.get(
        "train/logit_entropy",
        summary.get("train/entropy", "—"),
    )
    print(
        f"SUMMARY: {args.run_name} state={state} steps={steps} "
        f"train_cols={len(train_keys)} custom_telemetry={len(custom)} "
        f"loss={loss} logit_entropy={logit_ent}"
    )


if __name__ == "__main__":
    main()
