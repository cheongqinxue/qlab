#!/usr/bin/env python3
"""Push the pathology-lab W&B workspace layout (sections + panels).

Run manually when you change ``wandb_workspace.py`` — not part of training.
Do not pass ``--new-view`` unless you intentionally want another saved view;
repeated saves with ``--new-view`` duplicate workspace tabs in the W&B UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tutorials"))

from pathologies.instrumentation.wandb_workspace import (  # noqa: E402
    DEFAULT_PROJECT,
    DEFAULT_WORKSPACE_NAME,
    save_pathology_workspace,
)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Create or update the Pathology lab workspace view in W&B (manual, not run by run.sh)."
        ),
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (default: WANDB_ENTITY or wandb login default).",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"W&B project (default: {DEFAULT_PROJECT}).",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_WORKSPACE_NAME,
        help=f"Workspace title (default: {DEFAULT_WORKSPACE_NAME!r}).",
    )
    parser.add_argument(
        "--new-view",
        action="store_true",
        help=(
            "Save as an additional workspace view (creates a new tab). "
            "Default is update-in-place for the named view."
        ),
    )
    args = parser.parse_args()

    workspace = save_pathology_workspace(
        entity=args.entity,
        project=args.project,
        name=args.name,
        as_new_view=args.new_view,
    )
    url = getattr(workspace, "url", None)
    if url:
        print(f"Workspace saved: {url}")
        print(
            "Open that URL (or pick workspace view 'Pathology lab' in the UI). "
            "The default project Charts page is not this layout."
        )
    else:
        print("Workspace saved.")


if __name__ == "__main__":
    main()
