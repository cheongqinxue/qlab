#!/usr/bin/env python3
"""Export Adam ``exp_avg_sq`` heatmaps from pathology checkpoints to an HTML viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "tutorials"))

from pathologies.analysis.second_moment_export import export_viewer_data


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        required=True,
        help="Parent of checkpoint-step-* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Viewer directory (default: sibling second_moment_viewer under run .data).",
    )
    args = parser.parse_args()
    checkpoints_dir = args.checkpoints_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else checkpoints_dir.parent / "second_moment_viewer"
    )
    data_path = export_viewer_data(checkpoints_dir, output_dir)
    index_path = output_dir / "index.html"
    print(f"Wrote {data_path}")
    print(f"Open {index_path} via a local server, e.g.:")
    print(f"  cd {output_dir} && python -m http.server 8765")


if __name__ == "__main__":
    main()
