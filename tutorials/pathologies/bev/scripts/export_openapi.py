#!/usr/bin/env python3
"""Write OpenAPI 3 schema for the BEV API to docs/openapi.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bev.server.app import create_app


def main() -> None:
    """Export ``docs/openapi.json`` from the FastAPI app (no scan root required for schema)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "openapi.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if output differs from committed file",
    )
    args = parser.parse_args()

    app = create_app(scan_root=Path("/tmp/bev-openapi-scan"), store_dir=Path("/tmp/bev-openapi-store"))
    schema = app.openapi()
    serialized = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.output.is_file():
            print(f"Missing {args.output}", file=sys.stderr)
            raise SystemExit(1)
        if args.output.read_text(encoding="utf-8") != serialized:
            print(f"Stale {args.output} — run: uv run python {__file__}", file=sys.stderr)
            raise SystemExit(1)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
