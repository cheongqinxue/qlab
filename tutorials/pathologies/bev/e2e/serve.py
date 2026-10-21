#!/usr/bin/env python3
"""Start BEV against a committed offline W&B scan root for Playwright E2E."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import uvicorn

from bev.fixtures.paths import default_playwright_scan_root
from bev.server.app import create_app
from bev.server.watcher import ingest_all


def main() -> None:
    """Ingest fixture data and serve the API + static UI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("BEV_E2E_PORT", "18765")))
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=None,
        help="Override scan root (default: llama70b_update_rank_scan or BEV_SCAN_ROOT)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    scan_root = (args.scan_root or default_playwright_scan_root()).resolve()
    if not scan_root.is_dir():
        raise SystemExit(
            f"Scan root not found: {scan_root}\n"
            "Regenerate: tutorials/pathologies/bev/scripts/ensure-fixtures.sh"
        )

    store_dir = Path(tempfile.mkdtemp(prefix="bev-e2e-store-"))
    ingest_all(scan_root, store_dir)
    app = create_app(scan_root, store_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
