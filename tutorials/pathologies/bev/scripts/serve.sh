#!/usr/bin/env bash
# Serve BEV from the qlab repo root (or any cwd with tutorials/pathologies on disk).
set -euo pipefail
BEV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${BEV_ROOT}/../../.." && pwd)"
export PYTHONPATH="${BEV_ROOT}/backend${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --project "${REPO_ROOT}" python -m bev serve "$@"
