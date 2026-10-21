#!/usr/bin/env bash
# Run Python unit tests and Playwright UI smoke tests for BEV.
set -euo pipefail

BEV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QLAB_ROOT="$(cd "$BEV_ROOT/../../.." && pwd)"

cd "$QLAB_ROOT"
echo "==> pytest (BEV backend)" >&2
uv run pytest tutorials/pathologies/bev/backend/tests tutorials/pathologies/bev/sdk/tests

echo "==> Playwright E2E (BEV)" >&2
cd "$BEV_ROOT/e2e"
npm install
npx playwright install chromium
npm test
