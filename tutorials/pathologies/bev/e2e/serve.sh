#!/usr/bin/env bash
# Start BEV for Playwright (see e2e/playwright.config.js).
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
BEV_ROOT="$(cd "${E2E_ROOT}/.." && pwd)"
QLAB_ROOT="$(cd "${BEV_ROOT}/../../.." && pwd)"
export PYTHONPATH="${BEV_ROOT}/backend"
export BEV_E2E_PORT="${BEV_E2E_PORT:-18765}"

bash "${BEV_ROOT}/scripts/ensure-fixtures.sh"

if [[ ! -f "${BEV_ROOT}/static/index.html" ]]; then
  echo "Building BEV UI (static/)…" >&2
  (cd "${BEV_ROOT}/webui" && npm install && npm run build)
fi

cd "${QLAB_ROOT}"
exec uv run python "${E2E_ROOT}/serve.py" --port "$BEV_E2E_PORT"
