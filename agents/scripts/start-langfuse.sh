#!/usr/bin/env bash
# Start self-hosted Langfuse (creates agents_shared network). Idempotent.
#
# Usage (from anywhere):
#   ./agents/scripts/start-langfuse.sh

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

load_agents_env
LANGFUSE_UI_PORT="${LANGFUSE_UI_PORT:-3000}"

if langfuse_is_running; then
  echo "Langfuse already running (langfuse-web)."
  exit 0
fi

echo "Starting Langfuse stack…"
cd "${AGENTS_DIR}"
docker_compose_langfuse up -d

if ! wait_for_http "http://127.0.0.1:${LANGFUSE_UI_PORT}/api/public/health" "Langfuse" 90 3; then
  wait_for_http "http://127.0.0.1:${LANGFUSE_UI_PORT}" "Langfuse UI" 30 2
fi

echo "Langfuse UI: http://127.0.0.1:${LANGFUSE_UI_PORT}"
