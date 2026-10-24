#!/usr/bin/env bash
# Stop Langfuse stack (does not remove volumes).
#
# Usage:
#   ./agents/scripts/stop-langfuse.sh

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

cd "${AGENTS_DIR}"
echo "Stopping Langfuse…"
docker_compose_langfuse down --remove-orphans
echo "Langfuse stopped."
