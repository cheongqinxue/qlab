#!/usr/bin/env bash
# Stop Docker Ollama started via docker-compose.ollama.yml only.
# Does not stop a system-wide `ollama serve` you started yourself.
#
# Usage:
#   ./agents/scripts/stop-ollama.sh

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

cd "${AGENTS_DIR}"
if docker_compose_ollama ps -q ollama 2>/dev/null | grep -q .; then
  echo "Stopping Docker Ollama…"
  docker_compose_ollama down --remove-orphans
else
  echo "Docker Ollama compose stack is not running."
fi
