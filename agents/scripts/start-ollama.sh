#!/usr/bin/env bash
# Ensure Ollama API is reachable. Idempotent.
#
# Modes (OLLAMA_START_MODE):
#   auto   — use running API; else native `ollama` if installed; else Docker compose
#   host   — only check / start native ollama serve
#   docker — only docker-compose.ollama.yml
#
# Usage:
#   ./agents/scripts/start-ollama.sh

set -euo pipefail

# shellcheck source=_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

load_agents_env
MODE="${OLLAMA_START_MODE:-auto}"

if ollama_is_running; then
  echo "Ollama API already up: $(ollama_api_url)"
  exit 0
fi

start_docker_ollama() {
  echo "Starting Ollama via Docker…"
  cd "${AGENTS_DIR}"
  docker_compose_ollama up -d
  wait_for_http "$(ollama_api_url)/api/tags" "Ollama" 60 2
}

start_host_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    return 1
  fi
  echo "Starting native ollama serve in background…"
  nohup ollama serve >/dev/null 2>&1 &
  wait_for_http "$(ollama_api_url)/api/tags" "Ollama" 30 1
}

case "${MODE}" in
  docker)
    start_docker_ollama
    ;;
  host)
    start_host_ollama || {
      echo "Could not start host Ollama. Install from https://ollama.com or set OLLAMA_START_MODE=docker." >&2
      exit 1
    }
    ;;
  auto)
    if start_host_ollama 2>/dev/null; then
      :
    else
      start_docker_ollama
    fi
    ;;
  *)
    echo "Unknown OLLAMA_START_MODE=${MODE} (use auto|host|docker)" >&2
    exit 1
    ;;
esac

echo "Ollama API: $(ollama_api_url)/v1"
echo "Pull a model, e.g.: ollama pull qwen2.5:7b-instruct"
