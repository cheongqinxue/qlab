#!/usr/bin/env bash
# Start MemGraphRAG agent and stream logs.
#
# Usage:
#   ./agents/memgraphrag/run-agent.sh
#   ./agents/memgraphrag/run-agent.sh --rebuild-image
#   ./agents/memgraphrag/run-agent.sh --detach
#   ./agents/memgraphrag/run-agent.sh stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_run_common.sh
source "${SCRIPT_DIR}/_run_common.sh"

if [[ "${1:-}" == "stop" ]]; then
  if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") stop" >&2
    exit 1
  fi
  stop_memgraphrag_stack
  exit 0
fi

REBUILD_IMAGE=0
FOREGROUND=0
DETACH=0

for arg in "$@"; do
  case "${arg}" in
    --rebuild-image) REBUILD_IMAGE=1 ;;
    --foreground) FOREGROUND=1 ;;
    --detach | -d) DETACH=1 ;;
    -h | --help)
      sed -n '2,9p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${FOREGROUND}" -eq 1 && "${DETACH}" -eq 1 ]]; then
  echo "Use either --foreground or --detach, not both." >&2
  exit 1
fi

load_memgraphrag_env
ensure_agents_runtime_image "${REBUILD_IMAGE}"
ensure_agents_shared_network

COMPOSE=(docker compose -f memgraphrag/docker-compose.yml)

if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "Starting MemGraphRAG in foreground…"
  echo "Agent URL: ${MEMGRAPHRAG_AGENT_URL}"
  cd "${AGENTS_DIR}"
  exec "${COMPOSE[@]}" up --remove-orphans
fi

echo "Starting MemGraphRAG (detached)…"
(cd "${AGENTS_DIR}" && "${COMPOSE[@]}" up -d --remove-orphans)
wait_for_memgraphrag_health

echo "Agent URL: ${MEMGRAPHRAG_AGENT_URL}"

if [[ "${DETACH}" -eq 1 ]]; then
  echo "Follow logs: cd ${AGENTS_DIR} && ${COMPOSE[*]} logs -f agent"
  exit 0
fi

follow_agent_logs
