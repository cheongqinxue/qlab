#!/usr/bin/env bash
# Start H-MEM agent and stream logs.
#
# Usage:
#   ./agents/hmem/run-agent.sh
#   ./agents/hmem/run-agent.sh --rebuild-image
#   ./agents/hmem/run-agent.sh --detach
#   ./agents/hmem/run-agent.sh stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_run_common.sh
source "${SCRIPT_DIR}/_run_common.sh"

if [[ "${1:-}" == "stop" ]]; then
  if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") stop" >&2
    exit 1
  fi
  stop_hmem_stack
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

load_hmem_env
ensure_agents_runtime_image "${REBUILD_IMAGE}"
ensure_agents_shared_network

COMPOSE=(docker compose -f hmem/docker-compose.yml)

cd "${AGENTS_DIR}"
"${COMPOSE[@]}" up -d --remove-orphans

if [[ "${DETACH}" -eq 1 ]]; then
  wait_for_hmem_health
  echo "Agent URL: ${HMEM_AGENT_URL}"
  exit 0
fi

if [[ "${FOREGROUND}" -eq 1 ]]; then
  "${COMPOSE[@]}" up --remove-orphans
  exit 0
fi

wait_for_hmem_health
echo "Agent URL: ${HMEM_AGENT_URL}"
follow_agent_logs
