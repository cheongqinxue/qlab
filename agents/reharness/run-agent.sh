#!/usr/bin/env bash
# Start ReHarness (Qdrant + agent) and stream agent logs.
#
# Prerequisites:
#   1. agents/.env
#   2. Langfuse: cd agents && docker compose -f docker-compose.langfuse.yml up -d
#
# Usage:
#   ./agents/reharness/run-agent.sh
#   ./agents/reharness/run-agent.sh --rebuild-image
#   ./agents/reharness/run-agent.sh --foreground   # docker compose up (all service logs)
#   ./agents/reharness/run-agent.sh --detach       # up -d, health check, exit (no log follow)
#   ./agents/reharness/run-agent.sh stop             # stop agent + qdrant

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_run_common.sh
source "${SCRIPT_DIR}/_run_common.sh"

if [[ "${1:-}" == "stop" ]]; then
  if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") stop" >&2
    exit 1
  fi
  stop_reharness_stack
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
      sed -n '2,14p' "${BASH_SOURCE[0]}"
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

load_reharness_env
ensure_agents_runtime_image "${REBUILD_IMAGE}"
ensure_agents_shared_network

COMPOSE=(docker compose -f reharness/docker-compose.yml)

if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "Starting ReHarness in foreground (docker compose logs to this terminal)…"
  echo "Agent URL: ${REHARNESS_AGENT_URL}"
  cd "${AGENTS_DIR}"
  exec "${COMPOSE[@]}" up --remove-orphans
fi

echo "Starting ReHarness stack (detached)…"
(cd "${AGENTS_DIR}" && "${COMPOSE[@]}" up -d --remove-orphans)
wait_for_agent_health

echo "Agent URL: ${REHARNESS_AGENT_URL}"

if [[ "${DETACH}" -eq 1 ]]; then
  echo "Follow logs: cd ${AGENTS_DIR} && ${COMPOSE[*]} logs -f agent"
  echo "Stack is running in the background."
  exit 0
fi

follow_agent_logs
