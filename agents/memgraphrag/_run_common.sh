# Shared helpers for memgraphrag/run-agent.sh (source, do not execute).

MEMGRAPHRAG_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
AGENTS_DIR="$(cd "${MEMGRAPHRAG_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${AGENTS_DIR}/.." && pwd)"

load_memgraphrag_env() {
  if [[ ! -f "${AGENTS_DIR}/.env" ]]; then
    echo "Missing ${AGENTS_DIR}/.env — copy from .env-template and fill in values." >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1091
  source "${AGENTS_DIR}/.env"
  set +a
  MEMGRAPHRAG_AGENT_PORT="${MEMGRAPHRAG_AGENT_PORT:-8082}"
  MEMGRAPHRAG_AGENT_URL="${MEMGRAPHRAG_AGENT_URL:-http://localhost:${MEMGRAPHRAG_AGENT_PORT}}"
  AGENTS_RUNTIME_IMAGE="${AGENTS_RUNTIME_IMAGE:-agents-runtime:latest}"
}

ensure_agents_runtime_image() {
  local rebuild="${1:-0}"
  if [[ "${rebuild}" -eq 1 ]]; then
    echo "Building shared image ${AGENTS_RUNTIME_IMAGE} …"
    "${AGENTS_DIR}/scripts/build-image.sh"
    return
  fi
  if ! docker image inspect "${AGENTS_RUNTIME_IMAGE}" >/dev/null 2>&1; then
    echo "Docker image ${AGENTS_RUNTIME_IMAGE} not found." >&2
    echo "Build it once: ${AGENTS_DIR}/scripts/build-image.sh" >&2
    echo "Or pass --rebuild-image to build and start." >&2
    exit 1
  fi
}

ensure_agents_shared_network() {
  if docker network inspect agents_shared >/dev/null 2>&1; then
    return 0
  fi
  "${AGENTS_DIR}/scripts/start-langfuse.sh"
}

wait_for_memgraphrag_health() {
  local url="${MEMGRAPHRAG_AGENT_URL}/health"
  echo "Waiting for agent health at ${url} …"
  local attempt
  for attempt in $(seq 1 60); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      echo "Agent is up."
      return 0
    fi
    sleep 2
  done
  echo "Agent did not become healthy in time." >&2
  echo "Check logs: cd ${AGENTS_DIR} && docker compose -f memgraphrag/docker-compose.yml logs agent" >&2
  return 1
}

follow_agent_logs() {
  echo "Streaming agent logs from Docker (Ctrl+C stops follow; containers keep running)…"
  cd "${AGENTS_DIR}"
  exec docker compose -f memgraphrag/docker-compose.yml logs -f --tail=40 agent
}

stop_memgraphrag_stack() {
  echo "Stopping MemGraphRAG agent…"
  cd "${AGENTS_DIR}"
  docker compose -f memgraphrag/docker-compose.yml down --remove-orphans
  echo "MemGraphRAG stack stopped."
}
