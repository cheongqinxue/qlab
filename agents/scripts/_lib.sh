# Shared helpers for agents/scripts/* (source only, do not execute).

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${AGENTS_DIR}/.." && pwd)"

load_agents_env() {
  if [[ ! -f "${AGENTS_DIR}/.env" ]]; then
    echo "Missing ${AGENTS_DIR}/.env — copy from .env-template and fill in values." >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1091
  source "${AGENTS_DIR}/.env"
  set +a
}

wait_for_http() {
  local url="$1"
  local label="${2:-service}"
  local max_attempts="${3:-60}"
  local sleep_secs="${4:-2}"
  echo "Waiting for ${label} at ${url} …"
  local attempt
  for attempt in $(seq 1 "${max_attempts}"); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      echo "${label} is up."
      return 0
    fi
    sleep "${sleep_secs}"
  done
  echo "${label} did not become reachable in time." >&2
  return 1
}

docker_compose_langfuse() {
  docker compose -f "${AGENTS_DIR}/docker-compose.langfuse.yml" "$@"
}

docker_compose_ollama() {
  docker compose -f "${AGENTS_DIR}/docker-compose.ollama.yml" "$@"
}

langfuse_is_running() {
  docker_compose_langfuse ps --status running -q langfuse-web 2>/dev/null | grep -q .
}

ollama_api_url() {
  local host="${OLLAMA_HOST:-127.0.0.1}"
  local port="${OLLAMA_PORT:-11434}"
  echo "http://${host}:${port}"
}

ollama_is_running() {
  curl -sf "$(ollama_api_url)/api/tags" >/dev/null 2>&1
}

ensure_agents_shared_network() {
  if docker network inspect agents_shared >/dev/null 2>&1; then
    return 0
  fi
  echo "Docker network agents_shared missing — starting Langfuse stack…" >&2
  "${SCRIPTS_DIR}/start-langfuse.sh"
}
