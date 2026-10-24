#!/usr/bin/env bash
# End-to-end LLM smoke for all qlab agent harnesses (uses agents/.env + OPENAI_MODEL).
set -euo pipefail

AGENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${AGENTS_DIR}"

if [[ ! -f .env ]]; then
  echo "Missing agents/.env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

export LANGFUSE_TRACING_ENABLED="${LANGFUSE_TRACING_ENABLED:-false}"
export OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-default}"
export MEMGRAPHRAG_AGENTS=llm
export COVE_AGENTS=llm
export HMEM_AGENTS=llm

docker network inspect agents_shared >/dev/null 2>&1 || docker network create agents_shared

PASS=0
FAIL=0

record() {
  local label=$1
  shift
  if "$@"; then
    echo "[PASS] ${label}"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] ${label}"
    FAIL=$((FAIL + 1))
  fi
}

wait_health() {
  local url="$1"
  for _ in $(seq 1 90); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

reharness_llm() {
  ./reharness/run-agent.sh --detach
  wait_health "http://localhost:8080/health"
  curl -sf http://localhost:8080/health | jq -e '.dependencies.qdrant == "ok"' >/dev/null
  python3 - <<'PY'
import json
import sys
import urllib.request

req = urllib.request.Request(
    "http://localhost:8080/tasks/stream",
    data=json.dumps({"task": "Reply with one word: ready."}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    saw_llm = False
    for line in resp:
        line = line.decode().strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("type") in ("llm_message", "llm_delta", "interrupt"):
            saw_llm = True
        if event.get("type") == "interrupt":
            break
    if not saw_llm:
        sys.exit("no LLM events in stream")
PY
  ./reharness/run-agent.sh stop
}

hipporag_llm() {
  ./hipporag/run-agent.sh --detach
  wait_health "http://localhost:8081/health"
  curl -sf -X POST http://localhost:8081/ask \
    -H 'Content-Type: application/json' \
    -d '{"query":"Which Stanford professor works on Alzheimer'\''s?","top_k":3}' \
    | jq -e '.answer | length > 0' >/dev/null
  ./hipporag/run-agent.sh stop
}

memgraphrag_llm() {
  ./memgraphrag/run-agent.sh --detach
  wait_health "http://localhost:8082/health"
  curl -sf -X POST http://localhost:8082/corpus/load-toy >/dev/null
  curl -sf -X POST http://localhost:8082/index | jq -e '.facts >= 1' >/dev/null
  curl -sf -X POST http://localhost:8082/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"query":"When was Isaac Newton born?","top_k":2}' \
    | jq -e '.hits | length >= 1' >/dev/null
  ./memgraphrag/run-agent.sh stop
}

cove_llm() {
  ./cove/run-agent.sh --detach
  wait_health "http://localhost:8083/health"
  curl -sf -X POST http://localhost:8083/verify \
    -H 'Content-Type: application/json' \
    -d '{"query":"When did the Mexican American War end?"}' \
    | jq -e '.final_response | length > 0' >/dev/null
  ./cove/run-agent.sh stop
}

hmem_llm() {
  ./hmem/run-agent.sh --detach
  wait_health "http://localhost:8084/health"
  curl -sf -X POST http://localhost:8084/corpus/bootstrap-toy >/dev/null
  curl -sf -X POST http://localhost:8084/ask \
    -H 'Content-Type: application/json' \
    -d '{"query":"What is the cat'\''s name?"}' \
    | jq -e '.answer | test("Mochi"; "i")' >/dev/null
  ./hmem/run-agent.sh stop
}

echo "Model: ${OPENAI_MODEL:-unknown}  OPENAI_TEMPERATURE=${OPENAI_TEMPERATURE}"

record reharness reharness_llm
record hipporag hipporag_llm
record memgraphrag memgraphrag_llm
record cove cove_llm
record hmem hmem_llm

echo ""
echo "LLM E2E summary: ${PASS} passed, ${FAIL} failed"
exit $(( FAIL > 0 ? 1 : 0 ))
