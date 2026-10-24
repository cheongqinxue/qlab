#!/usr/bin/env bash
# One-shot local test: shared services (if needed) → MemGraphRAG agent (no UI yet).
#
# Usage:
#   ./agents/memgraphrag/run.sh
#   ./agents/memgraphrag/run.sh --no-ollama

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_LANGFUSE=1
START_OLLAMA=1

for arg in "$@"; do
  case "${arg}" in
    --no-langfuse) START_LANGFUSE=0 ;;
    --no-ollama) START_OLLAMA=0 ;;
    -h | --help)
      sed -n '2,7p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${START_LANGFUSE}" -eq 1 ]]; then
  "${AGENTS_DIR}/scripts/start-langfuse.sh"
fi

if [[ "${START_OLLAMA}" -eq 1 ]]; then
  "${AGENTS_DIR}/scripts/start-ollama.sh"
fi

exec "${SCRIPT_DIR}/run-agent.sh" --detach
