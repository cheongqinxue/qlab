#!/usr/bin/env bash
# One-shot local test: shared services (if needed) → CoVe agent (no UI yet).
#
# Usage:
#   ./agents/cove/run.sh
#   ./agents/cove/run.sh --no-langfuse
#   ./agents/cove/run.sh --no-ollama
#   ./agents/cove/run.sh --rebuild-image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_LANGFUSE=1
START_OLLAMA=1
EXTRA_AGENT_ARGS=(--detach)

for arg in "$@"; do
  case "${arg}" in
    --no-langfuse) START_LANGFUSE=0 ;;
    --no-ollama) START_OLLAMA=0 ;;
    --rebuild-image) EXTRA_AGENT_ARGS=(--rebuild-image --detach) ;;
    -h | --help)
      sed -n '2,8p' "${BASH_SOURCE[0]}"
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

exec "${SCRIPT_DIR}/run-agent.sh" "${EXTRA_AGENT_ARGS[@]}"
