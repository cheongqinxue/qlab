#!/usr/bin/env bash
# Build the shared agents Docker image (heavy uv sync — run when pyproject.toml / lock changes).
#
# Usage:
#   ./agents/scripts/build-image.sh
#   AGENTS_RUNTIME_IMAGE=my-agents:v1 ./agents/scripts/build-image.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${AGENTS_DIR}/.." && pwd)"
IMAGE="${AGENTS_RUNTIME_IMAGE:-agents-runtime:latest}"

echo "Building ${IMAGE} from ${REPO_ROOT} …"
docker build \
  -f "${AGENTS_DIR}/Dockerfile" \
  -t "${IMAGE}" \
  "${REPO_ROOT}"

echo "Done. Use in experiment compose: image: ${IMAGE}"
