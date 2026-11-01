#!/usr/bin/env bash
# Pull open-weight models from agents-llms.html compact/edge tier (Ollama tags).
# RTX 3090: models are sized for ~24 GB VRAM one at a time; disk ~80–100 GB total.
#
# Uses OLLAMA_MODELS from agents/.env (via start-ollama.sh) or your shell profile.
#
# Usage:
#   ./agents/scripts/pull-agent-eval-models.sh
#   ./agents/scripts/pull-agent-eval-models.sh --dry-run

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPTS_DIR}/_lib.sh"
load_agents_env 2>/dev/null || true
export OLLAMA_MODELS="${OLLAMA_MODELS:-$(ollama_models_dir)}"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# Maps to research/cheatsheets/agents-llms.html open-weight compact + one mid row.
MODELS=(
  ornith-1.5:9b          # Ornith-1.5-9B
  qwen3.5:9b             # Qwen3.5-9B (VLM)
  qwen3.5:4b             # Qwen3.5-4B
  gemma4:12b             # Gemma 4 12B Unified
  gemma4:e4b             # Gemma 4 E4B edge
  gpt-oss:20b            # GPT-OSS-20B
  granite4.2:8b          # Granite 4.2-8B
  granite4.2:3b          # Granite 4.2-3B
  lfm3:1.7b              # LFM3-1.7B
  lfm2.5:8b              # LFM2.5-8B-A1B (default tag)
  bonsai-2:9b-q2_0       # Ternary Bonsai 2 9B
  muse-glimmer:30b       # Muse Glimmer-30B (local agent VLM)
  nemotron-3-nano:30b-a3b-q4_0  # Nemotron 3 Nano Omni class (text+vision)
  qwen3.8:27b            # Qwen3.8-27B (mid tier; ~17–20 GB)
)

for tag in "${MODELS[@]}"; do
  if ollama show "$tag" >/dev/null 2>&1; then
    echo "skip (already local): $tag"
    continue
  fi
  if [[ "$DRY" -eq 1 ]]; then
    echo "would pull: $tag"
    continue
  fi
  echo "pulling: $tag"
  ollama pull "$tag"
done

echo "Done. ollama list:"
ollama list
