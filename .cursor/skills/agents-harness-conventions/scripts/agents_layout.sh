#!/usr/bin/env bash
# Discover qlab agents/ experiments and check cross-experiment *_core imports.
set -euo pipefail

SHARED_DIRS="agents_utils scripts"
SKIP_PKG_DIRS="tests eval papers ui config server __pycache__ .data"

find_repo_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/agents/README.md" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  echo "agents_layout.sh: could not find repo root (agents/README.md)" >&2
  return 1
}

is_shared_agent_dir() {
  local name=$1
  for s in $SHARED_DIRS; do
    [[ "$name" == "$s" ]] && return 0
  done
  [[ "$name" == .* ]] && return 0
  return 1
}

is_skip_pkg_dir() {
  local name=$1
  for s in $SKIP_PKG_DIRS; do
    [[ "$name" == "$s" ]] && return 0
  done
  [[ "$name" == .* ]] && return 0
  return 1
}

experiment_dirs() {
  local agents_root=$1
  local d name
  for d in "$agents_root"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    is_shared_agent_dir "$name" && continue
    [[ -f "$d/main.py" ]] || continue
    echo "$d"
  done
}

top_level_packages() {
  local exp_root=$1
  local sub name
  for sub in "$exp_root"/*/; do
    [[ -d "$sub" ]] || continue
    name="$(basename "$sub")"
    is_skip_pkg_dir "$name" && continue
    if [[ "$name" == *_core ]] || [[ -f "$sub/__init__.py" ]]; then
      echo "$name"
    fi
  done
}

register_core_owners() {
  local agents_root=$1 strict=$2
  local exp name pkg sub
  declare -gA CORE_OWNER=()
  for exp in $(experiment_dirs "$agents_root"); do
    name="$(basename "$exp")"
    if [[ "$strict" == "1" ]]; then
      while IFS= read -r pkg; do
        [[ -z "$pkg" ]] && continue
        CORE_OWNER["$pkg"]=$name
      done < <(top_level_packages "$exp")
    else
      for sub in "$exp"/*_core/; do
        [[ -d "$sub" ]] || continue
        pkg="$(basename "$sub")"
        CORE_OWNER["$pkg"]=$name
      done
    fi
  done
}

cmd_list() {
  local agents_root=$1 repo_root=$2
  local exp name rel pkgs
  for exp in $(experiment_dirs "$agents_root" | sort); do
    name="$(basename "$exp")"
    rel="${exp#"$repo_root"/}"
    pkgs="$(top_level_packages "$exp" | sort | paste -sd, -)"
    [[ -z "$pkgs" ]] && pkgs="(none detected)"
    printf '%s\t%s\t%s\n' "$name" "$rel" "$pkgs"
  done

  local env_template="$agents_root/.env-template"
  if [[ -f "$env_template" ]]; then
    echo
    echo "# ports from agents/.env-template"
    grep -E '^[A-Z0-9_]*(AGENT|UI|PORT)[A-Z0-9_]*=[0-9]+' "$env_template" \
      | sort -t= -k2 -n || true
  fi

  if [[ -d "$agents_root" ]]; then
    echo
    echo "# duplicate test module basenames across experiments (if any)"
    find "$agents_root" -path '*/tests/test_*.py' -printf '%f\n' 2>/dev/null \
      | sort | uniq -d | while read -r base; do
        echo "  ${base}: $(find "$agents_root" -path "*/tests/${base}" -printf '%h\n' \
          | sed "s|$agents_root/||;s|/tests||" | paste -sd, -)"
      done
  fi
}

cmd_check_imports() {
  local agents_root=$1 strict=$2
  register_core_owners "$agents_root" "$strict"
  local violations=0 pkg owner file pattern

  if ((${#CORE_OWNER[@]} == 0)); then
    echo "OK: no *_core packages registered (nothing to check)."
    return 0
  fi

  for pkg in "${!CORE_OWNER[@]}"; do
    owner="${CORE_OWNER[$pkg]}"
    pattern="(from|import)[[:space:]]+${pkg}([[:space:].]|$)"
    while IFS= read -r file; do
      [[ -z "$file" ]] && continue
      if [[ "$file" == "agents/${owner}/"* ]]; then
        continue
      fi
      echo "${file}: imports '${pkg}' (owned by experiment '${owner}')"
      violations=1
    done < <(rg -l --glob '*.py' -e "$pattern" agents 2>/dev/null || true)
  done

  if [[ "$violations" -eq 0 ]]; then
    echo "OK: no cross-experiment top-level imports detected."
    return 0
  fi
  return 1
}

usage() {
  echo "Usage: agents_layout.sh list|check-imports [--strict-packages]" >&2
  echo "  list              Experiments, packages, ports, pytest basename hints" >&2
  echo "  check-imports     Fail on cross-experiment *_core imports (see --strict-packages)" >&2
}

main() {
  local cmd=${1:-}
  local strict=0
  local AGENTS_ROOT_OVERRIDE=""
  shift || true
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --strict-packages) strict=1 ;;
      --agents-root)
        shift
        AGENTS_ROOT_OVERRIDE=${1:?}
        ;;
      *) echo "Unknown argument: $1" >&2; usage; return 2 ;;
    esac
    shift || true
  done

  local repo_root agents_root
  repo_root="$(find_repo_root)"
  agents_root="${AGENTS_ROOT_OVERRIDE:-$repo_root/agents}"

  case "$cmd" in
    list) cmd_list "$agents_root" "$repo_root" ;;
    check-imports) cmd_check_imports "$agents_root" "$strict" ;;
    *) usage; return 2 ;;
  esac
}

main "$@"
