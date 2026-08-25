#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".gitnexus/run.cjs" ]] && command -v node >/dev/null 2>&1; then
  # Keep the analyzer and the on-disk index on the same project-local runtime.
  # Mixing a global CLI/npx version with .gitnexus/run.cjs was the source of
  # repeated stale metadata and FTS schema mismatches on this repository.
  GITNEXUS=(node .gitnexus/run.cjs)
elif command -v gitnexus >/dev/null 2>&1; then
  GITNEXUS=(gitnexus)
elif command -v npx >/dev/null 2>&1; then
  GITNEXUS=(npx gitnexus)
else
  echo "GitNexus is unavailable: install its CLI or provide npx." >&2
  exit 127
fi

MODE="${1:-refresh}"
ANALYZE_ARGS=(
  analyze
  --pdg
  --index-only
  --max-file-size 1024
)

case "${MODE}" in
  status)
    if [[ ! -f ".gitnexus/gitnexus.json" ]]; then
      echo "GitNexus index is missing. Run: bash tools/maintain_gitnexus.sh refresh" >&2
      exit 2
    fi
    "${GITNEXUS[@]}" status
    ;;
  refresh|force)
    # A full rebuild is deterministic and repairs Ladybug/FTS inconsistencies.
    # Small edits are already allowed to defer refresh by AGENTS.md, so this
    # reliable default does not run on every tiny change.
    "${GITNEXUS[@]}" "${ANALYZE_ARGS[@]}" --force
    ;;
  incremental)
    "${GITNEXUS[@]}" "${ANALYZE_ARGS[@]}"
    ;;
  *)
    echo "Usage: $0 {status|refresh|force|incremental}" >&2
    exit 64
    ;;
esac
