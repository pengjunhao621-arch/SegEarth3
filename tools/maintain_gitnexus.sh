#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if command -v gitnexus >/dev/null 2>&1; then
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
  refresh)
    "${GITNEXUS[@]}" "${ANALYZE_ARGS[@]}"
    ;;
  force)
    "${GITNEXUS[@]}" "${ANALYZE_ARGS[@]}" --force
    ;;
  *)
    echo "Usage: $0 {status|refresh|force}" >&2
    exit 64
    ;;
esac
