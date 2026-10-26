#!/usr/bin/env bash
# Default: current TailRing source snapshots on the verified CANN9.2 environment.
# --legacy explicitly selects the previous sealed v0.27-derived candidate flow.
set -euo pipefail
CI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for arg in "$@"; do
  if [[ "$arg" == --legacy ]]; then
    args=()
    for item in "$@"; do [[ "$item" == --legacy ]] || args+=("$item"); done
    exec bash "$CI_DIR/build_legacy.sh" "${args[@]}"
  fi
done
exec bash "$CI_DIR/scripts/build-tailring-source-rebase.sh" "$@"
