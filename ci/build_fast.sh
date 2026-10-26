#!/usr/bin/env bash
# Official rc1/CANN9.2 upgrade must rebuild native components.
# Default: --fast safely falls back to the complete session recipe.
# Historical Python-only replacement is explicit: ci/build_fast.sh --legacy ...
set -euo pipefail
CI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$CI_DIR/build.sh" --fast "$@"
