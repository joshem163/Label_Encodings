#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sgformer_polyformer_nodeformer_bfq_mode.sh" --mode raw --gpu-sgformer 0 --gpu-polyformer 1 --gpu-nodeformer 2 "$@"
