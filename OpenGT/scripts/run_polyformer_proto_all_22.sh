#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

# PolyFormer runner wrapper for all 22 node-classification datasets
# with proto feature augmentation enabled.
#
# Usage:
#   bash scripts/run_polyformer_proto_all_22.sh --group all --gpu 0
#   SINGLE_SPLIT_REPEAT=3 bash scripts/run_polyformer_proto_all_22.sh --group d13 --gpu 1 --summary
#
# Run this from the repository root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GROUP="all"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--group" ]] && (( i + 1 < ${#args[@]} )); then
        GROUP="${args[$((i + 1))]}"
    fi
done

mkdir -p logs
LOG_FILE="logs/PolyFormer+Proto_group-${GROUP}.log"

echo "[LAUNCH] model=PolyFormer proto=true group=${GROUP} log=${LOG_FILE}"
bash "${SCRIPT_DIR}/run_model_all_22.sh" --model PolyFormer --proto true "$@" 2>&1 | tee "$LOG_FILE"
