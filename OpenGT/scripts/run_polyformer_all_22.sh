#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

# PolyFormer runner wrapper for all 22 node-classification datasets.
#
# Usage:
#   bash run_polyformer_all_22.sh --group all --gpu 0
#   bash run_polyformer_all_22.sh --group all --gpu 0 --proto true
#   SINGLE_SPLIT_REPEAT=3 bash run_polyformer_all_22.sh --group d13 --gpu 1 --summary

GROUP="all"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--group" ]] && (( i + 1 < ${#args[@]} )); then
        GROUP="${args[$((i + 1))]}"
    fi
done

mkdir -p logs
LOG_FILE="logs/PolyFormer_group-${GROUP}.log"

echo "[LAUNCH] model=PolyFormer group=${GROUP} log=${LOG_FILE}"
bash run_model_all_22.sh --model PolyFormer "$@" 2>&1 | tee "$LOG_FILE"
