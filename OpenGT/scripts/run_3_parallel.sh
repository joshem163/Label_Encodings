#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

# Run 3 models in parallel, one model per GPU.
# Models: SGFormer, CoBFormer, PolyFormer
#
# Usage:
#   bash run_3_parallel.sh
#   bash run_3_parallel.sh --group all
#   SINGLE_SPLIT_REPEAT=3 bash run_3_parallel.sh --group all --no-summary

GROUP="all"
SUMMARY=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --no-summary) SUMMARY=0; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODELS=(SGFormer CoBFormer PolyFormer)
GPUS=(0 1 2)

if [[ ${#MODELS[@]} -ne ${#GPUS[@]} ]]; then
    echo "MODELS and GPUS must have the same length"
    exit 1
fi

mkdir -p logs
PIDS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu="${GPUS[$i]}"
    log="logs/${model}_group-${GROUP}.log"

    cmd=(bash run_model_all_22.sh --model "$model" --group "$GROUP" --gpu "$gpu")
    if [[ "$SUMMARY" -eq 1 ]]; then
        cmd+=(--summary)
    fi

    echo "[LAUNCH] model=$model gpu=$gpu log=$log"
    "${cmd[@]}" >"$log" 2>&1 &
    PIDS+=("$!")
done

echo "[INFO] Launched ${#PIDS[@]} jobs. pids=${PIDS[*]}"

FAIL=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "[DONE] One or more model runs failed. Check logs/"
    exit 1
fi

echo "[DONE] All model runs completed successfully. Logs: logs/"
