#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

# Run SGFormer, NodeFormer, and PolyFormer in parallel on all 22 datasets
# with identical proto+lss augmentation settings.
#
# Usage:
#   bash scripts/run_sgformer_nodeformer_proto_lss_parallel_all22.sh
#   bash scripts/run_sgformer_nodeformer_proto_lss_parallel_all22.sh --gpu-sgformer 0 --gpu-nodeformer 1 --gpu-polyformer 2
#   bash scripts/run_sgformer_nodeformer_proto_lss_parallel_all22.sh --lss-backend gpu --lss-gpu-batch-size 512
#   bash scripts/run_sgformer_nodeformer_proto_lss_parallel_all22.sh --seeds 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_model_all_22.sh"

GROUP="all"
SUMMARY=1
GPU_SGFORMER=0
GPU_NODEFORMER=1
GPU_POLYFORMER=2
LSS_BACKEND="${LSS_BACKEND:-auto}"
LSS_GPU_BATCH_SIZE="${LSS_GPU_BATCH_SIZE:-256}"
RESULT_ROOT="${RESULT_ROOT:-results_proto_lss}"
SEEDS="${SINGLE_SPLIT_REPEAT:-3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-sgformer) GPU_SGFORMER="$2"; shift 2 ;;
        --gpu-nodeformer) GPU_NODEFORMER="$2"; shift 2 ;;
        --gpu-polyformer) GPU_POLYFORMER="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --lss-backend) LSS_BACKEND="$2"; shift 2 ;;
        --lss-gpu-batch-size) LSS_GPU_BATCH_SIZE="$2"; shift 2 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --no-summary) SUMMARY=0; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$GROUP" != "all" ]]; then
    echo "This launcher is intended for all 22 datasets. Use --group all."
    exit 1
fi

case "${LSS_BACKEND,,}" in
    auto|cpu|gpu) ;;
    *)
        echo "Invalid --lss-backend '$LSS_BACKEND' (use: auto|cpu|gpu)"
        exit 1
        ;;
esac
if ! [[ "$LSS_GPU_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$LSS_GPU_BATCH_SIZE" -lt 1 ]]; then
    echo "Invalid --lss-gpu-batch-size '$LSS_GPU_BATCH_SIZE' (use positive integer)"
    exit 1
fi
if ! [[ "$SEEDS" =~ ^[0-9]+$ ]] || [[ "$SEEDS" -lt 1 ]]; then
    echo "Invalid --seeds '$SEEDS' (use positive integer)"
    exit 1
fi

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/${RESULT_ROOT}"

PIDS=()
MODELS=(SGFormer NodeFormer PolyFormer)
GPUS=("$GPU_SGFORMER" "$GPU_NODEFORMER" "$GPU_POLYFORMER")

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu="${GPUS[$i]}"
    log="${ROOT_DIR}/logs/${model}_proto_lss_group-${GROUP}.log"

    cmd=(
        bash "$RUNNER"
        --model "$model"
        --group "$GROUP"
        --gpu "$gpu"
        --proto true
        --lss true
        --lss-backend "$LSS_BACKEND"
        --lss-gpu-batch-size "$LSS_GPU_BATCH_SIZE"
        --result-root "$RESULT_ROOT"
    )
    if [[ "$SUMMARY" -eq 1 ]]; then
        cmd+=(--summary)
    fi

    echo "[LAUNCH] model=$model gpu=$gpu proto=true lss=true seeds=$SEEDS split=60/20/20 lss_backend=$LSS_BACKEND lss_gpu_batch_size=$LSS_GPU_BATCH_SIZE result_root=$RESULT_ROOT log=$log"
    (
        cd "$ROOT_DIR"
        SINGLE_SPLIT_REPEAT="$SEEDS" "${cmd[@]}"
    ) >"$log" 2>&1 &
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
    echo "[DONE] One or more runs failed. Check logs in ${ROOT_DIR}/logs/"
    exit 1
fi

echo "[DONE] SGFormer + NodeFormer + PolyFormer proto+lss runs completed successfully."
echo "[DONE] Logs: ${ROOT_DIR}/logs/"
