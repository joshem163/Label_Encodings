#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# PolyFormer ablation runner over selected datasets and feature combinations.
# Default datasets:
#   cora, citeseer, amazon-ratings, coauthor-cs, tolokers
#
# Combinations:
#   raw                  (proto=False, lss=False)
#   raw+proto            (proto=True,  lss=False)
#   raw+lss              (proto=False, lss=True)
#   raw+proto+lss        (proto=True,  lss=True)
#
# Usage:
#   bash ablation/run_polyformer_ablation.sh
#   bash ablation/run_polyformer_ablation.sh --gpu 0 --seeds 3 --lss-backend gpu --lss-gpu-batch-size 512

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="PolyFormer"
GPU=0
SEEDS="${SINGLE_SPLIT_REPEAT:-3}"
LSS_BACKEND="${LSS_BACKEND:-auto}"
LSS_GPU_BATCH_SIZE="${LSS_GPU_BATCH_SIZE:-256}"
RESULT_BASE="${RESULT_BASE:-ablation/results}"
LOG_DIR="${LOG_DIR:-ablation/logs}"
SUMMARY=1

DATASETS=(cora citeseer amazon-ratings coauthor-cs tolokers)

resolve_cuda_visible_device() {
    local gpu_arg="$1"
    if [[ "$gpu_arg" =~ ^GPU- ]] || [[ "$gpu_arg" =~ ^MIG- ]] || [[ "$gpu_arg" =~ ^00000000: ]]; then
        echo "$gpu_arg"
        return 0
    fi
    if [[ "$gpu_arg" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_uuid
        gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_arg" 2>/dev/null | head -n1 | tr -d '[:space:]')"
        if [[ -n "$gpu_uuid" ]]; then
            echo "$gpu_uuid"
            return 0
        fi
    fi
    echo "$gpu_arg"
}

write_summary_csv() {
    local result_root="$1"
    local mode_name="$2"

    DATASETS_FOR_SUMMARY="$(IFS=,; echo "${DATASETS[*]}")"
    export DATASETS_FOR_SUMMARY MODEL RESULT_ROOT="$result_root" MODE_NAME="$mode_name"

    python - <<'PY'
import csv
import json
import os

model = os.environ["MODEL"]
mode_name = os.environ["MODE_NAME"]
datasets = [d for d in os.environ.get("DATASETS_FOR_SUMMARY", "").split(",") if d]
result_root = os.environ.get("RESULT_ROOT", "ablation/results")

rows = []
for ds in datasets:
    best = os.path.join(result_root, f"{ds}-{model}", "agg", "test", "best.json")
    if not os.path.exists(best):
        rows.append([mode_name, model, ds, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
        continue
    with open(best, "r") as f:
        m = json.load(f)
    rows.append([
        mode_name, model, ds,
        m.get("accuracy", "N/A"), m.get("accuracy_std", "N/A"),
        m.get("f1", "N/A"), m.get("f1_std", "N/A"),
        m.get("auc", "N/A"), m.get("auc_std", "N/A"),
    ])

os.makedirs(result_root, exist_ok=True)
out = os.path.join(result_root, f"summary_{model.lower()}_{mode_name}.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "model", "dataset", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "auc_mean", "auc_std"])
    w.writerows(rows)
print(f"Wrote {out}")
PY
}

run_mode() {
    local mode_name="$1"
    local proto_enabled="$2"
    local lss_enabled="$3"
    local result_root="$4"

    mkdir -p "${ROOT_DIR}/${result_root}" "${ROOT_DIR}/${LOG_DIR}"
    local log_file="${ROOT_DIR}/${LOG_DIR}/${MODEL}_${mode_name}.log"

    echo "[LAUNCH] mode=${mode_name} model=${MODEL} gpu=${GPU} seeds=${SEEDS} datasets=${DATASETS[*]} proto=${proto_enabled} lss=${lss_enabled} result_root=${result_root} log=${log_file}"

    (
        cd "$ROOT_DIR"
        echo "[ENV] model=${MODEL} gpu=${GPU} single_split_repeat=${SEEDS}"
        echo "[ENV] mode=${mode_name} proto=${proto_enabled} lss=${lss_enabled}"
        echo "[ENV] lss_backend=${LSS_BACKEND} lss_gpu_batch_size=${LSS_GPU_BATCH_SIZE}"
        echo "[ENV] split_policy=random_fixed_60_20_20"
        python - <<'PY'
import torch
print("[ENV] torch:", torch.__version__)
print("[ENV] torch_path:", torch.__file__)
print("[ENV] cuda_available:", torch.cuda.is_available())
PY

        failed=0
        for ds in "${DATASETS[@]}"; do
            cfg="configs/${MODEL}/${ds}-${MODEL}.yaml"
            if [[ ! -f "$cfg" ]]; then
                echo "[FAIL] Missing config: $cfg"
                failed=1
                continue
            fi

            best_json="${result_root}/${ds}-${MODEL}/agg/test/best.json"
            if [[ -f "$best_json" ]]; then
                echo "[SKIP] Existing aggregate found: $best_json"
                continue
            fi

            echo "[RUN] dataset=${ds} model=${MODEL} repeat=${SEEDS} gpu=${GPU} mode=${mode_name}"

            extra_args=(
                accelerator cuda
                train.auto_resume True
                out_dir "$result_root"
                train.sampler full_batch
                dataset.split_mode random
                gt.use_proto "$proto_enabled"
                gt.use_lss "$lss_enabled"
                gt.lss_backend "$LSS_BACKEND"
                gt.lss_gpu_batch_size "$LSS_GPU_BATCH_SIZE"
            )

            attempt=1
            max_attempts=2
            while (( attempt <= max_attempts )); do
                GPU_VISIBLE="$(resolve_cuda_visible_device "$GPU")"
                echo "[ENV] launch_cuda_visible_devices=${GPU_VISIBLE} (requested_gpu=${GPU})"
                if CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" python main.py --cfg "$cfg" --repeat "$SEEDS" "${extra_args[@]}"; then
                    if [[ ! -f "$best_json" ]]; then
                        echo "[FAIL] dataset=${ds} mode=${mode_name} finished without aggregate: $best_json"
                        failed=1
                    fi
                    break
                fi
                if (( attempt < max_attempts )); then
                    echo "[RETRY] dataset=${ds} mode=${mode_name} attempt=$((attempt+1))/${max_attempts}"
                    sleep 10
                else
                    echo "[FAIL] dataset=${ds} mode=${mode_name}"
                    failed=1
                fi
                attempt=$((attempt+1))
            done
        done

        if [[ "$SUMMARY" -eq 1 ]]; then
            write_summary_csv "$result_root" "$mode_name"
        fi

        exit "$failed"
    ) 2>&1 | tee "$log_file"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --lss-backend) LSS_BACKEND="$2"; shift 2 ;;
        --lss-gpu-batch-size) LSS_GPU_BATCH_SIZE="$2"; shift 2 ;;
        --result-base) RESULT_BASE="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --datasets)
            IFS=',' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        --no-summary) SUMMARY=0; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    echo "Invalid --datasets: empty list"
    exit 1
fi

# Handle common typo while preserving canonical dataset naming in configs.
for i in "${!DATASETS[@]}"; do
    if [[ "${DATASETS[$i]}" == "tokolers" ]]; then
        DATASETS[$i]="tolokers"
    fi
done

RAW_ROOT="${RESULT_BASE}/raw"
RAW_PROTO_ROOT="${RESULT_BASE}/raw_proto"
RAW_LSS_ROOT="${RESULT_BASE}/raw_lss"
RAW_PROTO_LSS_ROOT="${RESULT_BASE}/raw_proto_lss"

run_mode "raw" False False "$RAW_ROOT"
run_mode "raw_proto" True False "$RAW_PROTO_ROOT"
run_mode "raw_lss" False True "$RAW_LSS_ROOT"
run_mode "raw_proto_lss" True True "$RAW_PROTO_LSS_ROOT"

echo "[DONE] PolyFormer ablation completed."
echo "[DONE] Results:"
echo "  ${ROOT_DIR}/${RAW_ROOT}"
echo "  ${ROOT_DIR}/${RAW_PROTO_ROOT}"
echo "  ${ROOT_DIR}/${RAW_LSS_ROOT}"
echo "  ${ROOT_DIR}/${RAW_PROTO_LSS_ROOT}"
echo "[DONE] Logs: ${ROOT_DIR}/${LOG_DIR}"
