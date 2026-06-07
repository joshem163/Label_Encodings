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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE=""
GPU_SGFORMER=0
GPU_POLYFORMER=1
GPU_NODEFORMER=2
SEEDS="${SINGLE_SPLIT_REPEAT:-3}"
SUMMARY=1
LSS_BACKEND="${LSS_BACKEND:-auto}"
LSS_GPU_BATCH_SIZE="${LSS_GPU_BATCH_SIZE:-256}"
PROTO_RESULT_ROOT="${PROTO_RESULT_ROOT:-results_proto_lss}"
RAW_RESULT_ROOT="${RAW_RESULT_ROOT:-results}"

DATASETS=(flickr questions)
MODELS=(SGFormer PolyFormer NodeFormer)

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

multi_splits_count() {
    local cfg_file="$1"
    awk '
    BEGIN { in_rms=0; count=0 }
    /^[[:space:]]*#/ { next }
    /^[^[:space:]][^:]*:/ {
        if ($0 !~ /^run_multiple_splits:/) in_rms=0
    }
    /^run_multiple_splits:[[:space:]]*\[[^]]+\][[:space:]]*$/ {
        line=$0
        sub(/^run_multiple_splits:[[:space:]]*\[/, "", line)
        sub(/\][[:space:]]*$/, "", line)
        if (line ~ /[^[:space:]]/) {
            n=split(line, a, /,[[:space:]]*/)
            count=n
        }
        in_rms=0
        next
    }
    /^run_multiple_splits:[[:space:]]*\[[[:space:]]*\][[:space:]]*$/ { count=0; in_rms=0; next }
    /^run_multiple_splits:[[:space:]]*$/ { in_rms=1; next }
    in_rms==1 && /^[[:space:]]*-[[:space:]]*/ { count++; next }
    in_rms==1 && /^[^[:space:]]/ { in_rms=0 }
    END { print count }
    ' "$cfg_file"
}

write_summary_csv() {
    local result_root="$1"
    local model="$2"

    DATASETS_FOR_SUMMARY="$(IFS=,; echo "${DATASETS[*]}")"
    export DATASETS_FOR_SUMMARY MODEL="$model" RESULT_ROOT="$result_root"

    python - <<'PY'
import csv
import json
import os

model = os.environ["MODEL"]
datasets = [d for d in os.environ.get("DATASETS_FOR_SUMMARY", "").split(",") if d]
result_root = os.environ.get("RESULT_ROOT", "results")

rows = []
for ds in datasets:
    best = os.path.join(result_root, f"{ds}-{model}", "agg", "test", "best.json")
    if not os.path.exists(best):
        rows.append([model, ds, "N/A", "N/A", "N/A", "N/A"])
        continue
    with open(best, "r") as f:
        m = json.load(f)
    rows.append([
        model, ds,
        m.get("accuracy", "N/A"), m.get("accuracy_std", "N/A"),
        m.get("f1", "N/A"), m.get("f1_std", "N/A"),
    ])

os.makedirs(result_root, exist_ok=True)
out = os.path.join(result_root, f"summary_{model.lower()}_bfq.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "dataset", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"])
    w.writerows(rows)
print(f"Wrote {out}")
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --gpu-sgformer) GPU_SGFORMER="$2"; shift 2 ;;
        --gpu-polyformer) GPU_POLYFORMER="$2"; shift 2 ;;
        --gpu-nodeformer) GPU_NODEFORMER="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --datasets)
            IFS=',' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        --lss-backend) LSS_BACKEND="$2"; shift 2 ;;
        --lss-gpu-batch-size) LSS_GPU_BATCH_SIZE="$2"; shift 2 ;;
        --proto-result-root) PROTO_RESULT_ROOT="$2"; shift 2 ;;
        --raw-result-root) RAW_RESULT_ROOT="$2"; shift 2 ;;
        --no-summary) SUMMARY=0; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "proto_lss" && "$MODE" != "raw" ]]; then
    echo "Missing/invalid --mode (use: proto_lss|raw)"
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
if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    echo "Invalid --datasets: empty list"
    exit 1
fi

if [[ "$MODE" == "proto_lss" ]]; then
    PROTO_ENABLED=True
    LSS_ENABLED=True
    RESULT_ROOT="$PROTO_RESULT_ROOT"
else
    PROTO_ENABLED=False
    LSS_ENABLED=False
    RESULT_ROOT="$RAW_RESULT_ROOT"
fi

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/${RESULT_ROOT}"

PIDS=()
GPUS=("$GPU_SGFORMER" "$GPU_POLYFORMER" "$GPU_NODEFORMER")
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu="${GPUS[$i]}"
    log="${ROOT_DIR}/logs/${model}_${MODE}_bfq.log"

    echo "[LAUNCH] mode=${MODE} model=${model} gpu=${gpu} datasets=${DATASETS[*]} proto=${PROTO_ENABLED} lss=${LSS_ENABLED} result_root=${RESULT_ROOT} log=${log}"

    (
        cd "$ROOT_DIR"
        failed=0
        for ds in "${DATASETS[@]}"; do
            cfg="configs/${model}/${ds}-${model}.yaml"
            if [[ ! -f "$cfg" ]]; then
                echo "[SKIP] Missing config: $cfg"
                continue
            fi

            best_json="${RESULT_ROOT}/${ds}-${model}/agg/test/best.json"
            if [[ -f "$best_json" ]]; then
                echo "[SKIP] Existing aggregate found: $best_json"
                continue
            fi

            ms_count="$(multi_splits_count "$cfg")"
            if [[ "$ms_count" -gt 0 ]]; then
                run_repeat=1
                echo "[RUN] dataset=${ds} model=${model} repeat=${run_repeat} gpu=${gpu} (multi-split: ${ms_count})"
            else
                run_repeat="$SEEDS"
                echo "[RUN] dataset=${ds} model=${model} repeat=${run_repeat} gpu=${gpu} (single-split)"
            fi

            extra_args=(
                accelerator cuda
                train.auto_resume True
                out_dir "$RESULT_ROOT"
                train.sampler full_batch
                dataset.split_mode random
                gt.use_proto "$PROTO_ENABLED"
                gt.use_lss "$LSS_ENABLED"
                gt.lss_backend "$LSS_BACKEND"
                gt.lss_gpu_batch_size "$LSS_GPU_BATCH_SIZE"
            )

            attempt=1
            max_attempts=2
            while (( attempt <= max_attempts )); do
                GPU_VISIBLE="$(resolve_cuda_visible_device "$gpu")"
                echo "[ENV] launch_cuda_visible_devices=${GPU_VISIBLE} (requested_gpu=${gpu})"
                if CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" python main.py --cfg "$cfg" --repeat "$run_repeat" "${extra_args[@]}"; then
                    if [[ ! -f "$best_json" ]]; then
                        echo "[FAIL] dataset=${ds} model=${model} finished without aggregate: $best_json"
                        failed=1
                    fi
                    break
                fi
                if (( attempt < max_attempts )); then
                    echo "[RETRY] dataset=${ds} model=${model} attempt=$((attempt+1))/${max_attempts} after failure"
                    sleep 10
                else
                    echo "[FAIL] dataset=${ds} model=${model}"
                    failed=1
                fi
                attempt=$((attempt+1))
            done
        done

        if [[ "$SUMMARY" -eq 1 ]]; then
            write_summary_csv "$RESULT_ROOT" "$model"
        fi

        exit "$failed"
    ) >"$log" 2>&1 &

    PIDS+=("$!")
done

echo "[INFO] mode=${MODE} launched ${#PIDS[@]} jobs. pids=${PIDS[*]}"

FAIL=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "[DONE] mode=${MODE} had failures. Check logs in ${ROOT_DIR}/logs/"
    exit 1
fi

echo "[DONE] mode=${MODE} completed successfully."
if [[ "$MODE" == "proto_lss" ]]; then
    echo "[DONE] outputs: ${ROOT_DIR}/${PROTO_RESULT_ROOT}"
else
    echo "[DONE] outputs: ${ROOT_DIR}/${RAW_RESULT_ROOT}"
fi
