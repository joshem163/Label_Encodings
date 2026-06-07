#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

# Generic runner for one model across node-classification dataset groups.
#
# Usage:
#   bash run_model_all_22.sh --model GPS --group all --gpu 0
#   SINGLE_SPLIT_REPEAT=5 bash run_model_all_22.sh --model Graphormer --group d13 --gpu 1 --summary

MODEL=""
GROUP="all"
GPU=0
SUMMARY=0
SINGLE_SPLIT_REPEAT="${SINGLE_SPLIT_REPEAT:-3}"
PROTO_FLAG=""
PROTO_ENABLED="${ENABLE_PROTO:-0}"   # backward-compatible fallback
LSS_FLAG=""
LSS_ENABLED="${ENABLE_LSS:-0}"       # backward-compatible fallback
RESULT_ROOT="${RESULT_ROOT:-}"       # may be overridden by --result-root
SAMPLER="${SAMPLER:-full_batch}"     # train sampler override
LSS_BACKEND="${LSS_BACKEND:-auto}"   # auto|cpu|gpu
LSS_GPU_BATCH_SIZE="${LSS_GPU_BATCH_SIZE:-256}"
SAINT_BATCH_SIZE="${SAINT_BATCH_SIZE:-}"
ITER_PER_EPOCH="${ITER_PER_EPOCH:-}"
WALK_LENGTH="${WALK_LENGTH:-}"
TRAIN_PARTS="${TRAIN_PARTS:-}"

resolve_cuda_visible_device() {
    local gpu_arg="$1"

    # Preserve explicit UUID / PCI bus IDs provided by caller.
    if [[ "$gpu_arg" =~ ^GPU- ]] || [[ "$gpu_arg" =~ ^MIG- ]] || [[ "$gpu_arg" =~ ^00000000: ]]; then
        echo "$gpu_arg"
        return 0
    fi

    # Convert numeric GPU index to UUID to avoid scheduler/device-map ambiguity.
    if [[ "$gpu_arg" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_uuid
        gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_arg" 2>/dev/null | head -n1 | tr -d '[:space:]')"
        if [[ -n "$gpu_uuid" ]]; then
            echo "$gpu_uuid"
            return 0
        fi
    fi

    # Fallback to original value.
    echo "$gpu_arg"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --summary) SUMMARY=1; shift 1 ;;
        --proto) PROTO_FLAG="$2"; shift 2 ;;
        --lss) LSS_FLAG="$2"; shift 2 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --sampler) SAMPLER="$2"; shift 2 ;;
        --lss-backend) LSS_BACKEND="$2"; shift 2 ;;
        --lss-gpu-batch-size) LSS_GPU_BATCH_SIZE="$2"; shift 2 ;;
        --saint-batch-size) SAINT_BATCH_SIZE="$2"; shift 2 ;;
        --iter-per-epoch) ITER_PER_EPOCH="$2"; shift 2 ;;
        --walk-length) WALK_LENGTH="$2"; shift 2 ;;
        --train-parts) TRAIN_PARTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Missing required --model"
    exit 1
fi

if [[ -n "$PROTO_FLAG" ]]; then
    case "${PROTO_FLAG,,}" in
        true|1|yes|y) PROTO_ENABLED=1 ;;
        false|0|no|n) PROTO_ENABLED=0 ;;
        *) echo "Invalid --proto value '$PROTO_FLAG' (use true|false)"; exit 1 ;;
    esac
fi

if [[ -n "$LSS_FLAG" ]]; then
    case "${LSS_FLAG,,}" in
        true|1|yes|y) LSS_ENABLED=1 ;;
        false|0|no|n) LSS_ENABLED=0 ;;
        *) echo "Invalid --lss value '$LSS_FLAG' (use true|false)"; exit 1 ;;
    esac
fi

if [[ -z "$RESULT_ROOT" ]]; then
    if [[ "$PROTO_ENABLED" == "1" ]]; then
        RESULT_ROOT="results_proto"
    else
        RESULT_ROOT="results"
    fi
fi
mkdir -p "$RESULT_ROOT"

case "${SAMPLER,,}" in
    full_batch|neighbor|random_node|cluster|saint_rw|saint_node|saint_edge)
        ;;
    graphsaint_rw) SAMPLER="saint_rw" ;;
    graphsaint_node) SAMPLER="saint_node" ;;
    graphsaint_edge) SAMPLER="saint_edge" ;;
    *)
        echo "Invalid --sampler '$SAMPLER' (use: full_batch|neighbor|random_node|cluster|saint_rw|saint_node|saint_edge)"
        exit 1
        ;;
esac

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

# Prevent overlapping runs of the same model+result_root on the same node.
# This avoids false collisions with unrelated experiments using the same model.
LOCK_ROOT_TAG="$(echo "${RESULT_ROOT}" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"
LOCK_FILE="/tmp/opengt-${MODEL}-${LOCK_ROOT_TAG}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[LOCK] Another run_model_all_22.sh instance is already running for model=$MODEL result_root=$RESULT_ROOT (lock: $LOCK_FILE)"
    exit 1
fi

DATASETS_CCP=(cora citeseer pubmed)
DATASETS_ACW=(amazon-computers amazon-photo coauthor-cs coauthor-physics wikics corafull)
DATASETS_D13=(texas cornell wisconsin actor squirrel chameleon roman-empire amazon-ratings minesweeper tolokers questions flickr blogcatalog)
DATASETS_22=("${DATASETS_CCP[@]}" "${DATASETS_ACW[@]}" "${DATASETS_D13[@]}")

shopt -s nullglob
CFG_FILES=("configs/${MODEL}"/*-"${MODEL}".yaml)
if [[ ${#CFG_FILES[@]} -eq 0 ]]; then
    echo "No configs found under configs/${MODEL}/"
    exit 1
fi

declare -A AVAILABLE_DATASETS=()
for cfg_file in "${CFG_FILES[@]}"; do
    cfg_base="$(basename "$cfg_file")"
    ds="${cfg_base%-${MODEL}.yaml}"
    AVAILABLE_DATASETS["$ds"]=1
done

case "$GROUP" in
    ccp) CANDIDATES=("${DATASETS_CCP[@]}") ;;
    acw) CANDIDATES=("${DATASETS_ACW[@]}") ;;
    d13) CANDIDATES=("${DATASETS_D13[@]}") ;;
    all) CANDIDATES=("${DATASETS_22[@]}") ;;
    *) echo "Invalid --group '$GROUP' (use: ccp|acw|d13|all)"; exit 1 ;;
esac

DATASETS=()
MISSING_DATASETS=()
for ds in "${CANDIDATES[@]}"; do
    if [[ -n "${AVAILABLE_DATASETS[$ds]:-}" ]]; then
        DATASETS+=("$ds")
    else
        echo "[SKIP] Missing config: configs/${MODEL}/${ds}-${MODEL}.yaml"
        MISSING_DATASETS+=("$ds")
    fi
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No runnable datasets selected for group '$GROUP'."
    exit 1
fi

if [[ "$GROUP" == "all" && ${#DATASETS[@]} -ne 22 ]]; then
    echo "[FATAL] group=all requires all 22 datasets, but found ${#DATASETS[@]} for model=$MODEL."
    if [[ ${#MISSING_DATASETS[@]} -gt 0 ]]; then
        echo "[FATAL] Missing datasets: ${MISSING_DATASETS[*]}"
    fi
    exit 1
fi

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

FAILED=()

echo "[ENV] model=$MODEL gpu=$GPU single_split_repeat=$SINGLE_SPLIT_REPEAT"
echo "[ENV] proto_enabled=$PROTO_ENABLED lss_enabled=$LSS_ENABLED result_root=$RESULT_ROOT"
echo "[ENV] train_sampler=$SAMPLER"
echo "[ENV] lss_backend=$LSS_BACKEND lss_gpu_batch_size=$LSS_GPU_BATCH_SIZE"
echo "[ENV] parent_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[ENV] split_policy=random_fixed_60_20_20"
python - <<'PY'
import torch
print("[ENV] torch:", torch.__version__)
print("[ENV] torch_path:", torch.__file__)
print("[ENV] cuda_available:", torch.cuda.is_available())
PY

for ds in "${DATASETS[@]}"; do
    cfg="configs/${MODEL}/${ds}-${MODEL}.yaml"
    best_json="${RESULT_ROOT}/${ds}-${MODEL}/agg/test/best.json"
    if [[ -f "$best_json" ]]; then
        echo "[SKIP] Existing aggregate found: $best_json"
        continue
    fi
    ms_count="$(multi_splits_count "$cfg")"
    if [[ "$ms_count" -gt 0 ]]; then
        run_repeat=1
        echo "[RUN] dataset=$ds model=$MODEL repeat=$run_repeat gpu=$GPU (multi-split: $ms_count)"
    else
        run_repeat="$SINGLE_SPLIT_REPEAT"
        echo "[RUN] dataset=$ds model=$MODEL repeat=$run_repeat gpu=$GPU (single-split)"
    fi
    extra_args=(accelerator cuda train.auto_resume True out_dir "$RESULT_ROOT" train.sampler "$SAMPLER" dataset.split_mode random)
    if [[ -n "$SAINT_BATCH_SIZE" ]]; then
        extra_args+=(train.batch_size "$SAINT_BATCH_SIZE")
    fi
    if [[ -n "$ITER_PER_EPOCH" ]]; then
        extra_args+=(train.iter_per_epoch "$ITER_PER_EPOCH")
    fi
    if [[ -n "$WALK_LENGTH" ]]; then
        extra_args+=(train.walk_length "$WALK_LENGTH")
    fi
    if [[ -n "$TRAIN_PARTS" ]]; then
        extra_args+=(train.train_parts "$TRAIN_PARTS")
    fi

    # Optional feature toggles controlled via CLI/env. Flag should always win.
    if [[ "$PROTO_ENABLED" == "1" ]]; then
        extra_args+=(gt.use_proto True)
        echo "[PROTO] dataset=$ds gt.use_proto=True out_dir=${RESULT_ROOT}"
    else
        extra_args+=(gt.use_proto False)
        echo "[PROTO] dataset=$ds gt.use_proto=False out_dir=${RESULT_ROOT}"
    fi
    if [[ "$LSS_ENABLED" == "1" ]]; then
        extra_args+=(gt.use_lss True gt.lss_backend "$LSS_BACKEND" gt.lss_gpu_batch_size "$LSS_GPU_BATCH_SIZE")
        echo "[LSS] dataset=$ds gt.use_lss=True gt.lss_backend=${LSS_BACKEND} gt.lss_gpu_batch_size=${LSS_GPU_BATCH_SIZE} out_dir=${RESULT_ROOT}"
    else
        extra_args+=(gt.use_lss False gt.lss_backend "$LSS_BACKEND" gt.lss_gpu_batch_size "$LSS_GPU_BATCH_SIZE")
        echo "[LSS] dataset=$ds gt.use_lss=False out_dir=${RESULT_ROOT}"
    fi

    # SAN uses quadratic full-graph attention, which OOMs on large graphs.
    # Force sparse attention for large datasets while keeping small datasets unchanged.
    if [[ "$MODEL" == "SAN" ]]; then
        case "$ds" in
            pubmed|amazon-computers|amazon-photo|coauthor-cs|coauthor-physics|corafull|wikics|flickr|blogcatalog)
                extra_args+=(gt.full_graph False)
                echo "[SAN-MEM] dataset=$ds override=gt.full_graph:False"
                ;;
        esac
    fi

    attempt=1
    max_attempts=2
    while (( attempt <= max_attempts )); do
        GPU_VISIBLE="$(resolve_cuda_visible_device "$GPU")"
        echo "[ENV] launch_cuda_visible_devices=${GPU_VISIBLE} (requested_gpu=${GPU})"
        if CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" python main.py --cfg "$cfg" --repeat "$run_repeat" "${extra_args[@]}"; then
            break
        fi
        if (( attempt < max_attempts )); then
            echo "[RETRY] dataset=$ds model=$MODEL attempt=$((attempt+1))/${max_attempts} after failure"
            sleep 10
        else
            FAILED+=("$cfg")
        fi
        attempt=$((attempt+1))
    done
done

if [[ "$SUMMARY" -eq 1 ]]; then
export DATASETS_FOR_SUMMARY
DATASETS_FOR_SUMMARY="$(IFS=,; echo "${DATASETS[*]}")"
export MODEL
export RESULT_ROOT
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
out = os.path.join(result_root, f"summary_{model.lower()}_selected.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "dataset", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"])
    w.writerows(rows)
print(f"Wrote {out}")
PY
fi

echo ""
echo "========================================"
echo "Group=$GROUP, model=$MODEL"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "  $f"
    done
else
    echo "All runs succeeded."
fi
echo "========================================"
