#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

# Unified GPS runner for node classification datasets.
#
# Groups:
#   ccp      -> cora,citeseer,pubmed
#   acw      -> amazon-computers,amazon-photo,coauthor-cs,coauthor-physics,wikics,corafull
#   d13      -> texas,cornell,wisconsin,actor,squirrel,chameleon,roman-empire,amazon-ratings,minesweeper,tolokers,questions,flickr,blogcatalog
#   all      -> all available GPS configs
#
# Usage:
#   bash run_gps_all_22.sh --group all --gpu 0
#   bash run_gps_all_22.sh --group ccp --gpu 0 --summary

GROUP="all"
GPU=0
SUMMARY=0
MODEL="GPS"
SINGLE_SPLIT_REPEAT="${SINGLE_SPLIT_REPEAT:-3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --summary) SUMMARY=1; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
for ds in "${CANDIDATES[@]}"; do
    if [[ -n "${AVAILABLE_DATASETS[$ds]:-}" ]]; then
        DATASETS+=("$ds")
    else
        echo "[SKIP] Missing config: configs/${MODEL}/${ds}-${MODEL}.yaml"
    fi
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No runnable datasets selected for group '$GROUP'."
    exit 1
fi

FAILED=()

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

echo "[ENV] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
python - <<'PY'
import torch
print("[ENV] torch:", torch.__version__)
print("[ENV] torch_path:", torch.__file__)
print("[ENV] cuda_available:", torch.cuda.is_available())
PY

for ds in "${DATASETS[@]}"; do
    cfg="configs/${MODEL}/${ds}-${MODEL}.yaml"
    ms_count="$(multi_splits_count "$cfg")"
    if [[ "$ms_count" -gt 0 ]]; then
        run_repeat=1
        echo "[RUN] dataset=$ds model=$MODEL repeat=$run_repeat gpu=$GPU (multi-split config: $ms_count splits)"
    else
        run_repeat="$SINGLE_SPLIT_REPEAT"
        echo "[RUN] dataset=$ds model=$MODEL repeat=$run_repeat gpu=$GPU (single-split config)"
    fi
    CUDA_VISIBLE_DEVICES="$GPU" python main.py --cfg "$cfg" --repeat "$run_repeat" || FAILED+=("$cfg")
done

if [[ "$SUMMARY" -eq 1 ]]; then
export DATASETS_FOR_SUMMARY
DATASETS_FOR_SUMMARY="$(IFS=,; echo "${DATASETS[*]}")"
python - <<'PY'
import csv
import json
import os

model = "GPS"
datasets = os.environ.get("DATASETS_FOR_SUMMARY", "").split(",")
datasets = [d for d in datasets if d]

rows = []
for ds in datasets:
    best = os.path.join("results", f"{ds}-{model}", "agg", "test", "best.json")
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

os.makedirs("results", exist_ok=True)
out = os.path.join("results", "summary_gps_selected.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "dataset", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"])
    w.writerows(rows)
print(f"Wrote {out}")
PY
fi

echo ""
echo "========================================"
echo "Group=$GROUP, model=$MODEL, single_split_repeat=$SINGLE_SPLIT_REPEAT"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "  $f"
    done
else
    echo "All runs succeeded."
fi
echo "========================================"
