#!/usr/bin/env bash
# Run all GT and GNN baseline models on all node classification datasets
# Split: random 60-20-20 (configured in all YAML files)
# Usage: bash run_all.sh [--repeat N] [--model MODEL] [--dataset DATASET]

REPEAT=3
FILTER_MODEL=""
FILTER_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --repeat)   REPEAT="$2";          shift 2 ;;
        --model)    FILTER_MODEL="$2";    shift 2 ;;
        --dataset)  FILTER_DATASET="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODELS=(
    # GNN Baselines
    GCN GAT APPNP
    # Graph Transformers
    DIFFormer GPS GRIT SGFormer NodeFormer SAN SpecFormer Exphormer CoBFormer
    Graphormer Graphtransformer PolyFormer
    # GT + Positional Encoding variants
    GPS+RWSE GPS+GE GPS+WLSE
)

DATASETS=(
    cora citeseer pubmed
    amazon-computers amazon-photo
    coauthor-cs coauthor-physics
    wikics corafull
    actor cornell texas wisconsin
    chameleon squirrel wn-chameleon
)

FAILED=()
COUNT=0

for model in "${MODELS[@]}"; do
    [[ -n "$FILTER_MODEL" && "$model" != "$FILTER_MODEL" ]] && continue

    for dataset in "${DATASETS[@]}"; do
        [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]] && continue

        cfg="configs/${model}/${dataset}-${model}.yaml"
        if [[ ! -f "$cfg" ]]; then
            echo "[SKIP] No config: $cfg"
            continue
        fi

        echo "[RUN] model=$model dataset=$dataset repeat=$REPEAT"
        python main.py --cfg "$cfg" --repeat "$REPEAT"

        if [[ $? -ne 0 ]]; then
            echo "[FAIL] $cfg"
            FAILED+=("$cfg")
        fi
        ((COUNT++))
    done
done

echo ""
echo "========================================"
echo "Completed $COUNT experiment(s)."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "  $f"; done
else
    echo "All experiments succeeded."
fi
echo "========================================"
