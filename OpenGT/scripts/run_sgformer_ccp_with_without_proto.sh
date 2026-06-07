#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

# Run SGFormer on CCP datasets (cora, citeseer, pubmed)
# in two variants: without proto, then with proto.
#
# Usage:
#   bash scripts/run_sgformer_ccp_with_without_proto.sh --gpu 0
#   SINGLE_SPLIT_REPEAT=5 bash scripts/run_sgformer_ccp_with_without_proto.sh --gpu 1 --summary
#
# Run from repository root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU=0
SUMMARY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --summary) SUMMARY=1; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p logs

summary_arg=()
if [[ "$SUMMARY" -eq 1 ]]; then
    summary_arg=(--summary)
fi

for proto_mode in false true; do
    if [[ "$proto_mode" == "true" ]]; then
        suffix="proto"
    else
        suffix="noproto"
    fi

    result_root="results_sgformer_ccp_${suffix}"
    log_file="logs/SGFormer_ccp_${suffix}.log"

    echo "[LAUNCH] model=SGFormer group=ccp proto=${proto_mode} lss=false lss_backend=gpu gpu=${GPU} result_root=${result_root} log=${log_file}"
    bash "${SCRIPT_DIR}/run_model_all_22.sh" \
        --model SGFormer \
        --group ccp \
        --gpu "${GPU}" \
        --proto "${proto_mode}" \
        --lss false \
        --lss-backend gpu \
        --sampler full_batch \
        --result-root "${result_root}" \
        "${summary_arg[@]}" 2>&1 | tee "${log_file}"
done

echo "[DONE] SGFormer CCP runs completed for both proto settings."
