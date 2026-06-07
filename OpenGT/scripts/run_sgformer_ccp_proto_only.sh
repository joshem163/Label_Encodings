#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

# Run SGFormer on CCP datasets (cora, citeseer, pubmed) with proto enabled.
#
# Usage:
#   bash scripts/run_sgformer_ccp_proto_only.sh --gpu 0
#   SINGLE_SPLIT_REPEAT=1 bash scripts/run_sgformer_ccp_proto_only.sh --gpu 1 --summary
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
LOG_FILE="logs/SGFormer_ccp_proto.log"
RESULT_ROOT="results_sgformer_ccp_proto"

summary_arg=()
if [[ "$SUMMARY" -eq 1 ]]; then
    summary_arg=(--summary)
fi

echo "[LAUNCH] model=SGFormer group=ccp proto=true lss=true lss_backend=gpu gpu=${GPU} result_root=${RESULT_ROOT} log=${LOG_FILE}"
bash "${SCRIPT_DIR}/run_model_all_22.sh" \
    --model SGFormer \
    --group ccp \
    --gpu "${GPU}" \
    --proto true \
    --lss true \
    --lss-backend gpu \
    --sampler full_batch \
    --result-root "${RESULT_ROOT}" \
    "${summary_arg[@]}" 2>&1 | tee "${LOG_FILE}"

echo "[DONE] SGFormer CCP proto-only run completed."
