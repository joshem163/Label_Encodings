#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH --ntasks-per-node=4
#SBATCH -q regular
#SBATCH -J opengt_22d_8gpu
#SBATCH --mail-user=saikarthik.navuluru@utdallas.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4271_g
#SBATCH -t 8:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export PYTHONNOUSERSITE=1

cd /pscratch/sd/s/saik1999/Proto/OpenGT

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate GPM

echo "[ENV] host=$(hostname)"
echo "[ENV] conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "[ENV] python=$(which python)"
python - <<'PY'
import sys
print("[ENV] sys.executable:", sys.executable)
PY

if [[ "${CONDA_DEFAULT_ENV:-}" != "GPM" ]]; then
  echo "[FATAL] Conda env is not GPM; got '${CONDA_DEFAULT_ENV:-unset}'"
  exit 1
fi

REPEAT="${REPEAT:-1}"

DATASETS=(
  cora citeseer pubmed
  amazon-computers amazon-photo
  coauthor-cs
  wikics corafull
  texas cornell wisconsin actor
  squirrel chameleon
  roman-empire amazon-ratings minesweeper tolokers questions
  flickr blogcatalog
)

run_models() {
  local worker="$1"
  shift
  local models=("$@")
  local visible="${CUDA_VISIBLE_DEVICES:-UNSET}"

  echo "[W${worker}] START host=$(hostname) cuda_visible=${visible} models=${models[*]}"
  for model in "${models[@]}"; do
    local fail=0
    for ds in "${DATASETS[@]}"; do
      local cfg="configs/${model}/${ds}-${model}.yaml"
      if [[ ! -f "${cfg}" ]]; then
        echo "[W${worker}][SKIP] missing ${cfg}"
        continue
      fi
      echo "[W${worker}][RUN] model=${model} dataset=${ds} repeat=${REPEAT}"
      python main.py --cfg "${cfg}" --repeat "${REPEAT}" || fail=1
    done
    if [[ "${fail}" -ne 0 ]]; then
      echo "[W${worker}][FAIL] model=${model} had failures"
    else
      echo "[W${worker}][DONE] model=${model}"
    fi
  done
  echo "[W${worker}] COMPLETE"
}

run_worker() {
  local wid="${SLURM_PROCID:-0}"
  case "${wid}" in
    0) run_models "${wid}" Graphormer NodeFormer ;;
    1) run_models "${wid}" DIFFormer GPS ;;
    2) run_models "${wid}" GRIT SpecFormer ;;
    3) run_models "${wid}" Exphormer SAN ;;
    4) run_models "${wid}" SGFormer ;;
    5) run_models "${wid}" CoBFormer ;;
    6) run_models "${wid}" PolyFormer ;;
    7) run_models "${wid}" Graphtransformer ;;
    *) echo "[W${wid}] no assigned workload";;
  esac
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

echo "[MASTER] Launching 8 workers across 2 nodes (4 tasks/node, 1 GPU/task)"
srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=8 \
  bash /pscratch/sd/s/saik1999/Proto/OpenGT/script.sh --worker
echo "[MASTER] ALL DONE"
