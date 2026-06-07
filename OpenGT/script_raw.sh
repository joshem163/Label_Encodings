#!/bin/bash

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 3
#SBATCH -q regular
#SBATCH -J bfq_raw
#SBATCH --mail-user=saikarthik.navuluru@utdallas.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4271_g
#SBATCH -t 06:00:00

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

set -euo pipefail

ROOT_DIR="/pscratch/sd/s/saik1999/Proto/OpenGT"
cd "$ROOT_DIR"

module load conda
if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found after module load conda"
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate GPM

bash scripts/run_sgformer_polyformer_nodeformer_raw_bfq.sh --seeds 3 --datasets flickr,questions
