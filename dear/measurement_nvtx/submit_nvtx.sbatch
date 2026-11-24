#!/usr/bin/env bash
#SBATCH --job-name=HELIX_NVTX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpugpu
#SBATCH --output=helix_nvtx_%j.out
set -euo pipefail
# Repository root and a site-specific env script must be supplied at submission.
# That script loads the original Conda/CUDA/MPI/NCCL environment; no credentials
# or obsolete node/network addresses are embedded here.
: "${HELIX_PROJECT_DIR:?Set HELIX_PROJECT_DIR to the Dear repository root}"
: "${HELIX_ENV_SCRIPT:?Set HELIX_ENV_SCRIPT to the site environment setup script}"
set +u
source "$HELIX_ENV_SCRIPT"
set -u
export HELIX_REPORT_DIR="${HELIX_REPORT_DIR:-$HELIX_PROJECT_DIR/dear/logs/nvtx/$SLURM_JOB_ID}"
mkdir -p "$HELIX_REPORT_DIR"
export HELIX_HOSTFILE="$HELIX_REPORT_DIR/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $0 " slots=1"}' > "$HELIX_HOSTFILE"
export NWORKERS="${NWORKERS:-$SLURM_NNODES}"
exec bash "$HELIX_PROJECT_DIR/dear/measurement_nvtx/run_mpi_nvtx.sh" "$@"
