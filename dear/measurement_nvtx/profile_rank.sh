#!/usr/bin/env bash
# Run inside each MPI worker. Do not put nsys around mpirun on the login node.
set -euo pipefail
rank="${OMPI_COMM_WORLD_RANK:-${RANK:-0}}"
nsys_bin="${NSYS_BIN:-nsys}"
output_dir="${HELIX_REPORT_DIR:?Set HELIX_REPORT_DIR to a shared writable directory}"
mkdir -p "$output_dir"
exec "$nsys_bin" profile \
    --trace=cuda,nvtx,osrt --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="$output_dir/helix_${SLURM_JOB_ID:-local}_rank${rank}" \
    "$@"
