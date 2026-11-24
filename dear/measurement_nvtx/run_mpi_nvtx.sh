#!/usr/bin/env bash
# Existing Conda/CUDA/MPI/NCCL environment must already be loaded.
# Arguments before -- are extra MPI arguments; arguments after -- are BERT args.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
export HELIX_NVTX=1
export HELIX_CAPTURE_START="${HELIX_CAPTURE_START:-40}"
export HELIX_CAPTURE_STEPS="${HELIX_CAPTURE_STEPS:-40}"
export HELIX_CAPTURE_TAIL="${HELIX_CAPTURE_TAIL:-1}"
export HELIX_NVTX_AUTOGRAD="${HELIX_NVTX_AUTOGRAD:-1}"
export HELIX_REPORT_DIR="${HELIX_REPORT_DIR:-$PWD/logs/nvtx}"
export NSYS_BIN="${NSYS_BIN:-nsys}"
export DEAR_EVENT_SYNC="${DEAR_EVENT_SYNC:-1}"
export DEAR_OVERLAP_NEEDS_SYNC=0
mkdir -p "$HELIX_REPORT_DIR"
HELIX_REPORT_DIR="$(cd "$HELIX_REPORT_DIR" && pwd)"
export HELIX_REPORT_DIR

mpi_extra=()
while (($#)) && [[ "$1" != -- ]]; do mpi_extra+=("$1"); shift; done
if (($#)); then shift; fi
benchmark_extra=("$@")
if [[ -n "${HELIX_MPI_TCP_IF_INCLUDE:-}" ]]; then
    mpi_extra+=(--mca btl_tcp_if_include "$HELIX_MPI_TCP_IF_INCLUDE")
fi
if [[ ! "${NWORKERS:-2}" =~ ^[0-9]+$ ]] || (( ${NWORKERS:-2} < 2 )); then
    echo 'Use at least two workers: single-worker BERT bypasses the distributed codec/NCCL runtime.' >&2
    exit 2
fi
host_args=()
if [[ -n "${HELIX_HOSTFILE:-}" ]]; then
    host_args=(--hostfile "$HELIX_HOSTFILE")
fi
exports=()
for key in PATH LD_LIBRARY_PATH PYTHONPATH HELIX_NVTX HELIX_CAPTURE_START \
    HELIX_CAPTURE_STEPS HELIX_CAPTURE_TAIL HELIX_NVTX_AUTOGRAD HELIX_REPORT_DIR \
    NSYS_BIN DEAR_EVENT_SYNC DEAR_OVERLAP_NEEDS_SYNC NCCL_SOCKET_IFNAME \
    NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_DISABLE NCCL_DEBUG NCCL_LAUNCH_MODE \
    UCX_MEMTYPE_CACHE UCX_WARN_UNUSED_ENV_VARS SLURM_JOB_ID; do
    if [[ -v "$key" ]]; then exports+=(-x "$key"); fi
done
mpirun_bin="${MPIRUN:-${MPIPATH:+$MPIPATH/bin/}mpirun}"
python_bin="${PYTHON:-${PY:-python}}"
exec "$mpirun_bin" -np "${NWORKERS:-2}" --bind-to none --map-by slot \
    "${host_args[@]}" "${exports[@]}" "${mpi_extra[@]}" \
    bash "$script_dir/profile_rank.sh" "$python_bin" bert_benchmark.py \
    --model bert_base --batch-size 64 --sentence-len 64 \
    --compressor halfrankk --density 0.125 --compress-rank 16 \
    --nstreams 1 --threshold 67108864 \
    --compress-warmup 20 --rank-schedule fixed --compress-refresh-k 0 \
    --active-prefix-enabled 1 --embedding-policy word \
    --num-warmup-batches 30 --num-batches-per-iter 1 --num-iters 60 \
    --loss-log-every 0 --overlap-console 0 \
    "${benchmark_extra[@]}"
