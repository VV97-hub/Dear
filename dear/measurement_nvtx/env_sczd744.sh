#!/usr/bin/env bash
# Original sczd744 software/network settings. Adapt this file on a new cluster.
# Source from submit_nvtx.sbatch; this file contains no proxy or login secrets.
: "${HELIX_PROJECT_DIR:?Set HELIX_PROJECT_DIR to the Dear repository root}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py38-hvd
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8
source "$HELIX_PROJECT_DIR/configs/envs.conf"
source "$HELIX_PROJECT_DIR/setup_env.sh"
export PYTHON="$CONDA_PREFIX/bin/python"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MPIPATH/lib:$LD_LIBRARY_PATH"
export NSYS_BIN="${NSYS_BIN:-/data/home/sczd744/tools/nsight-systems-2026.5.1/target-linux-x64/nsys}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_bond_0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_LAUNCH_MODE="${NCCL_LAUNCH_MODE:-PARALLEL}"
export UCX_MEMTYPE_CACHE=n
export HELIX_MPI_TCP_IF_INCLUDE="${HELIX_MPI_TCP_IF_INCLUDE:-$ETH_MPI_BTC_TCP_IF_INCLUDE}"
