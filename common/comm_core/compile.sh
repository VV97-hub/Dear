#!/bin/bash
set -e

# Build comm_core with the same CUDA/OpenMPI stack used by the training scripts.
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py38-hvd

source /etc/profile.d/modules.sh
module load cuda/11.8
module load gcc/11.4.0
module load openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8

export CUDA_HOME=/data/apps/cuda/11.8
export CUDA_PATH=$CUDA_HOME
export CUDA_DIR=$CUDA_HOME

export MPI_HOME=/data/apps/openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8
export NCCL_HOME=${NCCL_HOME:-$CONDA_PREFIX}
export LD_LIBRARY_PATH=$MPI_HOME/lib:$CUDA_HOME/lib64:$NCCL_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python3 setup.py clean
python3 setup.py install
