#!/bin/bash
export http_proxy=http://10.244.6.36:8080
export https_proxy=http://10.244.6.36:8080

export OMPI_MCA_btl_openib_allow_ib=0
export OMPI_MCA_btl="^openib"
export LD_LIBRARY_PATH="/data/home/sczd744/.conda/envs/py38-hvd/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/data/home/sczd744/run/dear_pytorch-master:${PYTHONPATH}"

module load cmake/3.30 
module load cuda/11.8  
module load nccl/2.16.5_cu118 
module load gcc/11.4.0
# 新增，因为conda没有安装openmpi
module load openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8


# 环境设置
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py38-hvd
source /data/home/sczd744/run/dear_pytorch-master/setup_env.sh


# 检查
python -c "import torch;print(torch.cuda.is_available())"
which cmake
which pip
which python



# ===== 编译器设置 =====
# C++17 设置（多重保险）
export CXXFLAGS="-std=c++17"
export CFLAGS="-std=c11"
export CMAKE_CXX_STANDARD=17
export CMAKE_CXX_STANDARD_REQUIRED=ON
export CMAKE_CXX_FLAGS="-std=c++17"
export HOROVOD_CXX_FLAGS="-std=c++17"
export CC=/data/apps/gcc/11.4.0/bin/gcc
export CXX=/data/apps/gcc/11.4.0/bin/g++
export AR=$(which gcc-ar)
export CMAKE_AR=$(which gcc-ar)
export CMAKE_C_COMPILER=$CC
export CMAKE_CXX_COMPILER=$CXX
export CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17"

# 强制 CMake 使用 C++17
export HOROVOD_BUILD_ARCH_FLAGS="-std=c++17"
# 安装 horovod

HOROVOD_WITH_PYTORCH=1 \
HOROVOD_WITH_TENSORFLOW=0 \
HOROVOD_GPU_OPERATIONS=NCCL \
HOROVOD_CMAKE=/data/apps/cmake/3.30/bin/cmake \
HOROVOD_CUDA_HOME=/data/apps/cuda/11.8 \
HOROVOD_NCCL_HOME=/data/apps/nccl/2.16.5-1+cuda11.8 \
HOROVOD_NCCL_INCLUDE=/data/apps/nccl/2.16.5-1+cuda11.8/include \
HOROVOD_NCCL_LIB=/data/apps/nccl/2.16.5-1+cuda11.8/lib/libnccl.so \
CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" \
python -m pip install --no-cache-dir horovod==0.28.1

# 直接运行benchmarks.py
# python benchmarks.py