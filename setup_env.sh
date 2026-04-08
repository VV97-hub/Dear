#!/bin/bash
# setup_env.sh - DEAR 项目环境配置

# 设置 MPI 库路径
# export LD_LIBRARY_PATH="/data/apps/openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8/lib:${LD_LIBRARY_PATH}"

# 预加载 MPI C++ 库
# export LD_PRELOAD="/data/apps/openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8/lib/libmpi_cxx.so.40:${LD_PRELOAD}"

# 添加 PyTorch 库路径
TORCH_LIB=$(python -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))" 2>/dev/null)
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
fi

# 设置 PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
