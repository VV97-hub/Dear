#!/bin/bash
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"
gpuids="${gpuids:-0,1,2,3}"

export http_proxy=http://10.244.6.36:8080
export https_proxy=http://10.244.6.36:8080
source configs/envs.conf
# source /data/apps/miniforge3/bin/activate py38
# source /data/apps/miniforge3/bin/activate py310
# source /data/home/sczd744/.conda/envs/py38-hvd/bin/activate py38-hvd 没有这个路径
# conda activate py38-hvd
# module load openmpi/4.1.1
#module load cuda/12.1
# module load cuda/11.8
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py38-hvd
source setup_env.sh

# 客服要求新增，防止提交作业报错
#export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_allow_ib=0
export OMPI_MCA_btl="^openib"

# 明确指定 MPI 库路径
export LD_LIBRARY_PATH="/data/home/sczd744/.conda/envs/py38-hvd/lib:${LD_LIBRARY_PATH}"

# 设置 PYTHONPATH
export PYTHONPATH="/data/home/sczd744/run/dear_pytorch-master:${PYTHONPATH}"

# Debug：python路径和torch版本的巡查
echo "当前 Python 路径: $(which python)"
echo "当前环境: $CONDA_DEFAULT_ENV"
python -c "import sys; print('Python 可执行文件:', sys.executable)"
python -c "import torch; print('Torch 版本:', torch.__version__)"

if [ "$rdma" = "0" ]; then
    params="-mca pml ob1 \
            -mca btl tcp,vader,self \
            -x NCCL_DEBUG=VERSION \
            -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
            -x NCCL_IB_DISABLE=1 \
            -x HOROVOD_CACHE_CAPACITY=0 \
            -x CUDA_VISIBLE_DEVICES=${gpuids}"
else
    params="-mca pml ob1 -mca btl tcp,vader,self \
            -x LD_LIBRARY_PATH  \
            -x NCCL_IB_DISABLE=0 \
            -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
            -x NCCL_DEBUG=VERSION \
            -x HOROVOD_CACHE_CAPACITY=0"
fi


# 运行训练
# 也可以运行其他benchmark脚本，例如：


# examples/mnist/pytorch_mnist.py

# Debug: 监控GPU状态,包括温度、功耗、使用率等
# nvidia-smi


$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile configs/cluster${nworkers} -bind-to none -map-by slot \
        $params \
        $PY benchmarks.py