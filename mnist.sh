#!/bin/bash
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"
gpuids="${gpuids:-0,1,2,3}"

source configs/envs.conf

# 客服要求新增，防止提交作业报错
export OMPI_MCA_btl_openib_allow_ib=1

# 明确指定 MPI 库路径
export LD_LIBRARY_PATH="/data/apps/openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8/lib:${LD_LIBRARY_PATH}"

# 设置 PYTHONPATH
export PYTHONPATH="/data/run01/sczd744/dear_pytorch-master:${PYTHONPATH}"

if [ "$rdma" = "0" ]; then
    params="-mca pml ob1 -mca btl ^openib \
            -mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
            -x NCCL_DEBUG=VERSION \
            -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
            -x NCCL_IB_DISABLE=1 \
            -x HOROVOD_CACHE_CAPACITY=0 \
            -x CUDA_VISIBLE_DEVICES=${gpuids}"
else
    params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
            -mca btl_tcp_if_include ${IB_INTERFACE} \
            --mca btl_openib_want_fork_support 1 \
            -x LD_LIBRARY_PATH  \
            -x NCCL_IB_DISABLE=0 \
            -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
            -x NCCL_DEBUG=VERSION \
            -x HOROVOD_CACHE_CAPACITY=0"
fi


$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile configs/cluster${nworkers} -bind-to none -map-by slot \
        $params \
        $PY examples/mnist/pytorch_mnist.py
