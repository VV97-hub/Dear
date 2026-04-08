#!/bin/bash
nworkers="${nworkers:-8}"
bs="${bs:-64}"
dnn="${dnn:-resnet50}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
source ../configs/envs.conf

# ----------------------------------------自己增加的关于环境设置的命令----------------------------------------
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py38-hvd
source /data/home/sczd744/run/dear_pytorch-master/setup_env.sh
export OMPI_MCA_btl_openib_allow_ib=0
export OMPI_MCA_btl="^openib"
# 明确指定 MPI 库路径
export LD_LIBRARY_PATH="/data/home/sczd744/.conda/envs/py38-hvd/lib:${LD_LIBRARY_PATH}"
# 设置 PYTHONPATH
export PYTHONPATH="/data/home/sczd744/run/dear_pytorch-master:${PYTHONPATH}"

if [ "$dnn" = "bert" ] || [ "$dnn" = "bert_base" ]; then
    cmd="$PY bert_benchmark.py --model $dnn --sentence-len $senlen --batch-size $bs"
else
    cmd="$PY imagenet_benchmark.py --model $dnn --batch-size $bs "
fi

#autotune='-x HOROVOD_AUTOTUNE=1 -x HOROVOD_AUTOTUNE_WARMUP_SAMPLES=2 -x HOROVOD_AUTOTUNE_STEPS_PER_SAMPLE=5 -x HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES=10 -x HOROVOD_AUTOTUNE_LOG=auto.log'
autotune=''

#10GbE Config
if [ "$rdma" = "0" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
    -x NCCL_DEBUG=VERSION  \
    -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
    -x NCCL_IB_DISABLE=1 \
    $autotune \
    $cmd
elif [ "$rdma" = "1" ]; then
#100GbIB Config with RDMA 
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=VERSION \
    -x NCCL_LAUNCH_MODE=PARALLEL \
    $autotune \
    $cmd
else
#100GbIB Config with Ethernet
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=VERSION \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_NET_GDR_LEVEL=0 \
    -x NCCL_NET_GDR_READ=0 \
    -x NCCL_IB_CUDA_SUPPORT=0 \
    $cmd
fi

