#!/bin/bash
nworkers="${nworkers:-8}"
bs="${bs:-64}"
dnn="${dnn:-resnet50}"
compressor="${compressor:-none}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
nstreams="${nstreams:-1}"
mgwfbp="${mgwfbp:-0}"
asc="${asc:-0}"
threshold="${threshold:-0}"
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
    benchfile="bert_benchmark.py --model $dnn --sentence-len $senlen"
else
    benchfile="imagenet_benchmark.py --model $dnn"
fi

if [ "$compressor" = "none" ]; then 
    cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --nstreams $nstreams --threshold $threshold"
    if [ "$asc" = "1" ]; then 
        cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --nstreams $nstreams --mgwfbp --asc"
    fi
else
    cmd="$PY $benchfile --density 0.001 --compressor $compressor --batch-size $bs --nstreams $nstreams --threshold 67108864"
fi
echo $cmd

#10GbE Config
if [ "$rdma" = "0" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
    -x NCCL_DEBUG=VERSION  \
    -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
    -x NCCL_IB_DISABLE=1 \
    $cmd
elif [ "$rdma" = "1" ]; then
#100GbIB Config with RDMA
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=VERSION \
    $cmd
else
#100GbIB Config with Ethernet
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_NET_GDR_LEVEL=0 \
    -x NCCL_NET_GDR_READ=0 \
    -x NCCL_IB_CUDA_SUPPORT=0 \
    $cmd
fi

