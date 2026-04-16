#!/bin/bash
nworkers="${nworkers:-4}"
bs="${bs:-64}"
dnn="${dnn:-bert_base}"
# compressor的选项none、halfrankk、(topk、eftopk，gaussian，signum，efsignum，)
compressor="${compressor:-halfrankk}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
nstreams="${nstreams:-1}"
mgwfbp="${mgwfbp:-0}"
asc="${asc:-0}"
threshold="${threshold:-0}"
exclude_parts="${exclude_parts:-''}"
overlap_profile="${overlap_profile:-1}"
overlap_console="${overlap_console:-0}"
overlap_log_every="${overlap_log_every:-10}"
overlap_warmup="${overlap_warmup:-0}"
overlap_dir="${overlap_dir:-./overlap_logs}"
overlap_output="${overlap_output:-}"
source ../configs/envs.conf

### ----------------------------------------获取节点主机名----------------------------------------
echo "获取节点主机名"
scontrol show hostnames
GPUS=`nvidia-smi -L | wc -l`
HOSTFILE=../configs/cluster$SLURM_NNODES
rm $HOSTFILE
touch $HOSTFILE
for nodename in `scontrol show hostnames`
do
    echo "${nodename} slots=${GPUS}"
    echo "${nodename} slots=${GPUS}" >> ${HOSTFILE}
done
echo "HOSTFILE: ${HOSTFILE}"
cat $HOSTFILE

# 新增 new！测试检查GPU情况
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# 查看MPIRUN在哪里
which mpirun

# ----------------------------------------环境变量设置（下面是我增加的）----------------------------------------
# export http_proxy=http://10.244.6.36:8080
# export https_proxy=http://10.244.6.36:8080

export http_proxy=http://u-MtfrT7:vH5orjDV@127.0.0.1:3128
export https_proxy=http://u-MtfrT7:vH5orjDV@127.0.0.1:3128 

# -------------------------------------------查看NCCL报错详细信息----------------------------------------
export NCCL_DEBUG=WARN 
# export NCCL_DEBUG=INFO
unset NCCL_DEBUG
# export NCCL_DEBUG_SUBSYS=ALL
unset NCCL_DEBUG_SUBSYS
# 新增 new！：强制简单模式（快速定位）
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
unset NCCL_P2P_DISABLE=1
unset NCCL_IB_DISABLE=1

# ----------------------------------------自己增加的关于环境设置的命令----------------------------------------
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py38-hvd
source /data/home/sczd744/run/dear_pytorch-master/setup_env.sh
# export OMPI_MCA_btl_openib_allow_ib=0
# export OMPI_MCA_btl="^openib"

module load openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8

# 明确指定 MPI 库路径
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/data/apps/openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8/lib:$LD_LIBRARY_PATH"
# 设置 PYTHONPATH
export PYTHONPATH="/data/home/sczd744/run/dear_pytorch-master:${PYTHONPATH}"

# ----------------------------------------跨节点通信参数----------------------------------------
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

if [ "$overlap_profile" = "1" ] && [ -z "$overlap_output" ]; then
    mkdir -p "$overlap_dir"
    overlap_output="${overlap_dir}/overlap_${dnn}_${compressor}_bs${bs}_nw${nworkers}_sl${senlen}_job${SLURM_JOB_ID:-nojob}.log"
fi

# 前面层层包装起来，cmd{benchfile{选模型}}
if [ "$dnn" = "bert" ] || [ "$dnn" = "bert_base" ]; then
    benchfile="bert_benchmark.py --model $dnn --sentence-len $senlen --exclude-parts $exclude_parts"
    if [ "$overlap_profile" = "1" ]; then
        benchfile="$benchfile --overlap-profile --overlap-console $overlap_console --overlap-log-every $overlap_log_every --overlap-warmup $overlap_warmup --overlap-output $overlap_output"
    fi
else
    benchfile="imagenet_benchmark.py --model $dnn --exclude-parts $exclude_parts"
fi

if [ "$overlap_profile" = "1" ]; then
    echo "Overlap timing file: $overlap_output"
fi

if [ "$compressor" = "none" ]; then # 不压缩 TODO压缩是压缩什么内容？貌似compressor要么=none，要么=fp16
    cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --nstreams $nstreams --threshold $threshold"
    if [ "$asc" = "1" ]; then 
        cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --nstreams $nstreams --asc"
    fi
else # 压缩
    cmd="$PY $benchfile --density 0.125 --compressor $compressor --batch-size $bs --nstreams $nstreams --threshold 67108864"
fi
echo $cmd

# 下面三个命令都修改了 -map-by slot 改为 -map-by ppr:1:gpu。改为--map-by ppr:1:node （用不了）
# 下面三个命令都删去了 ： --oversubscribe
#10GbE Config
# 下面是跨节点的改发
# -x NCCL_DEBUG=VERSION \ 改成了 -x NCCL_DEBUG=INFO \ 可以看NCCL的报错信息
# -x NCCL_IB_DISABLE=1 改成了0，启用IB节点间通信
# -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \ 改成了bond0，${ETH_INTERFACE}是setup_env里面配置的变量，使用bond0进行通信。
# -hostfile ../configs/cluster$nworkers 改成了-H node1,node2,node3,node4 （第一行参数）取消用hostfile控制节点。因为localhost可以用，但gpu1、gpu2这些看不到
if [ "$rdma" = "0" ]; then
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none --map-by slot\
    -mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
    -x NCCL_DEBUG=VERSION  \
    -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_LAUNCH_MODE=PARALLEL \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
elif [ "$rdma" = "1" ]; then
#100GbIB Config with RDMA
# -x NCCL_DEBUG=VERSION \ 改成了 -x NCCL_DEBUG=INFO \ 
# 如果以后需要启用: -mca btl_tcp_if_include ${IB_INTERFACE}
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none --map-by slot\
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=INFO \
    -x NCCL_LAUNCH_MODE=PARALLEL \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
else
#100GbIB Config with Ethernet
# -x NCCL_DEBUG=VERSION \ 改成了 -x NCCL_DEBUG=INFO \ 
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none --map-by slot\
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
    -x NCCL_LAUNCH_MODE=PARALLEL \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
fi
