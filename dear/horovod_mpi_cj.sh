#!/bin/bash
nworkers="${nworkers:-4}"
bs="${bs:-64}"
# dnn参数选择模型和数据：cifar10_resnet18/cifar10_resnet34/cifar10_vgg16/cifar100_resnet18/cifar100_resnet34/cifar100_vgg16 （真实 CIFAR 训练任务——ACP测试精度的）
# GPT扩展实验：gpt_125m/gpt_160m/gpt_230m
# 原本的实验 bert_base /bert ————> 对应bert_benchmark  dnn=resnet18、dnn=vgg16 走的是 /mnt/c/Users/sg564/Desktop/Dear/dear/imagenet_benchmark.py:1，那是合成数据吞吐测试
dnn="${dnn:-bert_base}"
if [ -z "${data_dir+x}" ]; then # 只有当没有手动设置 data_dir 时，脚本才自动选择数据目录
    case "$dnn" in
        cifar100_*) data_dir="./cifar100_data" ;; # 以 cifar100_ 开头的模型，默认数据目录是./cifar100_data
        *) data_dir="./cifar10_data" ;;  # 其他则默认是./cifar10_data
    esac
fi
download_dataset="${download_dataset:-0}"
# compressor的选项none、halfrankk、(topk、eftopk，gaussian，signum，efsignum，)
compressor="${compressor:-halfrankk}"
compress_rank="${compress_rank:-}"
compress_warmup="${compress_warmup:-}"
compress_refresh_k="${compress_refresh_k:-}"
compress_min_numel="${compress_min_numel:-}"
rank_schedule="${rank_schedule:-}"
stable_rank_levels="${stable_rank_levels:-}"
update_norm_debug_every="${update_norm_debug_every:-}"
rank_reset_on_change="${rank_reset_on_change:-0}"
active_prefix_enabled="${active_prefix_enabled:-1}"
embedding_policy="${embedding_policy:-word}"
num_warmup_batches="${num_warmup_batches:-}"
num_batches_per_iter="${num_batches_per_iter:-}"
num_iters="${num_iters:-}"
loss_log_every="${loss_log_every:-0}"
convergence_log_every="${convergence_log_every:-0}"
convergence_output="${convergence_output:-}"
comm_stats_output="${comm_stats_output:-}"
comm_stats_every="${comm_stats_every:-1}"
epochs="${epochs:-}"
print_freq="${print_freq:-}"
base_lr="${base_lr:-}"
warmup_epochs="${warmup_epochs:-}"
lr_decay_epochs="${lr_decay_epochs:-}"
lr_decay_factor="${lr_decay_factor:-}"
cifar_workers="${cifar_workers:-}"
seed="${seed:-}"
gpt_data_file="${gpt_data_file:-./wikitext-local/train-00000-of-00001.parquet}"
gpt_tokenizer_dir="${gpt_tokenizer_dir:-./bert-base-uncased-local}"
max_train_tokens="${max_train_tokens:-}"
learning_rate="${learning_rate:-}"
weight_decay="${weight_decay:-}"
dear_event_sync="${dear_event_sync:-${DEAR_EVENT_SYNC:-1}}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
nstreams="${nstreams:-1}"
mgwfbp="${mgwfbp:-0}"
asc="${asc:-0}"
threshold="${threshold:-0}"
exclude_parts="${exclude_parts:-''}"
overlap_profile="${overlap_profile:-0}" # 控制是否打开时间测量
overlap_summary="${overlap_summary:-0}" # 以前设置的时间总结，但是现在没什么用了
overlap_timeline="${overlap_timeline:-1}" # 时间线功能打开，现在主要事件分析的工具
overlap_summary_mode="${overlap_summary_mode:-strict}"
overlap_timeline_mode="${overlap_timeline_mode:-light}" 
overlap_console="${overlap_console:-0}"
overlap_log_every="${overlap_log_every:-10}"
overlap_warmup="${overlap_warmup:-0}"
overlap_dir="${overlap_dir:-./overlap_logs}"
overlap_output="${overlap_output:-}"
overlap_timeline_output="${overlap_timeline_output:-}"
export DEAR_EVENT_SYNC="$dear_event_sync"
source ../configs/envs.conf

### ----------------------------------------获取节点主机名----------------------------------------
echo "获取节点主机名"
scontrol show hostnames
GPUS=`nvidia-smi -L | wc -l`
HOSTFILE=../configs/cluster${SLURM_NNODES}_${SLURM_JOB_ID:-$$}
rm -f "$HOSTFILE"
touch "$HOSTFILE"
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

if [ "$overlap_profile" = "1" ] && [ "$overlap_summary" = "1" ] && [ -z "$overlap_output" ]; then
    mkdir -p "$overlap_dir"
    overlap_output="${overlap_dir}/overlap_${dnn}_${compressor}_bs${bs}_nw${nworkers}_sl${senlen}_job${SLURM_JOB_ID:-nojob}.log"
fi

if [ "$overlap_profile" = "1" ] && [ "$overlap_timeline" = "1" ] && [ -z "$overlap_timeline_output" ]; then
    mkdir -p "$overlap_dir"
    if [ -n "$overlap_output" ]; then
        overlap_timeline_output="${overlap_output}.timeline.jsonl"
    else
        overlap_timeline_output="${overlap_dir}/overlap_${dnn}_${compressor}_bs${bs}_nw${nworkers}_sl${senlen}_job${SLURM_JOB_ID:-nojob}.timeline.jsonl"
    fi
fi

append_overlap_args() {
    local current="$1"
    if [ "$overlap_profile" = "1" ]; then
        current="$current --overlap-profile"
        if [ "$overlap_summary" = "1" ]; then
            current="$current --overlap-summary --overlap-summary-mode $overlap_summary_mode --overlap-output $overlap_output"
        fi
        if [ "$overlap_timeline" = "1" ]; then
            current="$current --overlap-timeline --overlap-timeline-mode $overlap_timeline_mode --overlap-timeline-output $overlap_timeline_output"
        fi
        current="$current --overlap-console $overlap_console --overlap-log-every $overlap_log_every --overlap-warmup $overlap_warmup"
    fi
    echo "$current"
}

append_acpr_args() {
    local current="$1"
    if [ -n "$compress_rank" ]; then
        current="$current --compress-rank $compress_rank"
    fi
    if [ -n "$compress_warmup" ]; then
        current="$current --compress-warmup $compress_warmup"
    fi
    if [ -n "$compress_refresh_k" ]; then
        current="$current --compress-refresh-k $compress_refresh_k"
    fi
    if [ -n "$compress_min_numel" ]; then
        current="$current --compress-min-numel $compress_min_numel"
    fi
    if [ -n "$rank_schedule" ]; then
        current="$current --rank-schedule $rank_schedule"
    fi
    if [ -n "$stable_rank_levels" ]; then
        current="$current --stable-rank-levels $stable_rank_levels"
    fi
    if [ -n "$update_norm_debug_every" ]; then
        current="$current --update-norm-debug-every $update_norm_debug_every"
    fi
    if [ "$rank_reset_on_change" = "1" ]; then
        current="$current --rank-reset-on-change"
    fi
    current="$current --active-prefix-enabled $active_prefix_enabled --embedding-policy $embedding_policy"
    echo "$current"
}

append_benchmark_args() {
    local current="$1"
    if [ -n "$num_warmup_batches" ]; then
        current="$current --num-warmup-batches $num_warmup_batches"
    fi
    if [ -n "$num_batches_per_iter" ]; then
        current="$current --num-batches-per-iter $num_batches_per_iter"
    fi
    if [ -n "$num_iters" ]; then
        current="$current --num-iters $num_iters"
    fi
    if [ "$loss_log_every" != "0" ]; then
        current="$current --loss-log-every $loss_log_every"
    fi
    if [ -n "$convergence_output" ]; then
        current="$current --convergence-output $convergence_output"
    fi
    if [ -n "$comm_stats_output" ]; then
        current="$current --comm-stats-output $comm_stats_output"
        current="$current --comm-stats-every $comm_stats_every"
    fi
    echo "$current"
}

append_cifar_args() {
    local current="$1"
    if [ -n "$epochs" ]; then
        current="$current --epochs $epochs"
    fi
    if [ -n "$base_lr" ]; then
        current="$current --base-lr $base_lr"
    fi
    if [ -n "$warmup_epochs" ]; then
        current="$current --warmup-epochs $warmup_epochs"
    fi
    if [ -n "$lr_decay_epochs" ]; then
        current="$current --lr-decay-epochs $lr_decay_epochs"
    fi
    if [ -n "$lr_decay_factor" ]; then
        current="$current --lr-decay-factor $lr_decay_factor"
    fi
    if [ -n "$cifar_workers" ]; then
        current="$current --workers $cifar_workers"
    fi
    if [ -n "$seed" ]; then
        current="$current --seed $seed"
    fi
    if [ -n "$print_freq" ]; then
        current="$current --print-freq $print_freq"
    fi
    if [ "$convergence_log_every" != "0" ]; then
        current="$current --convergence-log-every $convergence_log_every"
    fi
    if [ -n "$convergence_output" ]; then
        current="$current --convergence-output $convergence_output"
    fi
    if [ -n "$comm_stats_output" ]; then
        current="$current --comm-stats-output $comm_stats_output"
        current="$current --comm-stats-every $comm_stats_every"
    fi
    echo "$current"
}

append_gpt_args() {
    local current="$1"
    if [ -n "$num_warmup_batches" ]; then
        current="$current --num-warmup-batches $num_warmup_batches"
    fi
    if [ -n "$num_batches_per_iter" ]; then
        current="$current --num-batches-per-iter $num_batches_per_iter"
    fi
    if [ -n "$num_iters" ]; then
        current="$current --num-iters $num_iters"
    fi
    if [ -n "$learning_rate" ]; then
        current="$current --learning-rate $learning_rate"
    fi
    if [ -n "$weight_decay" ]; then
        current="$current --weight-decay $weight_decay"
    fi
    if [ -n "$max_train_tokens" ]; then
        current="$current --max-train-tokens $max_train_tokens"
    fi
    if [ -n "$seed" ]; then
        current="$current --seed $seed"
    fi
    if [ "$loss_log_every" != "0" ]; then
        current="$current --loss-log-every $loss_log_every"
    fi
    if [ -n "$convergence_output" ]; then
        current="$current --convergence-output $convergence_output"
    fi
    if [ -n "$comm_stats_output" ]; then
        current="$current --comm-stats-output $comm_stats_output"
        current="$current --comm-stats-every $comm_stats_every"
    fi
    echo "$current"
}

# 前面层层包装起来，cmd{benchfile{选模型}}
if [ "$dnn" = "bert" ] || [ "$dnn" = "bert_base" ]; then
    benchfile="bert_benchmark.py --model $dnn --sentence-len $senlen --exclude-parts $exclude_parts"
    benchfile=$(append_overlap_args "$benchfile")
    benchfile=$(append_acpr_args "$benchfile")
    benchfile=$(append_benchmark_args "$benchfile")
elif [ "$dnn" = "cifar10_resnet18" ] || [ "$dnn" = "cifar10_resnet34" ] || [ "$dnn" = "cifar10_vgg16" ] || [ "$dnn" = "cifar100_resnet18" ] || [ "$dnn" = "cifar100_resnet34" ] || [ "$dnn" = "cifar100_vgg16" ]; then
    benchfile="cifar_benchmark.py --model $dnn --exclude-parts $exclude_parts --data-dir $data_dir"
    if [ "$download_dataset" = "1" ]; then
        benchfile="$benchfile --download-dataset"
    fi
    benchfile=$(append_overlap_args "$benchfile")
    benchfile=$(append_acpr_args "$benchfile")
    benchfile=$(append_cifar_args "$benchfile")
elif [ "$dnn" = "gpt_125m" ] || [ "$dnn" = "gpt_160m" ] || [ "$dnn" = "gpt_230m" ]; then
    benchfile="gpt_benchmark.py --model $dnn --seq-len $senlen --exclude-parts $exclude_parts --data-file $gpt_data_file --tokenizer-dir $gpt_tokenizer_dir"
    benchfile=$(append_overlap_args "$benchfile")
    benchfile=$(append_acpr_args "$benchfile")
    benchfile=$(append_gpt_args "$benchfile")
else
    benchfile="imagenet_benchmark.py --model $dnn --exclude-parts $exclude_parts"
fi

if [ "$overlap_profile" = "1" ]; then
    echo "Overlap summary enabled: $overlap_summary"
    echo "Overlap timeline enabled: $overlap_timeline"
    echo "Overlap summary mode: $overlap_summary_mode"
    echo "Overlap timeline mode: $overlap_timeline_mode"
    if [ "$overlap_summary" = "1" ]; then
        echo "Overlap timing file: $overlap_output"
    fi
    if [ "$overlap_timeline" = "1" ]; then
        echo "Overlap timeline file: $overlap_timeline_output"
    fi
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
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile "$HOSTFILE" -bind-to none --map-by slot\
    -mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
    -x NCCL_DEBUG=VERSION  \
    -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_LAUNCH_MODE=PARALLEL \
    -x DEAR_EVENT_SYNC \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
elif [ "$rdma" = "1" ]; then
#100GbIB Config with RDMA
# -x NCCL_DEBUG=VERSION \ 改成了 -x NCCL_DEBUG=INFO \ 
# 如果以后需要启用: -mca btl_tcp_if_include ${IB_INTERFACE}
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile "$HOSTFILE" -bind-to none --map-by slot\
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=INFO \
    -x NCCL_LAUNCH_MODE=PARALLEL \
    -x DEAR_EVENT_SYNC \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
else
#100GbIB Config with Ethernet
# -x NCCL_DEBUG=VERSION \ 改成了 -x NCCL_DEBUG=INFO \ 
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile "$HOSTFILE" -bind-to none --map-by slot\
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
    -x DEAR_EVENT_SYNC \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
fi
