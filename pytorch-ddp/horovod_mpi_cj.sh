#!/bin/bash
nworkers="${nworkers:-8}"
bs="${bs:-64}"
dnn="${dnn:-resnet50}"
senlen="${senlen:-64}"
use_zero="${use_zero:-0}"
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
    script=bert_benchmark.py
    params="--model $dnn --sentence-len $senlen --batch-size $bs --use-zero $use_zero"
else
    script=imagenet_benchmark.py
    params="--model $dnn --batch-size $bs --use-zero $use_zero"
fi


# multi-node multi-GPU setting
node_rank=1  # launch node1, node2, ...
ngpu_per_node=4
node_count=$(expr $nworkers / $ngpu_per_node)

if [ $nworkers -lt 4 ]; then # single-node
    ngpu_per_node=$nworkers node_count=1 node_rank=$node_rank rdma=$rdma script=$script params=$params bash launch_torch.sh
else
    ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank rdma=$rdma script=$script params=$params bash launch_torch.sh
fi
