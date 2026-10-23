#!/bin/bash
set -euo pipefail

mkdir -p logs/acpr_scaling_20260706/perf logs/acpr_scaling_20260706/resource
LOG=logs/acpr_scaling_20260706/resource/submit_bert_helix_repeat_20260712.log
: > "$LOG"

echo "Submitting BERT Helix repeat runs at $(date)" | tee -a "$LOG"
echo "Repeats: S1N6, S1N8, S4N4, S4N6; P9/aggressive only; exclude g0005,g0015" | tee -a "$LOG"

BASE="bs=64 dnn=bert_base senlen=64 compressor=halfrankk compress_rank=16 compress_warmup=300 compress_refresh_k=0 rank_schedule=aggressive active_prefix_enabled=1 embedding_policy=word overlap_profile=0 loss_log_every=0 num_warmup_batches=500 num_batches_per_iter=10 num_iters=100"

submit_one() {
  local scale="$1"
  local nworkers="$2"
  local nnodes="$3"
  local gres="$4"
  local rep="$5"
  local name="${scale}_P9_acpr_aggressive_rep${rep}"
  local cmd="nworkers=${nworkers} ${BASE} sbatch -x g[0005,0015] -N ${nnodes} -p gpu --gres=gpu:${gres} --qos=gpugpu --job-name=${name} --output=logs/acpr_scaling_20260706/perf/${name}_%j.out horovod_mpi_cj.sh"
  echo "$cmd" | tee -a "$LOG"
  eval "$cmd" | tee -a "$LOG"
}

for rep in 1 2 3; do
  submit_one S1N6 6 6 1 "$rep"
  submit_one S1N8 8 8 1 "$rep"
  submit_one S4N4 16 4 4 "$rep"
  submit_one S4N6 24 6 4 "$rep"
done

squeue -u sczd744 -o "%.18i %.40j %.2t %.8D %R" | tee -a "$LOG"
