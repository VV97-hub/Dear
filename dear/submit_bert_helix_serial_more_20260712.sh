#!/bin/bash
set -euo pipefail
mkdir -p logs/acpr_scaling_20260706/perf logs/acpr_scaling_20260706/resource
LOG=logs/acpr_scaling_20260706/resource/submit_bert_helix_serial_more_20260712.log
: > "$LOG"

echo "Submitting additional serialized BERT Helix repeats at $(date)" | tee -a "$LOG"
echo "Goal: add stable repeats for S1N6/S1N8/S4N4 after current GPT/S4N4 jobs." | tee -a "$LOG"

BASE="bs=64 dnn=bert_base senlen=64 compressor=halfrankk compress_rank=16 compress_warmup=300 compress_refresh_k=0 rank_schedule=aggressive active_prefix_enabled=1 embedding_policy=word overlap_profile=0 loss_log_every=0 num_warmup_batches=500 num_batches_per_iter=10 num_iters=100"

submit_dep() {
  local scale="$1"
  local nworkers="$2"
  local nnodes="$3"
  local gres="$4"
  local tag="$5"
  local dep="$6"
  local name="${scale}_P9_acpr_aggressive_${tag}"
  local dep_arg=""
  if [ -n "$dep" ]; then dep_arg="--dependency=afterany:${dep}"; fi
  local cmd="nworkers=${nworkers} ${BASE} sbatch ${dep_arg} -x g[0005,0015] -N ${nnodes} -p gpu --gres=gpu:${gres} --qos=gpugpu --job-name=${name} --output=logs/acpr_scaling_20260706/perf/${name}_%j.out horovod_mpi_cj.sh"
  echo "$cmd" | tee -a "$LOG" >&2
  local out
  out=$(eval "$cmd")
  echo "$out" | tee -a "$LOG" >&2
  echo "$out" | awk "/Submitted batch job/ {print \$4}"
}

# Wait for the current long GPT dense run and the already submitted S4N4 job.
dep="1159781:1160810"

jid=$(submit_dep S1N6 6 6 1 serial4 "$dep")
dep="$jid"
jid=$(submit_dep S1N8 8 8 1 serial4 "$dep")
dep="$jid"
jid=$(submit_dep S4N4 16 4 4 serial2 "$dep")
dep="$jid"
jid=$(submit_dep S1N6 6 6 1 serial5 "$dep")
dep="$jid"
jid=$(submit_dep S1N8 8 8 1 serial5 "$dep")
dep="$jid"
jid=$(submit_dep S4N4 16 4 4 serial3 "$dep")
dep="$jid"

squeue -u sczd744 | tee -a "$LOG"
