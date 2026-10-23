#!/bin/bash
set -euo pipefail
mkdir -p logs/acpr_scaling_20260706/perf logs/acpr_scaling_20260706/resource
LOG=logs/acpr_scaling_20260706/resource/submit_helix_p9_p3_check_20260713.log
: > "$LOG"

echo "Submitting serialized Helix/P9 and no-prefix/P3 checks at $(date)" | tee -a "$LOG"
echo "Order: S1N8 P9 -> S4N6 P9 -> S1N8 P9 -> S4N6 P9 -> S1N8 P3 -> S4N4 P3 -> S1N8 P3 -> S4N4 P3" | tee -a "$LOG"

BASE="bs=64 dnn=bert_base senlen=64 compressor=halfrankk compress_rank=16 compress_warmup=300 compress_refresh_k=0 rank_schedule=aggressive embedding_policy=word overlap_profile=0 loss_log_every=0 num_warmup_batches=500 num_batches_per_iter=10 num_iters=100"
dep=""

submit_one() {
  local scale="$1"; local nworkers="$2"; local nnodes="$3"; local gres="$4"; local code="$5"; local active="$6"; local tag="$7"
  local name="${scale}_${code}_${tag}"
  local dep_arg=""
  if [ -n "$dep" ]; then dep_arg="--dependency=afterany:${dep}"; fi
  local cmd="nworkers=${nworkers} ${BASE} active_prefix_enabled=${active} sbatch ${dep_arg} -x g[0005,0015] -N ${nnodes} -p gpu --gres=gpu:${gres} --qos=gpugpu --job-name=${name} --output=logs/acpr_scaling_20260706/perf/${name}_%j.out horovod_mpi_cj.sh"
  echo "$cmd" | tee -a "$LOG" >&2
  local out
  out=$(eval "$cmd")
  echo "$out" | tee -a "$LOG" >&2
  dep=$(echo "$out" | awk "/Submitted batch job/ {print \$4}")
}

# Helix/P9 repeat checks, intended to verify or improve the current Fig. 10 values.
submit_one S1N8 8 8 1 P9_acpr_aggressive 1 verify1
submit_one S4N6 24 6 4 P9_acpr_aggressive 1 verify1
submit_one S1N8 8 8 1 P9_acpr_aggressive 1 verify2
submit_one S4N6 24 6 4 P9_acpr_aggressive 1 verify2

# No-prefix/P3 repeat checks, intended to test whether current no-prefix values are inflated.
submit_one S1N8 8 8 1 P3_no_active_prefix 0 lowcheck1
submit_one S4N4 16 4 4 P3_no_active_prefix 0 lowcheck1
submit_one S1N8 8 8 1 P3_no_active_prefix 0 lowcheck2
submit_one S4N4 16 4 4 P3_no_active_prefix 0 lowcheck2

squeue -u sczd744 | tee -a "$LOG"
