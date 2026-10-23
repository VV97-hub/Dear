#!/bin/bash
set -euo pipefail

LOG_ROOT=logs/helix_gpt_accuracy_20260712_long
mkdir -p "$LOG_ROOT/convergence" "$LOG_ROOT/resource"

echo "SUBMIT_TIME=$(date)" | tee "$LOG_ROOT/resource/submit_snapshot.log"
echo "PWD=$(pwd)" | tee -a "$LOG_ROOT/resource/submit_snapshot.log"
echo "CODE_SNAPSHOT=$(git rev-parse --short HEAD 2>/dev/null || echo no-git)" | tee -a "$LOG_ROOT/resource/submit_snapshot.log"

SBATCH_COMMON="sbatch -x g[0005,0015] -N 4 -p gpu --gres=gpu:1 --qos=gpugpu"
COMMON_ARGS="nworkers=4 dnn=gpt_230m senlen=128 bs=4 threshold=67108864 learning_rate=3e-4 weight_decay=0.01 max_train_tokens=2000000 overlap_profile=0 DEAR_RANK_CHANGE_DEBUG=0"
COMPRESS_ARGS="compressor=halfrankk compress_rank=16 compress_min_numel=16384 active_prefix_enabled=1 embedding_policy=word"
LONG_ARGS="num_warmup_batches=5000 num_batches_per_iter=10 num_iters=6000 loss_log_every=100 compress_warmup=5000"

submit_job() {
  local name="$1"
  local args="$2"
  echo "Submitting $name"
  env $COMMON_ARGS $args $LONG_ARGS convergence_output="$LOG_ROOT/convergence/${name}_%j.csv" \
    $SBATCH_COMMON --job-name="$name" --output="$LOG_ROOT/convergence/${name}_%j.out" horovod_mpi_cj.sh
}

submit_job GPTC0_gpt230m_dense_long "compressor=none active_prefix_enabled=1 embedding_policy=word"
submit_job GPTC1_gpt230m_fixed_long "$COMPRESS_ARGS compress_refresh_k=0 rank_schedule=fixed"
submit_job GPTC9_gpt230m_helix_long "$COMPRESS_ARGS rank_schedule=aggressive compress_refresh_k=0"
submit_job GPTC8_gpt230m_fullpq_long "$COMPRESS_ARGS compress_refresh_k=1 rank_schedule=fixed"
