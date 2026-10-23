#!/bin/bash
set -euo pipefail

cd /data/run01/sczd744/dear_pytorch-master/dear

LOG_ROOT=logs/helix_gpt_accuracy_20260712
mkdir -p ${LOG_ROOT}/convergence ${LOG_ROOT}/resource

# Fast resource strategy: only exclude confirmed bad nodes.
# If a strictly identical node set is required, replace this with:
# SBATCH_COMMON='sbatch -w g[0030-0032,0041] -N 4 -p gpu --gres=gpu:1 --qos=gpugpu'
SBATCH_COMMON='sbatch -x g[0005,0015] -N 4 -p gpu --gres=gpu:1 --qos=gpugpu'

COMMON_ARGS='nworkers=4 dnn=gpt_230m senlen=128 bs=4 threshold=67108864 learning_rate=3e-4 weight_decay=0.01 max_train_tokens=2000000 overlap_profile=0 DEAR_RANK_CHANGE_DEBUG=0'
COMPRESS_ARGS='compressor=halfrankk compress_rank=16 compress_min_numel=16384 active_prefix_enabled=1 embedding_policy=word'

# Accuracy/convergence run: no benchmark warmup, because warmup updates the model but is not logged.
# Total logged training = num_batches_per_iter * num_iters = 10 * 500 = 5000 steps.
# Compression starts after 500 steps; loss is logged every 20 steps.
C_ARGS='num_warmup_batches=0 num_batches_per_iter=10 num_iters=500 loss_log_every=20 compress_warmup=500'

{
  echo "Submitting GPT C-series accuracy experiments at $(date)"
  echo "Node policy: ${SBATCH_COMMON}"
  echo "Model: gpt_230m, seq_len=128, local_batch=4, workers=4"
  echo "C_ARGS: ${C_ARGS}"
} | tee ${LOG_ROOT}/resource/submit_gpt_c_accuracy_20260712.log

submit_job() {
  local cmd="$1"
  echo "----" | tee -a ${LOG_ROOT}/resource/submit_gpt_c_accuracy_20260712.log
  echo "$cmd" | tee -a ${LOG_ROOT}/resource/submit_gpt_c_accuracy_20260712.log
  eval "$cmd" 2>&1 | tee -a ${LOG_ROOT}/resource/submit_gpt_c_accuracy_20260712.log
}

# C0: dense fine-grained communication baseline.
submit_job "$COMMON_ARGS $C_ARGS compressor=none \
  convergence_output=${LOG_ROOT}/convergence/GPTC0_gpt230m_dense.csv \
  $SBATCH_COMMON --job-name=GPTC0_gpt230m_dense_acc \
  --output=${LOG_ROOT}/convergence/GPTC0_gpt230m_dense_%j.out horovod_mpi_cj.sh"

# C1: fixed-rank half-step compression. Tests the accuracy effect of one-factor communication with fixed rank.
submit_job "$COMMON_ARGS $C_ARGS $COMPRESS_ARGS compress_refresh_k=0 rank_schedule=fixed \
  convergence_output=${LOG_ROOT}/convergence/GPTC1_gpt230m_fixed.csv \
  $SBATCH_COMMON --job-name=GPTC1_gpt230m_fixed_acc \
  --output=${LOG_ROOT}/convergence/GPTC1_gpt230m_fixed_%j.out horovod_mpi_cj.sh"

# C9: Helix main path. Tests dynamic rank plus half-step factor communication.
submit_job "$COMMON_ARGS $C_ARGS $COMPRESS_ARGS compress_refresh_k=0 rank_schedule=aggressive \
  convergence_output=${LOG_ROOT}/convergence/GPTC9_gpt230m_helix.csv \
  $SBATCH_COMMON --job-name=GPTC9_gpt230m_helix_acc \
  --output=${LOG_ROOT}/convergence/GPTC9_gpt230m_helix_%j.out horovod_mpi_cj.sh"

# C8: full-factor PowerSGD-style refresh every compressed step. Tests whether stale factor state is the accuracy bottleneck.
submit_job "$COMMON_ARGS $C_ARGS $COMPRESS_ARGS compress_refresh_k=1 rank_schedule=fixed \
  convergence_output=${LOG_ROOT}/convergence/GPTC8_gpt230m_fullpq.csv \
  $SBATCH_COMMON --job-name=GPTC8_gpt230m_fullpq_acc \
  --output=${LOG_ROOT}/convergence/GPTC8_gpt230m_fullpq_%j.out horovod_mpi_cj.sh"

squeue -u sczd744
