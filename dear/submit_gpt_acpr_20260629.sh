#!/bin/bash
set -euo pipefail

cd /data/home/sczd744/run/dear_pytorch-master/dear

mkdir -p logs/acpr_gpt_20260629/perf \
         logs/acpr_gpt_20260629/convergence \
         logs/acpr_gpt_20260629/resource

SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
if [ -n "$SBATCH_DEPENDENCY" ]; then
  SBATCH_COMMON="sbatch --dependency=$SBATCH_DEPENDENCY -w g[0013,0030,0032,0045] -N 4 -p gpu --gres=gpu:1 --qos=gpugpu"
else
  SBATCH_COMMON='sbatch -w g[0013,0030,0032,0045] -N 4 -p gpu --gres=gpu:1 --qos=gpugpu'
fi

COMMON_ARGS='nworkers=4 dnn=gpt_230m senlen=128 bs=4 threshold=67108864 learning_rate=3e-4 weight_decay=0.01 max_train_tokens=2000000 overlap_profile=0 DEAR_RANK_CHANGE_DEBUG=0'
COMPRESS_ARGS='compressor=halfrankk compress_rank=16 compress_min_numel=16384 active_prefix_enabled=1 embedding_policy=word'

echo "Submitting GPT ACPR experiments at $(date)" | tee logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log
echo "Model: gpt_230m, seq_len=128, batch_size_per_worker=4, nodes=g[0013,0030,0032,0045]" | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

# ------------------------- P series: throughput-oriented runs -------------------------
P_ARGS='num_warmup_batches=200 num_batches_per_iter=10 num_iters=80 loss_log_every=0 compress_warmup=200'

eval "$COMMON_ARGS $P_ARGS compressor=none \
  $SBATCH_COMMON --job-name=GPTP0_gpt230m_dense \
  --output=logs/acpr_gpt_20260629/perf/GPTP0_gpt230m_dense_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

eval "$COMMON_ARGS $P_ARGS $COMPRESS_ARGS compress_refresh_k=0 rank_schedule=fixed \
  $SBATCH_COMMON --job-name=GPTP1_gpt230m_fixed \
  --output=logs/acpr_gpt_20260629/perf/GPTP1_gpt230m_fixed_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

eval "$COMMON_ARGS $P_ARGS $COMPRESS_ARGS compress_refresh_k=0 rank_schedule=aggressive \
  $SBATCH_COMMON --job-name=GPTP9_gpt230m_acpr \
  --output=logs/acpr_gpt_20260629/perf/GPTP9_gpt230m_acpr_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

eval "$COMMON_ARGS $P_ARGS $COMPRESS_ARGS compress_refresh_k=1 rank_schedule=fixed \
  $SBATCH_COMMON --job-name=GPTP8_gpt230m_fullpq \
  --output=logs/acpr_gpt_20260629/perf/GPTP8_gpt230m_fullpq_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

# ------------------------- C series: loss-over-time convergence runs -------------------------
C_ARGS='num_warmup_batches=0 num_batches_per_iter=10 num_iters=300 loss_log_every=10 compress_warmup=500'

eval "$COMMON_ARGS $C_ARGS compressor=none \
  convergence_output=logs/acpr_gpt_20260629/convergence/GPTC0_gpt230m_dense.csv \
  $SBATCH_COMMON --job-name=GPTC0_gpt230m_dense \
  --output=logs/acpr_gpt_20260629/convergence/GPTC0_gpt230m_dense_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

eval "$COMMON_ARGS $C_ARGS $COMPRESS_ARGS compress_refresh_k=0 rank_schedule=fixed \
  convergence_output=logs/acpr_gpt_20260629/convergence/GPTC1_gpt230m_fixed.csv \
  $SBATCH_COMMON --job-name=GPTC1_gpt230m_fixed \
  --output=logs/acpr_gpt_20260629/convergence/GPTC1_gpt230m_fixed_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

eval "$COMMON_ARGS $C_ARGS $COMPRESS_ARGS compress_refresh_k=0 rank_schedule=aggressive \
  convergence_output=logs/acpr_gpt_20260629/convergence/GPTC9_gpt230m_acpr.csv \
  $SBATCH_COMMON --job-name=GPTC9_gpt230m_acpr \
  --output=logs/acpr_gpt_20260629/convergence/GPTC9_gpt230m_acpr_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

eval "$COMMON_ARGS $C_ARGS $COMPRESS_ARGS compress_refresh_k=1 rank_schedule=fixed \
  convergence_output=logs/acpr_gpt_20260629/convergence/GPTC8_gpt230m_fullpq.csv \
  $SBATCH_COMMON --job-name=GPTC8_gpt230m_fullpq \
  --output=logs/acpr_gpt_20260629/convergence/GPTC8_gpt230m_fullpq_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260629/resource/submit_gpt_20260629.log

squeue -u sczd744
