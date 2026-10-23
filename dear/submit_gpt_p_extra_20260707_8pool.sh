#!/bin/bash
set -euo pipefail

cd /data/run01/sczd744/dear_pytorch-master/dear

mkdir -p logs/acpr_gpt_20260707_8pool/perf \
         logs/acpr_gpt_20260707_8pool/resource

SBATCH_COMMON='sbatch -w g[0013,0032,0045,0048,0030,0042,0066,0053] -N 4 -p gpu --gres=gpu:1 --qos=gpugpu'
COMMON_ARGS='nworkers=4 dnn=gpt_230m senlen=128 bs=4 threshold=67108864 learning_rate=3e-4 weight_decay=0.01 max_train_tokens=2000000 overlap_profile=0 DEAR_RANK_CHANGE_DEBUG=0'
P_ARGS='num_warmup_batches=200 num_batches_per_iter=10 num_iters=80 loss_log_every=0 compress_warmup=200'
COMPRESS_BASE='compressor=halfrankk compress_rank=16 compress_min_numel=16384 compress_refresh_k=0 rank_schedule=aggressive embedding_policy=word'

{
  echo "Submitting GPT extra P-series experiments at $(date)"
  echo "Model: gpt_230m, seq_len=128, batch_size_per_worker=4, nodes=g[0013,0032,0045,0048,0030,0042,0066,0053]"
  echo "Extra P-series: P3 no active prefix; P7 CPU synchronize handoff"
} | tee -a logs/acpr_gpt_20260707_8pool/resource/submit_gpt_p_extra_20260707_8pool.log

eval "$COMMON_ARGS $P_ARGS $COMPRESS_BASE active_prefix_enabled=0 \
  $SBATCH_COMMON --job-name=GPTP3_gpt230m_noap_8pool \
  --output=logs/acpr_gpt_20260707_8pool/perf/GPTP3_gpt230m_no_active_prefix_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260707_8pool/resource/submit_gpt_p_extra_20260707_8pool.log

eval "$COMMON_ARGS $P_ARGS $COMPRESS_BASE active_prefix_enabled=1 dear_event_sync=0 \
  $SBATCH_COMMON --job-name=GPTP7_gpt230m_cpusync_8pool \
  --output=logs/acpr_gpt_20260707_8pool/perf/GPTP7_gpt230m_cpu_sync_handoff_%j.out horovod_mpi_cj.sh" \
  | tee -a logs/acpr_gpt_20260707_8pool/resource/submit_gpt_p_extra_20260707_8pool.log

squeue -u sczd744
