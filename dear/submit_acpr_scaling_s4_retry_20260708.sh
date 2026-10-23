#!/bin/bash
set -uo pipefail
cd /data/run01/sczd744/dear_pytorch-master/dear
mkdir -p logs/acpr_scaling_20260706/perf logs/acpr_scaling_20260706/resource
LOG=logs/acpr_scaling_20260706/resource/submit_s4_retry_20260708.log
{
  echo "Submitting S4 retry jobs at $(date)"
  echo "Code snapshot: $(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
  echo "bert fix: $(grep -n \"Pin each local MPI rank\" bert_benchmark.py || true)"
} | tee "$LOG"

SBATCH_POOL="sbatch -w g[0013,0032,0045,0048,0030,0042,0066,0053] -p gpu --qos=gpugpu"
SBATCH_S4_N2="$SBATCH_POOL -N 2 --gres=gpu:4"
SBATCH_S4_N4="$SBATCH_POOL -N 4 --gres=gpu:4"
SBATCH_S4_N8="$SBATCH_POOL -N 8 --gres=gpu:4"

submit_cmd() {
  local cmd="$1"
  echo "----" | tee -a "$LOG"
  echo "$cmd" | tee -a "$LOG"
  eval "$cmd" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "FAILED rc=$rc: $cmd" | tee -a "$LOG"
  fi
}

COMMON="bs=64 dnn=bert_base senlen=64 overlap_profile=0 loss_log_every=0 num_warmup_batches=500 num_batches_per_iter=10 num_iters=100"
COMP="compressor=halfrankk compress_rank=16 compress_warmup=300 compress_refresh_k=0 embedding_policy=word"

for spec in \
  "S4N2 8 \$SBATCH_S4_N2" \
  "S4N4 16 \$SBATCH_S4_N4" \
  "S4N8 32 \$SBATCH_S4_N8"
do
  set -- $spec
  tag=$1
  nw=$2
  sbatch_cmd="${@:3}"
  submit_cmd "nworkers=$nw $COMMON compressor=none $sbatch_cmd --job-name=${tag}_P0_dense_retry --output=logs/acpr_scaling_20260706/perf/${tag}_P0_dense_retry_%j.out horovod_mpi_cj.sh"
  submit_cmd "nworkers=$nw $COMMON $COMP rank_schedule=fixed active_prefix_enabled=1 $sbatch_cmd --job-name=${tag}_P1_fixed_rank_retry --output=logs/acpr_scaling_20260706/perf/${tag}_P1_fixed_rank_retry_%j.out horovod_mpi_cj.sh"
  submit_cmd "nworkers=$nw $COMMON $COMP rank_schedule=aggressive active_prefix_enabled=1 $sbatch_cmd --job-name=${tag}_P9_acpr_aggressive_retry --output=logs/acpr_scaling_20260706/perf/${tag}_P9_acpr_aggressive_retry_%j.out horovod_mpi_cj.sh"
  submit_cmd "nworkers=$nw $COMMON $COMP rank_schedule=aggressive active_prefix_enabled=1 dear_event_sync=0 $sbatch_cmd --job-name=${tag}_P7_cpu_sync_handoff_retry --output=logs/acpr_scaling_20260706/perf/${tag}_P7_cpu_sync_handoff_retry_%j.out horovod_mpi_cj.sh"
  submit_cmd "nworkers=$nw $COMMON $COMP rank_schedule=aggressive active_prefix_enabled=0 $sbatch_cmd --job-name=${tag}_P3_no_active_prefix_retry --output=logs/acpr_scaling_20260706/perf/${tag}_P3_no_active_prefix_retry_%j.out horovod_mpi_cj.sh"
done

squeue -u sczd744 -o "%.18i %.9P %.40j %.8u %.2t %.10M %.6D %R" | tee -a "$LOG"
