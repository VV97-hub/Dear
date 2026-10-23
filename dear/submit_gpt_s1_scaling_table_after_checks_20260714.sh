#!/bin/bash
set -euo pipefail
mkdir -p logs/helix_gpt_scaling_20260714/perf logs/helix_gpt_scaling_20260714/resource
LOG=logs/helix_gpt_scaling_20260714/resource/submit_gpt_s1_scaling_table_after_checks_20260714.log
: > "$LOG"

echo "Submitting GPT S1 scaling jobs for Table 1 at $(date)" | tee -a "$LOG"
echo "Initial dependency: after BERT repeat-check chain job 1162182" | tee -a "$LOG"
echo "Columns: S1N2, S1N6, S1N8. Rows: Dense, Full-factor, CPU-sync, No active prefix, Helix." | tee -a "$LOG"

COMMON="dnn=gpt_230m senlen=128 bs=4 threshold=67108864 learning_rate=3e-4 weight_decay=0.01 max_train_tokens=2000000 overlap_profile=0 DEAR_RANK_CHANGE_DEBUG=0 num_warmup_batches=200 num_batches_per_iter=10 num_iters=80 loss_log_every=0 compress_warmup=200"
COMP="compressor=halfrankk compress_rank=16 compress_min_numel=16384 embedding_policy=word"
dep="1162182"

submit_one() {
  local scale="$1"; local workers="$2"; local nodes="$3"; local code="$4"; local args="$5"
  local name="GPT${scale}_${code}"
  local cmd="nworkers=${workers} ${COMMON} ${args} sbatch --dependency=afterany:${dep} -x g[0005,0015] -N ${nodes} -p gpu --gres=gpu:1 --qos=gpugpu --job-name=${name} --output=logs/helix_gpt_scaling_20260714/perf/${name}_%j.out horovod_mpi_cj.sh"
  echo "$cmd" | tee -a "$LOG" >&2
  local out
  out=$(eval "$cmd")
  echo "$out" | tee -a "$LOG" >&2
  dep=$(echo "$out" | awk "/Submitted batch job/ {print \$4}")
}

for spec in "S1N2 2 2" "S1N6 6 6" "S1N8 8 8"; do
  set -- $spec
  scale="$1"; workers="$2"; nodes="$3"
  submit_one "$scale" "$workers" "$nodes" P0_dense "compressor=none active_prefix_enabled=1 embedding_policy=word"
  submit_one "$scale" "$workers" "$nodes" P8_fullpq "${COMP} compress_refresh_k=1 rank_schedule=fixed active_prefix_enabled=1"
  submit_one "$scale" "$workers" "$nodes" P7_cpusync "${COMP} compress_refresh_k=0 rank_schedule=aggressive active_prefix_enabled=1 dear_event_sync=0"
  submit_one "$scale" "$workers" "$nodes" P3_noap "${COMP} compress_refresh_k=0 rank_schedule=aggressive active_prefix_enabled=0"
  submit_one "$scale" "$workers" "$nodes" P9_helix "${COMP} compress_refresh_k=0 rank_schedule=aggressive active_prefix_enabled=1"
done

squeue -u sczd744 | tee -a "$LOG"
