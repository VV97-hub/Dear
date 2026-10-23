#!/bin/bash
set -e

cd /data/home/sczd744/run/dear_pytorch-master/dear

ROOT=logs/sync_measurement_20260918
RUNNER=measurement_20260918/horovod_mpi_measure_20260918.sh
mkdir -p "$ROOT/perf" "$ROOT/nsys" "$ROOT/perfetto"

common_env=(
    nworkers=2 bs=64 dnn=bert_base senlen=64 compressor=halfrankk
    compress_rank=16 rank_schedule=update_norm_stable stable_rank_levels=16,12,8
    compress_refresh_k=0 active_prefix_enabled=1 rank_reset_on_change=0
    embedding_policy=word loss_log_every=0 rdma=0
)

submit_pair() {
    local category="$1"
    local event_sync="$2"
    local job_name="$3"
    shift 3
    env DEAR_EVENT_SYNC="$event_sync" dear_event_sync="$event_sync" \
        "${common_env[@]}" "$@" \
        sbatch -N 2 -p gpu --gres=gpu:1 --qos=gpugpu \
        --job-name="$job_name" \
        --output="$ROOT/$category/${job_name}_%j.out" \
        "$RUNNER"
}

case "${1:-}" in
    smoke)
        # One-GPU plumbing check only; it is not a paper result.
        common_env[0]="nworkers=1"
        env DEAR_EVENT_SYNC=1 dear_event_sync=1 \
            "${common_env[@]}" overlap_profile=0 nsys_profile=1 \
            compress_warmup=0 num_warmup_batches=0 \
            num_batches_per_iter=1 num_iters=1 \
            nsys_output_prefix="$ROOT/nsys/NSYS_PIPELINE_SMOKE" \
            sbatch -N 1 -p gpu --gres=gpu:1 \
            --job-name=NSYS_PIPELINE_SMOKE \
            --output="$ROOT/nsys/NSYS_PIPELINE_SMOKE_%j.out" \
            "$RUNNER"
        ;;
    perf)
        # End-to-end timing: 30 warm-up steps + 300 measured steps, no profiler.
        submit_pair perf 1 SYNC_PERF_EVENT \
            overlap_profile=0 nsys_profile=0 compress_warmup=20 \
            num_warmup_batches=30 num_batches_per_iter=1 num_iters=300
        submit_pair perf 0 SYNC_PERF_CPU \
            overlap_profile=0 nsys_profile=0 compress_warmup=20 \
            num_warmup_batches=30 num_batches_per_iter=1 num_iters=300
        ;;
    nsys_event)
        # Short CUDA trace: rank 0 only, five compressed training steps in total.
        submit_pair nsys 1 SYNC_NSYS_EVENT \
            overlap_profile=0 nsys_profile=1 compress_warmup=0 \
            num_warmup_batches=1 num_batches_per_iter=1 num_iters=3 \
            nsys_output_prefix="$ROOT/nsys/SYNC_NSYS_EVENT"
        ;;
    nsys_cpu)
        submit_pair nsys 0 SYNC_NSYS_CPU \
            overlap_profile=0 nsys_profile=1 compress_warmup=0 \
            num_warmup_batches=1 num_batches_per_iter=1 num_iters=3 \
            nsys_output_prefix="$ROOT/nsys/SYNC_NSYS_CPU"
        ;;
    perfetto)
        # Lightweight custom timeline; not used for performance numbers.
        submit_pair perfetto 1 SYNC_PERFETTO_EVENT \
            overlap_profile=1 overlap_timeline=1 overlap_timeline_mode=light \
            overlap_summary=0 overlap_console=0 overlap_warmup=1 \
            overlap_timeline_output="$ROOT/perfetto/SYNC_PERFETTO_EVENT.jsonl" \
            nsys_profile=0 compress_warmup=0 \
            num_warmup_batches=1 num_batches_per_iter=1 num_iters=5
        submit_pair perfetto 0 SYNC_PERFETTO_CPU \
            overlap_profile=1 overlap_timeline=1 overlap_timeline_mode=light \
            overlap_summary=0 overlap_console=0 overlap_warmup=1 \
            overlap_timeline_output="$ROOT/perfetto/SYNC_PERFETTO_CPU.jsonl" \
            nsys_profile=0 compress_warmup=0 \
            num_warmup_batches=1 num_batches_per_iter=1 num_iters=5
        ;;
    *)
        echo "Usage: $0 {smoke|perf|nsys_event|nsys_cpu|perfetto}" >&2
        exit 2
        ;;
esac
