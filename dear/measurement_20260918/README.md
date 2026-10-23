# Helix CUDA-event synchronization measurement suite (2026-09-18)

This suite is isolated from the mature Helix source tree. It reuses the verified
BERT-base/P2 configuration and changes only measurement controls and run length.

## Categories

| Category | Event job | CPU-sync job | Purpose |
|---|---:|---:|---|
| End-to-end performance | 1295159 | 1295160 | 30 warm-up + 300 measured steps; no profiler |
| Nsight Systems | 1295151 | 1295156 | rank-0 CUDA/NVTX/OSRT trace; five total training steps |
| Perfetto JSON | 1295154 | 1295155 | light custom timeline; display only |
| Pipeline smoke | 1295158 | — | one-GPU plumbing check; not a paper result |

All paired paper tests request two nodes with one RTX 4090 per node. The common
configuration is BERT-base, batch size 64, sentence length 64, halfrankk rank 16,
P2 update-norm schedule, active-prefix enabled, and word-embedding policy.

## Server locations

- Scripts: `/data/home/sczd744/run/dear_pytorch-master/dear/measurement_20260918/`
- Outputs: `/data/home/sczd744/run/dear_pytorch-master/dear/logs/sync_measurement_20260918/`
- Nsight CLI package: `/data/home/sczd744/tools/packages/NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb`

The Nsight package is unpacked into the allocated node's `/tmp` at job start.
Only rank 0 is profiled, so the shared filesystem stores reports rather than the
458 MiB expanded CLI installation.

## Submission commands

```bash
cd /data/home/sczd744/run/dear_pytorch-master/dear
measurement_20260918/submit_sync_measurement_suite.sh perf
measurement_20260918/submit_sync_measurement_suite.sh nsys_event
measurement_20260918/submit_sync_measurement_suite.sh nsys_cpu
measurement_20260918/submit_sync_measurement_suite.sh perfetto
```

## Status

At submission time the public partition had only one immediately available GPU,
so all two-GPU jobs were pending for `Priority`. Do not interpret this as a test
or script failure.
