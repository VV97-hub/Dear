#!/usr/bin/env bash
# 所有资源参数由 submit.sh 从 conf 读取后传给 sbatch，不在这里重复设置。
set -euo pipefail
if (($# != 3)); then
    echo 'Use: bash submit.sh [experiment.conf]; do not submit job.sbatch directly.' >&2
    exit 2
fi
# Slurm 会把本脚本复制到 spool，不能用 dirname "$0" 定位项目。
exec python3 "$2/submit_experiment.py" --run-job "$1" --run-dir "$3"
