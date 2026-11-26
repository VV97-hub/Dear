#!/usr/bin/env bash
# bash submit.sh --dry-run      只检查并预览，不申请 GPU
# bash submit.sh                提交 experiment.conf
# bash submit.sh other.conf     提交另一份配置
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$script_dir/submit_experiment.py" "$@"
