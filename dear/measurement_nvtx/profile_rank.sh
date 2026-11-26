#!/usr/bin/env bash
# Run inside each MPI worker. Do not put nsys around mpirun on the login node.
set -euo pipefail
rank="${OMPI_COMM_WORLD_RANK:-${RANK:-0}}"
nsys_bin="${NSYS_BIN:-nsys}"
output_dir="${HELIX_REPORT_DIR:?Set HELIX_REPORT_DIR to a shared writable directory}"
mkdir -p "$output_dir"
sample="${HELIX_NSYS_SAMPLE:-none}"
if [[ "$sample" == process-tree ]]; then
    # 在实际计算节点检查；登录节点的 perf 权限不代表计算节点的权限。
    cpu_status="$("$nsys_bin" status --environment 2>&1)"
    printf '[rank %s] CPU sampling check:\n%s\n' "$rank" "$cpu_status"
    if printf '%s\n' "$cpu_status" | grep -Eq 'CPU Profiling Environment \(process-tree\):[[:space:]]*Fail'; then
        echo "[rank $rank] CPU 热点采样不可用：当前节点 perf 权限或采样触发器不支持。请管理员检查 nsys status --environment；本次停止启动，避免生成缺少 CPU 采样的报告。仅测 GPU 可将 sample 改回 none。" >&2
        exit 1
    fi
fi
exec "$nsys_bin" profile \
    --trace="${HELIX_NSYS_TRACE:-cuda,nvtx,osrt}" --sample="$sample" \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="$output_dir/helix_${SLURM_JOB_ID:-local}_rank${rank}" \
    "$@"
