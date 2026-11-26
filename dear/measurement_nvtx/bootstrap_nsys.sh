#!/usr/bin/env bash
# 每个计算节点分别执行；沿用已通过冒烟测试的 2026.5.1 安装包解压方式。
set -euo pipefail
: "${NSYS_LOCAL_ROOT:?}" "${NSYS_PACKAGE:?}"
mkdir -p "$NSYS_LOCAL_ROOT"
if [[ ! -x "$NSYS_LOCAL_ROOT/target-linux-x64/nsys" ]]; then
    dpkg-deb --fsys-tarfile "$NSYS_PACKAGE" | tar -x -C "$NSYS_LOCAL_ROOT" \
        --strip-components=5 ./opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64
fi
"$NSYS_LOCAL_ROOT/target-linux-x64/nsys" --version
