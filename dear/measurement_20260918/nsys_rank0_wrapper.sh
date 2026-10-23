#!/bin/bash
set -e

rank="${OMPI_COMM_WORLD_RANK:-0}"
if [ "$rank" != "0" ]; then
    exec "$@"
fi

NSYS_ROOT="${NSYS_ROOT:?NSYS_ROOT is required}"
output_prefix="${NSYS_OUTPUT_PREFIX:?NSYS_OUTPUT_PREFIX is required}_rank0"
mkdir -p "$(dirname "$output_prefix")"

exec "$NSYS_ROOT/nsys" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output="$output_prefix" \
    "$@"
