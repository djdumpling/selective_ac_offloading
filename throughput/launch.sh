#!/usr/bin/env bash
# Convenience launcher for throughput/run_pipeline.py.
#
# Usage:
#   throughput/launch.sh <pp_size> <ac_mode> [extra args...]
#
# Examples:
#   throughput/launch.sh 2 full-ac
#   throughput/launch.sh 4 pipeline-aware --seq 4096 --mbs 1 --microbatches 8
#   throughput/launch.sh 4 no-ac --model llama7b

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <pp_size> <ac_mode> [run_pipeline.py args...]" >&2
    exit 1
fi

PP=$1
AC=$2
shift 2

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${PP}" \
    "${REPO_ROOT}/throughput/run_pipeline.py" \
    --pp "${PP}" --ac "${AC}" "$@"
