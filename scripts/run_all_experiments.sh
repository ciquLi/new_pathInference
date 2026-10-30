#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PATHVEIL_PYTHON:-/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
BATCH_DIR="${PATHVEIL_BATCH_DIR:-${PROJECT_ROOT}/runs/batches/priority_$(date -u +%Y%m%dT%H%M%SZ)}"
exec "${PYTHON}" -B "${SCRIPT_DIR}/run_priority_parallel.py" --batch-dir "${BATCH_DIR}" --max-downstream-jobs 3 "$@"
