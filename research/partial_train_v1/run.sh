#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATHVEIL_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=20261015
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$#" -eq 0 ]]; then
    exec "${PATHVEIL_PYTHON:-/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python}" -B "${PROJECT_ROOT}/research/partial_train_v1/dispatch.py"
fi
exec "${PATHVEIL_PYTHON:-/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python}" -B -m pathInference.research.partial_train "$@"
