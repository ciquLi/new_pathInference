#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATHVEIL_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20261015
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# PATS (Prefix-Anchored Tail Stitching) parameters.
# Edit these defaults when running the registered PATS dates. They are copied
# into the run identity, so changing one creates a distinct method identity.
export PATHVEIL_PATS_LOW_ROOT_LIMIT="${PATHVEIL_PATS_LOW_ROOT_LIMIT:-21}"
export PATHVEIL_PATS_RECURSION_TOPK="${PATHVEIL_PATS_RECURSION_TOPK:-3}"
export PATHVEIL_PATS_LOW_TAIL_MIN="${PATHVEIL_PATS_LOW_TAIL_MIN:-300}"
export PATHVEIL_PATS_SOURCE_THRESHOLD="${PATHVEIL_PATS_SOURCE_THRESHOLD:-0.75}"
export PATHVEIL_PATS_MINIMUM_PATHS="${PATHVEIL_PATS_MINIMUM_PATHS:-800}"
export PATHVEIL_PATS_QUERY_CAPACITY="${PATHVEIL_PATS_QUERY_CAPACITY:-200}"
# Training-side PATS generation forks two AS tasks after the shared environment
# is loaded. Test-side generation remains single-worker for label isolation.
export PATHVEIL_PATS_PATHGEN_TRAIN_THREADS="${PATHVEIL_PATS_PATHGEN_TRAIN_THREADS:-4}"
export PATHVEIL_PATS_PATHGEN_TEST_THREADS="${PATHVEIL_PATS_PATHGEN_TEST_THREADS:-4}"
PYTHON="${PATHVEIL_PYTHON:-/home/lxl/anaconda3/envs/rl-aspath-inference/bin/python}"
if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 YYYYMMDD [--check-only | --plan-only | --prepare-only | --upstream-only | --augmentation-only] [--run-dir PATH | --fresh-run] [--device cuda|cpu|auto]" >&2
    exit 2
fi
exec "${PYTHON}" -B "${PROJECT_ROOT}/scripts/run_registered_experiment.py" "$@"
