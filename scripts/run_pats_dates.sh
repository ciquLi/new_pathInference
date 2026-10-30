#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_experiment.sh"
BATCH_DIR="${PATHVEIL_BATCH_DIR:-${PROJECT_ROOT}/runs/batches/pats_$(date -u +%Y%m%dT%H%M%SZ)}"
DATES=(20260201 20260301 20260401)

if [[ ! -x "${RUNNER}" ]]; then
    echo "Missing executable formal entry: ${RUNNER}" >&2
    exit 2
fi

mkdir -p "${BATCH_DIR}"
printf 'PATS batch directory: %s\n' "${BATCH_DIR}"
printf 'Dates (sequential): %s\n' "${DATES[*]}"
printf 'Extra arguments: %s\n' "${*:-<none>}"

current_pid=""
cleanup() {
    if [[ -n "${current_pid}" ]] && kill -0 "${current_pid}" 2>/dev/null; then
        kill -INT "${current_pid}" 2>/dev/null || true
        wait "${current_pid}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

for date in "${DATES[@]}"; do
    log="${BATCH_DIR}/${date}.log"
    printf '%s log=%s\n' "${date}" "${log}"
    (
        printf '[%s] start %s\n' "$(date -u +%FT%TZ)" "${date}"
        exec "${RUNNER}" "${date}" "$@"
    ) >"${log}" 2>&1 &
    current_pid=$!
    if wait "${current_pid}"; then
        printf '%s completed\n' "${date}"
    else
        code=$?
        printf '%s exited with code %s (log: %s)\n' "${date}" "${code}" "${log}" >&2
        exit "${code}"
    fi
    current_pid=""
done
