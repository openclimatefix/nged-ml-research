#!/usr/bin/env bash
# Experiment 721, wave 1: stage 0, the probe of fold 2026-05, the synthetic fits of seed 0, the quick look.
#
# Usage:  SLOT_CPU=<n> DATA_PATH_INTERNAL=<data folder> ./wave1.sh [--list | STEP]
#   no argument   run every step in order and stop at the first failure
#   --list        print the steps (number, then the arguments of `e00721_cli.py`) and run nothing
#   STEP          run the one step with that number (a step that already finished is skipped by the fit)
#
# Every step goes through `run.sh` (one core, one thread per library, 12 GB, the GPU gates). A step
# that fails stops the script with that step's exit status; nothing is retried. The first GPU step
# prints the "fits starting" line and the last GPU step prints the "last fit ended" line of
# `run.sh`. Each job is a fit process followed by its predict process, because experiment 713's
# pipeline builds one phase per process. Wave 1 is the page's: the synthetic world at seed 0
# (`SG-A`, `SG-A-S`) and the quick look (`P0`, `SG-0`, `SG-A` in `W2`, fold 2026-05, seed 0).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
steps=(
    "stage0"
    "probe --device cuda --only-fold 2026-05"
    "fit --arm SG-A --world synthetic --fold synthetic --seed 0 --device cuda"
    "predict --arm SG-A --world synthetic --fold synthetic --seed 0 --device cuda"
    "fit --arm SG-A-S --world synthetic --fold synthetic --seed 0 --device cuda"
    "predict --arm SG-A-S --world synthetic --fold synthetic --seed 0 --device cuda"
    "fit --arm P0 --world W2 --fold 2026-05 --seed 0 --device cuda"
    "predict --arm P0 --world W2 --fold 2026-05 --seed 0 --device cuda"
    "fit --arm SG-0 --world W2 --fold 2026-05 --seed 0 --device cuda"
    "predict --arm SG-0 --world W2 --fold 2026-05 --seed 0 --device cuda"
    "fit --arm SG-A --world W2 --fold 2026-05 --seed 0 --device cuda"
    "predict --arm SG-A --world W2 --fold 2026-05 --seed 0 --device cuda"
    "quicklook"
)
first_gpu=2   # the probe
last_gpu=12   # the predict of SG-A in W2

if [ "${1:-}" = "--list" ]; then
    for i in "${!steps[@]}"; do echo "$((i + 1)) ${steps[$i]}"; done
    exit 0
fi

: "${SLOT_CPU:?set SLOT_CPU to the one cpu number the manager assigned}"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"

run_step() {
    local n="$1"
    local args="${steps[$((n - 1))]}"
    local marks=()
    [ "$n" -eq "$first_gpu" ] && marks+=(FITS_STARTING=1)
    [ "$n" -eq "$last_gpu" ] && marks+=(LAST_FIT_ENDED=1)
    echo "=== step ${n} of ${#steps[@]}: ${args} ($(date -u +%FT%TZ))"
    # shellcheck disable=SC2086
    env ${marks[@]+"${marks[@]}"} "$here/run.sh" --go e00721_cli.py $args || {
        status=$?
        echo "FAILED step ${n}: ${args} (exit ${status}); stopping, nothing was retried" >&2
        exit "$status"
    }
}

if [ -n "${1:-}" ]; then
    if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -lt 1 ] || [ "$1" -gt "${#steps[@]}" ]; then
        echo "STEP must be a number from 1 to ${#steps[@]}, or --list" >&2
        exit 64
    fi
    run_step "$1"
    exit 0
fi
for n in $(seq 1 "${#steps[@]}"); do run_step "$n"; done
echo "wave 1 done ($(date -u +%FT%TZ))"
