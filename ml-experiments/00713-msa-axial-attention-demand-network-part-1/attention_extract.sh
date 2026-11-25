#!/usr/bin/env bash
# Run the attention extract of experiment 713 for the arms M1, E, and P on one fold and seed, one process
# at a time, each under run.sh (12 GB memory cap, one BLAS thread). Pass `--random-init` first for the smoke test.
#
# Usage: attention_extract.sh [--random-init] [MONTH] [SEED]     (defaults: 2026-01 and 0)
# The script refuses to start while two or more other GPU processes run or the one-minute load is 24 or more.
# `e00713_attention.py` limits the process to 6 GB of GPU memory. Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment").
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
extra=()
if [ "${1:-}" = "--random-init" ]; then
    extra=(--random-init)
    shift
fi
# Enforce the manager's launch rules: fewer than two other GPU processes, and a one-minute load under 24.
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || { echo "nvidia-smi failed" >&2; exit 1; }
others=$(printf '%s\n' "$apps" | grep -c . || true)
if [ "$others" -ge 2 ]; then
    echo "$others other GPU processes are running; wait until fewer than two" >&2
    exit 1
fi
load=$(awk '{print int($1)}' /proc/loadavg)
if [ "$load" -ge 24 ]; then
    echo "the one-minute load is $load; wait until it is under 24" >&2
    exit 1
fi
month="${1:-2026-01}"
seed="${2:-0}"
for arm in M1 E P; do
    SLOT_MEM="${SLOT_MEM:-12G}" "$here/run.sh" e00713_attention.py extract "$arm" "$month" "$seed" "${extra[@]}"
done
