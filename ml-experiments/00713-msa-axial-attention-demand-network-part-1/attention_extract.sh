#!/usr/bin/env bash
# Run the attention extract of experiment 713 for the arms M1, E, and P on one fold and seed, one process
# at a time, each under run.sh (12 GB memory cap, one BLAS thread). Pass `--random-init` first for the smoke test.
#
# Usage: attention_extract.sh [--random-init] [MONTH] [SEED]     (defaults: 2026-01 and 0)
# Check `nvidia-smi` and `uptime` first: start only while fewer than two other GPU processes run and the
# one-minute load is under 24. Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment").
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
extra=()
if [ "${1:-}" = "--random-init" ]; then
    extra=(--random-init)
    shift
fi
month="${1:-2026-01}"
seed="${2:-0}"
for arm in M1 E P; do
    SLOT_MEM="${SLOT_MEM:-12G}" "$here/run.sh" e00713_attention.py extract "$arm" "$month" "$seed" "${extra[@]}"
done
