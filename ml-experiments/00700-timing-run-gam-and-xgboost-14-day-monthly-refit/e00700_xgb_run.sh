#!/usr/bin/env bash
# Experiment 700: refit the BG2 members (experiment 87) at the monthly cut 2026-01-01, timed.
# Usage: e00700_xgb_run.sh CONFIG [extra fit_demand args]  (run from the repository root)
# Needs DATA_PATH_INTERNAL=<sandbox root> so that the fit writes to the sandbox copy of
# `00087-*/work/anchor_preds`, from which the 2026-01-01 files were deleted first. The fit script
# skips every file that exists, so only the January refit is fitted.
set -euo pipefail
config=$1; shift
out="${DATA_PATH_INTERNAL%/sandbox}/timing"; tag="${config}${2:+_}${2:-}${3:+_}${3:-}"
mkdir -p "$out"
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> "$out/gpu_${tag}.csv"; sleep 5; done ) &
sampler=$!
trap 'kill $sampler' EXIT
/usr/bin/time -f "wall=%e user=%U sys=%S maxrss_kb=%M" -o "$out/time_xgb_${tag}.txt" \
  uv run --no-sync python ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations/e00087_fit_demand.py \
  --config "$config" --workers 6 --cores 12 "$@" > "$out/log_xgb_${tag}.txt" 2>&1
