#!/usr/bin/env bash
# Experiment 700: stage 1 then stage 2 of the GAM for origin months 2025-11 to 2026-06, one process
# per Primary, 8 at a time. Run from the repository root. `DATA_PATH_INTERNAL` is the sandbox root
# (the sandbox holds copies of the experiment 377 and 378 folders, so the stored files stay as they are).
set -uo pipefail
d=ml-experiments/00700-timing-run-gam-and-xgboost-14-day-monthly-refit
: "${DATA_PATH_INTERNAL:?set it to the sandbox root}"
export PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility
log="${DATA_PATH_INTERNAL%/sandbox}/timing"; mkdir -p "$log"
stage() {  # stage SCRIPT: all 15 Primaries, 8 processes at a time
  script=$1
  for s in "control D5" "control D9" "control D10" "control D11" "control D14" "control D16" "control D18" \
           "mva D2" "mva D3" "mva D4" "mva D8" "mva D12" "mva D19" "mva D21" "mva D22"; do echo "$s"; done |
    xargs -P 8 -L 1 bash -c 'E00377_SOURCE=$0 E00377_ARM=a0 /usr/bin/time -f "'"$script"' $1 wall=%e user=%U maxrss_kb=%M" \
      uv run --no-sync python '"$d"'/'"$script"' --sites $1 >> '"$log"'/year_'"$script"'_$1.log 2>&1'
}
stage e00700_gam_stage1.py
stage e00700_gam_stage2.py
echo GAM_YEAR_DONE
