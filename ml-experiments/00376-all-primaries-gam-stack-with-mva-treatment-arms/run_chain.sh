#!/usr/bin/env bash
# Usage: [ARM=a1] run_chain.sh SOURCE SITE... : fit, forecast, and build the stage-2 arms for the sites.
# SOURCE is `mva` (the eight new sites) or `control` (MW sites from experiment 111's stores).
set -euo pipefail
source=$1; shift
D=ml-experiments/00376-all-primaries-gam-stack-with-mva-treatment-arms
run() { E00376_SOURCE=$source E00376_ARM=${ARM:-a0} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue uv run --no-sync python "$@"; }
for site in "$@"; do
  run $D/e00376_fit.py --cadences monthly --sites "$site" --workers 1
  run $D/e00376_forecast.py --arms a2 --sites "$site" --workers 1
  run $D/e00376s2_run.py --sites "$site" --workers 1
done
