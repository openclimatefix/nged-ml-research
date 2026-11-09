#!/bin/bash
# usage: fits.sh: fit and export the prevailing rows of each fold month in order, as soon as its own real rows and the banks of all earlier months exist.
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries
H=$R/prevailing/head_data
prev=""
for M in 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07 2026-08 2026-09 2026-10; do
  until [ -f $H/bank_$M.npz ] && { [ -z "$prev" ] || [ -f $H/bank_$prev.npz ]; }; do sleep 30; done
  OMP_NUM_THREADS=2 $D/cpu_p.sh p667_fit.py $M > $R/prevailing/logs/fit_$M.log 2>&1 || { echo "$M fit FAILED"; exit 1; }
  $D/cpu_p.sh p667_export.py $M >> $R/prevailing/logs/fit_$M.log 2>&1 || { echo "$M export FAILED"; exit 1; }
  echo "$M fitted and exported $(date -u +%T)"
  prev=$M
done
