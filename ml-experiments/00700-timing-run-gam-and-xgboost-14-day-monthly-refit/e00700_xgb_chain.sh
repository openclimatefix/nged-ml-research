#!/usr/bin/env bash
# Experiment 700: the other BG2 members' January refits, one after another (the GPU is shared).
set -uo pipefail
d=ml-experiments/00700-timing-run-gam-and-xgboost-14-day-monthly-refit
$d/e00700_xgb_run.sh an --regime monthly
$d/e00700_xgb_run.sh pt --regime monthly
$d/e00700_xgb_run.sh n1h --regime monthly --offset
$d/e00700_xgb_run.sh an --regime monthly --offset
echo CHAIN_DONE
