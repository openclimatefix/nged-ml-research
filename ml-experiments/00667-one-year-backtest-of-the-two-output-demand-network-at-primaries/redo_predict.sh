#!/bin/bash
# usage: redo_predict.sh MONTH...: rebuild the predict-phase tables and forecasts of finished folds (models are kept)
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries
for m in "$@"; do
  mkdir -p $R/superseded/$m
  mv $R/tables/predict/T_${m}_s*.npz $R/superseded/$m/ 2>/dev/null
  mv $R/P7_D50_$m/work/fits/P5/${m}_d0_s?.parquet $R/superseded/$m/ 2>/dev/null
  $D/chain.sh $m ptable pfit
done
