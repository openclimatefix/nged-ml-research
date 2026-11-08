#!/bin/bash
# usage: run_arm.sh ARM [STAGE...]: one GPU process at a time. Stages (default: all): chain (K1 pretrain + step 2),
# k1 (K1 fit + inject), c2 (candidate fit + inject). Bottleneck arms (nbbn*) run only the c2 stage.
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd $R
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
source ml-experiments/00524*/env.sh
ARM=$1; shift
case $ARM in nbbn*) STAGES=${*:-c2};; *) STAGES=${*:-chain k1 c2};; esac
M="2025-10 2026-01 2026-06"
mkdir -p $DATA_PATH_INTERNAL/ml-experiments/00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate/logs
L=$DATA_PATH_INTERNAL/ml-experiments/00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate/logs
run() { # NET CMD
  echo "$(date -u +%FT%TZ) start $ARM $1 $2" >> $L/timing.log
  flock /tmp/e648.gpu.lock systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue .venv/bin/python $D/e00648_run.py $ARM $1 $2 --months $M --seeds 0 >> $L/${ARM}_$1_$2.log 2>&1 || { echo "FAILED $ARM $1 $2" >> $L/timing.log; exit 1; }
  echo "$(date -u +%FT%TZ) end $ARM $1 $2" >> $L/timing.log
}
for s in $STAGES; do
  case $s in
    chain) run k1 pretrain; run k1 step2 ;;
    k1) run k1 fit; run k1 inject ;;
    c2) run c2 fit; run c2 inject ;;
    *) echo "bad stage $s"; exit 2 ;;
  esac
done
