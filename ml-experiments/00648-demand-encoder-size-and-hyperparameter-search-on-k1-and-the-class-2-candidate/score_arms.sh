#!/bin/bash
# usage: score_arms.sh ARM...: wait for each arm's fits, then score K1 and the candidate (CPU, 16 GB cap)
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd $R
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
source ml-experiments/00524*/env.sh
E=$DATA_PATH_INTERNAL/ml-experiments/00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate
for a in "$@"; do
  until grep -q "end $a c2 inject\|FAILED $a\|ARM FAILED $a" $E/logs/timing.log; do sleep 30; done
  grep -q "end $a c2 inject" $E/logs/timing.log || { echo "skip $a" >> $E/logs/score.log; continue; }
  ln -sfn P5 $E/P7_D50_$a/work/fits/P5-removed
  S="systemd-run --user --scope -q -p MemoryMax=16G -p OOMPolicy=continue .venv/bin/python $D/e00648_score.py"
  case $a in nbbn*) ;; *) $S k1_$a --arm k1_$a >> $E/logs/score.log 2>&1;; esac
  $S c2_$a --arm c2_$a >> $E/logs/score.log 2>&1
  echo "scored $a" >> $E/logs/score.log
done
