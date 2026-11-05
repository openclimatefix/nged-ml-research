#!/bin/bash
# usage: world.sh WORLD "ARM ARM ..." "months" "seeds"   e.g. world.sh Wc "M0w ORA D5" "2025-12 2026-06" "0 1"
# One GPU process at a time: pretrain, fit and assemble each arm in order (the staged arm needs D5 first).
D="$(cd "$(dirname "$0")" && pwd)"
WORLD=$1; ARMS=$2; MONTHS=$3; SEEDS=$4
export WORLD
for arm in $ARMS; do
  $D/run.sh e00571_fit.py pretrain --arm $arm --months $MONTHS --seeds $SEEDS || exit 1
  $D/run.sh e00571_fit.py fit --arm $arm --months $MONTHS --seeds $SEEDS || exit 1
  $D/run.sh e00571_fit.py assemble --arm $arm --months $MONTHS --seeds $SEEDS || exit 1
  echo "WORLD $WORLD ARM $arm DONE"
done
