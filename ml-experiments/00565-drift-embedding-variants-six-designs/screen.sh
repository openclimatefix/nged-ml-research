#!/bin/bash
# Coarse screening (cadence h: refits on days 0 and 14), seeds 0 and 1, injected shift on seed 0. Waits for the step R fits.
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "00565-drift-embedding-variants-six-designs/stepR" >/dev/null; do sleep 30; done
for arm in C-M0 C-A10 D3m D3q D4 D4-null D5; do
  case $arm in C-M0|C-A10) ;; *) $D/run.sh pretrain-wt --arm $arm --seeds 0 1 || exit 1;; esac
  $D/run.sh fit --arm $arm --seeds 0 1 || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --stress || exit 1
  $D/run.sh assemble --arm $arm --seeds 0 1 || exit 1
  echo "SCREEN $arm DONE"
done
echo SCREEN ALL DONE
