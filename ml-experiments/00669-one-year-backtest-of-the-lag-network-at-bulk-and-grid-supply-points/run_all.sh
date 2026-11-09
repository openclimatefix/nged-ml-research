#!/bin/bash
# Experiment 669: the whole year, both groups, one GPU process at a time. Resumable: finished fits are skipped.
D="$(cd "$(dirname "$0")" && pwd)"
MONTHS="2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07 2026-08 2026-09 2026-10"
$D/gpu.sh e00669_testrows.py || exit 1
for g in bsp gsp; do
  for cmd in pretrain-wt step2 fit assemble; do
    $D/gpu.sh e00669_fit.py $cmd --group $g --months $MONTHS --seeds 0 1 || exit 1
  done
done
echo FITS DONE
