#!/bin/bash
# One GPU process at a time: every arm, both worlds, seeds 0 to 4, each cutoff in turn.
D=$(dirname "$0")
for c in 2026-06 2026-03 2025-07; do
  bash $D/run.sh e00644_fit.py fit --cutoff $c || exit 1
done
