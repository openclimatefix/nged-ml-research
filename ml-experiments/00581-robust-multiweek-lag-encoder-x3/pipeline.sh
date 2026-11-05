#!/bin/bash
# Resumable: every command skips finished cells. Order: X3-med, the v3+D5 injected baseline, then the placebo and X3-35.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh x3med pretrain
$D/run.sh x3med fit
$D/run.sh x3med assemble
$D/run.sh v3d5 inject
$D/run.sh x3med inject
for k in placebo x335; do
  $D/run.sh $k pretrain
  $D/run.sh $k fit
  $D/run.sh $k assemble
  $D/run.sh $k inject
done
echo PIPELINE DONE
