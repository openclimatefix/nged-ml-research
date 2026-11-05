#!/bin/bash
# P6 = P5 plus the W-T trailing-temperature columns (pre-training, step 2, fits, stress, assemble).
set -e
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh pretrain-wt --arm P6
$D/run.sh step2 --arm P6
$D/run.sh fit --arm P6
$D/run.sh fit --arm P6 --seeds 0 --stress
$D/run.sh assemble --arm P6
echo P6 DONE
