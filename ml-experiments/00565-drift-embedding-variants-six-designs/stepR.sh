#!/bin/bash
# Step R: A10 (experiment 559's code, run in place) for seeds 2 to 4, plus the seed 1 to 4 injected-shift fits.
# Seeds 0 and 1 of A10 are experiment 559's stored fits. One GPU process at a time.
R=/home/jack/dev/nged-ml-research/.claude/worktrees/drift-embedding-variants
RUN=$R/ml-experiments/00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty/run.sh
$RUN pretrain-wt --arm A10 --seeds 2 3 4 || exit 1
$RUN fit --arm A10 --seeds 2 3 4 || exit 1
$RUN assemble --arm A10 --seeds 2 3 4 || exit 1
echo STEP R SCORED FITS DONE
$RUN fit --arm A10 --seeds 1 2 3 4 --stress || exit 1
echo STEP R A10 STRESS DONE
