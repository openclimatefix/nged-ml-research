#!/bin/bash
# Injected-shift fits of M0 for seeds 1 to 4 (seed 0 is experiment 545's), experiment 554's code run in place; waits for stepR.sh.
R=/home/jack/dev/nged-ml-research/.claude/worktrees/drift-embedding-variants
while pgrep -f "00565-drift-embedding-variants-six-designs/stepR.sh" >/dev/null; do sleep 30; done
$R/ml-experiments/00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/run.sh fit --arm M0 --seeds 1 2 3 4 --stress || exit 1
echo M0 STRESS DONE
