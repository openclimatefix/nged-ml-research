#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh pretrain --arm M1 && $D/run.sh fit --arm M1 && $D/run.sh assemble --arm M1 && $D/run.sh fit --arm M1 --stress --seeds 0 || exit 1
$D/run.sh fit --arm M0lr && $D/run.sh assemble --arm M0lr && $D/run.sh fit --arm M0lr --stress --seeds 0 || exit 1
echo ALL DONE
