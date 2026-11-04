#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh pretrain --arm M0 && $D/run.sh fit --arm M0 && $D/run.sh assemble --arm M0 && $D/run.sh fit --arm M0 --stress --seeds 0 || exit 1
echo ALL DONE
