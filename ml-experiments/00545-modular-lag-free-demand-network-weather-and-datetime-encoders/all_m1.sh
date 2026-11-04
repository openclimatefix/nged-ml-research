#!/bin/bash
D=/home/jack/dev/nged-ml-research/.claude/worktrees/modular-arch-I/ml-experiments/00545-modular-lag-free-demand-network-weather-and-datetime-encoders
$D/run.sh pretrain --arm M1 && $D/run.sh fit --arm M1 && $D/run.sh assemble --arm M1 && $D/run.sh fit --arm M1 --stress --seeds 0 || exit 1
$D/run.sh fit --arm M0lr && $D/run.sh assemble --arm M0lr && $D/run.sh fit --arm M0lr --stress --seeds 0 || exit 1
echo ALL DONE
