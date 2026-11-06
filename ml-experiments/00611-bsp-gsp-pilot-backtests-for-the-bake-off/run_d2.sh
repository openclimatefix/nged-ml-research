#!/bin/bash
# Design 2: experiment 415's N1 lag network per group; one GPU process at a time.
cd "$(cd "$(dirname "$0")" && pwd)"
for g in bsp gsp; do
  ./run.sh e00611_lag.py --group $g || exit 1
done
echo D2 DONE
