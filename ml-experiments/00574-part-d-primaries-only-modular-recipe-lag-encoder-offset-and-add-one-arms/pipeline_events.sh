#!/bin/bash
# Real-event harness for the six arms, one GPU process at a time.
D=$(cd "$(dirname "$0")" && pwd)
for a in P0 P3E0 P2 P4E P5 P6; do
  bash $D/run.sh events --arm $a || echo "events $a FAILED"
  echo "events $a done"
done
echo "EVENTS DONE"
