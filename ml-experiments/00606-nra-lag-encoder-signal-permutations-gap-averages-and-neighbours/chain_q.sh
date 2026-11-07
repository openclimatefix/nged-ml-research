#!/bin/bash
# usage: chain_q.sh MONTHS_CSV ARM...; step 2, fit, assemble, and inject each paired arm (seeds 0 and 1)
D="$(cd "$(dirname "$0")" && pwd)"
M=(${1//,/ }); shift
for A in "$@"; do
  $D/run.sh $A step2 --months "${M[@]}" && $D/run.sh $A fit --months "${M[@]}" && $D/run.sh $A assemble --months "${M[@]}" && $D/run.sh $A inject --months "${M[@]}" || echo "$A FAILED"
  echo "$A DONE $(date -u +%H:%M)"
done
