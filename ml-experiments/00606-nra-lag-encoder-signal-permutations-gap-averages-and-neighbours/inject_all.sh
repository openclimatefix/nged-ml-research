#!/bin/bash
# usage: inject_all.sh MONTHS_CSV ARM...; inject the bank for arms whose fits exist (skips finished cells)
D="$(cd "$(dirname "$0")" && pwd)"
M=(${1//,/ }); shift
for A in "$@"; do
  $D/run.sh $A inject --months "${M[@]}" || echo "$A INJECT FAILED"
  echo "$A INJECT DONE $(date -u +%H:%M)"
done
