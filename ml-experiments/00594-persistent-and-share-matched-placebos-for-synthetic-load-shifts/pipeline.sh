#!/bin/bash
# usage: pipeline.sh ARM... ; fit, assemble, inject, one arm at a time
D="$(cd "$(dirname "$0")" && pwd)"
for a in "$@"; do
  for c in fit assemble inject; do "$D/run.sh" $c --arm $a || exit 1; done
done
