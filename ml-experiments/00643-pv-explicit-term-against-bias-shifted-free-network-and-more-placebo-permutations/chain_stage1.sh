#!/bin/bash
# stage 1: fbias, foff, and the new permutations for every cutoff, then the contrasts
HERE=$(cd "$(dirname "$0")" && pwd)
for c in 2026-06 2026-03 2025-07; do
  $HERE/run.sh e00643_fit.py fbias --cutoff $c || exit 1
  $HERE/run.sh e00643_fit.py foff --cutoff $c || exit 1
  $HERE/run.sh e00643_fit.py perm --cutoff $c --worlds step --arms perm1 perm3 perm4 perm5 || exit 1
done
$HERE/run.sh e00643_analyse.py contrasts --tag stage1
