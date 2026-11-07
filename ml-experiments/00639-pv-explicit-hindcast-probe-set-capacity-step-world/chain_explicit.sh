#!/bin/bash
# explicit arms for every cutoff, then the contrasts
cd "$(cd "$(dirname "$0")" && pwd)"; HERE=$PWD
for c in 2026-06 2026-03 2025-07; do
  $HERE/run.sh e00639_fit.py explicit --cutoff $c || exit 1
done
$HERE/run.sh e00639_analyse.py contrasts
