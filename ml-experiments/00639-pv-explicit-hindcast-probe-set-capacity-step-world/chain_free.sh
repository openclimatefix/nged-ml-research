#!/bin/bash
# free phase and probes for every cutoff, then the minimum detectable effect
cd "$(cd "$(dirname "$0")" && pwd)"; HERE=$PWD
for c in 2026-06 2026-03 2025-07; do
  $HERE/run.sh e00639_fit.py free --cutoff $c || exit 1
  $HERE/run.sh e00639_probe.py --cutoff $c || exit 1
done
$HERE/run.sh e00639_analyse.py mde
