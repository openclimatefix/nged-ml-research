#!/bin/bash
# Pre-train finishes first; then fit the two new months, predict the five old months with 572's weights, assemble.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "e00578_fit.py pretrain" >/dev/null; do sleep 10; done
$D/run.sh fit --months 2025-07 2025-08
$D/run.sh fit --months 2025-10 2025-12 2026-01 2026-03 2026-06
$D/run.sh assemble --months 2025-07 2025-08 2025-10 2025-12 2026-01 2026-03 2026-06
echo PIPELINE DONE
