#!/bin/bash
# Fold month 2026-05: pre-train and fit v3+D5, predict 2025-12 and 2026-01 with the stored weights (verified-shift twin), assemble; then the lag-free D5 on 2026-05.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh pretrain --months 2026-05
$D/run.sh fit --months 2025-12 2026-01 2026-05
$D/run.sh assemble --months 2025-12 2026-01 2026-05
$D/run_d5.sh pretrain-wt --months 2026-05
$D/run_d5.sh fit --months 2026-05
$D/run_d5.sh assemble --months 2026-05
echo PIPELINE DONE
