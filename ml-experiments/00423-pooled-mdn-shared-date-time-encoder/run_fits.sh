#!/usr/bin/env bash
# Experiment 423: run all fits in one GPU process (one feeder core, 8 GB memory cap).
# Usage: bash run_fits.sh   (from the worktree root). Resumable: finished seeds are skipped.
set -euo pipefail
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
export NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x
export OMP_NUM_THREADS=1
export PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights:ml-experiments/00419-pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports:ml-experiments/00423-pooled-mdn-shared-date-time-encoder
FIT="systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue uv run --no-sync python ml-experiments/00423-pooled-mdn-shared-date-time-encoder/e00423_fit.py --device cuda"
free -g | sed -n 2p
# 1. Pipeline check: R0 on 2025-12 (seed 0).
$FIT --arms R0 --months 2025-12
# 2. The five screen folds, 6 seeds (E0: 3 seeds).
$FIT --arms E1 E1n E2 E0
# 3. Event fold 2026-04: R0 (reference) and the encoder arms.
$FIT --arms R0 E1 E1n E2 E0 --months 2026-04
