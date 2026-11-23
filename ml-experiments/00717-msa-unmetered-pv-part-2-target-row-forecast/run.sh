#!/usr/bin/env bash
# Run one command of experiment 717 under the slot rules of the design (section 5.4): one thread for
# every numerical library, a 12 GB memory cap with no swap, the venv's Python called directly (never
# `uv run` inside the scope, because the snap build of uv moves its child out of the capped scope),
# and the folders of every experiment whose code this one imports on PYTHONPATH.
#
# Usage: run.sh e00716_fit.py HT 2025-11 0        (fits of part 2 run part 1's script and write under 00717's data folder)
#        run.sh e00717_score.py part2 --n-seeds 5
#        EXTRA_PYTHONPATH=<part 1 checkout>/ml-experiments/00716-... run.sh ...
#        SLOT_MEM=8G run.sh e00716_check_model.py
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# are set to dummy values here because the contracts package reads them at import. Check `free -g`
# first and wait when under 25 GB is available.
set -euo pipefail
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
# Part 2 imports part 1's code (e00716_*), which lives in its own worktree: EXTRA_PYTHONPATH names that folder.
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:?set EXTRA_PYTHONPATH to the 00716 folder of the part 1 checkout}"
script="$1"
shift
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt "${MIN_AVAILABLE_GB:-25}" ]; then
    echo "only ${available_gb} GB available; wait until ${MIN_AVAILABLE_GB:-25} GB is free (set MIN_AVAILABLE_GB to override)" >&2
    exit 1
fi
export PYTHONPATH="$here:$EXTRA_PYTHONPATH"
for d in "$root"/00*/; do
    export PYTHONPATH="$PYTHONPATH:${d%/}"
done
export DATA_PATH_INTERNAL
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
if [ -f "$here/$script" ]; then
    target="$here/$script"
elif [ -f "$EXTRA_PYTHONPATH/$script" ]; then
    target="$EXTRA_PYTHONPATH/$script"
else
    echo "$script is in neither $here nor $EXTRA_PYTHONPATH" >&2
    exit 1
fi
exec systemd-run --user --scope -q -p MemoryMax="${SLOT_MEM:-12G}" -p MemorySwapMax=0 -p OOMPolicy=continue \
    "$(dirname "$root")/.venv/bin/python" "$target" "$@"
