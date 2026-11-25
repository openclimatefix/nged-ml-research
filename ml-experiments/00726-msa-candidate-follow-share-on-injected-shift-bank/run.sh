#!/usr/bin/env bash
# Experiment 726: run one script of this folder under the workstation's rules.
#
# Usage:  run.sh cpu|gpu <script.py> [arguments...]
#
# Required environment: DATA_PATH_INTERNAL (the shared data folder) and, for the stored Candidate
# and MSA code, EXTRA_PYTHONPATH naming the frozen experiment 713 checkout's folder first and the
# pinned F4 checkout's folder second (the scripts assert both checkouts).
#
# Rules this wrapper enforces:
#   - one cgroup scope with MemoryMax=9G, no swap, and OOMPolicy=continue, so an out-of-memory kill
#     takes this job alone and never the session;
#   - one BLAS thread;
#   - the virtual environment's interpreter, called directly, so `uv` never re-syncs it;
#   - at least 25 GB of free memory and a load average below 24 before the job starts;
#   - for `gpu`, at most two GPU processes at once, each capped at GPU_CAP_GB (6) of GPU memory;
#   - a before and after snapshot of experiment 667's folder and experiment 713's stored Candidate
#     seeds and work folders: the job fails if either snapshot differs.
set -euo pipefail

mode="${1:?usage: run.sh cpu|gpu script.py [args]}"
script="${2:?usage: run.sh cpu|gpu script.py [args]}"
shift 2

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the shared data folder}"
data="$DATA_PATH_INTERNAL/ml-experiments"

free_gb="$(free -g | awk '/^Mem:/ {print $7}')"
if [ "$free_gb" -lt 25 ]; then
  echo "refusing to start: only ${free_gb} GB of memory is available (need 25)" >&2
  exit 3
fi
load="$(awk '{print int($1)}' /proc/loadavg)"
if [ "$load" -ge 24 ]; then
  echo "refusing to start: the load average is ${load} (limit 24)" >&2
  exit 3
fi

# every earlier experiment's folder is on the path, because a script may import any of them
pythonpath="$here"
for d in "$repo"/ml-experiments/00*/; do
  pythonpath="$pythonpath:${d%/}"
done
pythonpath="$pythonpath:${EXTRA_PYTHONPATH:-}"

snapshot() {
  # names, sizes, and modification times; set SNAPSHOT_HASH=1 to add a hash of the stored fits
  {
    find "$data"/00667-*/ -type f -printf '%p %s %T@\n' 2>/dev/null
    find "$data"/00713-*/candidate_seeds "$data"/00713-*/work -type f -printf '%p %s %T@\n' 2>/dev/null
    if [ "${SNAPSHOT_HASH:-0}" = 1 ]; then
      find "$data"/00667-*/ -path '*/fits/*' -name '*.parquet' -print0 2>/dev/null | xargs -0 -r sha256sum
    fi
  } | LC_ALL=C sort | sha256sum | cut -d' ' -f1
}

slot_cmd=()
if [ "$mode" = gpu ]; then
  # two lock files: a job takes the first one that is free and waits for none
  lock_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  got=""
  for n in 0 1; do
    exec {fd}>"$lock_dir/e00726_gpu_slot_$n.lock"
    if flock -n "$fd"; then
      got="$n"
      break
    fi
    exec {fd}>&-
  done
  if [ -z "$got" ]; then
    echo "refusing to start: both GPU slots are in use" >&2
    exit 4
  fi
  export GPU_CAP_GB="${GPU_CAP_GB:-6}"
  export E00667_PHASE=predict
else
  export E00667_PHASE="${E00667_PHASE:-predict}"
fi

before="$(snapshot)"
set +e
systemd-run --user --scope --quiet \
  -p MemoryMax=9G -p MemorySwapMax=0 -p OOMPolicy=continue \
  env \
    PYTHONPATH="$pythonpath" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 \
    NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}" \
    NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}" \
    NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}" \
    "$repo/.venv/bin/python" "$here/$script" "$@"
status=$?
set -e
after="$(snapshot)"
if [ "$before" != "$after" ]; then
  echo "FAIL: experiment 667's folder or experiment 713's stored Candidate seeds changed during $script" >&2
  exit 5
fi
exit "$status"
