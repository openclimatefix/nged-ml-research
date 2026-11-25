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
#   - one cgroup scope with MemoryMax=${MEMORY_MAX:-12G for e00726_msa_panel.py, 9G otherwise}, no swap, and OOMPolicy=continue, so an out-of-memory kill
#     takes this job alone and never the session;
#   - one BLAS thread;
#   - the virtual environment's interpreter, called directly, so `uv` never re-syncs it;
#   - at least 25 GB of free memory and a load average below 24 before the job starts;
#   - for `gpu`, at most two GPU processes at once, each capped at GPU_CAP_GB (6) of GPU memory;
#   - a before and after snapshot of the folders of experiments 581, 650, 667, 709, 713, 714, and
#     715 (names, sizes, times, and a hash of the stored fits): the job fails if the snapshot differs.
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

# the frozen checkouts come first, so a copy of experiment 713 or 715 on this branch can never shadow them
wt="$(dirname "$repo")"  # the folder that holds the frozen checkouts beside this worktree
if [ -z "${EXTRA_PYTHONPATH:-}" ]; then
  EXTRA_PYTHONPATH="$(echo "$wt"/frozen-713-7c29114a8/ml-experiments/00713-*):$(echo "$wt"/frozen-f4-ac3e768e4/ml-experiments/00715-*)"
fi
pythonpath="$here"
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
  pythonpath="$here:$EXTRA_PYTHONPATH"
fi
for d in "$repo"/ml-experiments/00*/; do
  pythonpath="$pythonpath:${d%/}"
done
unset PYTHONOPTIMIZE  # the gates are exceptions, but `assert`s elsewhere must stay live
export PYTHONDONTWRITEBYTECODE=1  # no job writes `__pycache__` into a frozen checkout

snapshot() {
  # names, sizes, and times of every folder this experiment reads, directories included, plus a
  # hash of the stored fits of experiments 667 and 713; set SNAPSHOT_HASH=0 to skip the hashes
  {
    find "$data"/00581-* "$data"/00650-* "$data"/00667-* "$data"/00709-* "$data"/00713-* \
      "$data"/00714-* "$data"/00715-* -printf '%y %p %s %T@\n' 2>/dev/null
    if [ "${SNAPSHOT_HASH:-1}" = 1 ]; then
      find "$data"/00667-*/ "$data"/00713-*/ -path '*/fits/*' -name '*.parquet' -print0 2>/dev/null \
        | xargs -0 -r sha256sum
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

# the memory cap defaults per job type: 12G for the MSA command (peak 8.0 GB in the smoke test), 9G otherwise
case "$script" in
  e00726_msa_panel.py) default_cap=12G ;;
  *) default_cap=9G ;;
esac
memory_max="${MEMORY_MAX:-$default_cap}"

before="$(snapshot)"
set +e
systemd-run --user --scope --quiet \
  -p MemoryMax="$memory_max" -p MemorySwapMax=0 -p OOMPolicy=continue \
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
  echo "FAIL: a folder of experiments 581 to 715 changed during $script" >&2
  exit 5
fi
exit "$status"
