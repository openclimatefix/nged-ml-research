#!/usr/bin/env bash
# Launch one command of experiment 732 under the slot rules of the page's "Compute" section.
#
# The wrapper starts nothing by default: it needs `--go` before the script name, so that a stray
# invocation only prints this help.
#
# Usage:  SLOT_CPUS=<cpu list, at most 6 cores, e.g. 16-21> run.sh --go e00732_cli.py <banks|neighbours|limits|overlays|tensor|solar_diag|gallery|panels|timing|fit|real|score|report|figures|loss_curves|series|chain> <arguments>
#         SLOT_CPUS=<list> FITS_STARTING=1 run.sh --go ...     prints the "fits starting" line for the manager
#         SLOT_CPUS=<list> LAST_FIT_ENDED=1 run.sh --go ...    prints the "last fit ended" line for the manager
#
# Rules it sets (a list of at most 6 cores replaces experiment 729's single core):
#   - slot A2: one GPU process of at most 6 GB, cores 16-21, `MemoryMax`/`SLOT_MEM` at most 14G;
#   - one scope at a time: `systemd-run --user --scope`, `MemoryMax=$SLOT_MEM`, `MemorySwapMax=0`,
#     `OOMPolicy=continue`, `AllowedCPUs=$SLOT_CPUS`;
#   - one thread per worker for every numerical library (OMP, MKL, OpenBLAS, Polars, and Rayon);
#   - `uptime` load below 24 and at least 25 GB available at launch, or the wrapper refuses;
#   - `taskset -c $SLOT_CPUS` as well, and `e00732_paths.pin_threads` asserts that `os.sched_getaffinity` shows at most 6 cores;
#   - the venv's Python called directly (never `uv run` inside the scope); `PYTHON` names a venv built
#     with `uv sync --group gpu` for the fits, and defaults to the `msa-f7-pv-part-1` worktree's venv,
#     which has torch; the wrapper stops with an error when the interpreter cannot import torch;
#   - for a command that uses the GPU (no `--device cpu`, and `CUDA_VISIBLE_DEVICES` not empty): the
#     `nvidia-smi` totals leave room for 6 GB under the total of 36 GB, at most one other GPU process
#     (of this experiment or another, counted together) holds the GPU, and a lock file allows one worker
#     of this experiment at a time. `banks`, `neighbours`, `limits`, `overlays`, `tensor`, `solar_diag`, `gallery`, `panels`, `real`, `score`, `report`, `figures`, and `series` run on the CPU and skip both checks;
#   - this folder on PYTHONPATH: `e00732_paths.py` adds the folders of experiment 730, and of the experiments that 730 imports, from this checkout.
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# default to dummy values because the contracts package reads them at import. Each fit holds at most
# 6 GB of the GPU (`e00729_train.limit_gpu_memory, called by e00732_cli`).
set -euo pipefail
if [ "${1:-}" != "--go" ] || [ -z "${2:-}" ]; then
    sed -n '2,30p' "$0" >&2
    echo "refusing to run: pass --go and a script name (nothing was started)" >&2
    exit 64
fi
shift
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
: "${SLOT_MEM:=14G}"
if ! [[ "$SLOT_MEM" =~ ^([0-9]+)G$ ]] || [ "${BASH_REMATCH[1]}" -gt 14 ]; then
    echo "SLOT_MEM must be at most 14G (got '$SLOT_MEM')" >&2
    exit 64
fi
: "${SLOT_CPUS:?set SLOT_CPUS to the cpu list the manager assigned (at most 6 cores)}"
if ! [[ "$SLOT_CPUS" =~ ^[0-9]+([-,][0-9]+)*$ ]]; then
    echo "SLOT_CPUS must be a cpu list such as 16-21 (got '$SLOT_CPUS')" >&2
    exit 64
fi
n_cpus=$(python3 -c "import sys; n=0
for p in sys.argv[1].split(','):
    a,_,b=p.partition('-'); n+=int(b or a)-int(a)+1
print(n)" "$SLOT_CPUS")
if [ "$n_cpus" -gt 6 ]; then
    echo "SLOT_CPUS names ${n_cpus} cores; the slot holds at most 6" >&2
    exit 64
fi
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
script="$1"
shift
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt 25 ]; then
    echo "only ${available_gb} GB available; at least 25 GB must be free at launch" >&2
    exit 1
fi
load=$(awk '{print int($1)}' /proc/loadavg)
if [ "$load" -ge 24 ]; then
    echo "one-minute load ${load} is 24 or above; wait" >&2
    exit 1
fi
# `e00732_paths.py` puts the sibling experiment folders on sys.path.
export PYTHONPATH="$here"
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
export SLOT_CPUS SLOT_MEM
export DATA_PATH_INTERNAL
if [ -n "${FITS_STARTING:-}" ]; then
    echo "fits starting: experiment 732, ${script} $*, core ${SLOT_CPUS}, $(date -u +%FT%TZ)"
fi
# Set PYTHON explicitly, for example `PYTHON=/path/to/venv/bin/python` (a venv built with `uv sync
# --group gpu`). The default venv path holds the text `msa-f7`, so the scorer check below would match
# a fit of this experiment that runs with the default interpreter. The check therefore ignores command
# lines that name an `e00732_` script, and it keeps refusing on any other match.
default_python="$(dirname "$root")/.claude/worktrees/msa-f7-pv-part-1/.venv/bin/python"
[ -x "$default_python" ] || default_python="$HOME/dev/nged-ml-research/.claude/worktrees/msa-f7-pv-part-1/.venv/bin/python"
python_bin="${PYTHON:-$default_python}"
if ! [ -x "$python_bin" ] || ! "$python_bin" -c 'import torch' 2>/dev/null; then
    echo "PYTHON='$python_bin' cannot import torch: set PYTHON to a venv built with 'uv sync --group gpu'" >&2
    exit 1
fi
uses_gpu=1
[ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ -n "${CUDA_VISIBLE_DEVICES}" ] || uses_gpu=0
for arg in "$@"; do [ "$arg" = "cpu" ] && uses_gpu=0; done
# these stages read files and compute on the CPU only: no GPU check and no GPU lock
case "${1:-}" in banks | neighbours | limits | overlays | tensor | solar_diag | gallery | panels | real | score | report | figures | loss_curves | series) uses_gpu=0 ;; esac
lock_prefix=""
if [ "$uses_gpu" = 1 ] && command -v nvidia-smi >/dev/null; then
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    if [ $((used_mb + 6144)) -gt 36864 ]; then
        echo "the GPU holds ${used_mb} MB; 6 GB more would pass the 36 GB total" >&2
        exit 1
    fi
    ours=0
    others=0
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'e00732'; then ours=$((ours + 1)); else others=$((others + 1)); fi
    done
    if [ $((ours + others)) -ge 2 ]; then
        echo "GPU processes: ${ours} of experiment 732 and ${others} of others; at most 2 GPU processes run at once across all agents, and this one would be the third" >&2
        exit 1
    fi
    if [ "$ours" -ge 1 ]; then
        echo "a process of experiment 732 already holds the GPU; the slot allows one GPU process of this experiment" >&2
        exit 1
    fi
    if pgrep -af 'e0071[67]_score|msa-f7' | grep -qv 'e00732_' ; then
        echo "a scorer of experiment 716 or 717, or an F7 job, holds the GPU: wait" >&2
        exit 1
    fi
    mkdir -p "$HOME/.cache"
    lock_prefix="flock -n $HOME/.cache/e00732_gpu.lock"
fi
status=0
# shellcheck disable=SC2086
systemd-run --user --scope -q -p MemoryMax="$SLOT_MEM" -p MemorySwapMax=0 -p OOMPolicy=continue \
    -p AllowedCPUs="$SLOT_CPUS" taskset -c "$SLOT_CPUS" \
    $lock_prefix "$python_bin" "$here/$script" "$@" || status=$?
if [ -n "${LAST_FIT_ENDED:-}" ]; then
    echo "last fit ended: experiment 732, ${script} exit ${status}, $(date -u +%FT%TZ)"
fi
exit "$status"
