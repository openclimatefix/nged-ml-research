#!/usr/bin/env bash
# Experiment 701, GAM side: refit, stage 1, stage 2, member responses, and experiment 378's arms for the
# origin months 2026-07 to 2026-10, then the export. Run from the repository root with the data
# folder variables set (see README.md). `DATA_PATH_INTERNAL` must be the sandbox,
# `<data>/ml-experiments/00701-*/gam/sandbox`, which `e00701_gam_sandbox.py` makes. At most 4 processes run
# at a time. Every stage replaces its months' rows, so the script can be run again after a failure.
set -uo pipefail
d=ml-experiments/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10
: "${DATA_PATH_INTERNAL:?set it to the sandbox folder}"
case "$DATA_PATH_INTERNAL" in */gam/sandbox) ;; *) echo "DATA_PATH_INTERNAL is not the GAM sandbox" >&2; exit 2 ;; esac
real_data="${DATA_PATH_INTERNAL%/ml-experiments/*}"
root="${DATA_PATH_INTERNAL%/sandbox}"
log="$root/timing"; mkdir -p "$log"
export MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base=ml-experiments
export PYTHONPATH=$base/00377-demand-v1-first-run-feasibility:$base/00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1:$base/00184-dp-ens-daylight-calibration:$base/00202-dp-ens-two-number-rolling-replay:$base/00319-primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days:$base/00320-primary-demand-default-2-night-aware-heuristic-blend-and-night-band-width:$base/00700-timing-run-gam-and-xgboost-14-day-monthly-refit:$d
control="D5 D9 D10 D11 D14 D16 D18"
mva="D2 D3 D4 D8 D12 D19 D21 D22"

check() {  # check STAGE: stop if any log of the stage holds a traceback
  if grep -l Traceback "$log"/run_"$1"_*.log 2>/dev/null; then echo "FAILED: $1" >&2; exit 1; fi
  echo "DONE: $1"
}
per_site() {  # per_site STAGE SCRIPT: one process per Primary, 4 at a time
  for s in $control; do echo "control $s"; done > "$log/sites.txt"
  for s in $mva; do echo "mva $s"; done >> "$log/sites.txt"
  xargs -P 4 -L 1 bash -c 'E00377_SOURCE=$0 E00377_ARM=a0 /usr/bin/time -f "'"$2"' $1 wall=%e user=%U maxrss_kb=%M" \
    uv run --no-sync python '"$d"'/'"$2"' --sites $1 > '"$log"'/run_'"$1"'_$1.log 2>&1' < "$log/sites.txt" || { echo "FAILED: $1 (a process exited non-zero)" >&2; exit 1; }
}

echo "== refit"
for src in control mva; do
  sites=$control; [ "$src" = mva ] && sites=$mva
  E00377_SOURCE=$src E00377_ARM=a0 uv run --no-sync python $d/e00701_gam_refit.py --sites $sites > "$log/run_refit_$src.log" 2>&1
done
check refit
echo "== stage 1"; per_site stage1 e00701_gam_stage1.py; check stage1
echo "== stage 2 (after stage 1 of all Primaries, because D8 reads its donors)"; per_site stage2 e00701_gam_stage2.py; check stage2
echo "== member responses"; per_site members e00701_gam_members.py; check members
echo "== arms (S and the daylight blend), 4 processes"
printf '%s\n' "control low" "control long" "mva low" "mva long" |
  xargs -P 4 -L 1 bash -c 'E00377_SOURCE=$0 /usr/bin/time -f "arms $0 $1 wall=%e user=%U maxrss_kb=%M" \
    uv run --no-sync python '"$d"'/e00701_gam_arms.py $1 > '"$log"'/run_arms_$0_$1.log 2>&1' || { echo "FAILED: arms (a process exited non-zero)" >&2; exit 1; }
check arms
echo "== export (real data folder)"
DATA_PATH_INTERNAL="$real_data" uv run --no-sync python $d/e00701_gam_export.py > "$log/run_export.log" 2>&1
check export
echo GAM_FINAL_TEST_DONE
