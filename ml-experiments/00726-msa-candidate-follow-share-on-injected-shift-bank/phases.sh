#!/usr/bin/env bash
# Experiment 726: the phases of the plan, one job at a time, through run.sh.
#
# Usage: phases.sh phase0|phase1|phase2|phase3|phase4 [month ...]
#
# Every job waits for the machine (run.sh refuses when the load is 24 or more, when under 25 GB is
# free, or when two GPU slots are taken) and is retried after a refusal. A job that fails for any
# other reason stops the phase, so nobody changes code before the failure is read. The wall time of
# every job is appended to `logs/phase_times.tsv` of this experiment's data folder.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run="$here/run.sh"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
data="$(echo "$DATA_PATH_INTERNAL"/ml-experiments/00726-*)"
[ -d "$data" ] || { echo "no data folder: $data" >&2; exit 2; }
mkdir -p "$data/logs"
max_refusals=720  # 12 hours of one-minute waits
phase="${1:?usage: phases.sh phaseN [month ...]}"
shift || true
months=("$@")
[ ${#months[@]} -eq 0 ] && months=(2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06)

job() { # job <mode> <label> <script> <args...>
  local mode="$1" label="$2" refusals=0
  shift 2
  while true; do
    local t0 t1 status
    t0=$(date +%s)
    "$run" "$mode" "$@"
    status=$?
    t1=$(date +%s)
    if [ "$status" -eq 3 ] || [ "$status" -eq 4 ]; then
      refusals=$((refusals + 1))
      if [ "$refusals" -ge "$max_refusals" ]; then
        echo "STOP: job '$label' was refused $refusals times" >&2
        exit "$status"
      fi
      sleep 60
      continue
    fi
    printf '%s\t%s\t%s\t%s\n' "$phase" "$label" "$status" "$((t1 - t0))" >>"$data/logs/phase_times.tsv"
    if [ "$status" -ne 0 ]; then
      echo "STOP: job '$label' exited with status $status" >&2
      exit "$status"
    fi
    return 0
  done
}
cand() { job gpu "cand $*" e00726_candidate_panel.py "$@"; }
msa() { job gpu "msa $*" e00726_msa_panel.py "$@"; }
chk() { job cpu "check $*" e00726_checks.py "$@"; }

phase0() {
  chk kp
  for m in 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06; do
    chk kc4 --month "$m"
    chk kw --month "$m"
  done
  # KC1: the zero world through the Candidate
  cand zero --root e667 --month 2025-11 --seeds 0 --passes 0
  cand zero --root e667 --month 2026-04 --seeds 0 --passes 0
  cand zero --root seeds713 --month 2025-11 --seeds 2 --passes 0
  cand zero --root seeds713 --month 2026-04 --seeds 2 --passes 0
  cand zero --root seeds713 --month 2026-02 --seeds 3 --passes 0
  chk compare kc1
  # KC2: experiment 650's events
  for m in 2025-11 2026-06; do cand e650 --root e667 --month "$m" --seeds 0 1 --passes 0 7; done
  chk compare kc2
  # KC3: experiment 709's bank
  for m in 2025-11 2026-06; do cand bank709 --root e667 --month "$m" --seeds 0 1 --passes 0 5; done
  chk compare kc3
  # KC6: one shifted Primary, and one set uncut and cut
  cand kc6_one --root e667 --month 2025-11 --seeds 0 --passes 0 --primary 0
  cand kc6_set --root e667 --month 2025-11 --seeds 0 --passes 0 --sets 0
  cand e581_host --root e667 --month 2025-11 --seeds 0 --passes 0 --sets 0
  chk compare kc6 --month 2025-11 --seed 0 --pass 0 --primary 0 --set 0
  # KM1: the zero world through the MSA command
  for m in 2025-11 2026-04; do
    for arm in M1 K E N; do msa zero "$arm" "$m" 0; done
    msa zero M1 "$m" 0 --mask-history
  done
  chk compare km1
  # KM3, then KM2
  msa all M1 2025-11 0 --passes 0 --check-km3
  chk compare km3
  for m in 2025-11 2026-04; do
    msa all N "$m" 0 --passes 0 12
    msa all M1 "$m" 0 --passes 0 12 --mask-history
  done
  chk compare km2
  # KM4
  msa host M1 2025-11 0 --passes 0 --sets 0
  msa all M1 2025-11 0 --passes 0
  chk compare km4 --month 2025-11 --set 0
  chk selftest
}

phase1() {
  for m in "${months[@]}"; do
    cand e581_host --root e667 --month "$m" --seeds 0 1
    cand e581_all --root e667 --month "$m" --seeds 0 1
    for s in 2 3 4; do cand e581_host --root seeds713 --month "$m" --seeds "$s"; done
  done
}

phase2() {
  for m in "${months[@]}"; do
    for s in 0 1 2 3 4; do msa all M1 "$m" "$s"; done
  done
}

phase3() {
  for m in 2025-11 2026-01 2026-04 2026-06; do
    for arm in K E; do for s in 0 1; do msa all "$arm" "$m" "$s"; done; done
  done
}

phase4() {
  for m in 2025-11 2026-01 2026-04 2026-06; do cand g2 --root e667 --month "$m" --seeds 0 1; done
}

case "$phase" in
  phase0 | phase1 | phase2 | phase3 | phase4) "$phase" ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
