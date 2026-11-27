#!/usr/bin/env bash
# Experiment 726: the memory and timing smoke test of phase 0 (design gate 4 and review H5).
#
# Usage: smoke.sh [month]      (default 2026-06: the fold with the most training rows)
#
# Environment: MEMORY_MAX (default 9G) is the memory cap of each job; EXTRA_PYTHONPATH defaults to the
# frozen experiment 713 checkout (7c29114a8) and the frozen F4 checkout (ac3e768e4), as in run.sh.
#
# The preflight KP runs first, because every job reads `candidate/sets.json`. Then three GPU jobs
# run through run.sh on one fold month, each of which logs its host peak and GPU peak (`peaks` in
# the job log under `smoke/logs/`):
#   1. the Candidate in the host-only world at root e667, seeds 0 and 1 in one process;
#   2. the Candidate in the host-only world at root seeds713, seed 2;
#   3. M1 on the all-shifted world, all 24 passes.
# The jobs write under `smoke/` of this experiment's data folder (E00726_SMOKE=1), so phase 1
# cannot skip them as finished. If any host peak is above 8 GB, ask the manager for 12 GB.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
month="${1:-2026-06}"
run="$here/run.sh"
export MEMORY_MAX="${MEMORY_MAX:-9G}"

echo "== 0. preflight KP"
"$run" cpu e00726_checks.py kp
export E00726_SMOKE=1

echo "== 1. Candidate e667, seeds 0 and 1, one process"
/usr/bin/time -f "wall %e s, max RSS %M kB" "$run" gpu e00726_candidate_panel.py e581_host \
  --root e667 --month "$month" --seeds 0 1 --passes 0 1
echo "== 2. Candidate seeds713, seed 2"
/usr/bin/time -f "wall %e s, max RSS %M kB" "$run" gpu e00726_candidate_panel.py e581_host \
  --root seeds713 --month "$month" --seeds 2 --passes 0 1
echo "== 3. M1, all-shifted world, 24 passes"
/usr/bin/time -f "wall %e s, max RSS %M kB" "$run" gpu e00726_msa_panel.py all M1 "$month" 0
echo "Host and GPU peaks are in the newest files under smoke/logs/ (key peaks)."
