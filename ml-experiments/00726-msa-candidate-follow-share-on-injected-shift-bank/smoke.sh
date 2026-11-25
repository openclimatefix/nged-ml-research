#!/usr/bin/env bash
# Experiment 726: the memory and timing smoke test of phase 0 (design gate 4 and review H5).
#
# Usage: smoke.sh [month]      (default 2026-06: the fold with the longest training window)
#
# Runs four jobs through run.sh on one fold month, each of which logs its host peak and GPU peak
# (`peaks` in the job log under `logs/`, and a final line on standard output for the scorer):
#   1. the Candidate in the host-only world at root e667, seeds 0 and 1 in one process;
#   2. the Candidate in the host-only world at root seeds713, seed 2;
#   3. M1 on the all-shifted world, all 24 passes;
#   4. the scorer on the CPU (it needs the files of jobs 1 to 3, so it scores only what exists).
# Pass the month with the most training rows if it is not the default. If any host peak is above
# 8 GB, ask the manager for the design's 12 GB before phase 1.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
month="${1:-2026-06}"
run="$here/run.sh"

echo "== 1. Candidate e667, seeds 0 and 1, one process"
/usr/bin/time -f "wall %e s, max RSS %M kB" "$run" gpu e00726_candidate_panel.py e581_host \
  --root e667 --month "$month" --seeds 0 1 --passes 0 1
echo "== 2. Candidate seeds713, seed 2"
/usr/bin/time -f "wall %e s, max RSS %M kB" "$run" gpu e00726_candidate_panel.py e581_host \
  --root seeds713 --month "$month" --seeds 2 --passes 0 1
echo "== 3. M1, all-shifted world, 24 passes"
/usr/bin/time -f "wall %e s, max RSS %M kB" "$run" gpu e00726_msa_panel.py all M1 "$month" 0
echo "== 4. the scorer on the CPU"
/usr/bin/time -f "wall %e s, max RSS %M kB" "$run" cpu e00726_score.py counted --plan two --draws 2000 || true
echo "Host and GPU peaks of jobs 1 to 3 are in the newest files under logs/ (key peaks)."
