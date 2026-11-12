#!/bin/bash
# usage: e00695_queue.sh LOGDIR [WORKERS]: fit every arm after A0 (A1id, A1, W0, W1, AH, A1n), one fold per
# worker process, WORKERS at a time (default 10; the GPU was saturated at 8). Each job logs to LOGDIR/job_ARM_MONTH.out
# and the queue writes LOGDIR/queue.out when it ends. Arm A0 was run before this queue as 8 single-fold
# `e00695_arm.sh A0 MONTH` jobs started together. The A1id arm needs `e00695_identity.py stack` first.
D="$(cd "$(dirname "$0")" && pwd)"
LOG=$1; W=${2:-10}
mkdir -p "$LOG"
{ echo "A1id 2025-11"
  for a in A1 W0 W1 AH A1n; do
    for m in 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06; do echo "$a $m"; done
  done; } > "$LOG/jobs.txt"
xargs -P "$W" -L 1 bash -c "\"$D/e00695_arm.sh\" \$0 \$1 > \"$LOG/job_\$0_\$1.out\" 2>&1" < "$LOG/jobs.txt"
echo QUEUEDONE > "$LOG/queue.out"
