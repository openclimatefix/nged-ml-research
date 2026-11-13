#!/bin/bash
# usage: e00696_queue.sh LOGDIR JOBFILE [WORKERS]: run `e00696_chain.sh ARM MONTH [COMMAND...]` for every line
# "ARM MONTH SEEDS_WITH_UNDERSCORES [COMMANDS]" of JOBFILE (SEEDS like 3_4_5), WORKERS at a time (default 8).
# Each job logs to LOGDIR/job_ARM_MONTH_SEEDS.out and ends with a line "ALLDONE" or "FAILED". Needs
# DATA_PATH_INTERNAL. Check `uptime` and `nvidia-smi` before raising WORKERS.
D="$(cd "$(dirname "$0")" && pwd)"
LOG=$1; JOBS=$2; W=${3:-8}
mkdir -p "$LOG"
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=d NGED_S3_BUCKET_SECRET=d
xargs -P "$W" -L 1 bash -c "s=\$(date +%s); SEEDS=\"\$(echo \$2 | tr _ ' ')\" \"$D/e00696_chain.sh\" \$0 \$1 \${@:3} > \"$LOG/job_\$0_\$1_\$2.out\" 2>&1 && echo \"ALLDONE \$(( \$(date +%s)-s ))s\" >> \"$LOG/job_\$0_\$1_\$2.out\" || echo FAILED >> \"$LOG/job_\$0_\$1_\$2.out\"" < "$JOBS"
echo QUEUEDONE > "$LOG/queue_$(basename $JOBS).out"
