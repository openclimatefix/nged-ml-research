#!/usr/bin/env bash
# Start the workers of one stage of experiment 737 inside one scope (called by run.sh).
#
# `systemd-run` expands `$` in the arguments of the command it starts, so this loop lives in its own
# file. Environment: WORKER_CPUS (space-separated cpu list), WORKERS, PYTHON_BIN, and RUN_PY.
set -uo pipefail
read -ra cpu <<< "$WORKER_CPUS"
pids=()
for ((i = 0; i < WORKERS; i++)); do
    pin="${cpu[$i]:-}"
    if [ "${#cpu[@]}" -ne "$WORKERS" ] || [ -z "$pin" ]; then pin="${WORKER_CPUS// /,}"; fi
    taskset -c "$pin" "$PYTHON_BIN" "$RUN_PY" "$@" --worker "$i" --of "$WORKERS" &
    pids+=($!)
done
worst=0
for pid in "${pids[@]}"; do
    wait "$pid" || { code=$?; [ "$code" -gt "$worst" ] && worst=$code; }
done
exit "$worst"
