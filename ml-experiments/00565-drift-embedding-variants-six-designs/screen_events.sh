#!/bin/bash
# Per-event refits (experiment 556's harness) for M0d and D5d, seed 0. Waits for screen_d5d.sh.
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "00565-drift-embedding-variants-six-designs/screen_d" >/dev/null; do sleep 30; done
$D/run_events.sh M0d || exit 1
$D/run_events.sh D5d || exit 1
echo EVENTS DONE
