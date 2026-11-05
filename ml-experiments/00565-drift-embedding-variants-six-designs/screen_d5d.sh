#!/bin/bash
# D5 in the daily harness (cadence d), seeds 0 and 1, injected shift on seed 0 (block ages 0 to 6). Waits for screen_d2.sh.
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "00565-drift-embedding-variants-six-designs/screen_d2" >/dev/null; do sleep 30; done
$D/run.sh fit --arm D5d --seeds 0 1 || exit 1
$D/run.sh fit --arm D5d --seeds 0 --stress || exit 1
$D/run.sh assemble --arm D5d --seeds 0 1 || exit 1
echo D5D DONE
