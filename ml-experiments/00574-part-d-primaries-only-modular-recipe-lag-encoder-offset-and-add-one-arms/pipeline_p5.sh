#!/bin/bash
# Final candidate P5 = P0 + embargoed three-step schedule (P4E) + B1 columns (P2): step 2, fits, stress, assemble.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh step2 --arm P5
$D/run.sh fit --arm P5
$D/run.sh fit --arm P5 --seeds 0 --stress
$D/run.sh assemble --arm P5
echo P5 DONE
