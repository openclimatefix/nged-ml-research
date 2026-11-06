#!/bin/bash
# Resumable. Usage: pipeline_probe.sh ARM [ARM ...]: the heads' predictions on the injected bank and the observed lags
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for a in "$@"; do $D/run.sh $a probe; done
echo PROBE DONE
