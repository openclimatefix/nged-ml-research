#!/bin/bash
# usage: pipeline_arm.sh ARM [months...]; step 2, fits (days 0 and 14, seeds 0 and 1), assemble
set -e
D="$(cd "$(dirname "$0")" && pwd)"
A=$1; shift
M=(); [ $# -gt 0 ] && M=(--months "$@")
$D/run.sh step2 $A "${M[@]}"
$D/run.sh fit $A "${M[@]}"
$D/run.sh assemble $A "${M[@]}"
echo $A DONE
