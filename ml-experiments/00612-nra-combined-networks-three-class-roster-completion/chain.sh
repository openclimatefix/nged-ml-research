#!/bin/bash
# usage: chain.sh NAME [--drop ...]: pre-training, step 2, fit, and the injected bank of an arm with a removed input
D="$(cd "$(dirname "$0")" && pwd)"
N=$1; shift
set -e
$D/gpu.sh e00612_arm.py $N pretrain "$@"
$D/gpu.sh e00612_arm.py $N step2 "$@"
$D/gpu.sh e00612_arm.py $N fit "$@"
$D/gpu.sh e00612_arm.py $N inject "$@"
