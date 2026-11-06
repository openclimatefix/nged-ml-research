#!/bin/bash
# usage: chain_arm.sh ARM months...; pretrain, step 2, fits, assemble, injected bank. Waits for a free GPU slot (at most 2 processes).
D="$(cd "$(dirname "$0")" && pwd)"
A=$1; shift
wait_slot() { while [ "$(pgrep -fc 'bin/python.*e00603_arm.py')" -ge 2 ]; do sleep 20; done; }
set -e
wait_slot; $D/run.sh pretrain $A --months "$@"
wait_slot; $D/pipeline_arm.sh $A "$@"
wait_slot; $D/run.sh inject $A --months "$@"
echo $A CHAIN DONE
