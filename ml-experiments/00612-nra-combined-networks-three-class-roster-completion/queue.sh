#!/bin/bash
# usage: queue.sh LOG cmd...: wait until fewer than 3 e00612 GPU processes run, then run cmd
LOG=$1; shift
while [ "$(pgrep -fc 'python .*e00612_(arm|c1|free)')" -ge 3 ]; do sleep 30; done
exec "$@" > $LOG 2>&1
