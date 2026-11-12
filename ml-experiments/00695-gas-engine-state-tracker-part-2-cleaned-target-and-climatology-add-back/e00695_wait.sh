#!/bin/bash
# usage: e00695_wait.sh LOGDIR PATTERN N: block until N files in LOGDIR matching PATTERN contain ALLDONE
until [ "$(cat "$1"/$2 2>/dev/null | grep -c ALLDONE)" -ge "$3" ]; do sleep 10; done
echo done
