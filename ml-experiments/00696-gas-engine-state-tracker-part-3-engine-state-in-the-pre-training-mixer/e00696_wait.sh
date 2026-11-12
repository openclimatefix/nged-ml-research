#!/bin/bash
# usage: e00696_wait.sh LOGDIR N: block until N job logs in LOGDIR end with ALLDONE or FAILED; print the failed ones
until [ "$(grep -lE '^(ALLDONE|FAILED)' "$1"/job_*.out 2>/dev/null | wc -l)" -ge "$2" ]; do sleep 20; done
grep -l '^FAILED' "$1"/job_*.out; echo done
