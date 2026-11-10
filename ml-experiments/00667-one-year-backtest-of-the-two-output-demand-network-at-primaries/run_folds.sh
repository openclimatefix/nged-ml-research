#!/bin/bash
# usage: run_folds.sh MONTH...: run the listed folds one after the other (start up to 3 of these in parallel)
D="$(cd "$(dirname "$0")" && pwd)"
for m in "$@"; do $D/chain.sh $m; done
