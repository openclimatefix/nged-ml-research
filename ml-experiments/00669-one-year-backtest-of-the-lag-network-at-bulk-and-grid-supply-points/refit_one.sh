#!/bin/bash
# usage: refit_one.sh GROUP MONTH: pretrain, step2, fit, assemble one fold (used to redo a fold).
D="$(cd "$(dirname "$0")" && pwd)"
for c in pretrain-wt step2 fit assemble; do $D/gpu.sh e00669_fit.py $c --group $1 --months $2 --seeds 0 1 || exit 1; done
echo REFIT DONE
