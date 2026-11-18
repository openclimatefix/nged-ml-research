#!/usr/bin/env bash
# Experiment 701: refit the XGBoost members of BG2 for the runs 2026-07-01 to 2026-10-31, then assemble.
# Usage, from the repository root (set DATA_PATH_INTERNAL and the dummy NGED_S3_* variables first):
#   ml-experiments/00701-*/e00701_xgb_run.sh monthly   # the monthly-only variant, about 1.5 hours
#   ml-experiments/00701-*/e00701_xgb_run.sh weekly    # then BG2 in full, about 2.5 hours more
# The fit skips every file that exists, so a stopped run resumes by rerunning the same command.
# The GPU is shared by 6 worker processes with 2 CPU threads each (12 cores, as in experiment 700).
set -euo pipefail
stage=$1
d=ml-experiments/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10
m=ml-experiments
export PYTHONPATH=$m/00087-bg2-best-guess-assembled-with-drop-one-ablations:$m/00033-best-guess-architecture-vs-production:$m/00048-weekly-retraining-confirmation-and-extension-demand-generators:$m/00700-timing-run-gam-and-xgboost-14-day-monthly-refit:$m/00377-demand-v1-first-run-feasibility:$d
export MLFLOW_DISABLE_AGENT_HINT=1
fit() { uv run --no-sync python "$d/e00701_fit_demand.py" --workers 6 --cores 12 "$@"; }
py() { uv run --no-sync python "$d/$1" "${@:2}"; }
case "$stage" in
  monthly)
    fit --config frozen
    fit --config frozen --offset
    fit --config n1h --regime monthly
    fit --config n1h --regime monthly --offset
    fit --config an --regime monthly
    fit --config an --regime monthly --offset
    fit --config pt --regime monthly
    py e00701_assemble.py prepare
    py e00701_assemble.py build --arms p_a3_monthly_in_weekly_slot
    py e00701_export.py --arms p_a3_monthly_in_weekly_slot
    ;;
  weekly)
    fit --config n1h --regime weekly
    fit --config n1h --regime weekly --offset
    py e00701_assemble.py prepare
    py e00701_assemble.py build --arms p_s6_bg2
    py e00701_export.py --arms p_s6_bg2
    ;;
  *) echo "stage is monthly or weekly" >&2; exit 2 ;;
esac
echo "stage $stage done"
