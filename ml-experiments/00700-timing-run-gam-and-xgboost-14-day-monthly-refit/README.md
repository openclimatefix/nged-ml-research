# Experiment 700: timing run of the GAM and XGBoost for one origin month

**This folder holds the scripts of a one-month timing and feasibility run, not an experiment with a
verdict.** The run re-created the GAM (Primary demand research default 3, experiments 377 and 378)
and the XGBoost model BG2 (experiment 87, arm `p_s6_bg2`) for the origin month 2026-01. It used 14
days of lead time, daily 12:00 UTC contract origins, a refit at 2026-01-01, and all Primaries that
each model covers. The scores sit beside steps 5 and 8 of experiment 677.

**Everything ran in a sandbox copy of the private data folder.** `sandbox/` under
`data/ml-experiments/00700-*/` holds symlinks to every stored folder except the copies of
experiments 377, 378, and 87's `anchor_preds`, so the run overwrote no stored file. Run each script
with `DATA_PATH_INTERNAL` set to the sandbox for the model stages, and to the real data folder for
`e00700_export.py`, `e00700_ladder_january.py`, `e00700_protocol.py`, and `e00700_extra.py`.

## Files

- `e00700_horizon.py` lengthens the GAM stage 1 window from 672 to 678 half-hours, so that a 12:00
  UTC origin reaches 14 days.
- `e00700_gam_refit.py`, `e00700_gam_stage1.py`, `e00700_gam_stage2.py`, and `e00700_gam_members.py`
  re-run the monthly T-all refit, stage 1, stage 2, and the member responses for 2026-01 and time
  each. Experiment 378's `e00378_arms.py` and `e00378_arms_low.py` then run unchanged in the sandbox.
- `e00700_xgb_run.sh` and `e00700_xgb_chain.sh` refit the BG2 members at 2026-01-01 with experiment
  87's `e00087_fit_demand.py`, timing each configuration and sampling the GPU.
  `e00087_assemble.py --parts demand --arms p_s6_bg2 p_a3_monthly_in_weekly_slot --no-store` then
  assembles BG2 and its monthly-only variant.
- `e00700_export.py` writes the January files in the layout of experiment 677.
  `e00700_ladder_january.py` cuts the stored steps 5 and 8 to the same month.
- `e00700_protocol.py` and `e00700_extra.py` score every arm on the shared rows, with the protocol
  slices, the MAE of the median by forecast day, coverage, pinball loss, and the verified D5 and D16
  load transfer.
