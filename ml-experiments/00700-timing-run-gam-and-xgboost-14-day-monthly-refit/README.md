# Experiment 700: timing run of the GAM and XGBoost, and their scores over 8 origin months

**This folder holds the scripts of a timing and feasibility run, not an experiment with a verdict.**
The run re-created the GAM (Primary demand research default 3, experiments 377 and 378) and scored
it, and the XGBoost blend BG2 (experiment 87, arm `p_s6_bg2`) with its monthly-only variant, over
the origin months 2025-11 to 2026-06. It used 14 days of lead time and daily 12:00 UTC origins. The
January 2026 refits of both systems were timed. The scores sit beside steps 5 and 8 of experiment
677. The write-up is `docs/ml-experiments/00700-*/result.md`.

**The GAM stages ran in a sandbox copy of the private data folder.** `sandbox/` under
`data/ml-experiments/00700-*/` holds symlinks to every stored folder except the copies of
experiments 377, 378, and 87's `anchor_preds`, so the run overwrote no stored file. Run the GAM
and XGBoost timing scripts with `DATA_PATH_INTERNAL` set to the sandbox, and run `e00700_export.py`,
`e00700_ladder_year.py`, `e00700_protocol.py`, `e00700_extra.py`, `e00700_report.py`, and
`e00700_plots.py` with `DATA_PATH_INTERNAL` set to the real data folder.

## Files

- `e00700_horizon.py` lengthens the GAM stage 1 window from 672 to 678 half-hours, so that a 12:00
  UTC origin reaches 14 days. Import it before `e00377_ens` and `e00377_run`.
- `e00700_gam_refit.py`, `e00700_gam_stage1.py`, `e00700_gam_stage2.py`, and `e00700_gam_members.py`
  re-run the monthly refit, stage 1, stage 2, and the member responses, and time each.
  `e00700_gam_year.sh` runs stage 1 and stage 2 for 15 Primaries over the 8 origin months, 8
  processes at a time. Experiment 378's `e00378_arms.py` and `e00378_arms_low.py` then run unchanged
  in the sandbox.
- `e00700_xgb_run.sh` and `e00700_xgb_chain.sh` refit the BG2 members at 2026-01-01 with experiment
  87's `e00087_fit_demand.py`, timing each configuration and sampling the GPU.
  `e00087_assemble.py --parts demand --arms p_s6_bg2 p_a3_monthly_in_weekly_slot --no-store` then
  assembles BG2 and its monthly-only variant.
- `e00700_export.py` writes the GAM and XGBoost forecast files in the layout of experiment 677. The
  XGBoost files are the stored arms of experiment 87, relabelled to the 12:00 UTC origin.
  `e00700_ladder_year.py` cuts the stored steps 5 and 8 to the same origins.
- `e00700_protocol.py` and `e00700_extra.py` score every arm on the shared rows, with the protocol
  slices, the MAE of the median by forecast day, coverage, pinball loss, and the verified D5 and D16
  load transfer.
- `e00700_report.py` prints every number in the write-up, and `e00700_plots.py` draws the dot,
  bar, lanes, and MAE-by-forecast-day charts.
- `e00700_report.py` output is saved as `report.txt` in the experiment's data folder.
