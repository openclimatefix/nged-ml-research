# Experiment 685 plan: event table on the final-test window, all Primaries

**Data product, not an experiment: no detector is tuned and no model is scored.** The maintainer
opened the final-test window (2026-07-01 to 2026-10-31) for the event table only. The method is
E1-J-sym with the opposite-sign partner test, exactly as in experiment 680.

## Steps

1. **ERA5 extension.** `e00685_fetch_era5.py` runs the existing stage-1 box fetch (single-level
   variables, same box) for 2026-08 to 2026-10, into this experiment's data folder. Source:
   ARCO-ERA5. Hours after 2026-06-30 carry ERA5T in ARCO (no final-ERA5 label); the stored
   2019-09 to 2026-07 files are untouched.
2. **Extended inputs.** `e00685_inputs.py` appends to experiment 103's `rows.parquet` the cleaned
   power of the 15 panel series for 2026-07-01 to 2026-10-31 (`present` where a cleaned reading
   exists, `in_pool` carried from the site's role, no lockdown, no hand-filtered mask, `dark` where
   CAMS irradiance is under 1 W/m²), and appends the ERA5 and CAMS demand-point rows. Weather
   used at forecast time of the lag-free model: ERA5 temperature at the target time and CAMS
   irradiance at the target time, both past-weather, as in experiment 327 (not available at
   forecast time; a lab setting).
3. **Refit arm C.** `e00685_refit.py` builds the feature cache on the extended grid (to 2026-10-31
   23:30) and refits experiment 327's arm C monthly at cuts 2026-06-01 (reproduction gate: must
   match the stored forecast to 1e-6 MW), 2026-07-01, 08-01, 09-01, and 10-01. It patches the
   module constants that close the window (`HOLDOUT_START`, `GRID_END`), in this script only.
4. **Event table.** `e00685_run.py` extends power and residual to 84,720 half-hours, removes the
   all-Primaries median with the scale fixed on the pre-July span, freezes the noise units and
   slow levels on the pre-July residual (`upto`), then runs E1-J-sym with experiment 680's
   unchanged parameters over the whole panel from 2022-01-01. Gate: events starting before
   2026-07-01 match the table experiment 680 stored; any difference is reported.
5. **Outputs** in the data folder: `candidate_events.parquet` and `.csv` (Primaries involved, class,
   onset, end or "open", depth in MW and noise units, confidence, boundary flag) and `SUMMARY.md`
   (priority rows first, then counts per Primary and per month).

## Limits

CPU only, 2 cores, `MemoryMax=16G`. No tuning, no scoring, no reading of window data beyond the
residual panel. Every row is a candidate, never a label; NGED's event labels are an incomplete dump.
