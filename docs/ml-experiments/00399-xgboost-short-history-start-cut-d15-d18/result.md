---
id: "00399"
slug: xgboost-short-history-start-cut-d15-d18
theme: retraining-and-training-history
also_relevant_to:
  - demand-inputs
status: planned
kind: change
protocol: demand-v1
size: >-
  medium. Stored: 60 single-series prediction files (private) and one score table. Designs: more
  than one defensible (how many months to cut, and whether to remove near-zero readings by value
  or by run), so the experiment sweeps them. Callers: imports experiment 87's fit modules and
  experiment 381's fit script, and no other folder. Production serving path and degradation rule:
  not touched.
hypothesis: >-
  Dropping the first months of the two short-history Primaries, D15 and D18, from the monthly
  XGBoost demand model's training rows lowers its error at those Primaries, because the readings
  taken while the meters were being commissioned teach the model a wrong level.
prediction: >-
  Written before any score is read. P1. At D18, cutting to 2025-03-01 (NC) lowers the CRPS in days
  1 to 10 by 3% or more against N0, and cutting 3 or 6 months (N3, N6) lowers it by a similar
  amount, because D18 looks commissioned about 7.5 weeks after its first reading. P2. At D15, no
  cut lowers the CRPS by more than 2%, because its level stepped up in July 2025, September 2025,
  and March 2026, and a start cut can remove only part of that bias. P3. Removing near-zero
  readings (NCZ, NCR) changes the CRPS by less than 1% against NC at both Primaries, and NCR
  differs from NCZ by less than NCZ differs from NC.
decision_rule: >-
  Twelve planned contrasts (each of N3, N6, NS, NC, NCZ, and NCR against N0, at each of D15 and
  D18), read on days 1 to 10 under `demand-v1`, report-only. D18 has 6 scored months and D15 has 4,
  so every figure is descriptive and no interval is claimed to establish a gain. Resampling is by
  whole months where a slice holds 3 or more months. An arm counts as helping at a Primary when its
  days 1 to 10 CRPS is lower than N0's in at least two thirds of the scored months and lower in
  the pooled figure, raw and at the 1.15 widening. The porting threshold is zero, because a cut
  date per Primary is a number production already holds as the settling rule. The result says
  nothing about the other 14 Primaries.
benchmark: >-
  N0, the stored monthly control (`monthly/control` of experiment 381: the plain recipe refitted
  monthly, no 1.15 widening). References: the GAM stack and BG2 (experiment 379's scores), the
  `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS estimated from the 11 quantiles, signed median bias, coverage at q01, q05, q95, and q99, and
  avoidable-curtailment cost, from `studies.eval_protocol`, in points of effective capacity; days 1
  to 10 and by lead band; by calendar month since the first reading.
code: ml-experiments/00399-xgboost-short-history-start-cut-d15-d18/
production_status: not-for-production
adopted_in: none
date: 2026-10-30
---

# Does cutting the first months of D15 and D18 help the monthly XGBoost demand forecast?

**This plan is written before any fit runs.** The question: do the first months of a new Primary's
readings, taken while the meter was being commissioned, hurt the monthly XGBoost demand forecast
at D15 and D18, the two Primaries whose history starts after the ENS archive does?

## Design

**Only D15 and D18 can change.** XGBoost trains from 2024-04-01, and every other Primary has
readings from before that date. Experiments 87 and 381 trim no start; the `new_meter_settling` rule
(D15 to 2025-03-20, D18 to 2025-03-04) is not applied in them.

Every arm is the monthly control recipe of experiment 381, refitted monthly, with
`colsample_bytree` and `colsample_bynode` at 1.0, seed 0. Only the training rows differ.

| Arm | Training rows |
|---|---|
| N0 | All rows from the first reading (stored; not refitted). |
| N3 | Drop each series' first 3 months. |
| N6 | Drop each series' first 6 months. |
| NS | Apply the `new_meter_settling` rule of `studies/quick_wins/cleaning_rules.py` (drop 6 weeks after the first 48 live half-hours, live meaning above 5% of capacity). |
| NC | Drop every row before 2025-03-01 (the maintainer's cut, judged by eye on the full-history chart). |
| NCZ | NC, and remove every reading with an absolute value of 0.009 MW or less. |
| NCR | NC, and remove only runs of 2 or more half-hours inside that band. |

**Removal acts everywhere in NCZ and NCR.** A removed half-hour leaves the training labels, the
scoring rows, and the three origin lags (the lag is set to missing, which XGBoost reads natively).
In N3, N6, NS, and NC the cut applies to training labels only, and the lags read the true readings.

**Fresh fits.** Scored refits are D18 from 2026-01 (6 refits) and D15 from 2026-03 (4 refits), 10
single-series fits per arm and 60 in all. N0 is read from
`work/arm_preds/monthly/control/` of experiment 381 (`D15_s0_*`, `D18_s0_*`), after a gate that
reproduces experiment 381's scores from the stored files. Actuals and flags come from experiment
379's `work/scores/rows_stack.parquet` (D15 is time series 12, D18 is time series 13).

**Power is read through `studies.power.scan_power` only.** Experiment 87's fit modules read a
stored `observed.parquet`. A check on 2026-10-30 found it equal to `scan_power` on every shared
half-hour at both Primaries (0 differing values), with 780 extra `scan_power` half-hours per series
that the plan expects to lie after the validation year. The experiment builds its power arrays from
`scan_power` restricted to valid times before 2026-07-01, and a gate asserts the extra half-hours
all fall after that date.

**Scoring.** Each arm is scored raw and at the 1.15 widening, on all rows and on the common set of
rows that NCZ keeps (NCZ changes the scored rows). The write-up says how many rows NCZ and NCR
remove inside the scored year; if NCZ removes none, it reports NCZ equal to NC. D18 steps up by
about 1.5 MW around 2026-07, after the scored year, so the experiment shows it as a descriptive
chart only. D18's dip on 2025-07-01 to 2025-07-03 (daily median 0.4 to 0.9 MW) is described and
not removed. The level anchor of experiment 395 is added to the best arm only if that experiment's
pull request has merged.

**Compute.** 3 CPU threads, `MemoryMax=8G`, no GPU, after the manager grants the slot. The fit
script is experiment 381's `e00381_fit.py` copied with a training-row filter, and sub-agents are
Sonnet only.

## Checks planned

- Reproduction gate: stored N0 files give experiment 381's D15 and D18 monthly-control scores.
- Leakage: the cut removes rows only, so every refit still sees valid times at least 3.5 hours
  before the refit (experiment 87's `known_rows` check).
- Median forecast minus mean forecast per site, and the error of the median forecast beside every
  CRPS contrast.
