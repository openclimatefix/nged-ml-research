---
id: "00392"
slug: xgboost-nra-load-shift-masked-targets
theme: lsc-detectors
also_relevant_to:
  - lagged-power-and-error-correction
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: other
protocol: demand-v1
size: >-
  complex. Stored: per-arm monthly forecasts, every fitted booster (about 27 MB each, under `work/boosters/<arm>_<form>/` in this experiment's private folder), mask tables
  and injected-transfer tables, all in this experiment's private folder; experiments 87, 379, 381,
  383, and 391 are read and never written. Designs: more than one defensible (which half-hours the
  detector marks, how long after an event to keep dropping rows, how injected transfers are built);
  the plan fixes each before any score is read. Callers: imports experiment 87's fit modules and
  experiment 383's merged folder, and reads experiment 391's stage 1 surprise tables. Production
  serving path and degradation rule: not touched.
hypothesis: >-
  Removing load-shift events from the training targets of the monthly XGBoost demand model gives a
  forecast that stays near the normal running arrangement (NRA) level while a transfer is under
  way, and a mask built by the "deep and long" detector does so almost as well as a mask built from
  NGED's 66 hand-labelled transfers. Giving the unmasked model the surprise of its lags and of its
  neighbours (arm R5) gives the same NRA behaviour with no detector.
prediction: >-
  Written before any score is read. P1. Outside events and the 48 hours after them, no masked arm
  differs from the unmasked control R0 by more than 0.02 points of effective capacity in CRPS
  (experiment 75 found +0.004 for the GAM). P2. Inside events, the lagless forms of R1 and R2
  follow under 25% of the labelled depth; the lagged form of R0 follows over 50%. P3. R2 recovers
  at least 75% of what R1 recovers in the injected-transfer check. P4. R5 follows under 50% of the
  depth in the lag-encroachment sweep at k = 3, between R0 and R2. P5. The random-block control
  moves the score outside events by under 0.01 points, so any gain of R1 or R2 comes from the
  removal.
decision_rule: >-
  Descriptive, no adoption claimed. Seven planned contrasts (C1 to C7 below), read at the 99.29%
  level (a Bonferroni correction across 7 at a 95% family level) with whole-event blocks on event
  slices and month blocks elsewhere. An arm passes as an NRA forecast when all four hold, with the
  tolerances fixed here: (1) no harm outside events, interval upper bound under +0.02 points
  (check 1); (2) the median follows at most 25% of the labelled depth inside events, lagless form
  (check 2) and at most 25% of D at k = 3 in the sweep; (3) at least 75% of an injected transfer's
  depth is recovered (check 3); (4) the 5% to 95% band covers 0.90 of event-level readings within
  0.03, and the pooled 5% to 95% band outside events is no more than 10% wider than R0's. Porting
  cost sets the threshold: a mask applied to training targets (R1, R2) adds a detector but no
  serving-path change, so it needs a gain whose interval excludes zero on the event slices plus
  check 1; R5 adds two column sets computed from one extra stage 1 model, so the same bar applies.
benchmark: >-
  R0 (unmasked, same recipe refitted monthly), BG2, `manual_heuristic`, `climatology`, and the
  stored forecasts of experiments 383 (weekly, interim comparison only) and 391 (monthly).
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS estimated from the 11 quantiles, pinball loss and coverage at q01, q05, q95, and q99, share
  of labelled depth followed, depth recovered on injected transfers, and avoidable-curtailment
  cost (experiment 383's definition), in points of effective capacity; headline band days 1 to 10,
  all 16 Primaries to 14 days, four lead bands.
code: ml-experiments/00392-xgboost-nra-load-shift-masked-targets/
production_status: not-for-production
adopted_in: none
---

# Planned: an NRA forecast from XGBoost trained with load-shift events removed from its targets

**Status: plan only. No fit has run, and none starts until the manager grants a compute slot.**

## What experiment 383 already settles

- **383's mask rule differs from R1's.** Its `lagless` arm drops training rows whose valid time
  lies in the *padded* hand-filtered mask of the feature cache (24 hours of padding on each side
  of an event, taken from `load_or_build_feature_cache`). R1 here drops rows inside an event and
  the 48 hours after it ends, with no padding before the onset. Its refits are weekly; this
  experiment refits monthly. Its forecasts are therefore not comparable with R0 arm for arm.
- **Consequence for R1.** The rules differ, so R1 is refitted monthly under the brief's rule
  (event plus 48 hours after), as R0's twin; the manager agreed on 2026-10-29. 383's stored
  `lagless` arm is not reused because its padding, its weekly cadence, and its single lagless form
  make it incomparable with R0 arm for arm.
- **Which NRA checks 383 reports.** Check 4 (tails and avoidable-curtailment cost) on
  contaminated-lag rows, during events, in the 7 days after, and on all rows, and a partial check 1
  (main score outside events, days 1 to 10, no lead bands). It does not report check 2 (share of
  labelled depth followed), check 3 (injected transfers), or the lag-encroachment sweep (its
  boosters were not saved). All three are new here, as are R2, R3, R4, and the random-block control.
- **Evidence read so far.** 383's lagless arm lowered CRPS on contaminated rows at the signed-MW
  six (rests on two events) and raised it at the MVA-only eight; nulling lags (c1) did nothing.

## Arms

Every arm is experiment 87's `n1h` XGBoost member, refitted **monthly** (maintainer rule,
2026-10-29), `colsample_bytree` and `colsample_bynode` fixed at 1.0, one seed, the same training
rows and parameters as R0. Each arm is fitted in a **lagged** form (three origin lags) and a
**lagless** form (no power lags). Weekly refits are not used, so every absolute and tail number is
labelled monthly; any winner is re-checked weekly before a release claim.

| Arm | Training rows dropped | Source of the mask | Role |
|---|---|---|---|
| R0 | none | none | control: lagged form is experiment 381's stored monthly `control` (see below); lagless form refitted here |
| R1 | valid time inside a hand-labelled transfer, or within 48 h after it ends | the 66 hand labels, events that ended before the fold's `train_end` | ceiling, never a candidate |
| R2 | same rule | the "deep and long" detector with the neighbour veto, built from data inside the fold's training window | candidate |
| Rc1, Rc2 | as many rows as R1 and R2, in random blocks of the same lengths | random | control for the smaller training set |
| R5 | none | none (surprise columns only) | stored: experiment 391's arm N1, not refitted |
| R3 | train, detect, mask, refit; two passes | detector | optional, last; stop rule: second pass changes under 5% of marked half-hours |
| R4 | as R2, plus experiment 391's neighbour surprise columns | detector | skipped unless 391 shows neighbour surprise helps |

- **Detector.** Taken from experiment 391's `e00391_detector.py` (plan PR 41, not yet merged); if
  391 has not merged when R2 is due, the file is copied into this folder with a note. The mask for
  a fold uses only half-hours before its `train_end`, and the detector's tuning months are fixed
  before any score is read.
- **R5 shares 391's N1 forecasts and surprise tables**; the plan does not refit it. The tables
  will be under `data/ml-experiments/00391-xgboost-lag-surprise-second-stage/surprise_tables/`
  (not yet written). N1 keeps the three lags, so R5 exists in the lagged form only, and the
  lagless contrast is R0 and R1's. Experiment 391 refits with no widening and scores raw and at
  1.15, which this experiment copies. R5 enters the sweep only if 391 keeps N1's boosters; the
  391 session is asked to keep one refit per quarter.
- **R0 lagged reuses experiment 381's monthly `control`** (`work/arm_preds/monthly/control/`, 192
  files, 16 Primaries by 12 refits, origins 2025-07-01 to 2026-06-01, no 1.15 widening), as the
  manager allowed. The recipe matches: 381's control uses `fd.input_columns(a7=False)` and
  experiment 87's `n1h` configuration with `common.N_ROUNDS`, the same modules this experiment's fit
  script calls, and drops no rows. A reproduction gate confirms it before use: one refit (D2,
  2025-07-01) is fitted here and must match the stored file within 0.0043 CRPS points, 383's gate.
  If it fails, R0 lagged is refitted. R0 lagless is always fitted here.
- **Widening.** Each arm's quantiles are stored raw, and every score is reported raw and at the
  1.15 widening factor, for R0 and every arm alike. The widening of 391's N1 is read from its page
  before any comparison with R5.
- **Porting cost:** R1 and R2 change only the training targets (R2 needs the detector in the
  training loop, not in serving); R5 adds a stage 1 model and two column sets to serving.

## Planned contrasts

C1 R1 minus R0. C2 R2 minus R0. C3 R2 minus R1 (does the detector mask match the oracle). C4 R5
minus R0. C5 R5 minus R2 (does surprise match the detector). C6 R1 minus Rc1. C7 R2 minus Rc2. Each
is read on four slices: outside events and the 48 hours after (harm), inside events, the 7 days
after, and all rows.

## The four NRA checks (pre-registered)

1. **No harm outside events:** arm minus R0, all 16 Primaries to 14 days, by lead band.
2. **NRA gap inside events:** for the half-hours inside a labelled transfer and the 48 hours after,
   the median's movement from its pre-event level divided by the label table's depth (MW), as a
   share of capacity. Low-confidence labels are reported separately.
3. **Injected transfers:** steps of known depth and duration (quartiles of the label table, 0.71,
   1.1, and 2.3 MW; both signs; durations drawn from the labelled events) are added to clean
   periods of the training history as in the `dp_nra` experiment, the arm is refitted with the
   injected events treated as labelled (R1) or run through the detector (R2), and the share of the
   clean original recovered is reported per arm. Restricted to the signed-MW six and four of the
   MVA-only eight; the subset is fixed before looking.
4. **Tails and cost:** coverage at q01, q05, q95, q99 and avoidable-curtailment cost during events,
   in the 7 days after, and across all rows.

## Lag-encroachment sweep

Prediction-only, on the saved boosters of every refit (path in the front matter's size line). The three lags are
ordered by recency: origin, day-earlier, week-earlier (the exact times come from experiment 87's
`origin_lags`). Levels k = 0 to 3. A step of depth D (2%, 5%, 10% of capacity and the three
quartile depths; both signs) is added to every lag inside the synthetic event. For R5 and R4 the
surprise and neighbour columns are recomputed from the shifted lags, with the neighbour carrying
the opposite step. Cases (a) event under way and (b) event ended are reported separately, by lead
band, per group and per Primary. The lagless forms must show share 0 and no widening, which tests
the code. A few real labelled events chosen by a rule fixed here (the longest event per Primary
with high confidence) are compared with the synthetic ones.

## Also reported

Per Primary, worst Primary, per group (signed-MW six, MVA-only eight, D15 and D18 report-only),
labelled events per Primary and per slice, D2 and D4 as placebos, the mean of the median minus the
mean forecast per site, and which audit triggers fired. Figures follow the `dataviz` skill and show
no metered generator's name or ID.

## Order and compute

1. R0 (lagged, lagless), then R1, then Rc1 and R2, Rc2. 2. R5 scoring from 391's stored output.
3. Checks 1, 2, 4. 4. Check 3 and the sweep. 5. R3 and R4 last. At most 3 threads, `OMP_NUM_THREADS=1`
per process, `MemoryMax=16G`, spawn start method. Fit counts: 5 arms (R0, R1, R2, Rc1, Rc2) in 2
forms, each with 12 to 14 monthly refits at 16 Primaries, plus injected-transfer refits for 10
Primaries. The manager sizes the slot from these counts.

## Open items

- A compute slot (3 threads), granted by the manager once the load falls. Until then only code and
  a 1-core smoke test run.
- Experiment 391's surprise tables and N1 boosters (above).

## Sources

Builds on experiment 383's oracle-mask arms, experiment 75 and 44 (masking in the GAM), experiment
100 (dropped half-hours in XGBoost), experiment 391's detector and surprise columns, and the
`dp_nra` experiment's injected-transfer scoring.
