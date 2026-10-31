---
id: "00423"
slug: pooled-mdn-shared-date-time-encoder
title: >-
  Does a small calendar encoder, shared by all 16 Primaries, forecast bank-holiday weeks better
  than the hand-made calendar columns of the pooled mixture density network?
theme: demand-inputs
also_relevant_to:
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: forecasts and fitted networks in this experiment's private folder only;
  experiments 397, 407, and 419 are read and never written. Production serving path: not touched
  (this repository holds none). Degradation rule: not touched. Designs: more than one defensible
  (whether holiday names enter, per-Primary or shared encoder), so the arms register each.
  Callers: the scripts import experiments 87, 48, 397, 407, and 419 and the `studies` evaluation
  code, whose callers sit in those folders.
hypothesis: >-
  A small encoder that turns the half-hour of day, day of week, day of month, day of year, and the
  bank-holiday name into one learned vector, shared across the 16 Primaries, lets the pooled plain
  mixture density network share what it learns about holidays across Primaries and across
  holidays of one kind, so it forecasts bank-holiday weeks better than the hand-made calendar
  columns and is no worse elsewhere.
prediction: >-
  Written before any fit and before any score is read. P1. The encoder with holiday names (E1)
  lowers the CRPS at days 1 to 10 on the bank-holiday-week rows by at least 0.10 points of capacity
  against the encoder with dates only (E2), at the signed-MW six. P2. Over all rows, E1 and E2 are
  each within the minimum detectable effect of the hand-made columns (R) at both groups, and the
  page says so rather than claiming equality. P3. Christmas and Easter weeks are reported per
  set and not tested, because the five screen folds hold one Christmas and a two-day sliver of
  Easter week. P4 (only if E1 or P1 is promising). A per-Primary encoder (E3) is worse than E1 on
  every row set, because each encoder then sees one sixteenth of the holiday days.
decision_rule: >-
  CRPS at days 1 to 10 in points of effective capacity, read per group (signed-MW six; MVA-only
  eight; D15 and D18 report-only), paired by fold and seed, with the month-block bootstrap of
  `studies.eval_protocol`, 20,000 resamples, at a 95% level Bonferroni-corrected over the planned
  contrasts. Planned contrasts. C1 E1 minus R, all rows. C2 E2 minus R, all rows. C3 E1 minus E2,
  bank-holiday-week rows. C4 E1 minus R on the Christmas set and on the Easter set (descriptive).
  C5 E3 minus E1 (run only if C3 is at most -0.05 or C1 is below zero). Seed noise is large
  (experiment 419: seeds 0 to 2 against 3 to 5 moved the MVA-only eight by -0.047), so every arm is
  fitted with seeds 0 to 5 and the reference R reuses the stored seeds 0 to 2 (experiment 407
  `ctrl`) and 3 to 5 (experiment 419 `ctrl_s3`). The minimum detectable effect is printed by the
  scorer before any contrast is read, as twice the standard deviation over the folds of the
  difference in the signed-MW six CRPS between `ctrl` and `ctrl_s3`, divided by the square root of
  two for six-seed means (experiment 419 found 0.0738 for three-seed means). A contrast smaller
  than the minimum detectable effect is reported as inside the noise. The hypothesis is supported
  if P1 holds with an interval wholly below zero and C1 is not above the minimum detectable
  effect; partly supported if P1 holds on its point estimate only; not supported otherwise. The
  encoder adds a module and a calendar table to the pipeline, so the smallest gain worth porting
  it is 0.03 points at the signed-MW six over all rows or 0.10 on the bank-holiday-week rows. No
  adoption is sought. The five screen folds are interim.
benchmark: >-
  The pooled plain network `ctrl` of experiment 407 (seeds 0 to 2) and `ctrl_s3` of experiment 419
  (seeds 3 to 5), on the same half-hours; BG2, `manual_heuristic`, and `climatology` as
  absolute-skill references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, raw and at the 1.15 widening, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00423-pooled-mdn-shared-date-time-encoder/
---

**Plan: can the pooled network learn how people use the calendar, instead of being handed it?**

## Design note (Sonnet read of the docs sketches and experiment 396's holiday arms)

- **The only encoder sketch is `TimeEncoder` in the production docs'
  [encoders page](https://openclimatefix.github.io/nged-substation-forecast/techniques/encoders/).**
  It names time of day, day of week, month, bank holidays, Christmas, and Easter, and gives no
  sizes or training recipe.
- **The pooled plain network gets 18 hand-made calendar columns today** (experiment 397,
  `e00397_prep.feature_names`): sine and cosine of the hour at 1 and 2 cycles, a one-hot weekday,
  `covid`, three Christmas columns, `bank_holiday`, and two distance columns, from
  `holidays.country_holidays("GB", subdiv="ENG")`. It never sees a holiday name, a day of month, or
  a day of year.
- **Experiment 396 (XGBoost)** adds a bank-holiday flag, a signed distance clipped at 7 days,
  day-before and day-after flags, and a holiday-type code (bank holiday, Easter, Christmas period,
  New Year, other).
- **Each named holiday has only one or two days in a training window.** The window is the 24 months
  before the refit, clipped at 2024-04-01.

| Holiday | 2025-10 | 2025-12 | 2026-01 | 2026-03 | 2026-06 |
|---|---|---|---|---|---|
| Christmas Day | 1 | 1 | 2 | 2 | 2 |
| Boxing Day | 1 | 1 | 2 | 2 | 2 |
| New Year's Day | 1 | 1 | 1 | 2 | 2 |
| Good Friday | 1 | 1 | 1 | 1 | 2 |
| Easter Monday | 2 | 2 | 2 | 2 | 2 |
| Early May, spring, summer | 2 | 2 | 2 | 2 | 2 |
| All bank holidays | 12 | 12 | 14 | 15 | 16 |

- **What that limits.** An embedding for "Christmas Day" is fitted on one or two days, so it can
  memorise them and not generalise. A holiday-name encoder can help only by tying holidays
  together: the five Monday bank holidays share one type with 8 to 10 days. A dates-only encoder
  cannot follow Easter between years. Any Easter result rests on two days, so the expected gain is
  small and sits in E1 against E2, and in shared against per-Primary encoders.

- **Experiment 396's holiday columns helped one metering group only.** In January they lowered
  the XGBoost CRPS by 0.046 points at the signed-MW group (interim, unaudited) and raised it at the
  MVA-only group. The plan therefore scores E1 and E2 separately for the signed-MW six and the
  MVA-only eight in every contrast, and does not expect the holiday-aware encoder to help the
  eight.

## Arms

All arms keep the weather, solar, lag, and `covid` inputs and the 8-wide Primary embedding of
experiment 397's plain network (`PlainMdn`, setting `g09`, summary scheme, 3 Gaussian components).

- **R (reference).** The 18 hand-made calendar columns; stored forecasts of experiments 407 and
  419, not refitted.
- **E1.** A 16-wide vector replaces the calendar columns. Inputs: local half-hour 0 to 47 (8-wide
  embedding), weekday (4), day of month (4), day of year (366 buckets, 4), holiday name (12
  categories, 4), and the two distance columns; then Linear 32, GELU, Linear 16. 3,440 parameters.
  The hour index uses the local half-hour, so the clock changes land on the same index.
- **E2.** E1 without the holiday name or distance columns. 3,200 parameters.
- **E3 (conditional, C5).** One E1 per Primary. 55,040 parameters.

The encoder trains end to end on the same loss, optimiser, early stopping, and seeds as R.

## Row sets and folds

Row sets use the local calendar date of the valid time, as in `e00396_arms.add_calendar`: bank
holiday day; bank-holiday week (Monday to Sunday); Christmas set (24 December to 1 January); Easter
set (Monday of Holy Week to Easter Monday).

| Row set (days in the test month) | 2025-10 | 2025-12 | 2026-01 | 2026-03 | 2026-06 |
|---|---|---|---|---|---|
| Bank-holiday days | 0 | 2 | 1 | 0 | 0 |
| Bank-holiday-week days | 0 | 10 | 4 | 2 | 0 |
| Christmas set | 0 | 8 | 1 | 0 | 0 |
| Easter set | 0 | 0 | 0 | 2 | 0 |

- **The five screen folds hold one Christmas and a two-day sliver of Easter week.** The plan adds
  the event fold 2026-04 (Good Friday 2026-04-03, Easter Monday 2026-04-06), which gives the Easter
  set 8 days, if the fold's weather inputs exist for the pooled network. Otherwise Christmas and
  Easter appear only as example charts.

## Standing practices followed

- **Monthly refits** on the 24-month window of experiment 397's folds; **cleaned data** (flagged
  exact-zero half-hours dropped, counts printed per fold); **D15 and D18 cut before 2025-03-01**;
  **ENS mean and spread passed once**; **fitted networks saved** under this experiment's private
  folder; **stored forecasts reused** (R) and the fits saved are the reason the cost is 60 fits,
  not 90.
- **Cost.** E1 and E2 times 5 folds times 6 seeds is 60 fits of 50 to 200 seconds each; the event
  fold adds 12 for E1 and E2 and 6 for R; E3 adds 30 if it runs. One GPU process under 8 GB, one
  feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`. The encoder indices are integers, precomputed
  once per refit.
- **Earlier experiments adopted.** Experiment 419: none of its input changes won, so the reference
  stays `ctrl`. Experiment 410: training at other origin hours is experiment 422's subject and is
  not adopted here. Experiments 406 and 403: the Primary embedding and pooling are kept.
- **Plan reviewed against the latest evidence** (`origin/main`, 2026-10-31): experiment 419 showed
  that seed noise moves the six by about 0.011 and the MVA-only eight by 0.047, so the plan uses
  six seeds and prints the minimum detectable effect; the planned experiments 418, 421, and 422
  change the training data or inputs of the same network; from their plan titles none changes the calendar
  columns, and a later experiment must merge with them.

## Results

Not yet run.
