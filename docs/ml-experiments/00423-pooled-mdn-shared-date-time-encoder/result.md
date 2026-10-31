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
  against the same encoder without the name (E1n), at the signed-MW six; P1 is read only if the
  printed minimum detectable effect on those rows is at most 0.10, and is otherwise reported as
  untestable. P2. Over all rows, E1, E1n, and E2 are each within the minimum detectable effect of
  the hand-made columns (R) at both groups; the page says "inside the noise" and does not claim
  equality. P3. E2 (day of year and day of month, no holiday information) is no better than E0
  (half-hour and weekday only) over all rows, because a day-of-year bucket seen once or twice can
  only be memorised. P4. The encoder does not help the MVA-only eight. P5 (only if C3 is at most
  -0.05 or C1 is below zero). A per-Primary encoder (E3) is worse than E1 on every row set.
decision_rule: >-
  CRPS at days 1 to 10 in points of effective capacity, the 1.15-widened quantiles as in
  experiment 419's scorer (its aggregation of seeds is copied), read per group (signed-MW six;
  MVA-only eight; D15 and D18 report-only), paired by fold and seed index, with the month-block
  bootstrap of `studies.eval_protocol`, 20,000 resamples, at a 95% level Bonferroni-corrected over
  the four tested contrasts. Tested contrasts. C1 E1 minus R, all rows. C2 E2 minus R, all rows.
  C3 E1 minus E1n, bank-holiday-week rows. C4 E2 minus E0, all rows (the memorisation control).
  Not tested, reported with intervals: E1 minus R on the Christmas set and on the Easter set; E3
  minus E1 (run only if C3 is at most -0.05 or C1 is below zero). Seed noise is large (experiment
  419: seeds 0 to 2 against 3 to 5 moved the MVA-only eight by -0.047), so E1, E1n, and E2 are
  fitted with seeds 0 to 5, E0 with seeds 0 to 2, and R reuses the stored seeds 0 to 2 (experiment
  407 `ctrl`) and 3 to 5 (experiment 419 `ctrl_s3`). The scorer prints, before any contrast is
  read, the minimum detectable effect on all rows and on the bank-holiday-week rows: twice the
  standard deviation over the folds of the difference in the signed-MW six CRPS between `ctrl` and
  `ctrl_s3`, divided by the square root of two for six-seed means. Experiment 419 found 0.0738 for
  three-seed means; the divided figure (about 0.052) is a lower bound, because it ignores the
  interaction of folds with seed sets, and it rests on five fold differences. A contrast smaller
  than the minimum detectable effect is reported as inside the noise. P1 holds only if the point
  estimate is at most -0.10 and the interval's upper end is below zero. The hypothesis is
  supported if P1 holds and C1 is not above the minimum detectable effect; partly supported if P1
  holds on its point estimate only; not supported otherwise. The bank-holiday-week rows are 14
  days of one Christmas and New Year week and a 2-day sliver of Easter week, in three test months,
  so the page says that a verdict on P1 is a one-Christmas result, and reports the sign of C3 in
  each of the three months. The encoder adds a module and a calendar table to the pipeline, so the
  smallest gain worth porting it is 0.03 points at the signed-MW six over all rows or 0.10 on the
  bank-holiday-week rows; the first cannot be shown with the minimum detectable effect above it, and
  the page says so. No adoption is sought. The five screen folds are interim.
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

All arms keep the weather, solar, lag, and `covid` inputs, the 8-wide Primary embedding, the
standardiser (fitted on the training rows; the encoder's integer indices bypass it), the trunk
width and depth, the optimiser, the held-out-weeks early stopping grouped by origin day
(experiment 397's `held_out_weeks`), and the row drops of experiments 407 and 419
(`e00419_fit.build_train`). Only the 18 calendar columns change.

- **R (reference).** The 18 hand-made calendar columns; stored forecasts of experiments 407 and
  419, not refitted on the five screen folds. R0 is a check: R refitted through the new code path
  (seed 0, fold 2025-12, and on 2026-04), which must reproduce the stored R within seed noise.
- **E0 (memorisation control).** Local half-hour (8-wide embedding) and weekday (4-wide) only; then
  Linear 32, GELU, Linear 16. Seeds 0 to 2.
- **E1n.** E0 plus the two distance columns (`nearest_days`, `nearest_inside`) and a bank-holiday
  flag, with no holiday name.
- **E1.** E1n plus the holiday name (12 categories: none, New Year's Day, Good Friday, Easter
  Monday, early May, spring, summer, Christmas Day, Boxing Day, other bank holiday, substitute
  day, 24 to 31 December non-holiday; the code lists them from the `holidays` package) as a
  4-wide embedding, and day of month (4-wide) and day of year (366 buckets, 4-wide).
- **E2.** The dates only: E0 plus day of month and day of year, with no holiday information of any
  kind. E2 minus E0 isolates the date columns, and E1 minus E1n isolates the holiday name.
- **E3 (conditional).** One E1 per Primary.

Each encoder outputs 16 wide and trains end to end on the same loss. The hour index is the local
half-hour, so the clock changes land on the same index.

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
  folder; **stored forecasts reused** (R), which saves 30 fits.
- **Cost.** E1, E1n, and E2 times 5 folds times 6 seeds is 90 fits and E0 adds 15, each of 50 to
  200 seconds. The 2026-04 event fold adds 21 and R0 adds 6; E3 adds 30 if it runs. One GPU
  process under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`. The encoder indices
  are integers, precomputed once per refit.
- **Earlier experiments adopted.** Experiment 419: none of its input changes won, so the reference
  stays `ctrl`. Experiment 410: training at other origin hours is experiment 422's subject and is
  not adopted here. Experiments 406 and 403: the Primary embedding and pooling are kept.
- **Plan reviewed against the latest evidence** (`origin/main`, 2026-10-31): experiment 419 showed
  that seed noise moves the six by about 0.011 and the MVA-only eight by 0.047, so the plan uses
  six seeds and prints the minimum detectable effect; the planned experiments 418, 421, and 422
  change the training data or inputs of the same network. Their plans do not mention the calendar
  columns, except one calendar term in 422's. This experiment's code is merged with theirs after
  they land.
- **Changes after the Sonnet plan review.** Added E0 (a control for calendar memorisation) and E1n
  (so that E1 minus E1n isolates the holiday name); required the minimum detectable effect on the
  bank-holiday-week rows; stated that a verdict on P1 is a one-Christmas result; dropped the
  "no worse" claim in favour of "inside the noise"; added R0, a refit of R through the new code
  path, to prove the reference matches; kept the descriptive contrasts outside the correction.

## Results

Not yet run.
