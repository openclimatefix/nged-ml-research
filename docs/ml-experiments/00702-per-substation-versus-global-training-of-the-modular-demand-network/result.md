---
id: "00702"
slug: per-substation-versus-global-training-of-the-modular-demand-network
title: >-
  Does training the modular demand neural network on one Primary substation at a time beat the
  single neural network trained across all 16 Primaries?
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  Anything else (not simple). (1) What gets stored: about 780 small neural networks (16 Primaries
  times 4 folds times 3 seeds times 4 per-substation arms, plus 12 extra global fits), their
  forecasts, and the scores, all private. (2) More than one defensible design: yes (where the
  network is split, what replaces the pooled statistics, the batch size, the blend rule), so each
  choice is fixed in this plan. (3) Spans code whose callers cannot be named without searching:
  yes (experiments 397, 418, 448, 510, 511, 559, 560, 572, 574, 581, 591, 613, 667, and 677).
hypothesis: >-
  A K1 demand neural network trained on one Primary's rows alone forecasts that Primary better
  than the one K1 neural network trained on the pooled rows of all 16 Primaries. The claim is
  tested at three split points of the three-step training schedule (before step 1, before step 2,
  and before step 3), and for an equal-weight blend of the global and the fully separate forecasts.
prediction: >-
  Written before any arm is fitted. Every number is a difference in pooled CRPS at leads of 0 to
  240 hours (the half-hours from 0 to 239.5 hours after the origin), in points of capacity, and
  "within x" means that both bounds of the contrast's 99% interval lie inside plus or minus x. P1:
  the fully separate arm (split-at-1) scores worse than the global arm, with a point estimate of
  0.05 to 0.30, and its largest losses are at D15 and D18, whose history starts on 2025-03-01. P2:
  split-at-2 and split-at-3 are each within 0.05 of the global arm. P3: the blend scores better
  than the mean of two global seeds, with a point estimate of 0.02 to 0.08. P4:
  split-at-2-plus-encoders is within 0.03 of split-at-2. The hypothesis is supported only if P1 or
  P2 fails in the direction of a per-substation gain.
decision_rule: >-
  Five planned contrasts, each read through a 99% interval (Bonferroni across five) from resampling
  7-day blocks of origin days and seeds: C1 split-at-1 minus global, C2 split-at-2 minus global, C3
  split-at-3 minus global, C4 blend minus the seed-matched global pair, C5
  split-at-2-plus-encoders minus split-at-2. A contrast is a gain when three conditions hold: the
  upper bound of the interval is below zero; the contrast is negative in each of the four test
  months; and the upper bound stays below zero with D15 left out and with D18 left out. A contrast
  is a loss under the mirror-image conditions (lower bound above zero, positive in each month, and
  still above zero without D15 and without D18). The verdict for an arm is `supported` when the
  contrast is a gain and the upper bound is at or below -0.05 points of capacity, `small gain` when
  the contrast is a gain and the upper bound is above -0.05, `refuted` when the contrast is a loss,
  and `inconclusive` otherwise. The threshold is 0.05 points because a per-substation arm stores,
  refits, and monitors 16 neural networks where the global arm has one. The four months are a
  screen: every verdict is interim, and an arm that reaches `supported` or `small gain` may advance
  to the other eight months of the backtest year. Those eight months include 2026-08, 2026-09, and
  2026-10, which lie in the final-test window, so the advance step needs the maintainer's explicit
  approval before it starts.
benchmark: >-
  The stored global K1 forecasts of experiment 677 (step 5 with injected events, monthly refits,
  seeds 0 to 2), with the manual heuristic and climatology of the same experiment on the same rows.
row_set_id: backtest_four_weather_months_2025_11_2026_01_2026_04_2026_07_daily_12utc
metric: >-
  CRPS (the quantile score over 11 levels) in points of capacity at leads of 0 to 240 hours, pooled
  over the 16 Primaries, with the normalised mean absolute error of the median forecast beside it.
  Also q01, q05, q95, and q99 coverage and pinball loss, every Primary on its own row, the lead
  bands of the demand protocol, and the copied share of a held-out injected load shift.
code: ml-experiments/00702-per-substation-versus-global-training-of-the-modular-demand-network/
---
Planned. The write-up follows the run. The rest of this page is the plan.

## The question

**The Primaries are forecast today by one neural network that learns from all 16 Primaries at
once, and this experiment asks whether 16 separate neural networks would forecast better.** The
network is K1, the modular demand neural network of [experiment
667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), described
on the [final models page](../final-models.md) under "Training regime". K1 has a weather encoder, a
date-time encoder, one shared lag encoder, and a mixer. K1 has no neighbour columns and no gap
signal columns, so no row of K1 reads another Primary's power. Every arm below uses K1 unchanged
except for the difference the arm names.

**Earlier experiments point in both directions.** For XGBoost models, [experiment
24](../00024-pooling-groupings-demand-xgboost/result.md) found that one model per Primary beat one
model across all series by 0.29 points. For the neural network, [experiment
677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md) found that
pre-training on six years of ERA5 and CAMS weather lowered the error by 0.59 points, and a
per-substation network has one sixteenth of those pre-training rows.

## What the global arm does, confirmed from the code

**The maintainer's description of the training schedule matches the code, with one addition: step
3 also injects synthetic load shifts into the lag inputs.** The schedule was read from
`e00574_fit.py`, `e00591_arm.py`, `e00613_arm.py`, `e00667_k1.py`, and `e00677_k1.py`.

- **Step 1** pre-trains every weight on ERA5 and CAMS pseudo-runs from 2019-09-01 (7.1 to 8.1
  million rows per fold), with AdamW, learning rate 0.0005, batch 8,192, 20% of the rows per epoch,
  up to 60 epochs, and patience 8.
- **Step 2** discards the pre-training mixer, draws a fresh mixer with seed 12345, freezes the
  weather and date-time encoders, and trains on up to 730 days of ENS rows with uniform weights and
  a 3-day embargo. The first ENS rows that the folds keep are from the run of 2024-04-10, because
  the first 9 days of runs fall in a purge, so the 2025-11, 2026-01, and 2026-04 folds hold 570,
  631, and 721 days. The lag encoder, the lag-identity embedding, the Primary embedding, and the newest
  offset knot stay trainable at learning rate 0.0002, and the fresh mixer trains at 0.0005.
- **Step 3** fine-tunes from the step-2 weights with the same frozen encoders, a 90-day half-life,
  a 7-day embargo, half weight on the next 7 days, and the synthetic lag-only load shifts of
  [experiment 581](../00581-robust-multiweek-lag-encoder-x3/result.md). With monthly refits there is
  one fine-tune per fold, at the first day of the test month.
- **The Primary embedding** (8 wide) is read by the weather encoder, the date-time encoder, the lag
  encoder, and the mixer. **The lag-identity embedding** (8 wide, 2 rows) tells the shared lag
  encoder whether a row is the day lag or the week lag, and holds nothing about a substation.
  **The offset** is one row of monthly knots per Primary, added to the three component means.

**Four statistics of the global arm are pooled across the 16 Primaries, and a per-substation
network must not use them.** They are the input standardiser (`Standardiser.fit` in
`e00397_prep.py`), the lag mean and standard deviation (`lag_stats` in `e00572_lags.py`), the scale
of the 14 thermal-lag and solar-gain columns (`pretrain_b1_columns` in `e00574_fit.py`), and the
median ensemble spread per lead band that fills the spread columns in pre-training (`spread_table`
in `e00448_run.py`). The early-stopping rows are pooled as well. The target mean and standard
deviation are already per Primary.

## Control arm

**The global arm is the stored forecast of experiment 677, and nothing is retrained for it.** Step
5 of experiment 677 with injected events at monthly refits is K1 with all three steps. Experiment
667 itself stores K1 only as far as step 2, because its step 3 is the Candidate's.

- **Forecasts:** `00677-*/backtests/step5w_with_injected_sp1.parquet`, 11 quantiles per (Primary,
  12:00 UTC origin, valid half-hour), seeds 0, 1, and 2, origins 2025-11-01 to 2026-10-31.
- **Weights:** `00677-*/chains/5/<month>/work/pretrained_P2/` and `step2/P5/` (steps 1 and 2), and
  `00677-*/arms/k1_5w_sp1/work/models/P5/` (step 3), for the 12 folds and seeds 0 to 2.
- **Match check before any new fit.** The scoring script of this experiment rescores the stored
  file and must reproduce experiment 677's stored monthly scores to within 0.01 points. The first
  run then refits the global arm once (seed 0, the pilot fold) through this experiment's code path
  with the per-substation switch off. The refit must score within the largest pairwise difference
  between the stored seeds 0 to 2 on that fold. If either check fails, the global arm is refitted
  for the four test months with seeds 0 to 2 and the stored file is not used.

## Arms

**The arms are a ladder of split points, so each step down the ladder isolates one phase of
training.** The maintainer's five proposals are all kept. One arm is added (split-at-3), and one
reference is added so that the blend is compared at an equal number of fits.

| Arm | Maintainer's name | Step 1 (ERA5 and CAMS) | Step 2 (fresh mixer) | Step 3 (fine-tune) | New fits per fold |
|---|---|---|---|---|---|
| global | Global | pooled | pooled | pooled | none (stored) |
| split-at-3 | added | pooled (stored) | pooled (stored) | per Primary | 16 times 3 seeds, step 3 only |
| split-at-2 | Hybrid 2 | pooled (stored) | per Primary | per Primary | 16 times 3 seeds, steps 2 and 3 |
| split-at-1 | Per-substation | per Primary | per Primary | per Primary | 16 times 3 seeds, all steps |
| split-at-2-plus-encoders | Hybrid 3 | pooled (stored) | per Primary, with two new encoders | per Primary | 16 times 3 seeds, steps 2 and 3 |
| blend | Hybrid 1 | global and split-at-1 | | | none |
| global-pair | added reference | pooled | pooled | pooled | 3 global seeds (3, 4, 5) |

**Split-at-3 is added because it is the cheapest per-substation change and it completes the
ladder.** Split-at-3 takes the stored global step-2 network and runs only the recency-weighted
fine-tune on one Primary's rows. The three adjacent differences then each isolate one phase:
split-at-3 minus global is the per-substation fine-tune, split-at-2 minus split-at-3 is the
per-substation fresh mixer, and split-at-1 minus split-at-2 is pooled pre-training together with
the Primary embedding and the pooled statistics. A fourth cell, per-substation pre-training
followed by pooled fine-tuning, is left out, because 16 separately pre-trained encoder sets cannot
feed one shared mixer without a new architecture.

**What happens to the embedding, the lag-identity embedding, the offset, and the statistics in each
arm:**

| Arm | Primary embedding | Lag-identity embedding | Offset knots | Input statistics | Frozen parts |
|---|---|---|---|---|---|
| global | 16 rows, learnt | learnt | 16 rows; only the newest knot trains in steps 2 and 3 | pooled | weather and date-time encoders in steps 2 and 3 |
| split-at-3 | the Primary's stored row, still trainable | stored, trainable | the Primary's row; newest knot trains | the fold's pooled statistics (the stored encoders were trained on them) | as global |
| split-at-2 | as split-at-3 | as split-at-3 | as split-at-3 | as split-at-3 | as global |
| split-at-1 | set to zero and frozen, so every part reads zeros and the layer shapes stay those of K1 | learnt per network | one row, all knots learnt in step 1 | fitted on the Primary's own training rows only | as global, on the Primary's own encoders |
| split-at-2-plus-encoders | as split-at-2 | as split-at-2 | as split-at-2 | as split-at-2 | the pre-trained encoders; the two new encoders train |

- **Split-at-1 never reads another Primary's data in any phase.** The input standardiser, the lag
  statistics, the scale of the thermal-lag and solar-gain columns, the spread fill of pre-training,
  the early-stopping rows, and the gradient rows all come from the one Primary. The weather is that
  of the Primary's own H3 resolution-5 cell (a hexagonal grid cell). Two Primaries in one cell read
  the same weather, which is public weather and not substation data.
- **Split-at-1 refits the four statistics with the functions the global arm uses, on one Primary's
  rows.** `Standardiser.fit` runs on the Primary's training rows of the fold, `lag_stats` on its
  lag values, the scale of the thermal-lag and solar-gain columns on its pre-training rows, and
  `spread_table` on its ENS rows. Each set of statistics is stored per fold and Primary with a
  hash, and the prediction phase checks the hash.
- **Zeroing the embedding is how "no substation-ID embedding" is done.** With one Primary an
  embedding row would be a constant input, which the first-layer biases already supply. Setting the
  row to zero and freezing it removes the embedding exactly and keeps the stored code paths
  (experiment 677's `LadderMdn` removes input groups the same way).
- **Split-at-2 and split-at-3 keep the embedding row,** because the frozen pre-trained encoders
  were trained with that row as an input.
- **The new encoders of split-at-2-plus-encoders** are a second weather encoder (7 weather means
  and the 14 thermal-lag and solar-gain columns, to 64 hidden units, to 32 outputs) and a second
  date-time encoder (21 columns, to 64, to 32), with no embedding input. Their 64 outputs enter the
  mixer beside the 145 existing inputs, so the mixer has 209 inputs. The new encoders start from a
  random draw with the fit's seed, train in step 2 at 0.0005 and in step 3 at 0.0002, and never
  pre-train. The lag encoder keeps reading the frozen pre-trained encoders only. C5 therefore
  measures the two new encoders together with a mixer that is 64 inputs wider.
- **The synthetic load shifts of step 3 are the same in every arm.** `synthetic_clean` in
  `e00581_bank.py` draws the schedule from the seed and from the first and last grid index of the
  training span, for all 16 Primaries in order. The global arm takes that span from its pooled
  rows. Every per-substation arm instead computes the span from the fold's constants (the cut-off,
  the 730-day window, and the first kept ENS target, 2024-04-10 09:30 UTC), asserts that the span
  equals the global arm's, calls `synthetic_clean` with the 16-row power array, and keeps the one
  Primary's row. The schedule then depends on no other Primary's data, and a Primary's shifted
  lags are equal across arms. A check asserts both.

**The blend rule is fixed here: the mean of the two forecasts, quantile by quantile, sorted.** The
blend averages each of the 11 quantiles of the global forecast and the split-at-1 forecast with
equal weights and sorts the result within the row, which is the rule that already averages seeds.
The blend holds six fits (three global seeds and three split-at-1 seeds), so its planned contrast
is against six global seeds (0 to 5), never against three. The seed-level pairs for the interval
are the mean of global seed `s` and split-at-1 seed `s` against the mean of global seeds `s` and
`s + 3`, for `s` of 0, 1, and 2. The global seeds 3, 4, and 5 are new fits of the unchanged global
chain. They are fitted by a copy of experiment 677's `chain.sh` (step `5w`, span 1) in this
experiment's folder, which reads the fold's stored statistics from experiment 677 and writes only
to this experiment's data folder. The blend against the three-seed global arm is reported as
exploratory.

**Every per-substation phase uses a batch of 512 rows, one sixteenth of the global batch, and the
batch is fixed here.** A Primary has about one sixteenth of the pooled rows. With 20% of the rows
per epoch, the global arm takes about 173 to 198 gradient steps per epoch in step 1 and about 114
in steps 2 and 3. At the global batch of 8,192 a per-substation epoch would take about 11 to 12
steps in step 1 and about 7 in steps 2 and 3, so the comparison would measure the step count. A
batch of 512 gives a per-substation network about the same number of steps per epoch as the global
network. Learning rates, the epoch limit, patience, dropout, weight decay, and the 20% row fraction
are unchanged. The first run also fits split-at-1 and split-at-3 at a batch of 8,192, and those
fits are a diagnostic that changes no setting.

**Early stopping uses the Primary's own held-out weeks, and a phase with too few held rows trains
for a fixed number of epochs.** The held weeks are the same absolute weeks as the global arm's (one
week in 12). When a Primary has fewer than two held weeks with at least 1,000 rows each in a phase,
that phase trains for 60 epochs in step 1, 20 epochs in step 2, and 10 epochs in step 3, and keeps
the last epoch's weights. The three counts are constants of this plan, like the learning rates.
They are near the stopping epochs in the stored logs of the global arm (53 to 60, 17 to 25, and 9
to 19). A fixed count was chosen over a hold-out of the Primary's newest two weeks, because with a
90-day half-life the newest rows carry the most weight in step 3 and a short history has few of
them. No run-time quantity of another Primary enters the rule. D15 and D18 in the 2025-11 fold,
with eight months of history, are the likely cases.

## Test months

**The four test months are 2025-11, 2026-01, 2026-04, and 2026-07, selected by the rule below from
stored weather.** The selection was fixed before any arm of this experiment was fitted or scored.
`e00702_months.py` prints the table. The weather columns are ERA5 means over the grid cells of the
16 Primaries. Each anomaly is the month's mean minus the mean of the same calendar month in 2019-09
to 2025-10, in standard deviations of those earlier yearly means. Six earlier years set each
standard deviation, so an anomaly ranks the 12 months and is not a climatological statistic. The
last three columns are the CRPS of the stored control at leads of 0 to 240 hours, in points of
capacity: over the 16 Primaries, at D15 alone, and at D18 alone. The script also writes, for each
month, the largest fall in the control's pooled CRPS from leaving one Primary out, which rule 5
reads.

| Month | Temperature (°C) | Temperature anomaly | Shortwave (W m⁻²) | Shortwave anomaly | Wind (m s⁻¹) | Wind anomaly | Demand (share of capacity) | Control CRPS | D15 CRPS | D18 CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2025-11** | 8.74 | +0.64 | 34.9 | -0.46 | 6.22 | +2.46 | 0.58 | 3.32 | 8.8 | 2.1 |
| 2025-12 | 7.38 | +1.08 | 22.5 | +0.27 | 6.22 | +0.71 | 0.61 | 4.02 | 9.3 | 3.1 |
| **2026-01** | 4.18 | -0.52 | 27.1 | -2.01 | 5.88 | +0.51 | 0.65 | 3.98 | 8.0 | 2.1 |
| 2026-02 | 6.59 | +0.13 | 44.5 | -2.21 | 5.61 | -0.53 | 0.61 | 3.50 | 9.1 | 2.1 |
| 2026-03 | 7.97 | +1.13 | 114.7 | +0.12 | 5.22 | -0.04 | 0.55 | 5.13 | 30.5 | 2.6 |
| **2026-04** | 9.91 | +0.69 | 196.5 | +0.46 | 5.22 | +0.55 | 0.48 | 4.01 | 11.3 | 2.3 |
| 2026-05 | 13.46 | +0.91 | 229.3 | +0.74 | 4.42 | -0.08 | 0.43 | 3.97 | 15.8 | 2.3 |
| 2026-06 | 17.39 | +1.36 | 225.1 | -0.32 | 4.82 | +0.62 | 0.42 | 3.45 | 10.7 | 3.5 |
| **2026-07** | 19.01 | +1.65 | 245.6 | +3.54 | 4.71 | +0.72 | 0.44 | 4.95 | 15.4 | 15.4 |
| 2026-08 | 18.42 | +0.96 | 203.7 | +1.85 | 4.21 | -0.81 | 0.43 | 4.67 | 12.8 | 12.7 |
| 2026-09 | 16.57 | +1.42 | 125.3 | -0.39 | 4.93 | +0.48 | 0.46 | 4.06 | 12.9 | 8.9 |
| 2026-10 | 12.39 | +0.59 | 73.4 | +0.72 | 5.59 | +0.80 | 0.52 | 4.01 | 9.3 | 6.8 |

**The selection rule takes one month per season, each for a different weather scenario:**

1. **Winter (December to February), the cold scenario:** the month with the lowest temperature
   anomaly. January 2026 is the only month of the 12 that was colder than normal, and has the
   highest demand.
2. **Summer (June to August), the sunny scenario:** the month with the highest shortwave anomaly,
   which is July 2026.
3. **Autumn (September to November), the dull and windy scenario:** the month with the highest wind
   anomaly minus shortwave anomaly, which is November 2025 (+2.92, against +0.87 and +0.08).
4. **Spring (March to May), the mild shoulder scenario:** the month with the highest temperature
   anomaly. The first choice is March 2026 (+1.13), which rule 5 excludes, so April 2026 is
   selected. April was named as the spring fallback before the plan review. May 2026 has the
   next-highest anomaly (+0.91) and passes rule 5, but May carries a larger D15 excess than April:
   leaving D15 out lowers the control's pooled CRPS by 0.79 points in May and by 0.48 in April.
5. **A month is ineligible when leaving one Primary out lowers the control's pooled CRPS at 0 to
   240 hours by more than 1.0 point.** The rule excludes only March 2026, where leaving D15 out
   lowers the control's CRPS by 1.69 points, from 5.13 to 3.44. D15 has a level step from the week
   of 2026-03-09, of about -0.8 MW to +0.5 MW, with no event in NGED's log, and March would have
   measured how each arm follows that one step. The largest fall in any other month is 0.79 points
   (D15 in May). In July 2026 leaving D18 out lowers the control's CRPS by 0.68 points, about the
   same as leaving D15 out of July (0.67), which has no step at D15.
6. **No two selected months may be adjacent,** so that no forecast of one month, with its 14-day
   lead, is valid in another. November, January, April, and July satisfy the constraint.

**Rule 5 was added after the control's scores were seen.** The plan review found D15's step in
the control's March scores, and the threshold of 1.0 point was set with the 12 leave-one-out falls
in view. Rules 1 to 4 and rule 6 read only weather and the calendar. No score of a new arm existed
when any rule was written. The control's CRPS at 0 to 240 hours averages 4.06 points over the four
months and 4.09 over the 12.

**July 2026 is the maintainer's choice of months, and it carries two caveats.** The maintainer
asked for four months from 2025-11 to 2026-10, and July 2026 lies in the final-test window (2026-07
to 2026-10) that the `ml-experiment` skill reserves for a finalist. Experiments 620, 667, 677, and
701 have already scored that window. D18 also has a level step in the week of 2026-07-06, from
about 2.1 MW to about 3.4 MW, and its CRPS rises from 2.1 to 3.5 points in the earlier months to
15.4 in July. No summer month avoids both problems: July and August both carry D18's step, and
June, the only summer month before the step, is not sunny (shortwave anomaly -0.32). The decision
rule therefore requires every verdict to hold with D18 left out, and with D15 left out, and the
write-up states both caveats.

**The advance step needs the maintainer's explicit approval before it starts.** An arm that reaches
`supported` or `small gain` would next be scored on the other eight months, and three of those
months (2026-08, 2026-09, and 2026-10) are in the final-test window. This plan asks for July only.

**The K1 design was tuned on five development months that include January 2026.** The tuning used
the global configuration, which favours the global arm in that month. Each planned contrast is
therefore also reported with January left out.

## Measurement

- **Rows.** A row is a (Primary, 12:00 UTC origin, valid half-hour). A row is scored only if every
  arm and every seed has a forecast, the manual heuristic and climatology have a forecast, and the
  cleaned observed power exists. The origin days are the days of the four test months. The headline
  leads are the half-hours from 0 to 239.5 hours after the origin (`lead_h < 240`, the filter in
  `e00702_months.py`), which this page calls "0 to 240 hours". The leads of 240 hours and more
  (days 11 to 14) are reported beside them.
- **Forecast scored.** Each arm's forecast is the mean over its seeds, quantile by quantile, sorted
  within the row. No band widening and no clamp is applied to any arm.
- **Metrics.** CRPS as the quantile score over 11 levels and the absolute error of the median, both
  in points of capacity (percent of the Primary's effective capacity), from
  `studies.eval_protocol` where its functions accept the 11-quantile contract rows, and otherwise
  from `e00668_score_lib.py` as experiment 677 used. One function scores every arm.
- **Intervals.** Per-seed losses are paired on Primary, origin, and valid time with
  `experiment_store.losses_for_bootstrap`. The decision interval resamples 7-day blocks of origin
  days and one seed, as experiment 677 did, with 20,000 resamples so that each 0.5% tail holds 100
  draws. A second interval resamples whole months and seeds with
  `studies.bootstrap.bootstrap_difference_at_level`. Four months give that interval few distinct
  values, so the decision rule uses the plain statement the interval approximates: the contrast has
  the same sign in each of the four months. Forecasts from neighbouring 7-day blocks overlap in
  valid time by up to 10 days, so the blocks are not independent and the block interval is narrower
  than an interval over independent weather episodes would be. The four-month sign condition is
  the guard against that.
- **Departure from the plan in the interval code.** The scoring script pairs the per-seed losses
  with its own `align` function and draws both intervals with `block_seed_bootstrap` in
  `e00702_core.py`, not with `experiment_store.losses_for_bootstrap` and
  `studies.bootstrap.bootstrap_difference_at_level`. The library functions resample the groups
  named in a `month` column and pair two arms on the same seed. The decision interval needs 7-day
  blocks of origin days, and contrast C4 pairs each seed of the blend with two seeds of the global
  arm. `block_seed_bootstrap` resamples 7-day blocks for the decision interval and calendar months
  for the second interval, and draws one seed in each resample.
- **Per-Primary reporting.** Every Primary has its own row for every arm, with the difference from
  the global arm and its interval. The write-up also gives the count of Primaries each arm improves,
  the pooled contrast with each Primary left out in turn, and three groups: the six signed-MW
  Primaries, the eight MVA-only Primaries, and D15 with D18.
- **History groups.** `e00702_history.py` prints each Primary's first cleaned reading in the power
  array that the fits train on, and writes the table to `history_first_clean_reading.csv` in the
  experiment's data folder. The array is `e00422_prep.cleaned_arrays()`: `e00510_common.clean_stack`
  returns it, `pretrain_wt`, `step2`, and `fit_arm` of `e00574_fit.py` read it through
  `clean_stack`, and `e00667_fold.py` and `e00677_fold.py` patch `clean_stack` to hide the held-out
  month. Eleven Primaries start on 2019-09-17 and D14 starts on 2019-10-21.
  D16 starts on 2020-12-22 and D9 on 2023-08-23. D15 and D18 start on 2025-03-01, the cut of
  experiment 399. The fit pipeline applies no other start cut: the 19-month cut that the
  data-processing page lists for D4 is not in this array, and the later start dates on the final
  models page for D11, D10, D5, and D14 do not match it. C1 is reported for the 12 Primaries with
  six years of history or more, for D16 and D9, and for D15 with D18.
- **Other reports.** The lead bands of the demand protocol, coverage and pinball loss at q01, q05,
  q95, and q99, cold nights in January, and the median forecast minus the mean forecast per Primary.
- **Injected load shifts.** A network fine-tuned on one Primary's recent rows may copy more of a
  load shift. Each per-substation arm is therefore run on the held-out bank of injected shifts of
  experiment 677 (`chain.sh ... 5wbank`), and the copied share is reported beside the global arm's
  0.071. The four load-shift scenarios of the `ml-experiment` skill are reported from
  `e00677_slices.py`, with the caveat that NGED's event log is incomplete.
- **Figures.** A day-ahead forecast and one full 14-day forecast against observed power for two
  Primaries, the per-Primary contrast chart, and the ladder of split points with paired intervals.
  Six time series plots of load-shift events follow, three at signed-MW Primaries and three at
  MVA-only Primaries. Each plot marks the start and end of the event and shows the global arm and
  each per-substation arm from 14 days before to at least 7 days after. D18's step of July 2026 is
  one of the six. The plots carry the caveat that NGED's log may miss events, so the event-free
  rows may hold unlabelled events.

## Leakage and look-ahead checks

**Each check is an assertion in the fit scripts or a test in the first run, and the fit stops when
one fails.**

- **Origin-time weather.** The fits keep the assertions of `e00574_fit.py`: every training origin
  has an ensemble summary file issued 9 hours earlier, and every test origin satisfies
  `nwp_init_time <= power_fcst_init_time`. A new truncated-archive test rebuilds the inputs of 20
  sampled test origins with every ensemble summary file issued after the origin's own run removed,
  and asserts that the weather, thermal-lag, solar-gain, and expected-weather inputs are unchanged.
  Experiment 404 was voided by weather look-ahead, so this test runs before the first scored fit.
- **ERA5 and CAMS are pre-training inputs only.** No fine-tuning or test row reads ERA5. The
  pre-training rows end at least 3.5 hours before the refit time, and the script asserts it.
- **Held-out power.** In the training phase the cleaned power is NaN from the first day of the test
  month (`e00667_fold.py`), and the prediction phase checks the standardiser's hash.
- **Isolation of split-at-1.** A test replaces every finite value of the power, the weather inputs,
  and the CAMS inputs of the other 15 Primaries with random numbers and leaves every missing value
  missing, so the pooled span of the rows is unchanged. The test rebuilds the one Primary's tensors
  and statistics for all three steps and asserts that they are identical bit for bit. The test runs
  on the first-run fold and on the 2025-11 fold for D15 and D18, where the fixed-epoch rule is most
  likely to apply, and asserts there that the epoch count of each phase is unchanged.
- **Lag age.** The newest lag is at least 24 hours before the 09:00 UTC native origin
  (`check_lag_ages`).
- **Columns.** The report prints each arm's input columns and trainable parameter names, so a
  reviewer can check the two tables above.

## First run

**The first run uses the fold 2026-05, which is not a test month, and three Primaries.** The three
are chosen by rule: the full-history signed-MW Primary with the largest capacity, the full-history
MVA-only Primary with the largest capacity, and D15. Every number from the first run is labelled
interim and enters no contrast. The first run checks:

1. the two match checks of the control arm;
2. the truncated-archive test and the isolation test;
3. that the synthetic shifts of a Primary are equal across arms;
4. a diagnostic of the batch size: split-at-1 and split-at-3 are fitted with seed 0 at batches of
   512 and 8,192, and the validation loss and the 2026-05 scores of both are reported, with no
   setting changed by the result;
5. that each arm's forecasts for the three Primaries beat climatology on 2026-05, as a guard
   against a broken fit;
6. the wall-clock time, the CPU time of refitting split-at-1's statistics, the peak host memory,
   and the GPU memory of each phase, which size the full run;
7. which Primaries and phases use the fixed epoch counts.

## Compute

**The full run needs under 10 GPU-hours on the workstation's RTX A6000, and no download or
spend.** The estimate rests on the logs of experiments 667 and 677: the pooled pre-training takes 67
to 83 seconds per seed and fold, step 2 takes 11 to 20 seconds, and step 3 takes 7 to 13 seconds. At
a batch of 512 a per-substation phase takes about as many gradient steps as the pooled phase, so
the time per network is about the pooled time or less.

| Arm | Fits (4 folds) | Estimated GPU time, one after another |
|---|---:|---:|
| split-at-1 | 192 networks, 3 steps each | 4 to 5 hours |
| split-at-2 | 192 networks, 2 steps each | 1.5 hours |
| split-at-2-plus-encoders | 192 networks, 2 steps each | 2 hours |
| split-at-3 | 192 networks, 1 step each | 0.6 hours |
| global seeds 3 to 5 | 12 networks, 3 steps each | 0.3 hours |
| held-out bank, 4 arms | inference only | 0.5 hours |
| batch diagnostic (first run) | 6 networks at each batch | 0.1 hours |

- **CPU time of split-at-1's statistics.** Refitting the standardiser, the lag statistics, and the
  two scales takes an estimated 1 to 2 minutes on one core per Primary and fold, so about 1 to 2
  CPU-hours for 64 Primary-folds. The first run measures the figure.
- **Parallel work.** One process per fold builds the fold's rows once and loops over the 16
  Primaries and 3 seeds. Each process needs up to 14 GB of host memory (the cap that experiment 677
  needed) and under 3 GB of GPU memory. Three processes at a time fit in 3 CPU cores, 42 GB of host
  memory, and 9 GB of the GPU. Four folds therefore run in two waves, three folds and then one, and
  the run takes about 5 hours of wall-clock time. The session asks the manager for the slot before
  starting.
- **Storage.** Every fitted network is saved as a state file in the private data folder, about 780
  files of under 1 MB each, with one forecast file per arm, fold, and seed.

## Code to reuse

- **`e00677_k1.py`, `e00677_fold.py`, `e00677_export.py`, `e00677_score.py`, `e00677_slices.py`,
  `e00677_tails.py`, `chain.sh`, and `gpu.sh` of experiment 677:** the K1 chain for one fold, the
  monthly fold with the held-out power hidden, the export to 12:00 UTC contract rows, and the
  scoring. The new fit script is a copy of `e00677_k1.py` with a `--primary` argument.
- **`e00574_fit.py`:** `pretrain_wt`, `step2`, `fit_arm`, `train`, and the weather assertions, run
  unchanged with module state patched, as experiments 591, 613, 667, and 677 do.
- **`e00591_arm.py` and `e00591_model.py`:** arm A1, the shared lag encoder with the lag-identity
  embedding and the synthetic shifts.
- **`e00559_model.py`, `e00560_model.py`, `e00572_model.py`, and `e00574_model.py`:** the encoders,
  the mixer, and the offset. The second encoder pair of split-at-2-plus-encoders is a new subclass
  in this experiment's folder.
- **`e00677_model.py`:** the pattern for removing an input group by multiplying it by zero.
- **`e00397_prep.py`, `e00448_run.py`, and `e00572_lags.py`:** the row builders that hold the four
  pooled statistics. Split-at-1 patches `primaries()` and the statistics from this experiment's
  folder, and edits no earlier experiment's script.

## Standing practices and earlier advice

- **Followed:** monthly refits; cleaned data with the `substation_zero` half-hours dropped; the cut
  of D15 and D18 before 2025-03-01; every fitted network saved; stored predictions reused as the
  control; a training window of up to 24 months (730 days), shorter in the three folds before
  2026-05 because the ENS rows start on 2024-04-10; a screen on four months, one per season, with
  every number interim; the tails and the example forecasts; a climatology row; the normalised MAE
  of the median in every table.
- **Followed from earlier advice:** check the sign of a contrast in every month before reading its
  pooled interval; months and not seeds set the interval width; judge a lag-reading network on
  the copied share of an injected shift and on event-free error together (experiment 677); report
  a pooled contrast with each series left out (the audit triggers of the `ml-experiment` skill).
- **Departed from:** July 2026 is scored although it lies in the final-test window, because the
  maintainer chose the range of months. XGBoost is not fitted. The avoidable-curtailment cost is
  not reported, because the four months are a screen.

## Open risks

- **Four months are four weather episodes.** A gain must hold in all four months, and a real gain
  confined to one season would read as inconclusive.
- **The batch of 512 changes the gradient noise** as well as the step count. The first run's
  diagnostic at 8,192 covers one fold, one seed, and three Primaries.
- **A fixed epoch count has no validation guard.** Where a Primary has fewer than two held weeks
  of 1,000 rows or more, the phase runs for 60, 20, or 10 epochs with no early stopping. D15 in the
  fold 2025-11 has about one ninth of the pre-training rows of a full-history Primary, so 60 epochs
  give D15 about one ninth of the gradient steps. Step 1 can therefore under-fit D15 and D18, and
  no held-out loss reports that. The first run lists which Primaries and phases use the fixed
  counts.
- **D15 and D18 each carry a level step,** D15 in March 2026 (not scored) and D18 in July 2026
  (scored). The leave-one-out conditions guard the verdicts, and the pooled point estimates still
  include D18's July.
- **The global arm's hyperparameters were tuned for pooled data.** No per-substation arm is tuned
  beyond the batch size, which favours the global arm.
- **Split-at-1 minus split-at-2 bundles three differences:** pooled pre-training, the embedding, and
  the pooled statistics. The experiment does not separate them.
- **The stored control was fitted by experiment 677's code,** and the new arms by a patched copy.
  The match checks bound the difference, and GPU fits are not bit-reproducible.
