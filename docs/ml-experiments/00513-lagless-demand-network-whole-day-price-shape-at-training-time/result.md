---
id: "00513"
slug: lagless-demand-network-whole-day-price-shape-at-training-time
title: >-
  Can the lag-free demand network use the whole-day electricity price shape at training time to
  forecast better, when no price exists at serving time?
theme: demand-inputs
also_relevant_to:
  - generator-forecasting
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: pretrained and fine-tuned lag-free networks and their raw quantile forecasts, under
  this experiment's private data folder only; every earlier experiment is read and never written.
  Serving path: not touched (this repository holds none). Degradation rule: not touched. Designs: more
  than one defensible (how the day's shape is defined, what a missing price day does, how the
  placebo day is drawn, which control isolates a battery effect from a national-state effect), so the
  plan fixes each before any score is read. Callers: this folder's scripts import experiments 397,
  422, 428, 448, 510, and the `studies` evaluation code. One Sonnet plan review, one Sonnet code review
  before the first fit that produces quoted numbers, one Sonnet results review, and two Sonnet write-up
  reviews.
hypothesis: >-
  The Market Index price of a whole local day, known exactly at training time, carries information
  about price-following embedded batteries and gas engines behind a Primary, so a lag-free demand
  network that sees it in training forecasts those Primaries better. At serving time no price
  exists, so this experiment first asks the upper-bound question: does the true shape, given at
  both training and scoring time, help at all?
prediction: >-
  Written before any score is read. P1: the oracle (PO) beats the no-shape control (P0) by 0.00 to
  0.06 points of CRPS at both groups, below or near the minimum detectable effect, because price
  terms are 3 to 5% of the error at the one Primary where they were measured. P2: the shuffled-day
  placebo (PP) recovers at least half of the oracle's gain, because a typical-shape day carries
  day-type information. P3: the national demand-and-wind control (PN) gains at least as much as the
  oracle. P4: at the Primaries with a known battery or gas engine (D2, D5, D8, D9, D11, D14) the
  oracle's gain is no larger than at the other Primaries. A single-fold, single-seed timing run
  (2025-10, seed 0, pinball loss only, no contrast and no interval) already scored the oracle,
  placebo, and national control all better than P0, which fits P2 and P3.
decision_rule: >-
  See "Pre-registered rules". Continue to the training-time-only arms only if either group's
  Bonferroni interval for PO minus P0 excludes zero and the point estimate is at most minus the
  minimum detectable effect of a 3-seed mean. A gain that PP or PN reproduces is a day-type or
  national-state effect, not a battery effect. A gain of any size is worth porting only if a
  training-time-only arm recovers it, because no price exists at serving time.
benchmark: >-
  P0 (no shape columns, same loop, same seeds) for every contrast; PP and PN for what the oracle
  holds beyond a day type and the national state. K2-S1 and A-pool are not scored: this experiment
  asks about a difference between lag-free arms.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q99 pinball loss, and CRPS by Primary, by half-hour of
  day, and by horizon band.
---

## Plan (status: planned; no score has been read)

**This plan was written before any fit that produced a quoted number.** The take-home message, the
verdict, and the results are added when the experiment finishes. The plan covers Part 1 of the brief
(the price shape) only; whether the 0/1 lockdown flag is still needed is a separate experiment.

## Question

Batteries charge when the electricity price is low and discharge when it is high, and a gas engine
runs when the price is high. We know the price of the whole delivery day exactly when we train, and
we know none of it when we forecast days 2 to 10. We ask whether a lag-free demand neural network
that is shown the day's price shape during training forecasts better than one that is not. The first
test is an upper bound, the **oracle**, which sees the true shape at training and at scoring. It is
never deployable. If the oracle shows no gain that the placebo and national controls do not already
explain, no training-time-only variant can recover a gain, and the experiment stops there.

The idea comes from the maintainer. The lag-free network is that of experiments
[448](../00448-s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool/result.md) and
[510](../00510-lagless-demand-network-weekly-finetune-and-recency-weights/result.md), pretrained on
ERA5 and CAMS rows and fine-tuned monthly on ensemble forecast rows. No earlier experiment tested a
forward whole-day price shape for demand: experiment 25's shape columns look back 24 hours only.

## Arms

All arms use experiment 510's `L-M` recipe (monthly refit, 182.6-day half-life weight, 3 mixture
components, hidden width 64, 2 layers, dropout 0.1) with the same rows, loop, folds, and seeds.
Every arm, the control included, is pretrained by this experiment's own script, so the arms differ
in their input columns and nothing else. The shape columns widen the input from 37 to 48, so the
pretrained weights of experiment 448 do not load, and no new column is zero-initialised.

| Arm | Extra input columns (11) | What it tests |
|---|---|---|
| P0 | none | The control. |
| PO | The ten shape features and a missing flag, from the true Market Index price of the target half-hour's local day, at training and scoring. | Upper bound: does the true day shape help at all? |
| PP | The same columns of a seeded other day of the same calendar month, day type, and length (the shuffled-day placebo). | What a typical shape for that kind of day gives. |
| PN | The same ten features of national demand minus national wind, of the target local day. | What realised national state gives: the realised price encodes realised national demand, wind, and weather error. |
| P-drop, P-aux, P-teach | Conditional on the stop rule; defined below. | Whether the oracle's gain survives with no price at serving time. |

**The oracle is an upper bound on the information in the realised national state, not a battery
result.** A passing oracle that PN matches means the realised national state helps, not that
batteries do.

## The ten shape features

A **delivery day** is a local settlement day (Europe/London midnight to midnight, so it starts at
23:00 UTC in summer) and has 46, 48, or 50 half-hours. Every feature uses only the target day's own
values: there is no trailing, centred, or multi-day mean. Six features vary per half-hour and four
are constant within the day.

- **Per half-hour:** `rank` (position within the day, scaled 0 to 1), `minus_mean` (value minus the
  day's mean), `hours_from_min` (signed hours since the day's minimum, divided by 12),
  `hours_to_max` (signed hours until the day's maximum, divided by 12), `cheapest_4h` and
  `dearest_4h` (1 on the 8 lowest and the 8 highest half-hours of the day).
- **Per day:** `day_min`, `day_max`, `day_spread` (maximum minus minimum), and
  `evening_minus_overnight` (mean of the 16:00 to 19:00 local half-hours minus the mean of the 00:00
  to 05:00 local half-hours).
- **Dropped from the brief's list:** the slope-sign-change count (noise on half-hourly data) and the
  day's mean relative to a 7-day mean (it needs a multi-day mean).
- **Scaling:** the level features (`minus_mean`, `day_min`, `day_max`, `day_spread`,
  `evening_minus_overnight`) are z-scored with the mean and standard deviation of the non-missing
  half-hours before the fold's refit, so a test month never sets its own scale.

**Price source.** `market_index.parquet` is the Elexon Market Index (provider `APXMIDP`), not
verifiably the day-ahead price; N2EX day-ahead exists only from 2025-09. Rows are deduplicated on
`(time, dataProvider)`, and rows with zero volume count as missing.

**Missing-price days.** A day with any missing half-hour is a missing day: its ten shape columns are
0 and its missing flag is 1, in training and at scoring. The grid holds 2,527 local days; the
checks print how many are missing (101 for price, 50 for the national series). The scored rows
exclude the
half-hours of missing days, for every arm, so the contrasts stay paired on one row set; the
number of excluded half-hours is reported.

**Shuffled-day placebo.** Each non-missing day takes the shape of another non-missing day with the
same calendar month (any year), the same day type (Monday to Friday, Saturday, Sunday), and the same
length in half-hours. Days of one group are shuffled once with a seeded generator (`seed = 513`) and
each takes the next day in that order, so a donor is never the day itself and is used once. The same
draw serves training and scoring. A group of one day, and every missing day, takes the missing flag.
The checks print the number of unmatched days (1).

**National control.** The same ten features of national demand outturn (`INDO`) minus `FUELHH` wind
generation, per half-hour, so they describe the day's residual demand shape.

## Folds, seeds, and cost

- **Folds and seeds.** The five folds of experiments 448, 449, and 510 (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06), seeds 0, 1, and 2. July 2026 is unused.
- **Fit count.** 4 arms x 5 folds x 3 seeds = 60 pretrainings and 60 fine-tunings. The conditional
  arms add about 55 pretrainings and 55 fine-tunings (one at a time, if the stop rule passes).
- **Timed cost.** One fold, seed 0, all four arms: feeder 13 s (rows and shape tables), pretraining
  17 to 24 s per arm, fine-tune 3 to 6 s, so about 5 minutes per fold and 0.4 GPU-hours for the 60
  pretrainings and 60 fine-tunings. One GPU process under 8 GB, with `MemoryMax=8G` and
  `OMP_NUM_THREADS=1`.

## Scoring and contrasts

- **Rows and scorer.** Experiment 510's pipeline on experiment 470's row set
  (`experiment_470_intersection_v1`), rows with no labelled event, leads up to 10 days, forecast
  quantiles raw (no widening). "The eight" are the eight Primaries that report only apparent power
  (D15 and D18 report-only); "the six" are the six that report signed megawatts.
- **Contrasts (negative favours the first arm).** Gating, against P0: PO, PP, PN, and, if run,
  P-drop, P-aux, P-teach (6 contrasts at 2 groups, so 12 CRPS tests: Bonferroni count 12; the family
  of the whole price-and-lockdown brief is 18 with Part 2's 6). Descriptive, at the same interval
  width: PO minus PP and PO minus PN.
- **Minimum detectable effect.** From the seed spread of P0 (three seeds, five folds), printed by
  `e00513_score.py baseline` before any contrast. It is scaled to a 3-seed mean. Experiments 448 to
  510 report 0.057 to 0.164 points, so a pooled pass of 0.05 or less is unlikely to be resolved: the
  price terms are 3 to 5% of the error at the one Primary where they were measured.
- **Intervals.** Month-block bootstrap with five blocks gives near min-to-max intervals; the page
  says so and gives the sign per fold beside every interval.
- **Slices.** Per Primary and by half-hour of day, with the Primaries that have a known battery or
  gas engine (D2, D5, D8, D9, D11, D14) against the other Primaries; evening-peak and overnight
  half-hours against the rest; horizon bands day 1, days 2 to 3, and days 4 to 10 beside days 1 to
  10; the half-hours of missing price days are excluded and counted. A pooled gain at Primaries
  with no known asset is a
  leak check, not a finding. The chart of the oracle gain by Primary and by half-hour of day uses
  the OCF palette and no mark in the background colour.

## Pre-registered rules

- **Stop rule.** Continue to P-drop and P-aux only if, at either group, the Bonferroni interval of
  PO minus P0 excludes zero and its point estimate is at most minus the 3-seed minimum detectable
  effect. Otherwise stop and report.
- **Reading a passing oracle.** If PN or PP reproduces at least 0.75 of the oracle's gain at a
  group, call that gain a national-state or day-type effect. A battery reading needs the oracle to
  beat both, at the Primaries with a known asset.
- **P-drop (conditional).** The shape columns plus a missing flag as inputs. Training hides the shape
  (flag on, columns zero) on a random 50% to 100% of the days, the fraction drawn per batch. Scoring
  always sends the flag as missing, so the serving input is constant.
- **P-aux (conditional).** No price input. An extra 10-output head predicts the day's standardised
  shape features from the network's own inputs, with a loss weight tuned on folds 2025-10 and
  2026-01 only and scored on the other three folds, or a fixed grid of weights reported in full and
  counted in the Bonferroni family.
- **P-teach (demoted).** Runs only if the oracle passes and P-drop and P-aux both fail to recover
  its gain, and only after asking the manager, because it needs the Discuss session's approval. A
  teacher per fold and seed, equal hidden widths (64) for teacher and student.
- **Excluded (maintainer ruling).** No price and no predicted price shape at serving time.

## Standing practices followed

Cleaned data and the D15 and D18 start cut (through experiment 428's rows); rows with no labelled
event; July 2026 unused; no XGBoost work; metered generators are not shown (this experiment scores
substations); titles say "load-shift", never "switching"; every interim number is unaudited.

## Implementation details (deleted when this ships)

Code in `ml-experiments/00513-.../`: `e00513_shape.py` (feature definitions and the placebo draw),
`e00513_data.py` (the price, placebo, and national tables on experiment 428's grid),
`e00513_checks.py` (feature and placebo checks, run before any fit), `e00513_fit.py` (pretrain,
fine-tune, forecast), `e00513_score.py` (experiment 510's scorer pointed here).
