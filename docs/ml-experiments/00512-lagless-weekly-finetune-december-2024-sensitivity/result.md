---
id: "00512"
slug: lagless-weekly-finetune-december-2024-sensitivity
title: >-
  Does a second December (2024, with only 7 months of ensemble history) show the same weekly-refit
  and recency-weight gains for the lag-free demand network?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: other
protocol: demand-v2
size: >-
  simple. Stored: forecasts and networks of the fold 2024-12 under this experiment's private data
  folder only; every earlier experiment is read and never written. Serving path: not touched (this
  repository holds none). Degradation rule: not touched. Designs: one (experiment 510's harness
  unchanged on another month). Callers: this folder's scripts import experiment 510's modules. One
  Sonnet review of the page.
hypothesis: >-
  The gains that experiment 510 found for the lag-free demand neural network in the months of
  2025-10 to 2026-06, refitting weekly instead of monthly and weighting recent rows with a 90-day
  half-life, also appear in December 2024, where the network has only 7 months of
  ensemble-forecast (ENS) history to fine-tune on.
prediction: >-
  None was written before the scores were read: this is a sensitivity run that repeats experiment
  510's harness on one more month, and the page reports whether each sign matches experiment
  510's.
decision_rule: >-
  Descriptive. One fold gives no Bonferroni-corrected claim; intervals are 95% origin-day block
  bootstraps without multiplicity correction. The page reports whether each sign matches
  experiment 510's and does not adopt anything.
benchmark: >-
  L-M (monthly refit) and L-W (weekly refit, 182.6-day weight) of the same fold; no K2-S1, A-pool,
  or BG2 exists for this month.
row_set_id: fold_2024_12_all_rows_with_actual
metric: >-
  CRPS in points of effective capacity, by group (the eight and the six), week of the month, and
  Christmas week.
verdict: partly supported
best_arm: >-
  L-Ww (weekly refit, 90-day half-life): CRPS 5.659 points at the eight MVA-only Primaries and
  4.277 at the six signed-MW Primaries over December 2024, against 5.772 and 4.695 for L-M.
headline: >-
  In December 2024 the weekly refit scored 0.110 points better than the monthly refit at the eight
  and 0.336 better at the six; in Christmas week the gain was 0.018 at the eight (interval
  includes zero) and 0.704 at the six. The 90-day weight improved the score by 0.002 at the eight
  (interval includes zero) and 0.081 at the six.
lessons: >-
  The weekly gain has the same sign in both Decembers at the six, and it was larger in 2024 (0.70
  points in Christmas week, against 0.28 in 2025). At the eight the Christmas-week gain was close
  to zero in 2024 and 0.31 in 2025, so the weekly gain is not a stable property of Christmas week.
  The 90-day weight helped at the six in both Christmas weeks, and at the eight only in 2025.
take_home_message: >-
  A second Christmas confirms the direction of experiment 510's weekly-refit gain at the six
  signed-MW Primaries (0.70 points in 2024, 0.28 in 2025) but not at the eight MVA-only Primaries
  (0.02 and 0.31), and the 2024 result came with only 7 months of ensemble history. In the first
  week of the month the weekly and monthly fits are the same, so the weekly gain is zero there.
  Two Christmases are still two events, so this is more support, not a test.
advice:
  - >-
    Consider the weekly refit for the lag-free network in the first-release comparison, since its
    sign held at the six in both Christmases (Results table, figure 1).
  - >-
    Do not read the 90-day weight as settled: it helped at the six in both Christmases but not at
    the eight in 2024 (Results table).
production_status: candidate
adopted_in: none
---

**A second December (2024) shows the same direction as experiment 510 for the weekly refit at the
six signed-MW Primaries, but not at the eight MVA-only Primaries.** In Christmas week 2024 the
weekly refit scored 0.704 points of continuous ranked probability score (CRPS) better than the
monthly refit at the six (0.28 better in 2025) and 0.018 better at the eight (0.31 better in
2025). The 90-day recency weight scored 0.121 points better than the 182.6-day weight at the six
and 0.005 worse at the eight (interval includes zero).

Experiment 510 refitted the lag-free demand neural network (it reads no lagged power) every month
(L-M) or every week (L-W), with a recency weight whose half-life is 182.6 days, or every week with a
90-day half-life (L-Ww). Its months had at least 12 months of archived ECMWF ensemble (ENS)
forecasts to fine-tune on. This page runs the same harness, unchanged, on December 2024, where only
7 months of ENS forecasts exist before the first refit (the stored data begins on 2024-05-01).
Pretraining on ERA5 and CAMS estimates of past weather was repeated for this fold with two seeds.
There is no K2-S1 or A-pool for this month, so the page compares each arm only with the other arms.
Scores are in points of effective capacity (the rated capacity of a Primary; one point is 1%) of
each Primary's capacity), over all rows
that have an actual reading, with no filter for labelled events. "The eight" are the eight Primaries
that report only apparent power in megavolt-amperes; "the six" are the six that report signed
megawatts.

Verdict: partly supported. The weekly refit beat the monthly refit over the month at both groups,
with the six gain larger than the minimum detectable effect of 0.123 that experiment 510 found for
five folds; the 90-day weight helped only at the six.

## Figures

![Weekly-over-monthly and 90-day-over-182.6-day CRPS differences in December 2024, by slice, with
intervals](figure_1_weekly_and_recency_december_2024.svg)

## Results

- **Setup.** Fold 2024-12, five weekly refits (days 1, 8, 15, 22, and 29), seeds 0 and 1, widening
  1.15, 3-component lag-free network with experiment 448's pretraining recipe. Leak check: 0 rows
  past any cut. About 20 fine-tunes. The rows are 119,040 half-hours at the eight and 89,280 at the
  six.
- **CRPS over the whole month (points).** Eight: L-M 5.772, L-W 5.662, L-Ww 5.659. Six: L-M 4.695,
  L-W 4.358, L-Ww 4.277.
- **Contrasts (points; 95% origin-day block bootstrap, no multiplicity correction).**

| Contrast | Eight | Six |
|---|---|---|
| L-W minus L-M, whole month | -0.110 [-0.215, -0.026] | -0.336 [-0.465, -0.209] |
| L-Ww minus L-W, whole month | -0.002 [-0.036, +0.033] | -0.081 [-0.126, -0.040] |
| L-W minus L-M, Christmas week | -0.018 [-0.085, +0.055] | -0.704 [-0.810, -0.587] |
| L-Ww minus L-W, Christmas week | +0.005 [-0.007, +0.016] | -0.121 [-0.165, -0.092] |

- **Two Christmas weeks.** Christmas week is valid time 22 to 31 December. Experiment 510's
  December 2025 values, on all rows, were: L-W minus L-M -0.31 at the eight and -0.28 at the six;
  L-Ww minus L-W -0.47 and -0.24. Experiment 510's brief put the noise floor of a single Christmas
  week at about
  0.26 (six) to 0.27 (eight) points. The six-group gains of both Christmases lie above that floor,
  and the eight-group gain of 2024 lies below it.
- **Week of the month (origin days).** L-W minus L-M is exactly 0 in days 1 to 7, because the first
  weekly fit is the monthly fit. At the six it is +0.004, -0.860, -0.391, and -0.565 in days 8 to
  14, 15 to 21, 22 to 28, and 29 to 31; at the eight -0.043, -0.119, +0.049 (interval includes
  zero), and -0.879. L-Ww minus L-W is +0.143 (eight) and +0.078 (six) in days 1 to 7: with 7 months
  of history, the 90-day weight scored worse in the first week, and better in days 8 to 28 at the
  six (not in days 29 to 31).
- **Not scored here.** Tail coverage, avoidable-curtailment cost, and event rows. No labelled-event
  filter was applied, so the numbers include any event in the month.

## Checks

- **Leak check.** The harness printed 0 rows past any cut for every refit and variant before any
  fit.
- **Week-1 identity.** L-M and L-W agree exactly for origins in days 1 to 7, as in experiment 510.
- **Review.** One Sonnet review of the page against the score table.

## Caveats

- One fold is one December. The intervals are origin-day block bootstraps, ignore dependence across
  days, and are not corrected for multiplicity.
- The ENS history is 7 months, against at least 12 for experiment 510's folds. The gains may be
  larger or smaller for that reason, and this page cannot separate that from the month.
- The rows include labelled events and use no event filter, unlike experiment 510's headline rows.
  The December 2025 comparison values are from experiment 510's Christmas slices (rowset main16, all
  classes of row).
- This adds a second Christmas for the weekly-versus-monthly and recency contrasts. It adds nothing
  for the comparison of the lag-free candidate with K2-S1, because K2-S1 does not exist for this
  month.
- NGED's event labels are incomplete, so rows treated as clean may hold unlabelled events.

## What to try next

- Refit K2-S1 weekly as well, to test whether the weekly gain is specific to lag-free networks.
- Give December 2024 a longer fine-tuning history from the ENS backfill (2021-03 to 2024-03), to
  separate the effect of 7 months of history from the effect of the month.
