---
id: "00687"
slug: causal-detector-features-for-the-prevailing-forecast-heads
title: >-
  Do causal step statistics, the neighbour mirror, and a return-to-level statistic let the small
  prevailing heads react sooner to a load shift than extra lagged power inputs do?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. (1) It stores per-origin feature tables, per-fit head weights (a few thousand parameters
  each), and one private Parquet file of predictions per arm; (2) more than one design would
  defensibly test the hypothesis (features only, features plus an explicit flag, extra lags), so
  the arms are fixed here and run side by side; (3) it reuses code from experiments 606, 610, 647,
  650, 652, and 660, whose callers are named below; (4) it touches no serving path or degradation
  rule; (5) it changes no contract.
hypothesis: >-
  Per-lag causal step statistics in noise units, the gap with the all-Primaries median removed, the
  opposite-sign neighbour gap, the routine-step rate, and a return-to-level statistic carry
  information about whether a lag lies inside a load shift that the gap summaries of experiment 606
  and 610 do not, and a small prevailing head that reads them reacts sooner to the start and the end
  of a shift than the same head reading extra lagged power inputs (the arm of experiment 681).
prediction: >-
  Written before any fit or score. P1 (separability, no refit): on the stored bank of experiments
  650 and 652, the area under the receiver operating characteristic curve (AUROC) of "this lag lies
  inside an injected shift" rises by 0.05 or more when D1 to D5 are added to experiment 606's
  summaries, on each of banks M, S, and X. P2 (reaction time): the features-only arm covers half of
  the new level at least 1 day sooner than the departure head of experiment 652 (R-haz) at the start
  of an injected shift and at its end, and at least 1 day sooner than the extra-lags arm. P3: in-force
  scaled CRPS improves by 0.30 points of capacity or more against R-haz, and near-end and
  just-ended scaled CRPS by 0.30 or more. P4: the false-alarm rate on event-free rows (the share of
  rows where the prevailing output departs from the NRA output by 3 points of capacity or more)
  rises by at most 1 percentage point against R-haz, and event-free CRPS by at most 0.02 points. P5:
  the placebo arm (features computed for an unrelated Primary) gains less than a third of the
  features-only arm's reaction-time gain.
decision_rule: >-
  Stop rule (from the spec, fixed before any score is read): if the AUROC gain is under 0.03 on every
  bank and no prevailing stratum (in force, near the end, just ended) gains 0.15 points of capacity of
  scaled CRPS over R-haz, the features are reported as not useful and the lag encoder is not
  refitted. Otherwise the verdict reads two contrasts. Primary contrast: features-only minus R-haz,
  reaction time at the start and end of injected shifts of the held-out family, with a 95% interval
  from a paired bootstrap over events; support needs the interval to exclude zero in the
  predicted direction with a gain of at least 1 day (origins are daily, so reaction time resolves to
  1 day, and a port adds a feature block to the head and nothing else). Secondary: the in-force and
  near-end scaled CRPS contrasts (month-block 95% interval over the 5 development months),
  descriptive and uncorrected for multiplicity. The veto: event-free CRPS and false-alarm rate
  within the limits of P4, and the placebo gain below a third of the features-only gain. Any failed
  condition is reported plainly with no rescue tuning.
benchmark: >-
  Experiment 652's R-haz head (arm B0, refitted here from the same code so that features and
  baseline share rows and seeds, and checked against 652's stored scores), experiment 681's
  extra-lags arm (arm L1, built here on the same banks because experiment 681 has run on the
  backtest folds only), K1 as context, and the placebo.
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  AUROC of lag-inside-shift; reaction time in days to cover half of the new level after an injected
  shift starts and ends; CRPS (twice the mean pinball loss over 11 quantiles) and normalised MAE of
  the median for both outputs by lead band; q05 to q95 coverage; q01, q05, q95, and q99 pinball
  loss; false-alarm rate on event-free rows.
---

**Plan only. Nothing here has been run.**

## Question and approach

**The question.** A load shift is a period when the network operator moves part of one primary
substation's demand onto or off a neighbouring one. The prevailing output of the two-output demand
network should follow such a shift, and today it takes days to do so (experiment 668 measured 181
and 52 hours at the verified D5 and D16 transfer). Experiment 681 tests one remedy, extra lagged
power inputs. This experiment tests another: give the small prevailing head the statistics that a
simple step detector computes, and let the network learn how to use them.

**Approach.** The work has two steps. Step 1 runs on the CPU with no refit and tests whether the
features separate "this lag lies inside an injected shift" from "it does not" better than the
gap summaries the lag encoder reads today. Step 2 refits the small heads of experiment 652 (a
two-layer network of 64 units, a few thousand parameters) with the features as extra inputs. Both
steps use the stored banks of experiments 650 and 652, whose injected shifts are known exactly.

## Features (all causal; D numbering follows the spec)

All power is a share of effective capacity. The gap at a half-hour is the cleaned power minus the
lag-free median `T` of the same half-hour (experiments 598 and 606). The noise unit of a Primary is
1.4826 times the median absolute deviation of its 6-hour block-mean gap, measured on the
uninjected cleaned power of the five development months.

- **D1, common-mode-removed gap.** The gap at the lag's time minus the median gap over all Primaries
  at that time, in noise units.
- **D2, opposite-sign neighbour gap.** The largest of minus the sign of the own gap times the
  neighbour's gap, over the electrical neighbours, in noise units, and the onset-time difference of
  that best neighbour.
- **D3, routine-step rate.** The share of the last 60 days of 6-hour blocks whose step exceeds 3
  noise units (the Primary's own history, ending before the lag).
- **D4, causal step statistics.** The mean gap over the last w hours minus the mean over the w hours
  before, for w of 6, 24, and 72, in noise units, ending at the lag's time and, for the prevailing
  head, at the newest reading.
- **D5, causal cumulative-sum detector** on D1: the time since the detector last crossed 4 noise
  units, and the suspected depth at that crossing. Used as a feature, not a hard flag.
- **D6, return to level.** (Current D1 minus D1 at the suspected onset) divided by the suspected
  depth, and the opposite-sign neighbour's reversal.

Lag-time features (D1 to D5 at the day lag and the week lag) read only half-hours at least 24
hours before the origin. The prevailing head's newest-reading features (D4 to D6) read half-hours
up to 4 hours before the origin, the same freshness as the `fresh` signal of experiment 598; a lab
arm uses 6 hours before the origin to measure what lower latency would be worth. A unit test
perturbs the power after the latest permitted half-hour and checks that no feature changes.

## Step 1: separability, no refit

- **Rows.** Every bank origin (Primary, 09:00 UTC origin) of the five development months, all passes.
  The label for a lag is whether the lag's own half-hour lies inside the Primary's injected shift.
- **Banks.** M is the stored bank (each shift mirrored on one or two neighbours, experiments 581 and
  650). X removes the mirror: the neighbour signals read the injected Primary's own shifted power
  and the neighbours' unshifted power. S approximates a 60/40 split by scaling the mirror to 60% of
  its stored strength, because the stored bank cannot be re-split; this is a post hoc approximation
  and the page says so. Experiment 651's own banks S and X are not reused, because they hold no
  per-origin gap tables for the development months.
- **Scorers.** Logistic regression on (a) experiment 606's summaries (the `gap`, `ew`, and `nb`
  column groups already in each bank's origin features), and (b) those plus D1 to D5. The folds
  hold out the scored month, as in experiment 652. AUROC with a month-block bootstrap interval.
  A third scorer adds the routine-step rate alone (D3) and a fourth shuffles the features across the
  Primary's origins (a placebo).
- **Stop rule.** As in the decision rule.

## Step 2: the refit

- **Heads.** B0 is experiment 652's R-haz exactly as fitted there (same code, folds, seeds 0 and 1,
  training on the other four months, duration mixture m50). F1 is B0 plus the feature block (D1 to
  D6 at the day lag, week lag, and newest reading; features only, with 50% signal dropout as in
  experiment 610). F2 is F1 plus the explicit flag (the D5 detector's binary state), a comparison
  arm. L1 is B0 plus experiment 681's L1 block (cleaned-power residuals at 6, 9, 12, 18, 24, 36, 48,
  and 72 hours before the origin, each at most 6 hours before the origin), built here on the 650
  banks because experiment 681 scores the backtest folds only. P is F1 with every feature taken from
  an unrelated Primary of the same metering group (a fixed map that skips the Primary and its
  electrical neighbours). F1-6h is F1 with the newest-reading features moved to 6 hours before the
  origin, a lab arm.
- **Fits.** 6 arms x 5 months x 2 seeds = 60 small fits, each about 10 seconds to a minute on one
  CPU thread in experiment 652 (40 fits took 6 minutes on 8 processes). CPU only, at most 4
  processes, no GPU. Fitted weights are saved to the experiment's data folder.
- **Rows scored.** The held-out family and the training family of experiment 650's banks, and the
  real event-free rows, exactly as experiment 652 scores them, so the stored R-haz scores can be
  reproduced as a gate (B0 minus 652's stored R-haz scaled CRPS within 0.01 points of capacity).
- **Reaction time.** For each injected event, the first origin day on or after the onset (or end)
  at which the head's median at leads of 24 to 48 hours differs from K1's median by at least half of
  the shift's depth at the valid time (for the end, by at most half of the in-force departure).
  Resolution is 1 day. Reported as a median over events with a bootstrap interval over events, and
  by event size (depth) and duration class.
- **Verified transfers.** D5 and D16 (2025-12-29) are scored descriptively on the real rows of
  experiment 647's files: one episode, no interval.
- **Scoring.** Everything experiment 652 reports: scaled CRPS by stratum (event-free, far, middle,
  near, just ended) and lead day, with equal width scaling, unscaled beside it, normalised MAE of the
  median, q05 to q95 coverage, tail pinball loss at q01, q05, q95, and q99, the avoidable-curtailment
  cost in three row sets (inside an event, the 7 days after, all rows), and the false-alarm rate on
  event-free rows.

## Standing practices followed

- Monthly or no refit cadence applies (small heads, one per scored month); no weekly refits.
- Cleaned data throughout, as in experiment 650's stored clean stack; D15 and D18 rows before
  2025-03-01 are cut; the dropped half-hours per month are reported.
- Stored predictions reused: experiment 652's R-haz scores as a gate, K1 rows from experiment 650.
  Fitted here: B0 (to share the code path and seeds with F1), F1, F2, L1, P, F1-6h.
- Weather: the lag-free table `T` is built from ERA5 means and CAMS shortwave at the valid time in
  experiment 598; the features read no weather themselves. The lags are past power, which exists at
  forecast time up to the freshness stated above.
- Data before 2026-07-01 only; an assertion on every index read.
- Reviews: one Sonnet review of this plan, one fresh Sonnet review of the code before the first fit, a
  fresh Sonnet re-derivation of the headline numbers, and one Sonnet prose review. No Opus.
- Open to a result that the features do not separate shifts better than the summaries: the stop
  rule then ends step 2.

## Figures planned

- Take-home first: reaction time at the start and at the end of the shift, by arm, with intervals.
- AUROC by bank, with and without D1 to D5.
- In-force, near-end, just-ended, and event-free scaled CRPS by arm.
- Six time series of real and injected shifts (both metering groups, generators anonymised), a light
  grey rectangle over each event, the NRA panel above the prevailing panel, bands shaded, 1 px lines,
  at least 280 px per panel; two example forecasts (day-ahead and full 14-day horizon).

## Compute

CPU only, at most 4 cores, `MemoryMax=16G`. Step 1 about 1 hour; step 2 about 1 hour of wall-clock
for 60 fits on 4 processes. No GPU hours are asked for.
