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
  Medium. (1) It stores per-origin feature tables, per-fit head weights (about 9,000 parameters
  each), and one private Parquet file of predictions per arm; (2) more than one design would
  defensibly test the hypothesis (features only, features plus an explicit flag, extra lags), so
  the arms are fixed here and run side by side; (3) it reuses code from experiments 598, 606, 647,
  650, and 652 (their scripts are imported, and no caller outside this folder exists); (4) it
  touches no serving path or degradation rule; (5) it changes no contract.
hypothesis: >-
  Per-lag causal step statistics in noise units, the gap with the all-Primaries median removed, the
  opposite-sign neighbour gap, the routine-step rate, and a return-to-level statistic carry
  information about whether a lag lies inside a load shift that the gap summaries of experiments 606
  and 610 do not, and a small prevailing head that reads them reacts sooner to the start and the end
  of a shift than the same head reading extra lagged power inputs (the arm of experiment 681).
prediction: >-
  Written before any fit or score. P1 (separability, no refit): on the stored bank of experiments
  650 and 652, the area under the receiver operating characteristic curve (AUROC) of "this lag lies
  inside an injected shift" rises by 0.05 or more when D1 to D5 are added to experiment 606's
  summaries, on each of the three banks (stored, mirror at 60%, no mirror). P2 (reaction time, enters
  the verdict): the features-only arm F1 covers half of the new level at least 1 day sooner than
  the departure head of experiment 652 (B0) at the start of an injected shift and at its end.
  Descriptive only: F1 against the extra-lags arm L1. P3 (descriptive): in-force
  scaled CRPS improves by 0.30 points of capacity or more against B0, and near-end and
  just-ended scaled CRPS by 0.30 or more. P4: the false-alarm rate on event-free rows (the share of
  rows where the prevailing output departs from the NRA output by 3 points of capacity or more)
  rises by at most 1 percentage point against B0, and event-free CRPS by at most 0.02 points against
  both B0 and K1. P5:
  the placebo arm (features computed for an unrelated Primary) gains less than a third of the
  features-only arm's reaction-time gain.
decision_rule: >-
  Stop rule (from the spec, fixed before any score is read): if the AUROC gain is under 0.03 on every
  bank and no prevailing stratum (in force, near the end, just ended) gains 0.15 points of capacity of
  scaled CRPS over B0, the features are reported as not useful and the fine-tune of K1 with the
  features (about 2 GPU-hours, not part of this plan) is not run. Step 2 always runs, because it is
  cheap and the stop rule needs its stratum scores. The verdict reads two contrasts.
  Primary contrast: F1 minus B0, reaction time at the start and end of injected shifts of the held-out family, with a 95% interval
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

**Plan only. Nothing here has been run.** Revised after the plan review (findings and triage are
in the pull request).

## Question and approach

**The question.** A load shift is a period when the network operator moves part of one primary
substation's demand onto or off a neighbouring one. The prevailing output of the two-output neural
network for demand should follow such a shift, and today it takes days to do so (experiment 668
measured 181 and 52 hours at the verified D5 and D16 transfer). Experiment 681 tests one remedy,
extra lagged power inputs. This experiment tests another: give the small prevailing head the
statistics that a simple step detector computes, and let the neural network learn how to use them.

**Approach.** Step 1 runs on the CPU with no refit and tests whether the features separate "this
lag lies inside an injected shift" from "it does not" better than the gap summaries the lag encoder
reads today. Step 2 refits the small heads of experiment 652 (two layers of 64 units, about 9,000
parameters) with the features as extra inputs. Both steps use the stored banks of experiments 650
and 652, whose injected shifts are known exactly. The size of the bank's concurrent shifts is a
limit stated under "Banks".

## Features (all causal; D numbering follows the spec)

All power is a share of effective capacity. The gap at a half-hour is the cleaned power minus the
lag-free median `T` of the same half-hour (experiments 598 and 606). The noise unit of a Primary,
for each block width, is 1.4826 times the median absolute deviation of consecutive non-overlapping
block-mean gap differences, measured on the uninjected cleaned power. The unit uses all five
development months, including the scored one. It is a scale of unshifted power and not a target, and
a sensitivity check recomputes it without the scored month and reports the largest change.

- **D1, common-mode-removed level.** The block mean of the gap minus the median over all Primaries
  of the uninjected block mean at that time, in noise units, for blocks of 6 hours, 1 day, and 3
  days. Only the scored Primary carries its injected shift, as in real operation, where few shifts
  run at once.
- **D2, opposite-sign neighbour step.** The largest over the electrical neighbours of minus the sign
  of the Primary's own step times the neighbour's step (6-hour and 1-day), and the hours between the
  Primary's last 2-noise-unit step and the neighbour's.
- **D3, routine-step rate.** The share of the previous 60 days whose common-mode-removed 6-hour step
  exceeded 3 noise units, computed on the shifted series, so injected steps count.
- **D4, causal step statistics.** The block mean ending at the cutoff minus the block mean ending
  one block earlier, for blocks of 6 hours, 1 day, and 3 days, in noise units (raw and
  common-mode-removed).
- **D5, cumulative-sum detector** over the previous 24 six-hour blocks of D1 (drift 1.5, threshold
  6 noise units): the depth at the excursion's extreme, hours since the excursion began, and the sum
  over its threshold. Its start window is fixed (6 days before the cutoff). The explicit flag of
  arm F2 is "the sum exceeds the threshold".
- **D6, return to level.** (Current level minus the extreme) divided by the excursion's size: 0 at
  the extreme and -1 when the level is back to where the excursion began.

Cutoffs. The day-lag and week-lag features end at the lag's own time (24 hours and 7 days before the
origin). The newest-reading features end 6 hours before the origin, the same delay as experiment
681's L1 block (its stated production delay). A lab arm uses 4 hours before the origin, the
freshness of the `fresh` signal of experiment 598, to measure what lower latency would be worth. A
unit test perturbs the power after each cutoff and checks that no feature changes, and each table
build asserts that no index at or after the origin minus the cutoff delay is read.

## Banks (not experiment 651's banks)

Experiment 650's banks inject one independent shift per Primary and pass, with one or two
receivers at a random gain of 0.5 to 1 (15% of events have no receiver, and for those the stored
and no-mirror banks are identical). Step 1 uses three versions: **stored** (neighbour signals carry
the mirror), **mirror x0.6** (the mirror at 60% of its stored strength, an approximation of a
split; it is not experiment 651's 60/40 split over two neighbours, which the stored bank cannot
reproduce), and **no mirror**. In every version the features of a Primary read that Primary's
shifted power, the neighbours' unshifted power plus the mirror (if any), and a median over the
other Primaries' unshifted power. The bank's other 15 simultaneous shifts therefore do not reach the
features, but they remain in the stored 58-column origin features of experiment 606's summaries
that the heads and the baseline logistic scorer read. This asymmetry is a limit, stated in the
write-up.

## Step 1: separability, no refit

- **Rows.** Every Primary and origin of the five development months and 16 passes. A row has one
  label per lag: whether that lag's half-hour lies inside the Primary's injected shift (day-lag and
  week-lag labels are scored separately). The held-out family (passes 200 to 203) is reported
  separately from the training family.
- **Scorers.** Logistic regression with standardised inputs and missing values set to zero beside a
  missing flag, on (a) experiment 606's summaries, (b) those plus D1 to D5, (c) those plus D1 to D5
  of an unrelated Primary of the same metering group (the placebo, the same map as arm P). Folds
  hold out the scored month. AUROC with a month-block bootstrap (5 blocks, so coarse) and an
  event-block bootstrap.
- **Stop rule.** As in the decision rule.

## Step 2: the refit

- **Heads, all 2 seeds x 5 months.** B0 is experiment 652's R-haz (m50 mixture), refitted with
  the same code, rows, and seeds. L1 is B0 plus experiment 681's L1 block (cleaned-power residuals
  at 6, 9, 12, 18, 24, 36, 48, and 72 hours before the origin), built here on experiment 650's
  banks because experiment 681 scores a different bank. F1 is B0 plus all features D1 to D6 at the
  day lag, week lag, and 6-hour cutoff. F1-D46 is B0 plus D4 to D6 at the 6-hour cutoff only (the
  spec's "with and without D4 to D6"). F2 is F1 plus the explicit flag. P is F1 with every feature
  taken from an unrelated Primary of the same metering group (a fixed map that skips the Primary
  and its neighbours). F1-4h is F1 with the 4-hour cutoff, a lab arm. B-haz (the no-fit survival
  curve) is not refitted: it has no fitted weights, and experiment 652's stored scores serve as
  its reference.
- **The survival gate.** Experiment 652's head sets the survival probability to zero when the age
  of the departure at the day lag is under 0.5 days, so a shift that began after the day lag leaves
  the prevailing output equal to the NRA output whatever the head reads. For the feature arms the
  age input becomes the larger of the day-lag age and the age (in days) of the 6-hour-cutoff
  detector excursion. L1's age becomes the larger of the day-lag age and 1 day when the mean of its
  6, 9, 12, 18, and 24-hour residuals exceeds 5 points of capacity. B0 keeps the stored age. A
  further sensitivity run gives B0 the same extended age rule using the 6-hour residual mean.
- **Dropout.** The feature block of F1, F1-D46, F2, F1-4h, and P gets 50% row-wise dropout during
  training (experiment 610's recipe); L1's lag block gets the same, and B0 has none. One run of F1
  without dropout (F1-nd) separates the effect of the features from the effect of the dropout.
- **Fits.** 9 arms x 5 months x 2 seeds = 90 small fits (experiment 652 took 6 minutes for 40 on 8
  processes). CPU only, at most 4 processes, no GPU. Fitted weights are saved to the experiment's
  data folder.
- **Gate.** B0 minus experiment 652's stored R-haz scaled CRPS must agree within 0.01 points of
  capacity; otherwise the experiment stops and reports the mismatch.
- **Reaction time.** For each injected event, the first origin day on or after the onset (or end)
  at which the head's median at leads of 24 to 48 hours differs from the NRA output (K1 run on the
  shifted lags, the bank's stored `q`) by at least half of the shift's depth at the valid time (at the
  end, by at most half of the in-force departure). Resolution is 1 day. An event whose head never
  reacts before the shift ends (at the start) or before 14 days pass (at the end) is censored at
  that point. The page reports the reacting share, the median over events of the censored time with a
  bootstrap interval over events, and the restricted-mean difference. Experiment 681 excluded
  non-reacting events and used a different bank, so its published numbers are not comparable, and
  only the arm L1 refitted here is the contrast.
- **Verified transfers.** D5 and D16 (2025-12-29) are scored descriptively on the real rows: one
  episode, no interval.
- **Scoring.** CRPS and normalised MAE of the median by lead band (24 to 48 hours, 2 to 5 days, and
  5 to 10 days) and by stratum (event-free, far, middle, near, just ended), scaled and unscaled;
  q05 to q95 coverage; tail pinball loss at q01, q05, q95, and q99; the avoidable-curtailment cost in
  three row sets (inside an event, the 7 days after, all rows); and the false-alarm rate on
  event-free rows. The NRA output is unchanged by construction (a head on frozen K1 quantiles), so
  it is reported once as an identity check.

## Standing practices followed

- Each head is fitted once per scored month on the other four months (small heads, no refit
  cadence), as in experiment 652.
- Cleaned data throughout, as in experiment 650's stored clean stack; D15 and D18 rows before
  2025-03-01 are cut (all development months are later); the dropped half-hours per month are
  reported.
- Stored predictions reused: experiment 652's R-haz scores as the gate, K1 rows from experiment 650.
  Fitted here: B0, L1, F1, F1-D46, F2, P, F1-4h, F1-nd.
- Weather: the lag-free table `T` comes from experiment 598 and was built from the weather forecast
  at the forecast time; the features read no weather themselves, only past power up to the cutoff.
- Data before 2026-07-01 only, asserted on every index range.
- Reviews: one Sonnet plan review (done), one fresh Sonnet code review before the first fit, a
  fresh Sonnet re-derivation of the headline numbers, and one Sonnet prose review. No Opus.

## Figures planned

- Take-home first: reaction time at the start and at the end of a shift, by arm, with intervals.
- AUROC by bank, with and without D1 to D5.
- In-force, near-end, just-ended, and event-free scaled CRPS by arm.
- Six time series of real and injected shifts (both metering groups, generators anonymised), a light
  grey rectangle over each event, the NRA panel above the prevailing panel, bands shaded, 1 px lines,
  at least 280 px per panel; two example forecasts (day-ahead and full 14-day horizon).

## Compute

CPU only, at most 4 cores, `MemoryMax=16G`. Step 1 about 1 hour; step 2 about 1 hour of wall-clock
for 90 fits on 4 processes. No GPU hours are asked for.
