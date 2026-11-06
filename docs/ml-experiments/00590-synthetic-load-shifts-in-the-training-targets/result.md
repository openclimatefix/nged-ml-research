---
id: "00590"
slug: synthetic-load-shifts-in-the-training-targets
title: >-
  Does the gain from fine-tuning on synthetic load-shifts survive when the synthetic load-shift also
  moves the training targets, as a real load-shift does, and does it teach the network to follow
  shifts in its lag inputs?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: finished
kind: other
protocol: demand-v2
size: >-
  medium. Stored: three new arms, one new fine-tune per fold month, refit day, and seed (60 small
  neural-network fits, private data folder), plus tables and figures. More than one design could
  test the hypothesis (see "Arms"); the arms follow the manager's brief. Spans experiments 574, 581,
  584, and 587, whose code this experiment wraps. Serving path and degradation rule: not touched
  (this repository holds none). Cost: well under the 1.5 GPU-h ceiling (fits of about 10 seconds
  each, row building of about 45 seconds per fold month, inference on the injected bank).
hypothesis: >-
  In experiments 581 and 584 the synthetic load-shift is added to the lag inputs only, so the
  network learns that its lags can lie. A real load-shift also moves the target on the same days.
  Applying the synthetic shift to the targets as well teaches the network that a shift in its lags
  predicts a shift in the target, so it follows lag shifts more and the coverage gain after a
  load-shift ends shrinks; a mixture of lag-only and lag-plus-target shifts keeps part of the gain.
prediction: >-
  Written before any score. P1: T1 (shift in lags and targets on 5% of the time) follows the
  lag-only injected events more than T0 (lag-only, stored as C1 in experiment 584) by at least 0.05
  of the depth in the 7 days after the end, and covers the NRA on at least 0.03 fewer rows. P2: T2
  (mixture) sits between T0 and T1 on both, and covers the NRA on at least 0.82 of rows in the 7
  days
  after the end. P3: T3 (placebo noise) covers the normal running arrangement (NRA) on at least 0.03
  fewer rows than T0, so the
  gain of T0 over C0 comes from the shifts and not from noisy lags. P4: event-free CRPS at 24-120 h
  of T1, T2, and T3 (continuous ranked probability score, CRPS) is within the minimum detectable
  effect of T0. P5: the refit-only share of
  experiment 587 (the share of a shift in the training power that the weights absorb) is higher for
  T1 than for T0 by at least 0.03 of depth, because T1's weights have seen target shifts.
decision_rule: >-
  Counted contrasts: 12 (T1, T2, and T3 minus T0, each for four measures: coverage of the NRA in
  q05-q95 in the 7 days after the end of the injected 15 and 30 day events, median follow ratio in
  the
  same window, event-free CRPS at the eight MVA Primaries, and event-free CRPS at the six MW
  Primaries). Intervals are at the Bonferroni level for 12 tests (event-block bootstrap with 2000
  draws for the injected bank, five-block month bootstrap for event-free CRPS), and the half-width
  of each interval is printed beside it as the minimum detectable effect. T1 or T2 beats T0 on a
  contrast only if the difference favours it (higher coverage, lower follow ratio, lower CRPS) by
  more than the minimum detectable effect. The experiment adopts nothing. A leak or slice cost is a
  flag unless it actively hurts: worse than T0 by more than max(0.03, the slice minimum detectable
  effect) with the interval excluding zero, or a harmful sign in 4 of 5 fold months.
benchmark: >-
  T0 = arm C1 of experiment 584 (stored forecasts and weights; the P5 recipe fine-tuned with
  lag-only
  synthetic load-shifts on 5% of each Primary's time), and C0 (experiment 572's v3+D5, stored) as
  the
  reference without synthetic shifts.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and q01-q99 by phase (before, days 0-3, day 4 to the end, 0-7 days
  after, 7-14 days after) on the injected bank and on the verified December pair, median follow
  ratio by phase, safe-side share, pinball loss at q01, q05, q95, q99, width, event-free CRPS in
  points of capacity, calibration on event-free rows, and the experiment 587 refit-only share for T0
  and T1.
verdict: partly supported
best_arm: >-
  T2 (lag-only shifts on 2.5% of the time plus lag-and-target shifts on 2.5%): q05-q95 band
  contains the normal running arrangement on 0.899 (0.890 to 0.908) of rows 0 to 7 days after an
  injected 15 or 30 day load-shift ends, against 0.860 for T0, but with a band 29% wider (29.2
  against 22.7% of capacity).
headline: >-
  Putting the synthetic load-shift in the targets as well makes the network follow lag shifts more
  (median follow ratio 0.080 for T1 against 0.049 for T0 after the end, and 0.202 against 0.108
  while the event lasts) and raises coverage after the end (0.896 against 0.860), but the extra
  coverage comes with a band 39% wider that also over-covers event-free rows at the eight MVA
  Primaries (0.919 against 0.883, nominal 0.90). Random lag noise with no shift (T3) covers 0.876,
  more than T0's 0.860.
lessons: >-
  T1 and T2 beat T0 on coverage after the end by more than the minimum detectable effect, and lose
  on following. Their bands are wider on event-free rows too, so the gain may be general widening
  and not an event-specific response; no width-matched control was run. T3 matching T0 says the gain
  of the fine-tune on shifted lags over
  plain P5 (0.781) comes from perturbing the lags during fine-tuning, and a constant shift adds
  little beyond noise on the injected bank. The verified December pair, two Primary-events, does not
  separate the arms.
take_home_message: >-
  Training on synthetic load-shifts that also move the targets raises coverage after an injected
  load-shift ends from 0.86 to 0.90, but only by making the band 29% to 39% wider on average, and
  the training also makes the forecast follow shifts in its lag inputs about twice as much while the
  event lasts. Random
  lag noise with no shift gets 0.88 on its own, so most of the coverage gain over a network trained
  on clean lags comes from perturbing the lags, not from the shifts themselves.
advice:
  - >-
    Compare any synthetic-load-shift fine-tune with a noise-only placebo on the same lag columns
    before crediting the shifts, because random lag noise gave 0.876 coverage after the end against
    0.860 for lag-only shifts (T3 against T0).
  - >-
    Do not add the synthetic load-shift to the training targets as a way to raise coverage after
    the end, because it widens the band on event-free rows too (width 25.0 against 21.6% of
    capacity at the eight MVA Primaries) and nearly doubles following during the event (T1 against
    T0).
  - >-
    Score any band-widening change against a band widened by a constant factor to the same mean
    width, because T1 and T2 gain 0.036 and 0.039 coverage (672-event table) for 8.8 and 6.5 points
    more width.
production_status: not-for-production
adopted_in: none
---

**Putting the synthetic load-shift in the training targets as well raises coverage after a
load-shift ends from 0.86 to 0.90, but the neural network gets there by widening its band and by
following shifts in its lag inputs about twice as much while the event lasts (1.6 times as much
after it ends).** A placebo that adds random noise to the
lags, with no shift, covers 0.88 on its own. On the injected bank, all of the gain of fine-tuning on
perturbed lags over a neural network fine-tuned on clean lags (0.78) therefore comes from the
perturbation and not from the shifts. The effect on the verified December pair is too small to read.

A row is one forecast half-hour for one primary substation (a Primary), 24 to 120 hours ahead. The
eight MVA Primaries are metered in MVA only, and the six MW Primaries in signed MW. A load-shift
moves part of one Primary's demand onto another, so the recent power readings (the lag inputs) no
longer show the normal running arrangement (NRA), the power the Primary would have drawn without the
load-shift. The follow ratio is the change in the median forecast caused by a shift in the lags,
divided by the size of the shift; 0 means the median ignores the shift. CRPS is the continuous
ranked probability score. [Experiment
584](../00584-nra-candidate-pack-x6/result.md) fine-tunes a neural-network forecast on lag inputs
that carry synthetic load-shifts while the targets stay clean (arm T0 here). A real load-shift also
moves the target. We repeated the fine-tune with the same synthetic load-shifts also applied to the
targets (T1), to half of the shifted time (T2), and replaced by random noise (T3), and scored each
on
the 1,248 injected lag-only load-shifts (960 of 15 and 30 days, which are 672 Primary-events in the
table below) of [experiment
581](../00581-robust-multiweek-lag-encoder-x3/result.md). Verdict: **partly supported**. Nothing
here
is adopted for production.

## Figures

![Share of the NRA inside q05-q95 after the end of an injected load-shift, by arm](coverage_after_the_end.svg)

Coverage of the NRA after the end; T1, T2, and T3 sit above T0 and P5.

![Share of the NRA inside q05-q95 by phase of the injected load-shift](coverage_by_phase.svg)

Coverage by phase; the arms differ most from day 4 of the event to its end.

![Follow ratio of the median by phase](follow_by_phase.svg)

Follow ratio by phase; T1 follows the lags most and T3 least.

![Coverage against the width of the band](coverage_against_width.svg)

Coverage against band width; T1 and T2 sit to the right of T0 as well as above it.

![Tail pinball loss after the end](tail_pinball.svg)

Tail pinball loss after the end; T1 and T2 have the highest tail loss.

![Event-free rows: each arm minus T0](event_free_contrasts.svg)

Event-free CRPS, coverage, and tail loss against T0; T1 and T2 over-cover and lose slightly on CRPS.

The next six figures are example forecasts. Day-ahead forecasts around four injected events (fixed
rule: month 2025-10, the lowest-numbered
Primary of each metering group, 15 days at 50% of capacity, up and down) and around the verified
December pair.

![Eight MVA Primaries, 15 days up 50%](example_injected_mva-only_up.svg)

![Eight MVA Primaries, 15 days down 50%](example_injected_mva-only_down.svg)

![Six MW Primaries, 15 days up 50%](example_injected_signed-mw_up.svg)

![Six MW Primaries, 15 days down 50%](example_injected_signed-mw_down.svg)

![D5, verified load-shift](example_verified_D5.svg)

![D16, verified load-shift](example_verified_D16.svg)

Stored forecasts cover 0 to 120 hours only, so no 14-day horizon plot exists. NGED care about 1 to
10
days, so every number here is a short-lead result.

## Arms

- **P5 (reference, not an arm of the question):** [experiment
574](../00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md)'s
  P5 recipe fine-tuned on clean lags, forecasting the same injected bank.
- **T0 (control):** experiment 584's C1: the P5 recipe fine-tuned with lag-only synthetic
  load-shifts on 5% of each Primary's time (depths 15%, 30%, 50% of capacity; lengths 3, 7, 15, 30
  days; random sign). Stored forecasts and weights, not refitted.
- **T1:** the same schedule applied to the lags and to the targets of every row whose valid time
  lies
  inside an event.
- **T2:** lag-only shifts on 2.5% of the time and lag-and-target shifts on a further 2.5%. The
  lag-only set is a prefix of T0's event sequence.
- **T3 (placebo):** independent Gaussian noise on the lags, one draw per Primary and half-hour, with
  the standard deviation of the T0 shift schedule at that Primary. The targets stay clean. The noise
  has the variance of the shifts and not their persistence.
- All arms: 20 fine-tunes each (five fold months, refits on days 0 and 14, seeds 0 and 1); step-2
  weights (the second of the three fine-tuning steps) and the B1 standardisation statistics are
  experiment 574's. The target moments stay those of the clean power.

## Results

- **Row set:** `experiment_470_intersection_v1`, five fold months (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06), 24 to 120 h, July 2026 unused, D15 and D18 cut before 2025-03-01, cleaned data.
  Coarse refits (days 0 and 14); the contrast could depend on the cadence. Counted contrasts: 12.
- **Coverage after the end** (injected 15 and 30 day events, 672 Primary-events, 0 to 7 days after,
  q05-q95, 95% event interval):

| Arm | Both groups | Eight MVA Primaries | Six MW Primaries | Mean width, % of capacity |
|---|---|---|---|---|
| P5 | 0.781 (0.768 to 0.795) | 0.785 | 0.775 | 22.96 |
| T0 | 0.860 (0.850 to 0.870) | 0.859 | 0.861 | 22.69 |
| T1 | 0.896 (0.887 to 0.905) | 0.897 | 0.894 | 31.52 |
| T2 | 0.899 (0.890 to 0.908) | 0.898 | 0.900 | 29.18 |
| T3 | 0.876 (0.867 to 0.884) | 0.877 | 0.873 | 22.38 |

- **Paired contrasts against T0, all 960 events of 15 and 30 days (a different weighting from the
  table above), 0 to 7 days after the end, Bonferroni level of 12** (difference, with the half-width
  of its interval as the minimum detectable effect): coverage T1 +0.029 (0.007), T2 +0.022 (0.006),
  T3 +0.006 (0.003); median follow ratio T1 +0.031 (0.004), T2 +0.011 (0.002), T3 -0.009 (0.001),
  where lower is better. T1 and T2 therefore beat T0 on coverage and lose on following. Plain P5
  covers 0.033 fewer rows than T0 and follows 0.024 more (descriptive).
- **Following by phase, both groups** (median follow ratio; 0 means the median ignores the lag
  shift): days 0-3 T0 0.055, T1 0.098, T2 0.073, T3 0.046; day 4 to the end T0 0.108, T1 0.202, T2
  0.146, T3 0.095; 0-7 days after T0 0.049, T1 0.080, T2 0.060, T3 0.041; 7-14 days after, 0.000 for
  every arm. The 7 days before the event and the first days of the month have no scored origins.
- **Coverage by phase, both groups** (follow table, 960 events): day 4 to the end T0 0.821, T1
  0.874,
  T2 0.879, T3 0.836; 7-14 days after T0 0.869, T1 0.903, T2 0.902, T3 0.866.
- **Tails, 0 to 7 days after the end, both groups** (mean pinball loss, % of capacity; lower is
  better):

| Arm | q01 | q05 | q95 | q99 |
|---|---|---|---|---|
| P5 | 0.209 | 0.869 | 0.969 | 0.268 |
| T0 | 0.175 | 0.695 | 0.858 | 0.241 |
| T1 | 0.272 | 0.880 | 1.039 | 0.311 |
| T2 | 0.258 | 0.789 | 0.997 | 0.313 |
| T3 | 0.172 | 0.675 | 0.817 | 0.228 |

- **Safe side of q05 or q95** (share of the NRA at or above q05 for an upward shift, or at or below
  q95 for a downward shift; nominal 0.95): P5 0.793, T0 0.876, T1 0.902, T2 0.907, T3 0.895.
- **Event-free rows, 24 to 120 h** (578,209 rows at 16 Primaries; each arm minus T0, month-block
  bootstrap at the Bonferroni level of 12, points of capacity; minimum detectable effect in
  brackets). CRPS (2 × the mean pinball loss over the 11 stored quantile levels): eight MVA
  Primaries T1 +0.094 (0.105), T2 +0.034 (0.080), T3 +0.023 (0.022); six MW Primaries T1 +0.118
  (0.195), T2 +0.129 (0.209), T3 +0.047 (0.037). Absolute CRPS (eight MVA, six MW): T0 3.081 and
  2.465, T1 3.178 and 2.572. Coverage of q05-q95: eight MVA T0 0.883, T1 0.919, T2 0.912, T3 0.882;
  six MW T0 0.880, T1 0.921, T2 0.921, T3 0.876. Mean q05-q95 width: eight MVA T0 21.6, T1 25.0, T2
  24.1; six MW T0 17.5, T1 20.6, T2 21.2.
- **Verified December pair, 0 to 7 days after the end, 24 to 120 h** (descriptive; one transfer,
  two Primaries): D16 P5 0.775, T0 0.824, T1 0.799, T2 0.816, T3 0.830; D5 P5 0.292, T0 0.369, T1
  0.441, T2 0.284, T3 0.382. D5 before the event: P5 0.505, T0 0.428, T1 0.798, T2 0.608, T3 0.322.
  The arms differ by less than the spread between the two Primaries (0.284 to 0.441 at D5 against
  0.799 to 0.830 at D16 for the four arms).
- **Predictions:** P1 fails on both halves: T1 follows 0.031 more than T0 (predicted at least 0.05)
  and covers 0.029 more rows (predicted at least 0.03 fewer). P2 holds on following (T2 follows
  0.060, between T0's 0.049 and T1's 0.080) and on the 0.82 floor, but not on coverage: T2 covers
  0.899, above T1's 0.896, in the 672-event table, and 0.894 against T1's 0.901 in the paired
  contrast, so the order of T1 and T2 depends on the weighting and the two are indistinguishable.
  P3 fails (T3 covers 0.016 more than T0 in the 672-event table and 0.006 more in the paired
  contrast, not 0.03 fewer). P4 holds for T1 and T2 (within the minimum detectable effects) and
  fails for T3: +0.023 against 0.022 at the eight MVA Primaries and +0.047 against 0.037 at the six
  MW Primaries, with intervals excluding zero. P5 is not tested. The verdict "partly supported"
  rests on P2 (following) and P4 holding for T1 and T2, and P1 and P3 failing.
- **Not computed:** experiment 587's absorption measure for T0 and T1 (it needs experiment 587's
  refit on shifted training power run on top of the synthetic fine-tune, which was judged more work
  than the main question needs); the avoidable-curtailment cost (needs node-level forecasts with
  generators); a
  width-matched control for T1 and T2; BG2, `manual_heuristic`, and climatology rows; weekly refits.

## Checks

- **Median minus mean forecast, and the error of the median:** not computed; only the 11 quantiles
  are stored, so the mean forecast is not stored.
- **Triggers fired:** the placebo T3 matching the treatment T0 (P3 failed) is a surprise, which
  fires
  the audit rule for an unexpected result. The explanation tested here is the plain-P5 reference
  (0.781): perturbing the lags in fine-tuning accounts for the gain, and a constant shift adds 0.006
  beyond noise in the paired contrast. No other trigger fired. Leaving one Primary out was not run.
- **Reviews:** a fresh Sonnet agent reviewed the fit wrapper before the first fit: no leakage, and
  it
  traced that patching the target after the rows are built reaches every fine-tune row, including
  the
  early-stopping rows. It suggested running the schedule check for every arm and storing each arm's
  schedule description; both were applied. A fresh Sonnet agent re-derived the headline numbers (see
  the PR body). A prose review of this page follows.

## Caveats

- **Only one verified transfer pair is in the fold months**, so every real-event number is
  descriptive.
- **Injected events** add a constant shift to the lag inputs only, so the injected bank rewards a
  network that ignores the lags. It penalises T1 and T2 for following lag shifts that, for a real
  load-shift, would be correct. The real December pair does not settle which view is right.
- **Width is not controlled:** T1 and T2 cover more partly because their bands are wider, on event
  rows and on event-free rows. No width-matched control was run.
- **T3's noise** is independent per half-hour. A persistent placebo (for example a random shift of
  zero mean) is not tested.
- **T1 uses the same event schedule as T0**, so the two arms differ in the targets; the T0 fits were
  not rerun.
- **Unlogged events:** NGED's logs may miss load-shift events, so clean rows may hold events no
  label
  covers, the event-free results are optimistic, and the real-event slices undercount.
- **Horizon:** stored forecasts cover 0 to 120 h; NGED care about 1 to 10 days.

## What the next experiment should answer

Whether the coverage gain of T0 over plain P5 survives a width-matched control for the placebo and
the shifts, and whether a placebo with persistent zero-mean shifts, which has the persistence of the
shifts but not their offset, performs as well as T0.

Scripts: `ml-experiments/00590-synthetic-load-shifts-in-the-training-targets/` (`e00590_arm.py`
wraps
experiment 574's fit as experiment 584 did; `e00590_score.py` wraps experiment 584's scoring;
`e00590_follow.py` computes follow ratios with experiment 581's code; `e00590_report.py` prints
every
table above; `e00590_figures.py` draws the figures). Code reused from experiments 572, 574, 578,
580, 581, 582, 583, and 584. Fitted weights (`.pt`) and tables are in
`data/ml-experiments/00590-synthetic-load-shifts-in-the-training-targets/` (private); the T1, T2,
and
T3 weights are under `t1`, `t2`, and `t3` in `work/models/P5`.
