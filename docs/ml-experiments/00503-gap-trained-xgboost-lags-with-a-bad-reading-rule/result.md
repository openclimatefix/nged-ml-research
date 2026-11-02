---
id: "00503"
slug: gap-trained-xgboost-lags-with-a-bad-reading-rule
title: >-
  Does training the XGBoost demand forecast with 10% of its power lags randomly missing, plus a rule
  that treats an exactly-zero, stuck, or implausibly distant lag as missing, shield the median
  forecast from a dropout, a stuck reading, and a spike in the lags, without losing accuracy on rows
  with no labelled event?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
date: 2026-11-02
kind: other
protocol: demand-v2
size: >-
  complex. Stored: 24 boosters, 24 forecast tables, and the sweep summaries in this experiment's
  private data folder. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched, but rule (ii) is a hard mask of lagged power, so the write-up states what
  serving would need. Designs: the training-gap rate (10%), the 6-times-MAD distance, and the 12-step
  stuck run are stated defaults, so more than one design would test the idea. Callers: the scripts
  import experiment 87's, 383's, 391's, 392's, 402's, and 465's modules.
hypothesis: >-
  An XGBoost demand forecast trained with each of its three power lags independently missing with
  probability 0.10 can ignore a lag that a serving rule sets to missing. The rule sets a lag to
  missing when it reads exactly zero, lies in a run of at least 12 identical half-hourly readings,
  or lies more than 6 times the Primary's stage-1 residual median absolute deviation (MAD) from the
  lagless forecast at the lag time. Together they cut the shift of the median under a dropout, a
  stuck reading, and a spike of 6 MAD or more to under 0.10 of the corruption, at a loss within
  ±0.11 points of capacity on rows with no labelled event.
prediction: >-
  Written before any score is read. Scores are CRPS in points of effective capacity at the six
  Primaries metered in signed megawatts. P1. For arm G-A (rules (i) and (iii)) and G-B (rules (i),
  (ii), and (iii)), the median shift under a 6 h dropout, a 24 h stuck reading, and a spike of 6 MAD
  or more falls below 0.10 of the corruption, with the false-alarm rate of rules (i) and (iii)
  under 1% of real lag values. The recall of rules (i) and (iii) on the injections is 100% by
  construction, so it is not a test. P2. G, G-A, and G-B each have the whole Bonferroni 95% interval
  of their difference from arm (a) inside ±0.11 on rows with no labelled event. P3. Rule (ii) sets
  under 20% of the lags to missing under a genuine step of 5% of capacity at 3 lags, so the hard
  mask at 6 MAD is not a detector of load-shift events. P4. No arm beats the fixed 50/50 mixture
  of arms (a) and (b) on rows with no labelled event.
decision_rule: >-
  Gap training costs nothing at serving, and rules (i) and (iii) need no table, so the smallest gain
  worth porting G-A is zero on the injected-corruption shifts at a loss inside ±0.11 on rows with no
  labelled event. Rule (ii) needs the stage-1 median at the lag times, so G-B must cut the shift by
  at least 0.05 of the corruption beyond G-A. Stop if the clean-row loss exceeds +0.11, or if the
  median shift under a dropout or a spike of 6 MAD or more stays above 0.25 of the corruption for
  G-A and G-B. A rule that shields without an accuracy gain is reported as "more robust, not more
  accurate". Contrasts: G, G-A, and G-B each minus (a) and minus (b), on rows with no labelled event
  and on harm rows at the six: 3 x 2 x 2 = 12 at a Bonferroni family level of 95%. The (a-rule)
  and (f) contrasts are descriptive. Equivalence means the whole interval inside ±0.11 on rows with
  no labelled event; harm rows are descriptive because their intervals span about 1 point.
benchmark: >-
  Arm (a), experiment 402's stored R0 lagged forecast (`work/arm_preds/r0_lagged`); arm (b),
  experiment 392's stored R0 lagless forecast; arm (f), their fixed 50/50 mixture; and the CUDA
  repeat of arm (a), "(a-rep)", fitted by experiment 502 and shared, which gives the noise floor.
row_set_id: experiment_392_monthly_four_folds
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from the 11 quantiles of
  the 51-member mixture, on the intersection of finite actuals (flagged zeros dropped) and all arms'
  forecasts, valid time before 2026-07-01.
code: >-
  ml-experiments/00503-gap-trained-xgboost-lags-with-a-bad-reading-rule/
---

**Plan only: no fit has run.** The experiment asks whether a forecast trained to cope with missing
power lags can be protected from bad readings by a short rule an operator can read: "a lag that
reads exactly zero, repeats for 6 hours, or lies far from the lagless forecast is ignored". Experiment
465 found that median lags shield a stuck meter but not a dropout or a spike.

## Design

- **Recipe.** Experiment 392's R0 arm as copied by experiment 402's `e00402_refit.py` and experiment
  465's `e00465_fit.py`: monthly refits at 2025-10-01, 2026-01-01, 2026-03-01, and 2026-06-01, 500
  rounds, `eta` 0.05, 11 quantiles, the control member for training, the 51 members pooled at scoring,
  flagged zeros dropped, D15 and D18 cut before 2025-03-01 (not applicable: the six are signed-MW
  Primaries, so the cut is moot).
- **Gap-trained model G.** Each of the three lag values of every training row is independently set to
  missing with probability 0.10. A seeded generator per Primary and refit draws the mask once, and the
  mask is saved beside the booster. 6 Primaries x 4 refits = 24 fits, on CUDA like arm (a).
- **Serving rules**, applied to a lag value before prediction, using only readings no younger than
  the newest usable one: (i) the reading at the lag time is exactly zero or flagged; (ii) the reading
  at the lag time is more than 6 times the Primary's stage-1 residual MAD away from experiment 391's
  out-of-sample stage-1 median at that time (off, and counted, where the table has no row); (iii) the
  lag time lies in a run of at least 12 identical readings.
- **Arms.** (a) the stored lagged forecast; (a-rule) the stored boosters of (a) with rules (i) and
  (iii), and a variant with all three; (b) the stored lagless forecast; G; G-A (G with (i) and (iii));
  G-B (G with (i), (ii), and (iii)); (f) the fixed 50/50 mixture of (a) and (b).
- **Reuse.** Arms (a) and (b), and their boosters, are stored. The CUDA repeat of (a) is fitted by
  experiment 502 and shared, so this experiment fits only G (24 boosters).
- **Compute.** One GPU process first; a second only after measuring peak host memory and finding
  more than 20 GB available. The timing fit reports peak memory and seconds per fit before the rest.
- **Reviews.** A fresh Sonnet agent reviews the code before the first fit that produces quoted
  numbers; later reviews follow the `ml-experiment` skill's order.

## Scoring and tests

- **MDE first.** The difference between arm (a) and (a-rep) is printed before any contrast.
- **Row classes.** Rows with no labelled event; harm rows; rows inside real events (NRA-corrected at
  depth 1.0, with 0.75 and 1.25 as sensitivities, and metered); the 7 days after events; cold nights;
  rows where a rule fires and rows where none fires. Absolute scores accompany every contrast, with
  the 12 to 36 h band separately.
- **Tails and curtailment.** Coverage and pinball loss at q01, q05, q95, and q99 (raw and widened by
  1.15), and avoidable curtailment at f = 0.9, 0.75, and 0.5.
- **Bad-data injections, first on 2 Primaries at 2 refits.** A 24 h stuck reading, a 6 h dropout to
  zero, and a spike of 4 to 6, 6 to 10, and 10 to 20 times the stage-1 residual MAD, at lead bands
  0 to 24, 24 to 72, and 72 to 168 h, injected into the Primary's own readings before the lags are
  rebuilt. Arm (a) is re-run under this spike definition because experiment 465 injected spikes in
  units of the week-on-week median change. Reported per arm: the median shift as a share of the
  corruption, and the share of injected cases in which each rule fires.
- **Genuine step.** Experiment 465's readings-level sweep (events of 6 h to 7 days, 5% of capacity,
  both signs) on G, G-A, and G-B, reporting the median shift as a share of the step, the band-width
  ratio, and the share of lags that rule (ii) sets to missing.
- **Real data.** How often each rule fires in the real rows of the four months, by Primary, in rows
  with no labelled event and inside real events.
- **Plots.** A day-ahead and a 14-day forecast for (a), (f), and G-B at one real labelled event and
  at one injected dropout, spike, and stuck case; and the rule-firing figure.

## Serving cost of each rule

- Gap training: one model per pass, no table.
- Rules (i) and (iii): no table, only the last 7 days of readings.
- Rule (ii): the stage-1 median at the lag times, a stage-1 pass over the last 7 days per series.
  It is the maintainer's held hard mask of lagged power in an extreme form (6 MAD, much wider than
  the 90% band).

## Caveats

NGED's event labels are incomplete, so rows with no labelled event may hold unlogged load shifts,
the harm table is optimistic, and real-event slices undercount. Precedent: experiment 460 (lag
dropout in a neural network) was worse on rows with no labelled event (+0.08 against S1); it is a
neural network and not a repeat. Experiment 465 skipped missing readings inside a median but did not
train with gaps.
