---
id: "00514"
slug: load-shift-probability-detector-and-calibration
title: >-
  Can a small calibrated classifier give each Primary half-hour a probability of sitting inside a
  load-shift event, using only data before a refit?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
size: >-
  medium. Stored: injected-world tables, the fitted classifier, and the frozen probability tables,
  under this experiment's private data folder only. Serving path: not touched (this repository
  holds none). Degradation rule: not touched. Designs: more than one defensible (how a pulse is
  scored, what a hard negative is, how an open pulse is treated), so the plan fixes each before any
  number is read. Callers: reads experiment 507's lagless mixture tables, experiment 400's event
  table, experiment 508's dip table, and experiment 510's and 428's clean-power helpers; no earlier
  script is edited. No neural network and no GPU. Reviews: one Sonnet plan review, one Sonnet code
  review before the first run, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  Four signals computed from the scaled residual series (a step at each end of a pulse, a return to
  the original level, edges at the same times in other series, and the symmetry of the sizes) can be
  combined by a logistic regression or a depth-limited tree into a probability that is calibrated on
  injected load-shift pulses, including held-out pulse shapes, seeds, and months, and that stays low
  on genuine level changes, on the D8 engine dips, and on rows with no injected event.
prediction: >-
  Written before any number is read. P1: the expected calibration error on held-out worlds is under
  0.05 at both prevalences (about 7% and about 2% of rows inside a pulse). P2: at p of at least 0.5,
  the recall of injected pulses of at least 1 times the residual spread that last at least 6 hours
  is 0.8 or more, with at most 2 false flags per Primary-year. P3: injected genuine level changes
  that have run for 30 days or more reach p of at least 0.5 on under 10% of their rows. P4: the
  mean p over real rows with no labelled event is under 0.10, so the weight (1 - p) loses under 10%
  of the training weight. P5: the area under the curve of p on the 184 labelled rows against matched
  rows with no labelled event is at least 0.60 (a lower bound, because the matched rows hold
  unlabelled events). P6: p stays under 0.5 on at least 80% of the rows of D22's 13 dip days and
  D8's 16 unexplained dips. Any of these can fail.
decision_rule: >-
  The p tables are frozen and written before any fine-tune score is read (the soft-weight fine-tune
  is a separate later experiment, which this one does not run). The tables go to that experiment
  only if P1 holds at both prevalences on held-out shapes, seeds, and months, and P4 holds. If P1
  fails, the page reports the achieved calibration error and the manager decides; the classifier is
  not tuned further after the freeze. P2 to P3 and P5 to P6 are reports and do not gate. No
  confirmatory hypothesis test is run here (Bonferroni count 0 in this experiment; the soft-weight
  fine-tune carries the count of 10 CRPS tests, 30 with the pinball tests). Every recall and
  false-flag rate carries a bootstrap interval over Primary-months.
benchmark: >-
  Experiment 456's step scan (1.74 false steps per Primary-year, from a daily 7-day scan that does
  not transfer to half-hourly pulses of 1 hour or more) and experiment 459's leak share (median
  0.53 against a placebo of 0.54), as references for the false-flag rate and the symmetry signal;
  experiments 217, 302, 295, 332, and 392 as the earlier detector and masking results.
row_set_id: not_applicable_no_forecast
metric: >-
  Brier score, expected calibration error, and reliability curve of p per row (and per candidate
  pulse); recall of injected pulses by size, duration, and shape; false flags per Primary-year;
  share of rows with p of at least 0.5; Kish effective sample size of the weight (1 - p)^g;
  area under the curve of p on labelled events.
---

**Plan only.** The results sections are written after the run.

## Question

Experiment 510 masked the NGED-labelled load-shift events from the training rows of the lag-free
demand network and scored 0.079 points of CRPS worse at the eight MVA-only Primaries and 0.032
worse at the six signed-MW Primaries (dropping 6.0% to 7.6% of rows; Bonferroni intervals exclude
zero). Its leak of an injected shift was 6.18 points of capacity against 6.11 without the mask, but
that null holds by construction: the injected shift carries no label, so a label mask cannot touch
it. A binary mask therefore showed a cost for dropping rows and said nothing about the leak. A soft
weight might do better, because a false flag at weight 0.1 costs less than a dropped row. The soft
weight needs a probability that each half-hour sits inside a load-shift event. This experiment builds
that probability from data before the refit, calibrates it on injected pulses, and freezes it. A
second experiment will then use the frozen tables as weights in a lag-free fine-tune.

## Data and residual series

- **Source.** Experiment 507's lagless mixture tables (`work/lagless_era5_mixture/month=2025-10,
  2026-01, 2026-03, 2026-06`, seed 0; each holds 16 Primaries from 2019-09 to the month's refit as
  `crossfit` rows, then 62 days of `further` rows). The residual of a Primary is the cleaned power
  over effective capacity (experiment 510's `clean_stack`, flagged exact-zero half-hours dropped, D15
  and D18 cut before 2025-03-01) minus `q50`. The scaled residual `z` divides by the Primary's
  median band half-width (`q95 - q05`) / 2 over its `crossfit` rows. The spread `s` of a Primary is
  1.4826 times the median absolute deviation of `z` over those rows. At the eight, the series is
  apparent power, so the sign of a shift cannot be recovered across zero (reported by group).
- **Folds and cutoffs.** The four folds are 2025-10, 2026-01, 2026-03, and 2026-06. The fifth fold of
  experiment 511 (2025-12) has no 507 table. This experiment drops it and says so. Rebuilding it
  costs 7 fits (GPU) and is left to the fine-tune experiment. The cutoffs are every day at 00:00 UTC
  of each fold month (123 cutoffs: 31 + 31 + 31 + 30); the 25 weekly cutoffs (days 1, 8, 15, 22, 29
  over 5 folds) are a subset, of which 20 exist.
- **Causality.** At the cutoff `c` the detector sees `z` only for half-hours at least 3.5 hours before
  `c` (experiment 510's cut). A script perturbs every `z` at or after the cutoff and checks that the
  stored p is unchanged, and prints the check before any other number. Every fold's scale statistics
  (spread, noise scales of the scan) come from the residual history before that fold's refit.
- **Built-in steps.** Each fold table has a junction between `crossfit` and `further` rows and five
  internal edges between the six cross-fit blocks (experiment 507's `leaks.csv` found 1 to 9
  Primaries with a junction jump above the usual q90). Each block edge is detrended by matching the
  median of the 7 days either side (history only). The junction is detrended causally at each cutoff
  by matching the median of the first up to 7 observed days after it to the 7 days before, and when
  fewer than 2 days of `further` rows exist the rows after the junction are left out of the
  detector. The report counts the flags that touch a junction or edge (within 1 day).
- **Partner series.** The other 15 Primaries (scaled residuals) and the 4 bulk and grid supply points
  (the power minus its median over the same half-hour of the 4 earlier weeks, a causal profile
  difference, because no lagless forecast exists for them).

## The detector

- **Candidate pulses.** For each Primary and each duration `d` in {2, 4, 8, 16, 32, 64, 128, 256,
  512, 672} half-hours (1 hour to 14 days) and each start `a`, compare the mean `z` inside `[a, a+d)`
  with the mean in the flank before and the flank after (`f = clip(d, 12, 336)` half-hours each).
  Signal 1 (step) is the size of each edge, scaled by the robust noise of that statistic over the
  fold's history; a ramp of up to 4 half-hours is allowed because the edge is a difference of means.
  Signal 2 (return) is the closed-pulse score `min(|edge_in|, |edge_out|)` with the same sign at
  both edges, and the return ratio `|after - before| / min(|edge_in|, |edge_out|)`. Candidates above
  a low admission score are thinned by greedy suppression of overlapping spans.
- **Open pulses.** A pulse whose end is not yet observed at the cutoff (no flank after, or fewer than
  4 half-hours of it) is scored as an open step from its start to the cutoff, for starts within the
  last 672 half-hours. It enters the classifier with an `open` flag, so a step in the last 14 days is
  scored high by training, not by hand. A step with no return for 30 days or more has a closed score
  near zero and is reported as "unreturned". It is not masked.
- **Signal 3 (synchrony).** For each candidate, count the partner series with an edge of the opposite
  sign within 2 half-hours (1 hour) of its start, of its end, and of both, with the partner edge
  scaled by its own history. Count the same-sign edges too, because a same-sign edge in many series
  at one time is a weather or data artefact. A shifted-date placebo (partner series shifted by 7 to
  60 days) gives the chance rate of the counts.
- **Signal 4 (symmetry).** For a candidate with a synchronised partner, compare the size of the
  donor's change in MW with the sum of the partners' changes in MW (log of the ratio of magnitudes).
  Experiment 459 found a median leak share of 0.53 against a placebo of 0.54, so this is a low-power
  signal, reported against the placebo and never used alone.
- **Shape correlation.** For candidates of 24 half-hours or more, the correlation of the pulse's
  residual with the Primary's own median daily profile over the 28 days before. It is a feature and
  a column of the mask table.
- **Tiers.** Tier A: a candidate with a synchronised partner (both edges) and a symmetry ratio inside
  a stated tolerance. Tier B: a single-series pulse. Tier A symmetry is a diagnostic and does not
  gate tier A (the tolerance and the tier shares are reported, not tuned).
- **Classifier.** A logistic regression on the standardised feature set (log score, log duration,
  open flag, return ratio, edge asymmetry, amplitude, counts of opposite-sign and same-sign partner
  edges at start, end, and both, symmetry ratio and its flag, shape correlation) and a depth-3 tree
  on the same features. The choice between them is made on the validation worlds' Brier score. The
  chosen model is recalibrated with a one-variable Platt fit on those validation worlds. No neural
  network. The row probability is the largest candidate probability that covers the row (0 if no
  candidate does). A candidate is a positive if at least half of its rows lie inside an injected
  pulse.

## Series-level injector

Not experiment 507's window-level generator (its pull request has not cleared review). A world is a
60-day window of all 20 series built from real residual history, with a random cutoff inside the
window.

- **Pulses.** Sizes from 0.4 to 3.5 times the donor's spread (log-uniform), durations from 2 to 672
  half-hours (log-uniform), both signs, 1 to 4 per world, with a ramp of up to 4 half-hours at each
  edge. Shapes: constant-MW box, own-demand-shaped, and (held out from training) circuit-profile
  (a random daily profile with morning and evening bumps or an Economy 7 night step).
- **Mirrored pairs.** Half the pulses add the opposite change at 1 or 2 neighbours of the donor
  (from experiment 395's corrected neighbour table), scaled by the capacity ratio and by a leak
  share drawn from 0.3 to 1.2 (experiment 459 measured 0.53). The rest have no mirror (tier B).
- **Real events stay out.** Window starts avoid the labelled events (padded 2 days) of the donor and
  of the neighbours; labelled rows are never injected over and never used as training negatives.
- **Hard negatives.** (a) Genuine level changes: a step with no return, injected at a random time,
  size 0.4 to 3.5 times the spread; (b) D8's 55 engine-explained dip days from experiment 508 that
  fall before 2025-07 (real rows, label 0); (c) step-free surrogates: windows with no injection,
  from months before 2025-07. D22's 13 dip days and D8's 16 unexplained dip days are held out from
  every negative set and from training.
- **Training, validation, and held-out sets.** Residual history ending before 2025-07 for training.
  Training worlds: months 2021-01 to 2024-06, two shapes, seeds 0 to 1999. Validation worlds
  (classifier choice and the Platt fit): months 2024-07 to 2024-12, seeds 2000 to 2499. Held-out
  worlds for calibration, never used to fit or choose anything: (i) held-out shape (circuit-profile),
  (ii) held-out seeds (2500 to 2999), (iii) held-out months (2025-01 to 2025-06). Evaluation grids
  for recall use sizes 0.5, 1, 2, and 3 times the spread and durations 1 hour, 6 hours, 1 day, 3 days,
  and 14 days. Training prevalence about 7% of rows inside a pulse (the labelled share with the
  pad); calibration is also reported at about 2%.
- **The national same-sign shift of experiment 511.** The fine-tune experiment applies it to all 16
  Primaries at once with one sign, so signals 3 and 4 are silent and the size lies outside the
  classifier's training range. This experiment evaluates p on that world and reads it as an
  extrapolation test (what share of the shifted rows receives p of at least 0.5).

## Outputs, frozen before any fine-tune score is read

- Per fold and cutoff, a float32 array of p over (Primary, target index) for the 730 days before the
  cutoff, keyed by (Primary, target_idx). Only the last 30 days before each cutoff are recomputed;
  older rows are taken from the previous cutoff's table, which saw at least 29 days after them. The
  first cutoff of a fold computes the whole history.
- The mask table (Primary, start, end, tier, size, shape correlation, open flag, and the cutoff it is
  valid for) at p of at least 0.5.
- The classifier, the feature scalers, the thresholds, and the freeze commit, written to the private
  folder; a file of SHA-256 sums of the p tables.

## Reports

- **Calibration.** Reliability curve, Brier score, and expected calibration error per row and per
  candidate, on training worlds (a fit statistic), validation worlds, and each held-out axis, at
  prevalences of about 7% and 2%.
- **Injected recall and false flags.** By size, duration, shape, and tier, with bootstrap intervals
  over Primary-months and the replicate counts as numbers. Genuine level changes called pulses.
- **Real rows.** Share of rows with p of at least 0.5 on rows with no labelled event, per
  Primary-year, by tier; the Kish effective sample size `(sum w)^2 / sum w^2` of `w = (1 - p)^g`
  (g = 1 and 2, floor 0.1) with and without p over experiment 511's training windows; the share of
  the unlabelled rows whose weight falls under 0.5.
- **Report only, after the freeze.** Area under the curve of p on the 184 labelled rows against
  matched rows with no labelled event (day-block bootstrap), and p on the 34 events with a measured
  deviation. NGED's labels are incomplete, so the matched rows hold unlabelled events and the area
  is a lower bound. D8 and D22 dip weights.
- **Group split.** At the eight the sign of a shift is not recoverable across zero; every figure is
  reported for the six and for the eight.

## Compute and cost

CPU only, at most 2 cores, one process per core, `OMP_NUM_THREADS=1`. The scan is vectorised by
cumulative sums. The whole run is expected to take about 1 to 3 hours; the first fold's time is
reported to the manager. Memory under 8 GB.

## Standing practices followed

Cleaned data and the D15 and D18 cut before 2025-03-01; no model fitted other than the
logistic regression and the tree; July 2026 unused; titles say "load-shift", never "switching";
NGED's labels are called incomplete ("rows with no labelled event"); no generator time series is
shown. The earlier results the plan follows: experiments 510 (a label mask costs CRPS and cannot
speak on an unlabelled leak) and 511 (the baseline cadence, daily, and its leak by block age), and
the earlier detector results of experiments 217, 302, 295, 332, and 392. Serving dependency: the
weights are training-time only and the classifier runs at each refit, not at serving, but
production would still need the lag-free residual series available at each refit.

## Implementation details (deleted when this ships)

Code in `ml-experiments/00514-load-shift-probability-detector-and-calibration/`: `e00514_data.py`
(residual and partner series, junction detrending, labels), `e00514_detect.py` (scan, candidates,
features), `e00514_inject.py` (series-level injector), `e00514_train.py` (worlds, classifier,
calibration), `e00514_tables.py` (the p tables, the causality check, real-row reports),
`e00514_report.py` (figures, numbers).
