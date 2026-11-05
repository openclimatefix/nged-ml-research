---
id: "00571"
slug: drift-offset-and-embedding-hybrid-known-answer-worlds
title: >-
  Does a slow output offset take level shifts while a drift embedding takes changes in behaviour?
  Known-answer synthetic worlds first, then the real data
theme: retraining-and-training-history
also_relevant_to:
  - differentiable-physics-long-history-and-drift
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: injected-world target arrays, fitted weights, and window forecasts under this
  experiment's private data folder only; experiments 545, 554, 565, and 566 are read only. More than
  one defensible design: four hybrid constructions, which is the question. Callers: the scripts
  import experiments 397, 418, 422, 428, 448, 510, 511, 559, and 565. Serving path and degradation
  rule: not touched (this repository holds none). One Sonnet review of the plan and code before the
  first fit, one Sonnet review of the write-up against the printed report.
hypothesis: >-
  A slow per-Primary output offset (D5 of experiment 565) absorbs level shifts, and a drift
  embedding that modulates the encoders (A10) absorbs changes in behaviour, such as a growing
  temperature sensitivity or a changed daily shape. Experiment 565's joint fit of both (the hybrid)
  was worse than the offset alone, but a joint fit lets the two components compete for the same slow
  variance, so that failure does not refute the hypothesis. Building the hybrid so the components
  cannot compete (staged, multiplicative-only, or orthogonal) should recover both kinds of injected
  change in worlds where the answer is known.
prediction: >-
  Written before any score. S0: the joint hybrid's two level paths correlate at 0.5 or higher
  (fails below 0.3). W-level: in the abrupt-step world D5 recovers at least 60% of the step truth and
  A10 at most 40%. W-behaviour: in the sensitivity world A10 recovers at least 50% and D5 at most
  25%. W-both: in the combined world the selected hybrid recovers at least 60% of both and beats D5
  and A10 each by at least the minimum detectable effect on the injected Primaries. W-compete: the
  joint hybrid recovers at least 15 percentage points less than D5 on steps or than A10 on
  sensitivity in the combined world. W0: every hybrid is within the minimum detectable effect of D5
  in the clean control. Neighbour leak: on the 10 untouched Primaries no arm moves CRPS by more than
  the minimum detectable effect. Real data: the selected hybrid beats D5 by at least 0.05 on cold
  winter nights or on evenings at either group, is not hurt on any other slice by more than the
  larger of 0.03 and that slice's minimum detectable effect, and has an absorbed share on the 74 real
  events at most 0.01 above D5's.
decision_rule: >-
  A world is readable only if the oracle gains at least 50% of the injected variance on the injected
  Primaries and the clean control shows no recovery by any arm (all within the minimum detectable
  effect of the base); a world that fails is dropped before any hybrid is scored. The selected
  hybrid is the one with the highest minimum of (step recovery, sensitivity recovery) in the combined
  world among the staged, multiplicative, and orthogonal hybrids; it must beat the joint hybrid's
  minimum or the hypothesis is reported as unsupported at this design and the real-data check is
  skipped. On real data a slice gain counts when the Bonferroni month-block interval (5 blocks, near
  the minimum to maximum of the five folds) excludes zero. 20 counted tests: 12 synthetic contrasts
  (3 worlds x 2 contrasts x 2 groups) and 8 real-data contrasts (4 slices x 2 groups). The threshold
  for porting a hybrid is the larger of 0.03 and the slice's minimum detectable effect, because the
  hybrid adds a second slow component to each Primary.
benchmark: >-
  D5 (the incumbent), A10, and the joint hybrid (H) of experiment 565, the coarse M0 of experiment
  554, and, in each synthetic world, an oracle fit of the base network given the true injected
  component as an input.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 24-120 h at the eight MVA Primaries and the six MW
  Primaries (D15 and D18 report-only); recovery fraction of the injected component against the
  oracle; injected-shift leak and absorbed share of depth on the 74 real events.
---

**Plan, written before any fit.** Experiment 565 found that adding the drift embedding to the
output offset removed the offset's gain. This experiment asks whether that failure came from the two
components competing for the same slow variance, by (1) building worlds with a known injected
answer, (2) building the hybrid four ways, and (3) checking the best construction on the real data.

## Step 0 result (CPU and one GPU pass over experiment 565's stored hybrid fits)

**The prediction S0 failed: the embedding and the offset do not compete through correlated level
paths.** Over the five fold months and seeds 0 and 1 (fits of day 14), the mean per-Primary
correlation between the embedding's level path and the offset's level path is 0.09 at the eight MVA
Primaries, 0.04 at the six MW Primaries, and 0.04 at D15 and D18 over all months, and 0.21, 0.15, and
0.67 over the last 12 months. The embedding carries 93% to 99% of the slow variance (embedding path
variance 0.15 to 0.56 in standardised units squared, offset 0.006 to 0.007). So the joint hybrid is
dominated by the embedding, not split between two paths. The staged and multiplicative-only
constructions below address this (they stop the embedding having a level channel), but the
correlation-based reading of "competition" is not supported, and this plan says so before the worlds
are built.

## Design

- **Worlds** (real ERA5 weather and real Primary series; injection into the cleaned power over
  capacity, so training, pre-training, and scoring actuals all carry it). Injected Primaries: 6 (3 MW,
  3 MVA) with the fewest labelled events, not D8 or D14. Wa: 3 abrupt steps per Primary (3% to 6% of
  effective capacity, random sign, at least 60 days apart, one in the 15 months before a fold's start
  and one 20 to 40 days before it, none in a scored month); Wa-in has one step inside the scored
  month (report-only). Wb1: extra load of s(t) times the heating degrees below 15 degrees Celsius,
  s rising linearly from 0 (2023-01) to 0.4% of capacity per degree (2025-06), then constant. Wb2
  (reference arms only): an evening bump, 17:00 to 21:00 local, amplitude rising from 0 to 3% of
  capacity. Wc: Wa and Wb1 together. W0: no injection. Folds: 2025-12 and 2026-06 for the synthetic
  worlds, all five for the real data.
- **Arms** on the lag-free base: D5, A10, H-joint (565's hybrid), H-staged (offset fitted first and
  frozen, then the embedding and mixer fitted), H-film (the embedding is a per-channel scale of 1 plus
  delta on the weather encoder's output, dimension 4, strong L2 shrinkage, no additive path, offset
  fitted jointly), H-orth (H-film plus a penalty that makes the embedding's mean effect on the mixture
  location zero within every Primary and month, and a penalty on the squared covariance of the two
  level paths; covariance rather than correlation because the first penalty drives the embedding's
  level path towards zero, where a correlation is unstable). H-staged-S (the abrupt-step offsets of
  addendum 24) is not run: its known-answer test has not yet been run.
- **Oracle:** the base network given the true injected component, standardised by the Primary's
  target standard deviation, as one extra input to the mixer in pre-training and fine-tuning.
- **Recovery** = (CRPS of the base without drift minus the arm's) / (the base's minus the oracle's) on
  the injected Primaries at 24-120 h, per fold. Seeds: 2 for Wc and the real data, 1 for Wa, Wb1, W0.
- **Timing asserts** (row by row, one printed example per arm): the weather comes from the ensemble run
  at or before the origin minus 9 h (the harness assertions of experiments 510 and 565); the base has
  no lag column; the injected truth enters only the target and, in the oracle only, the oracle input.
- **Standing practices followed:** cleaned data everywhere, D15/D18 cut, saved fitted models, stored
  predictions reused (565's D5, A10, and H on the real data), coarse refits (days 0 and 14). Departed
  from: none. Earlier advice followed: carry slow change as an offset (565); the A10 embedding added
  to D5 lost the gain (565), which this experiment tests rather than repeats.

## Cost and stop rules

Ceiling 4 GPU-hours (unmeasured guess in the brief: oracles 0.4, synthetic cells 2.2, real-data check
1.2). The first cell's cost is measured and the total projected; the manager is told before 4 GPU-h
is exceeded. If no hybrid beats H-joint's minimum recovery in Wc, the real-data check is skipped.
