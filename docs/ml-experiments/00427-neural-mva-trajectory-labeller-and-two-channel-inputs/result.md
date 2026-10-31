---
id: "00427"
slug: neural-mva-trajectory-labeller-and-two-channel-inputs
title: >-
  Does a whole-day sign labeller, with separate MW and MVA input channels, improve the export-gated
  neural network's signed forecast at Primaries where the meter hides the sign?
theme: production-reference-and-best-guess
also_relevant_to:
  - solar-disaggregation
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: 105 labeller state files and their scored tables (step 0), then fitted networks
  (state files) and per-row forecasts of the hidden pair and of the MVA-only eight, all in this
  experiment's private folder. Designs: more than one is defensible (a recurrent or a dilated
  convolutional labeller; soft labels in the likelihood or as inputs only; masking the measured
  power or not), so the plan registers each as an arm and fixes the choice of labeller architecture
  by a pilot on the visible Primaries before step 0 reads the hidden pair. Callers: the code imports
  experiments 87, 397, 405, 412, 417, and 420. Serving path and degradation rule: not touched (this
  repository holds neither).
hypothesis: >-
  A small labeller that reads a whole day of folded readings and outputs a probability that each
  half-hour is reverse flow, trained on the visible signed Primaries with solar farms injected,
  gives the export-gated neural network of experiment 420 (arm A) a better sign signal than the gate
  alone. Feeding the neural network separate MW and MVA input channels with a measured flag, and
  training it with MW masked on pseudo-Primaries that contain injected reverse flow, lets the
  network treat the hidden Primaries and the visible Primaries as the same kind of input. Together
  (arm TDL) the two ideas lower the signed CRPS on reverse-flow half-hours below arm A's.
prediction: >-
  Written before any fit. Step 0 (labeller only, no mixture density network fit). L1. At level 1.0,
  on the sunny slice of the hidden pair, the labeller's row-level area is at least 0.85 and its
  recall at a matched 5% false-reverse rate is at least 0.60 (arm A's row-level ranking area is
  0.930 and its recall at that rate is 0.623, so the labeller is predicted to match A's ranking,
  and the new information is a probability that is calibrated per row). L2. With no injection,
  `pi > 0.5` falls on at most 2% of sunny-midday half-hours at each of D5, D9, D11, D14, and D16.
  Network stage. S1. TDL beats A on reverse-flow rows by at least the minimum detectable effect at
  level 1.0 or 2.0. S2. TDL's q01 coverage on reverse-flow rows is closer to the nominal 0.01 than
  A's 0.062. S3. TDL's false-reverse rate on non-reverse sunny summer middays is below A's 66.8%. The
  prediction most at risk is S1 at level 2.0, where A already scores 7.05 points; the largest
  expected gain is at level 1.0, where A's recall is 0.524.
decision_rule: >-
  Step 0 gate, registered: the networks that use the labeller (L, TDL, TDL-perm) run only if L1 and
  L2 both hold. If either fails, run only T, TD, and the A refits (128 fits) and report that the
  labeller failed. Stage 1 (rotation r1, seed 0, T, TD, L, TDL, plus TDL's second seed, at all
  three settings) stops the experiment if TDL's point gain over A is zero or negative at both
  levels, or if TDL fails C4 or C5 on r1. C1 to C6 are signed CRPS in points of capacity, days 1 to
  10, with 95% Primary-month block intervals. C1: TDL beats P0 on reverse-flow rows by at least the
  minimum detectable effect, pooled and at each hidden Primary, at both levels. C2: TDL beats A on
  reverse-flow rows by at least the minimum detectable effect, pooled, at level 1.0 or 2.0 (Holm
  correction over the two), and is not worse than A by more than that figure at the other level.
  C3: TDL's q01 and q99 pinball losses on reverse-flow rows are no worse than A's (upper interval
  end at most +0.05). C4: with no injection, D5, D9, D11, D14, and D16 are within +0.05 of P0
  pooled and none is above +0.20; on the visible Primaries' measured rows TDL is within +0.05 of A
  pooled. C5: reading CRPS at the MVA-only eight is within +0.05 of A and of P0 pooled. C6: at level
  1.0 TDL beats TDL-perm by at least the minimum detectable effect. The minimum detectable effect is
  2.8 times the month-block bootstrap standard deviation of TDL's seed-0 against seed-1 difference
  on reverse-flow rows, per level, written down before any arm-against-arm score is read; a
  contrast whose figure exceeds 0.5 points is reported as underpowered. If C1 to C6 pass, TDL
  replaces A as the candidate for the ensemble-weather serving test, under the label "inferred,
  laboratory, gated". If C2 fails, A stays the candidate and architecture work on the neural MVA
  forecast stops. Any other passing arm is reported as not shown. No porting threshold applies,
  because the experiment is a laboratory test and nothing here ports to production.
benchmark: >-
  P0 and A of experiment 420 (stored forecasts, after a row-key identity check) plus two refits of A
  in this experiment's folds. T against A, TD against T, L against A, TDL against TD, the flag
  probe, and the false-reverse rate on both of experiment 420's denominators are reported with no
  bar.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Signed CRPS (twice the mean pinball loss over the 11 quantile levels) in points of effective
  capacity on rows with reverse flow, days 1 to 10, pooled over the six hidden Primaries; reading
  CRPS at the MVA-only eight; for the labeller, the row-level area and the recall at a matched 5%
  false-reverse rate on the sunny slice (CAMS unit shape at least 0.3 and clear-sky index at least
  0.7).
date: 2026-10-31
code: ml-experiments/00427-neural-mva-trajectory-labeller-and-two-channel-inputs/
---
**Status: planned. Nothing has been fitted. This page records the plan and the changes made to it
after reading the latest results.**

**This experiment asks whether the export-gated neural network of experiment 420 can be improved
where it is weakest.** Eight Primary substations report only the size of the power that flows, not
its direction (signed power is positive for demand and negative for reverse flow, where embedded
solar generation exports power up the electricity network). [Experiment
420](../00420-neural-mva-export-gated-signed-final/result.md) hid the direction at six Primaries
where it is known, added a real solar farm's recorded output to some of them, and found that a
neural network with a Primary-level export gate (arm A) forecast reverse-flow half-hours much better
than the same network with the gate shut. Arm A has three weaknesses: it finds only 52.4% of
reverse-flow half-hours when the injected farm is the size of the Primary's typical sunny midday
reading (level 1.0), its lower tail is too narrow (the 1% quantile covers 6.2% of reverse-flow rows),
and it calls reverse flow on 66.8% of non-reverse sunny summer middays. This experiment tests two
ideas against arm A: a labeller that reads a whole day of readings and gives each half-hour a
probability of reverse flow, and separate input channels for measured MW and for MVA.

## Changes after reading the latest evidence

- **The 0.70 area belongs to the hand-made day-flip diagnostic, not to arm A.** The day-flip
  diagnostic scored 0.702 in step 0 of experiment 420 and 0.73 in experiment 417. Arm A's own
  row-level ranking area is 0.930 on the sunny slice. The bar for step 0 (an area of 0.85 and a
  recall of 0.60 at a 5% false-reverse rate) is therefore set against arm A's 0.930 and 0.623, and
  the labeller can add value mainly through calibration and the level-1.0 recall, not through
  ranking.
- **[Experiment
  424](https://github.com/openclimatefix/nged-ml-research/pull/81) found that arm A survives
  ensemble weather** (a gain of 15.7 points over the gate-shut network, against 19.2 in the
  laboratory test), and that the network trained on recorded weather and applied to ensemble weather
  scored 5.5 points better than the network retrained on ensemble weather. This experiment keeps
  recorded CAMS and ERA5 inputs, so that every contrast is against experiment 420's stored arm A.
  A serving test of TDL on ensemble weather follows only if C2 passes, and that test must also
  score the laboratory-trained network applied to ensemble inputs.
- **[Experiment
  422](../00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2/result.md)
  found that training at several origin hours removes most of the loss from late readings.** The
  network stage keeps experiment 420's origin hour, so the comparison with A stays paired. Only the
  labeller trains on days cut at a random hour, so it can read a lag day that ends at the newest
  usable observation. The advice not to add age inputs is followed.
- **Experiments 408, 412, and 417 set the history.** [Experiment
  417](../00417-hidden-sign-mva-export-gate-zero-crossing-labels/result.md) counted eight earlier
  treatments of the hidden-sign problem that failed. Its hand-built zero-crossing labeller scored
  0.96 on injected against placebo days at level 2.0 and 0.58 at level 1.0, and its hard labels did
  not lift the fits. The registered bar here is a probability on every gate-open row, not hard labels.
- **Standing rules followed:** monthly refits, cleaned data, the D15 and D18 cut on 2025-03-01 (not
  used here: both are report-only), saved models, stored predictions reused (P0 and A from experiment
  420 after a row-key identity check), tails reported beside CRPS, example forecasts plotted, draft
  pull request. Advice from experiments 417 and 420 that the plan follows: score a gate where it
  opens, score against the ungated network as well as the shut one, add no shared PV-proportional
  variance.

## Design

- **Step 0: the labeller (105 fits, no mixture density network).** One labeller per rotation, test
  month, and left-out visible Primary: five per rotation and month (one trained on the four visible
  Primaries, four each leaving one visible Primary out). It reads 48 half-hours of a UTC day:
  reading `r`, `y' = sqrt(max(r^2 - 4, 0))`, CAMS `u` and `k`, ERA5 temperature, a demand reference
  (median reading by half-hour over overcast days of the same 28-day block and day type), the day
  type, and a mask. It outputs the logit of `P' < 0` and an estimate of `P'`. The architecture is a
  width-5 convolution feeding a two-layer bidirectional GRU of 32 units, or a dilated convolution
  stack of the same width; a pilot on the left-out visible Primaries (rotation r1, test months
  2025-09 and 2026-06, level 1.0) picks the one with the higher pooled area, and the choice is
  written down before the 105 run. If the GRU fit takes over 5 minutes, the dilated stack is used.
- **Training copies.** Each visible Primary-day is copied with a farm injected at power factor 1, at
  a level drawn from 0 to 2.5 (15% of copies at 0, the placebo days), with `Q` drawn from 0, 2, and
  4 points, capped at 70% of the farm's peak on 30% of copies, and cut at a random hour on 50% of
  copies. Loss: cross-entropy on the sign (weight 0.25 where `|P'| < 2`) plus a Huber loss on `P'`.
  Every eighth week is held out for early stopping.
- **No leakage from the hidden pair.** Each rotation injects only the farms its hidden pair does not
  use (r1: B and D; r2: A, C, and D; r3: B and D). A leave-one-out labeller also drops the left-out
  Primary's own farm, so its level-1.0 score is clean. The hidden pair's signed truth is used only to
  score.
- **Step 0 gate.** L1 and L2 above, measured on the hidden pair's training-window days, pooled over
  rotations and months. The page also reports the day-level area (a day's score is its largest `pi`
  on the sunny slice) beside design I's 0.702 and design C's 0.58, the area against the number of
  training Primaries and clean farms in a supplementary run, and the fit time.
- **Network stage (338 fits, after the manager confirms).** Arms: P0 and A (stored, plus two refits
  of A), T (A with two channels and a measured flag), TD (T with pseudo-Primaries and MW masking), L
  (A with the labeller's soft labels in the likelihood and its probabilities as inputs), TDL (the
  candidate, two seeds), and TDL-perm (TDL with `pi` permuted across days within a Primary-month, at
  level 1.0). The soft-label likelihood is `log[pi f(-y') + (1 - pi) f(y')]` with `pi` clipped to
  0.02 to 0.98, and `pi = 0.5` recovers the folded density, so A is nested. `pi` enters the network
  only from readings at or before the run's newest usable observation, asserted as `origin_lags`
  asserts its lags.
- **Compute.** One GPU process (about 1.4 GB of GPU memory), one feeder core, `MemoryMax=8G`,
  `OOMPolicy=continue`, `OMP_NUM_THREADS=1`; stop below 12 GiB in the `available` column of `free
  -g`. Each fit writes its own file atomically and the runs are resumable.

## What the next experiment should answer

To be written after the results.
