---
id: "00591"
slug: shared-lag-encoder-auxiliary-shift-depth-loss
title: >-
  Does a lag encoder shared across lags, trained with an auxiliary loss that asks it to predict the
  depth of the load-shift injected into each lag, follow injected load-shifts less and cover the
  normal running arrangement better after they end than the best model we have (C1 of experiment
  584)?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted weights, forecasts, auxiliary-head predictions, and scores in the private
  data folder only (two pre-training and step-2 chains, shared and separate encoders, plus 20 small
  daily fine-tunes per arm). More than one defensible design (the loss weight, whether the head is
  fed back to the mixer, per-lag or pooled depth label). Spans code whose callers need a search
  (experiments 574, 581, 583, 584, and 586). Serving path and degradation rule: not touched (this
  repository holds none). Ceiling 2 GPU-h; the plan states the wall-clock estimate after the first
  chain is timed.
hypothesis: >-
  The maintainer's request: a lag encoder with weights shared across the lags, trained with an
  auxiliary loss that predicts the signed depth of the synthetic load-shift applied to each lag, is
  forced to pick up the difference between the expected value and the actual lag. The network then
  follows an injected load-shift less (the median forecast moves less with the shifted lag) and its
  q05-q95 band contains the normal running arrangement (NRA) on more rows after a load-shift ends
  than experiment 584's C1 does, because the encoder can tell a shifted lag from an ordinary lag.
prediction: >-
  Written before any fit or score. P1 (the prediction that can fail): A2 or A3 halves the following
  of an injected shift relative to A0 (C1), to below 0.05 of the injected depth, and lifts the
  coverage of the NRA in the 0 to 7 days after an injected 15 or 30 day shift ends above 0.90
  without the gated band, at a CRPS cost inside the minimum detectable effect (MDE) of the contrast.
  P2: A1 (sharing and a lag ID, no auxiliary loss) is within max(0.03, MDE) of A0 in CRPS, as
  experiment 586 found for the same wiring on a different base. P3: the auxiliary head recovers the
  injected depth on held-out injected events with R squared of at least 0.7 at depth 0.30 and 0.50
  and lower at depth 0.15. P4: A4 (separate encoders) is within max(0.03, MDE) of A2 (sharing does
  not matter). Any of these may fail.
decision_rule: >-
  Every event metric is descriptive (event-block bootstrap), as in experiment 584. CRPS contrasts
  are month-block bootstrap intervals at the Bonferroni level of the counted tests, with the
  half-width printed beside each as the MDE. Counted tests: A1 minus A0, A2 minus A1, A3 minus A1,
  A3 minus A2, and A4 minus A2 (A5 minus A2 only if A5 runs), each at the eight MVA Primaries and
  the six MW Primaries, at 24-120 h, so 10 tests (12 with A5). The auxiliary loss is kept if
  following is below 0.05 of the injected depth and after-end coverage without the band is 0.90 or
  more with CRPS no worse than A0 by more than max(0.03, MDE) with the interval excluding zero. The
  threshold for porting is zero loss beyond the MDE, because the change adds one small head and a
  loss term to the daily fine-tune and nothing to forecasting. Stop rule from the brief: if A1 is
  worse than A0 by more than max(0.03, MDE) with the interval excluding zero in the first fold pair
  (2025-10 and 2025-12), run only A2 and A3 to see whether the auxiliary loss repairs it.
benchmark: >-
  A0, experiment 584's stored C1 forecasts (P5 recipe, daily fine-tune on lag-only synthetic
  load-shifts, seeds 0 and 1, coarse refits), with C0 (experiment 572) as the reference.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and the one-sided safe-side share by phase, following of the
  injected shift (experiment 581's lag-following measure), CRPS in points of effective capacity at
  24-120 h at the eight MVA Primaries and the six MW Primaries, tail pinball loss at q01, q05, q95,
  and q99, event-free calibration, and the auxiliary head's recovery of the injected depth (R
  squared, detection rate, onset delay, false triggers per Primary-year).
---

**Plan only; the write-up replaces this page when the fits have run.**

We ask whether a neural network that reads power from earlier days can be made to notice when those
days carry a load-shift (a block of demand moved onto or off a substation). The network already
reads each lag with one encoder. We give the encoder shared weights, so one encoder reads one lag at
a time, and add a second training target: for every training row the encoder's output must also
predict how far the synthetic load-shift (that we injected into the lag) moved that lag, as a share
of the substation's capacity.

## Arms

All arms use the P5 recipe of experiment 574 on the five fold months with refits on days 0 and 14,
seeds 0 and 1, and the daily fine-tune on lag-only synthetic load-shifts of experiment 584's C1.

- **A0 = C1 (stored).** Control. Separate lag-encoder weights, no auxiliary loss.
- **A1.** One shared lag encoder with a learnt lag-ID embedding (experiment 586's S1 wiring on the P5
  base). No auxiliary loss.
- **A2.** A1 plus the auxiliary loss, with weight lambda chosen so that its gradient norm on the lag
  encoder is about the same size as the training loss's (one-batch probe on 2025-10, seed 0, after
  step 2; the value is printed in the write-up).
- **A3.** A1 plus the auxiliary loss with 3 times A2's lambda.
- **A4.** A separate encoder per lag, plus the auxiliary loss at A2's lambda (does sharing matter).
- **A5 (only if A2 or A3 helps).** A2 whose head output also enters the mixer (a learnt detector
  inside the network). It starts from A2's step-2 weights, with two zero-initialised mixer inputs.

## Design

- **Where the auxiliary loss acts.** In the daily fine-tune only, on rows whose lag carries a
  synthetic shift or none (label zero). Pre-training and step 2 use clean lags, so they have no
  depth label. The weights of A1, A2, A3, and A5 share one pre-training and step-2 chain; A4 has its
  own. Experiment 584's stored step-2 weights have the wrong shape for a shared encoder, so both
  chains are fitted here.
- **Label.** The signed shift (share of capacity) added to the grid value that each lag reads, from
  experiment 581's `synthetic_clean` schedule (same seed rule as C1). Zero where the lag carries no
  shift. One label per lag. A lag masked by block dropout or missing has no label (weight zero).
- **Leakage guard.** The head reads only the lag encoder's output, which depends on inputs available
  at the forecast origin. The depth is a training label only and is never an input, except in A5,
  where the head's own prediction enters the mixer.
- **Loss.** Mean squared error between the head output and the label, multiplied by lambda, added to
  the weighted mixture negative log-likelihood. The head is a small two-layer network applied to the
  lag encoder's output (one head per encoder).
- **Reuse.** A0 and C0 forecasts are reused. The injected-bank forecasts, the lags-removed twins (for
  the gated band), and the scoring come from experiment 584's code unchanged.

## Standing practices followed

Monthly refits are not used because the brief fixes coarse refits (days 0 and 14); cleaned data and
the D15 and D18 cut before 2025-03-01 apply; every booster or network state is saved in the private
data folder; draft pull request; July 2026 unused; weather from the run issued at or before the
forecast origin minus 9 h; lag inputs at least 24 h old. Experiments 586 and 584 found that the
lag-encoder wiring moves CRPS by less than the design can detect, so the plan states the MDE beside
every CRPS contrast and treats following, coverage, and the head's recovery of the depth as the
main evidence.
