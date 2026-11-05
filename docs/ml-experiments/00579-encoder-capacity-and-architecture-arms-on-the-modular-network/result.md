---
id: "00579"
slug: encoder-capacity-and-architecture-arms-on-the-modular-network
title: >-
  Is the modular demand network limited by the size of its encoders, by what they are given, or by
  its architecture?
theme: demand-inputs
also_relevant_to:
  - lagged-power-and-error-correction
  - differentiable-physics-network
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted weights, forecasts, and scores in the private data folder only. More than
  one defensible design: how each part of the network is enlarged. Callers: the code copies
  experiments 560 and 572 and imports 397, 428, 510, 511, 545, 554, 558, 559, and 565 (more than two
  callers, found by search). Serving path and degradation rule: not touched (this repository holds
  none). Cost: ceiling about 5 GPU-h (the manager's limit).
hypothesis: >-
  The weather encoder and the datetime encoder of the modular network are small (about 2,500
  weights each). The datetime encoder has the most to learn for demand forecasting, so capacity may
  limit it, and the inputs it is given (hour harmonics 1 and 2, a day-of-week one-hot, solar
  geometry, holiday columns) may limit it too. The weather encoder has only 7 inputs, so capacity
  should not limit it.
prediction: >-
  Written before any score. DL (larger datetime encoder) gains at least 0.05 CRPS points of capacity
  at 24-120 h at one group, with the gain largest on cold winter nights and Christmas week (fails if
  within 0.03). WL (larger weather encoder) is within 0.03 of the base (fails if it gains 0.05 or
  more). XL beats the arm below it by under 0.02. F1 gains at least 0.03 on cold winter nights or
  Christmas week; F2 (Economy 7 proxy) gains at least 0.03 on cold winter nights. P1 and P2 each
  gain at least 0.03 at one group. K5 and K8 lower the summed tail pinball loss by at least 2% with
  CRPS within the minimum detectable effect (MDE) and no component weight under 0.02. RB, GB, ML,
  A1, A2, and EW are each within the MDE of the base, and A2 improves the weakest 4 Primaries. C-tied
  is within the MDE of the base. Each passer's real-event absorbed share is within 0.005 of the base's
  and its injected shift within 1.0 point of capacity. The larger datetime encoder's probe CRPS is
  at least 2% lower than the base's.
decision_rule: >-
  Stage 1 (seed 0, screening, descriptive): an arm passes the screen if its CRPS at 24-120 h
  improves on the base by at least that group's MDE at the eight MVA Primaries or the six MW
  Primaries with at least 4 of 5 folds agreeing in sign; or its summed tail pinball (5%, 95%, 99%
  quantiles) improves by at least 2% with CRPS not worse than the MDE; or it improves one
  pre-registered slice by at least that slice's MDE without actively hurting another (worse than the
  control by more than the larger of 0.03 and the slice's MDE, with the interval excluding zero on
  the harmful side or the harmful sign in 4 of 5 folds). Stage 2 (seed 1; at most 3 passers; the
  real-event harness): the counted tests, at most 16 (3 passers x 2 groups x 2 measures = 12, plus
  C-tied against the base and the large arm against C-tied at 2 groups = 4). Stage 1 contrasts are
  screening with unadjusted month-block intervals (28 CRPS contrasts and the same 28 for tail
  pinball). A change that needs a new pre-training adds weights and a pipeline step, so a passer
  needs the MDE gain; a gain inside the MDE is reported but not a candidate.
benchmark: >-
  B = experiment 572's v3+D5 (stored seeds 0 and 1: forecasts in `seed_preds/V3D5` and pre-trained
  weights in `pretrained_V3D5`), with M0, E2, and L-1d as references.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 24-120 h (lead bands 0-24 and 120-240 h descriptive) at
  the eight MVA Primaries and the six MW Primaries; summed pinball loss at the 5%, 95%, and 99%
  quantiles; slices; leak (real-event absorbed share, injected 7-day shift) for passers.
---

*Planned; every number below is a plan, not a result.*

## Plan

**Base.** B is experiment 572's v3+D5 network (the 560 lag encoder with capped 24 h lags and block
dropout, plus the D5 per-Primary output offset), coarse refit (one fit per fold on days 0 and 14),
five folds of 511, stored weights for seeds 0 and 1. The encoders are frozen in fine-tuning, and the
mixer is fresh. B's forecasts and weights are reused, not refitted.

**What the base datetime encoder holds (Step 0).** The 21 columns are `cos_zenith`, `sin_azimuth`,
`cos_azimuth` (solar geometry); `sin_hour`, `cos_hour`, `sin_hour_2`, `cos_hour_2` (local time of
day, harmonics 1 and 2); `dow_1` to `dow_7` (day-of-week one-hot); `covid`; `christmas_inside`,
`christmas_days`, `christmas_tent`; `bank_holiday`, `nearest_days`, `nearest_inside` (holidays). It
holds no time-of-day harmonic above 2, no day-of-week harmonic, and no day-of-year column. F1 adds
only those 14 columns (hour harmonics 3 and 4, day-of-week harmonics 1 and 2, day-of-year harmonics
1 to 3), and F2 adds the Economy 7 proxy as a fifteenth column. A day-of-year column can memorise
the calendar position, so F1 and F2 are read with that caveat (the stage 1 pilot found calendar columns
change error by 1.1 to 3.4 points); the leave-D8-out and event-free slices are the check.

**Arms** (each adds one change to B; seed 0 first): WL, DL, WDL (2 hidden layers of 256 and 64
outputs in the weather encoder, the datetime encoder, or both); XL (3 layers of 512, 128 outputs)
only for an encoder whose large arm helps; EW (Primary embedding of 16 in both encoders, the lag
encoder, and the mixer; 32 only if 16 passes); ML (mixer of 3 layers of 256); K5, K8; RB (residual
pre-layer-norm blocks in both encoders and the mixer); GB (gated GEGLU blocks in the same places);
F1, F2, F3 (school terms; report-only); P1 (separate 16-wide Primary embedding for the datetime
encoder); P2 (learnt half-hour-of-day embedding per Primary, 48 x 8, added to the datetime input);
A1 (FiLM conditioning of both encoders by the Primary embedding); A2 (rank-4 per-Primary adapter on
the mixer's last layer); U (encoders unfrozen at 0.1 times the learning rate). Controls: C-seed
(B's seed 1, stored) and C-tied (the largest passing encoder arm with the extra width as a
weight-tied copy of the base units).

**Which arms re-use the stored pre-training.** ML, K5, K8, and A2 change only the mixer, the
mixture, or an adapter, so they fine-tune from B's stored encoders. Every arm that changes an
encoder or the Primary embedding needs a new pre-training on the ERA5 and CAMS rows. The lag
encoder reads the encoders through weight-shared copies, so a wider encoder widens its input
automatically.

**Timing asserts** (as 572's code, printed once per arm): the weather comes from the ensemble run
at or before the origin minus 9 h; every lag is at or before the origin minus 3.5 h (the base uses
lags at least 24 h old); every datetime column is a function of the valid time alone (the F1, F2,
F3, and P2 columns are built from the grid index of the valid time and nothing else).

**Standing practices followed.** Cleaned data everywhere; D15 and D18 rows before 2025-03-01 cut;
July 2026 unused; every fitted model saved; stored forecasts reused as controls. Practices not
followed: monthly refits (the brief fixes a coarse refit per fold).

**Cost and cut order.** Step 0 times one base-size pre-training and one large-encoder (DL)
pre-training and projects the total before the other arms run. If the projection exceeds 5 GPU-h,
the manager is told first, and arms are cut in this order: F3, U, GB, A2, P2, XL, K8, A1, EW, then
seeds 2 to 4.

**Statistics.** Five-fold month-block bootstrap with 5 blocks (near the minimum to maximum of the
folds), per-fold signs, the MDE beside every contrast, leave-D8-out beside every contrast on the
eight MVA Primaries. Bonferroni counts: Stage 1 28 CRPS and 28 tail-pinball contrasts (screening,
unadjusted), Stage 2 at most 16.

## Result

*To be written after the scores.*
