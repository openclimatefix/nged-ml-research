---
id: "00586"
slug: lag-encoder-sharing-and-lag-id-embedding
title: >-
  Does one lag encoder shared by every lag, told which lag it is reading, forecast better than one
  encoder that reads all lags at once or a separate encoder for each lag?
theme: lagged-power-and-error-correction
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted weights, forecasts, and scores in the private data folder only. More than
  one defensible design (how the encoder is told which lag it reads: a learnt lag-ID embedding or a
  fixed encoding of the lag's period; how the per-lag outputs are joined). Spans code whose callers
  need a search (experiments 560, 565, 572, 581, and 585). Serving path and degradation rule: not
  touched (this repository holds none). Cost: ceiling 3 GPU-h; projected about 1.5, because
  experiment 572's whole v3+D5 fit took 8 minutes per arm and each new arm also pre-trains.
hypothesis: >-
  The maintainer's hunch: one lag encoder with weights shared across the lags, plus a way of telling
  it which lag it is reading, is the best wiring. Experiment 560 found that a shared encoder with no
  lag identity (b_shared) scored worse than one encoder reading both lags at once (v3). The shared
  encoder there could tell the day lag from the week lag only through the lag's age and through its
  position at the mixer, so adding the identity should close the gap or reverse it.
prediction: >-
  Written before any fit or score. All arms use the v3+D5 recipe of experiment 572 (the day-lag and
  week-lag encoder, plus the slow output offset), coarse refits, seeds 0 and 1. P1: S1 (shared encoder
  plus a learnt lag-ID embedding) beats S2 (shared, no identity) by at least 0.03 points of effective
  capacity at 24-120 h at the eight MVA Primaries. P2: S2 is worse than S0 (the control, one encoder
  reading both lags) by at least 0.03 there, as experiment 560 found without the offset (+0.070). P3: S1
  is within 0.03 of S0 or better; S1 beating S0 by more than the contrast's minimum detectable effect
  (MDE) is what the hunch needs. P4: S3 (a separate encoder per lag) is within 0.03 of S1. P5: S4
  (a fixed encoding of each lag's period instead of a learnt ID) is within 0.03 of S1. P6: the
  injected-shift leak of S1 is at most the leak of S0. Any of these may fail. A "within 0.03" claim is
  withdrawn wherever the printed MDE exceeds 0.03.
decision_rule: >-
  The maintainer's keep rule. An arm is kept if it improves all rows at one group by at least that
  contrast's MDE with a Bonferroni interval excluding zero, or improves a pre-registered slice (cold
  winter nights) by at least that slice's MDE. A leak or slice cost is a flag, not a gate, unless it
  actively hurts: worse than the control by more than the larger of 0.03 and the slice MDE with the
  interval excluding zero, or the harmful sign in 4 of 5 folds. Counted contrasts: S1 minus S2, S1
  minus S3, S1 minus S0, and S4 minus S1, each at the eight MVA Primaries and the six MW Primaries
  at 0-24 h, 24-120 h, and 120-240 h, so 4 x 2 x 3 = 24 tests. Intervals are 5-block month
  bootstraps near the minimum to maximum of the folds, with the two-seed MDE beside each. S2 minus
  S0 and S3 minus S0 are descriptive. The wide layout (stage 2) is run only after experiment 585
  merges, or from its branch if the code there is stable; its counted contrasts are S1w minus S0w and
  S3w minus S0w, 12 more tests, reported separately. Threshold for porting: zero loss beyond the MDE,
  because the change moves weights between existing encoders and adds only a small embedding table.
benchmark: >-
  S0, experiment 572's stored v3+D5 forecasts (seeds 0 and 1, coarse refits), with its lags-masked
  twin; experiment 565's lag-free D5 network and M0 as report-only references; the climatology and
  the manual heuristic as in the protocol.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 0-24 h, 24-120 h, and 120-240 h at the eight MVA-only
  Primaries and the six signed-MW Primaries (D15 and D18 report-only); tail pinball loss and
  coverage at q01, q05, q95, and q99; cold winter nights; the injected-shift leak; and the
  lag-following measures of experiment 581 on its injected bank.
---

# Experiment 586: one lag encoder shared by every lag, told which lag it is reading

**Status: planned.** This page is committed before any code runs.

## Question

Experiment 560 found that one lag encoder reading the day lag and the week lag together (v3) beat a
shared encoder that reads one lag at a time. That shared encoder was never told which lag it was
reading, except through the lag's age. This experiment adds the missing piece (a lag identity) and
compares the wirings on experiment 572's v3+D5 recipe.

## Arms (stage 1, two lags)

- **S0:** v3+D5, one encoder reads both lags and the four expected-weather embeddings, one embedding
  goes to the mixer. Experiment 572's stored forecasts, seeds 0 and 1.
- **S1:** one encoder with weights shared across the lags reads one lag at a time (value, missing
  flag, age, that lag's expected weather and datetime embeddings, the Primary embedding) plus a
  learnt 8-number embedding of the lag kind (day or week). The two outputs are concatenated, day first.
- **S2:** as S1 without the lag-ID embedding (experiment 560's b_shared on the D5 recipe).
- **S3:** a separate encoder per lag, each with its own weights (experiment 560's b_unshared on the D5
  recipe).
- **S4:** as S1 with a fixed encoding of the lag's nominal period in days (1 and 7) in place of the
  learnt ID: log2 of the period and its sine and cosine at 4 frequencies, 9 numbers. The lag's age and
  datetime embedding are already inputs of S2, so S4 against S2 isolates the explicit period.

S2 and S3 are refitted on the D5 recipe because experiment 560's b_shared and b_unshared have no
offset and used 1 seed. Experiment 560's own numbers stay a report-only reference.

## Stage 2 (wide layout, conditional)

The day, week, 2-week, and year-ago lags (experiment 585's Y1c layout), arms S0w (one encoder reads
all), S1w (shared plus lag ID, outputs concatenated in a fixed order with block dropout), and S3w
(separate). Run after experiment 585 merges, or from its branch if the code is stable. Experiment 585's
code is read, never edited.

## Standing practices followed

Monthly-equivalent coarse refits (days 0 and 14 of each of the five folds), as experiment 572. Cleaned
data and the D15 and D18 cut at 2025-03-01 come with the copied pipeline. Fitted weights are saved in
the experiment's data folder. Weather from the run issued at or before the origin minus 9 h; day and
week lags at least 24 h old; July 2026 is unused. Reuse: S0 and the references are stored forecasts;
S1 to S4 are fitted (4 arms x 5 folds x 2 seeds, plus pre-training and the stress fits).
