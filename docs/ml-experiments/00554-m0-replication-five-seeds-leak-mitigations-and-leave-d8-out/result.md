---
id: "00554"
slug: m0-replication-five-seeds-leak-mitigations-and-leave-d8-out
title: >-
  Does the modular lag-free demand network (M0) keep its gain over L-1d with 5 seeds, which of its
  two changes raises its injected-shift leak, and does the gain survive without D8?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fitted network weights, window forecasts, and stress-run forecasts under this
  experiment's private data folder only; no earlier experiment's files are written. Serving path:
  not touched (this repository holds none). Degradation rule: not touched. Designs: more than one
  defensible for the two diagnostic arms (which layers of the single network count as its mixer, and
  whether M0-sep keeps the spread and lead inputs), so the plan fixes each before any score is read.
  Callers: this folder's scripts import experiments 397, 428, 448, 510, 511, and 545 (the scorer and
  the stress script are 545's). One Sonnet code review before the first scored run.
hypothesis: >-
  The modular network M0 (a weather encoder, a datetime encoder, and a fresh mixer at fine-tuning)
  keeps most of its gain over the single network L-1d when both are averaged over 5 seeds instead of
  2, and the gain does not depend on D8 alone. One of M0's two changes (the separate encoders, or the
  fresh mixer) raises its injected-shift leak, and a longer recency half-life or a 3-day refit cadence
  brings the leak down to the production limit.
prediction: >-
  Written before any score is read. P1: M0's CRPS gain over L-1d, pooled over leads of 24 to 120 h at
  the eight, is at least 0.10 points of capacity with 5 seeds, and M0lr's is too. P2: at least one
  mitigation (M0-hl180 or M0-c3) brings the injected-shift leak (all leads, mean of the five folds,
  seed 0) to at most 6.0 points of capacity. P3: M0's gain pooled over 24 to 120 h at the seven
  Primaries of the eight other than D8 is at least 0.05. The experiment fails if M0's gain in P1 is
  under 0.07, if the leak of both mitigations stays above 6.5, or if the gain without D8 is under
  0.03. Stop rule: if M0lr's pooled gain is under 0.03 after 5 seeds, the mitigation arms and the
  two diagnostic arms are not run.
decision_rule: >-
  A gain counts when the Bonferroni month-block interval (5 blocks, near the minimum to maximum of
  the five folds; 28 counted tests: M0 and M0lr against L-1d, times 7 lead bands, times 2 row sets)
  excludes zero, and the pooled 24-120 h point estimate meets the P1 or P3 threshold. The thresholds
  of 0.10 (whole gain) and 0.05 (without D8) are set by the cost of porting a new network class and
  a new pretraining recipe, and sit near the minimum detectable effect of 0.10 to 0.14, so a pass
  reads "interval excludes zero" and the threshold is a point-estimate floor. The leak limit of 6.0
  points of capacity is the release checklist's.
benchmark: >-
  L-1d refitted here with the same 5 seeds (seeds 0 and 1 are experiment 511's stored fits), so
  both sides average the same number of seeds. R-DW (experiment 539), A-pool (experiment 425), and
  the stored 2-seed L-1d of experiment 511 are report-only references.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by the seven lead bands of experiment 521, at the eight
  MVA-only Primaries and the six signed-MW Primaries, with D15 and D18 report-only.
verdict: partly supported
best_arm: >-
  M0 (modular network, fresh mixer at fine-tuning): CRPS 4.433 points at the eight MVA-only Primaries
  over leads of 24 to 120 h, against 4.609 for the 5-seed L-1d.
headline: >-
  With 5 seeds on both sides, M0 beat L-1d by 0.176 points of capacity at 24 to 120 h at the eight
  (Bonferroni interval -0.304 to -0.095), and M0lr by 0.136 (-0.301 to -0.043). Without D8 the
  gains fall to 0.091 for M0 (-0.211 to +0.008) and 0.067 for M0lr (-0.212 to +0.081), and D8 alone
  gains 0.767. The leak mitigations and the two diagnostic arms were not run.
lessons: >-
  The modular network's gain is not a 2-seed accident and not mainly a learning-rate effect, but
  one Primary supplies more than half of it. The other seven Primaries still gain by a point
  estimate above the 0.05 floor, with an interval that includes zero. Which of the two changes
  raises the injected-shift leak is still unknown.
take_home_message: >-
  The modular demand network that reads weather and calendar through separate encoders still beats
  the single network at 1 to 5 days ahead at the eight MVA-only Primaries when both are averaged
  over 5 seeds (0.18 points of capacity), and most of that remains with the slower learning rate.
  About half of the gain comes from one Primary, D8, and the other seven gain by about 0.09 with an
  interval that includes zero, so the gain is real but narrower than the headline. The experiments
  that would find which part of the network raises the injected-shift leak were not run.
advice:
  - >-
    Compare any modular-network arm with a single-network control that averages the same number of
    seeds, because 5-seed L-1d is 0.022 better than 2-seed L-1d at the eight (Results).
  - >-
    Report the eight with and without D8 beside every gain of a lag-free demand network, because D8
    alone gains 0.767 and the other seven gain 0.091 (Results, figure 2).
  - >-
    Test the leak mitigations (a 180-day half-life, a 3-day refit cadence) before adopting M0,
    because the injected-shift leak was not re-measured here (Not done).
production_status: not-for-production
adopted_in: none
---

**The modular demand network M0 still beats the single network L-1d at leads of 1 to 5 days when
both are averaged over 5 seeds, by 0.176 points of capacity at the eight MVA-only Primaries (CRPS,
lower is better), and the gain mostly survives a slower learning rate. About half of it comes from
one Primary, D8: without D8 the gain is 0.091, with an interval that includes zero. The leak
mitigations and the diagnostic arms were not run, so what raises M0's injected-shift leak is still
unknown.**

We asked whether the gain of experiment 545 holds with more seeds. M0 has a weather encoder, a
datetime encoder, and a fresh mixer at fine-tuning; M0lr is the same with the mixer at L-1d's
learning rate. The control is L-1d refitted here for seeds 2 to 4, so that both sides average 5
seeds (seeds 0 and 1 are the stored fits of experiments 511 and 545). Scores are CRPS in points of
effective capacity on rows with no labelled event. "The eight" are the MVA-only Primaries and "the
six" the signed-MW Primaries. The maintainer cut stage 2 (M0-hl180, M0-c3, M0-sep, M0-mix) after
this stage scored, so prediction P2 was not tested.

Verdict: partly supported. P1 held (M0 0.176 and M0lr 0.136, both at least 0.10, with intervals
excluding zero). P3 held on its point estimate (0.091 without D8, at least 0.05) but its interval
includes zero. P2 was not tested. No failure condition fired, and the stop rule did not fire
(M0lr's gain is 0.136, above 0.03).

## Results

CRPS minus L-1d (5 seeds each), negative favours the modular arm, over leads of 24 to 120 h
pooled. Brackets give the Bonferroni month-block interval (5 blocks, near the minimum to maximum of
the five folds; 28 counted tests) and the number of folds in which the arm is worse. The minimum
detectable effect (MDE) is 0.111 at the eight, 0.088 at the eight without D8, and 0.091 at the six.

| Set | M0 | M0lr | M0 minus M0lr |
|---|---|---|---|
| Eight | -0.176 (-0.304 to -0.095; 0/5) | -0.136 (-0.301 to -0.043; 0/5) | -0.040 (-0.105 to +0.007; 2/5) |
| Eight without D8 | -0.091 (-0.211 to +0.008; 2/5) | -0.067 (-0.212 to +0.081; 1/5) | -0.025 (-0.090 to +0.026; 2/5) |
| D8 alone | -0.767 (-1.421 to +0.016; 1/5) | -0.626 (-1.248 to +0.201; 1/5) | -0.142 (-0.212 to -0.012; 1/5) |
| Six | -0.061 (-0.177 to +0.103; 1/5) | -0.021 (-0.154 to +0.213; 1/5) | -0.040 (-0.119 to -0.001; 0/5) |

![CRPS minus L-1d by lead band for M0 and M0lr](figure_1_gain_vs_l1d.svg)

**The gain is not a 2-seed accident.** In the bands the pre-registration names, M0 minus L-1d is
-0.216, -0.182, and -0.153 at 24-48 h, 48-72 h, and 72-120 h (experiment 545, 2 seeds: -0.208,
-0.182, -0.137), and no fold is worse. M0lr keeps -0.172, -0.140, and -0.116.

**One Primary carries about half of the gain.** M0 gains at 7 of the 8 Primaries (D4 is worse, by
0.108), and D8 gains 0.767 against at most 0.34 for any other of the eight. Without D8 the bands
are -0.134, -0.097, and -0.067, and only 24-48 h has an interval that excludes zero. The gain is not
carried by D8 alone, but it is not shown at the other seven.

![CRPS minus L-1d at leads of 24 to 120 h for each of the eight Primaries](figure_2_gain_by_primary.svg)

**M0 reaches R-DW and sits close to A-pool without reading power.** At the eight over 24-120 h, M0
scores 4.433, R-DW 4.442, A-pool 4.395, the 5-seed L-1d 4.609, the 2-seed L-1d 4.631, and the manual
heuristic 5.521 (M0 minus A-pool +0.038, interval -0.101 to +0.162). At the six the gain is smaller
and its interval includes zero (M0 3.245, the 5-seed L-1d 3.306, A-pool 3.280).

**Averaging 5 seeds helps L-1d a little.** The 5-seed L-1d is 0.022 better than the 2-seed L-1d at
the eight (-0.040 to +0.001).

## Checks

- **Origin-time weather, asserted row by row.** For every one of the 153 test origins the script
  asserts `nwp_init_time <= power_fcst_init_time`, `power_fcst_init_time == origin`, and that the
  weather-run summary file exists; every training target is after its origin and its weather run
  exists 9 hours before. Example, 2025-10 test row: target 2025-10-03 07:00 UTC, origin 2025-10-01
  09:00 UTC, weather run 2025-10-01 00:00 UTC (`checks_*_s2-3-4.json`: 24,979,347 training rows and
  1,142,520 test rows per arm).
- **Inputs** are experiment 545's: no power-derived column, the 9 lag columns removed.
- **Reuse.** Seeds 0 and 1 of M0, M0lr, and L-1d are the stored fits of experiments 545 and 511
  (unchanged code); seeds 2 to 4 were fitted here, with new pretrained weights for M0 (all five
  folds) and for L-1d in fold 2025-12.
- **Not done.** Stage 2 (M0-hl180, M0-c3, M0-sep, M0-mix) and the injected-shift leak were not run,
  so P2 is untested; the leak figures of experiment 545 (L-1d 4.49, M0 7.60, M0lr 7.71, seed 0) are
  unchanged. The climatology was not scored here, and no forecast-example figure was drawn. One
  Sonnet reviewer read the code before the first fit and found no bug; the write-up was checked
  against the printed report by its author only. The 5-seed gain over 2 seeds is not paired with
  a 5-seed A-pool or R-DW, which keep their own seeds.

## What the next experiment should answer

Whether a 180-day half-life or a 3-day refit cadence brings M0's injected-shift leak under 6.0, and
which part of the network raises it. Compute used: about 3.5 GPU-hours (pretraining and 1,377
fits).

The code is in `ml-experiments/00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/`
and reuses experiments 397, 428, 448, 510, 511, 521, 539, and 545.
