---
id: "00554"
slug: m0-replication-five-seeds-leak-mitigations-and-leave-d8-out
title: >-
  Does the modular lag-free demand network (M0) keep its gain over L-1d with 5 seeds, which of its
  two changes raises its injected-shift leak, and does the gain survive without D8?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: planned
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
---

**Planned.** Arms, all with 5 seeds (0 to 4), daily refit and a 90-day recency half-life unless the
name says otherwise: L-1d (matched control); M0 and M0lr (experiment 545's arms, whose seeds 0 and 1
are experiment 545's stored fits); M0-sep (separate encoders, pretrained mixer kept at fine-tuning,
so it reads no ensemble spread or lead time); M0-mix (the single network of L-1d, whose last two
linear layers are re-initialised at fine-tuning and trained at M0's mixer learning rate of 5e-4);
M0-hl180 (M0 with a 180-day half-life); and M0-c3 (M0 refitted every third day, on days 0, 2, 5, 8,
and so on of each month). The injected-shift leak uses seed 0 and experiment 511's stress world.
