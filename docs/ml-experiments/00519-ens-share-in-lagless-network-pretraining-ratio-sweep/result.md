---
id: "00519"
slug: ens-share-in-lagless-network-pretraining-ratio-sweep
title: >-
  How large a share of ECMWF ENS rows should the lag-free demand network's pretraining contain?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fitted network weights and window forecasts under this experiment's private data
  folder only; experiments 510's and 511's models and forecasts are read, never written. Serving
  path: not touched (this repository holds none). Degradation rule: not touched. Designs: more
  than one defensible (how a batch share, a flag-column control, and early stopping with no ERA5 term
  are defined), so the plan fixes each before any score is read. Callers: this folder's scripts
  import experiments 428, 448, 470, 510, and 511 and the `studies` evaluation code. One Sonnet plan
  review, one Sonnet code review before the first fit, one Sonnet results review, and two Sonnet
  write-up reviews.
hypothesis: >-
  Experiment 511's mixed pretraining (L-mix: half of every batch from ENS rows, half from ERA5 and
  CAMS rows) scored worse than ERA5-and-CAMS pretraining alone, although ENS rows are what the
  network sees when serving. A smaller ENS share, or no source flag, gives a pretraining at least as
  good as ERA5-and-CAMS alone; the L-mix deficit comes from the 50% share and its double use of the
  fine-tune rows, not from the flag.
prediction: >-
  Written before any score is read. P1: no ENS share beats the 0% control by 0.03 or more at the
  eight or at the six (every difference is inside the minimum detectable effect, 0.164 at the eight
  and 0.123 at the six). P2: 10% and 25% are within 0.03 of the 0% control (non-inferior) and 75% and
  ENS-only are worse than 25%. P3: removing the flag changes the score by less than 0.03. P4: the
  two-stage control is no better than the best mixed share.
decision_rule: >-
  See "Pre-registered rules". A share counts as an improvement only if its paired contrast against
  the 0% flag-column control has a Bonferroni interval (32 CRPS tests) that excludes zero and a point
  estimate of at most -0.03 at the six or at the eight, with q99 pinball loss within +0.02. A share
  is non-inferior if the upper Bonferroni bound is below +0.03. A null on every share is a result:
  ERA5-and-CAMS pretraining is as good as any ENS mixture at weekly cadence.
benchmark: >-
  The 0% flag-column control (this experiment) for every ratio contrast; L-Ww (experiment 510,
  weekly, 90-day half-life, 37 inputs) and L-mix (experiment 511, the 50% share); K2-S1 and A-pool
  for the gap to the lagged network; BG2, climatology, and `manual_heuristic` for scale.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the six signed-MW and the eight MVA-only Primaries, q01,
  q05, q95, and q99 coverage and pinball loss, by horizon band.
production_status: not-for-production
adopted_in: none
---

*Plan, written before any fit. Interim and unaudited until the write-up replaces this text.*

## Question

Experiment 511's L-mix pretrained the lag-free network on ERA5, CAMS, and ENS rows together and
scored worse than L-Ww, which pretrains on ERA5 and CAMS rows only: +0.062 [-0.015, +0.148] at the
eight (not significant) and +0.045 [+0.009, +0.081] at the six, in points of CRPS. We ask what ENS
share in the pretraining batches is best, and whether the deficit comes from the share, the source
flag, or the double use of rows.

## What experiment 511 built (Step 0)

- **Share.** Each pretraining batch held 4,096 ERA5-or-CAMS rows and 4,096 ENS rows, drawn
  uniformly: 50% ENS by batch share, but about 36 to 40% by row count (4.2 to 5.0 million ENS rows
  against 7.5 to 7.9 million ERA5-and-CAMS rows per fold). There were no row weights in pretraining.
- **Overlap.** Every pretraining ENS row also lies in the 730-day fine-tune window, so L-mix saw
  those rows twice. This is a design choice, not a test leak: `pretrain_mix` asserts that no ENS or
  ERA5/CAMS row lies past the cutoff minus 3.5 h or has an origin past the cutoff.
- **Flag and spread.** The source flag is a 38th input: 0 for ERA5-and-CAMS rows, 1 for ENS rows,
  and 1 at fine-tuning and scoring. ERA5 and CAMS rows carry the fold's median standardised ENS
  spread by lead band, as in experiment 418.
- **ENS history.** 18, 20, 21, 23, and 26 months before the five fold cutoffs (the 730-day window
  caps it at 24).

## Arms (all weekly cadence, 90-day half-life, restart from the month's pretrained weights)

Shares are shares of each 8,192-row pretraining batch. An epoch is 20% of the ERA5-and-CAMS training
rows at every share, so all arms take the same number of steps per epoch.

- **F0, the 0% control.** The 38-input network, the flag column fixed at 0 in every pretraining
  row (all rows are ERA5 and CAMS), flag 1 at fine-tuning and scoring, as in L-mix. F0 differs from
  L-mix only in the absence of ENS rows in pretraining, and it keeps the seed pairing. L-Ww (37
  inputs) anchors the cadence only.
- **E10, E25, E75, E100.** Same, with 10%, 25%, 75%, and 100% of every batch drawn from the ENS rows.
- **E50 is L-mix.** Experiment 511's 2 seeds x 5 folds of fits are reused.
- **Early stopping at any share** uses the held-out NLL averaged with the batch shares as weights,
  so 50% reproduces L-mix's mean of the two sources, and E100 uses the ENS term alone.
- **Flag ablation, fixed in advance at 25% and 50%.** NF25 and NF50: no flag column (37 inputs),
  same pretraining mix. Z25 and Z50: the pretrained weights of E25 and L-mix, with the flag set to
  0 (not 1) at fine-tuning and scoring. "Flag 1 at fine-tuning" is what L-mix did.
- **TS, the two-stage control.** Experiment 510's ERA5-and-CAMS pretrained weights (37 inputs),
  then one epoch (20% of the ENS training rows) on ENS rows only at 0.1 times the pretraining
  learning rate, keeping the stage-2 weights only if the held-out ENS NLL fell; then the usual
  fine-tune with no flag.

## Pre-registered rules

- **Tests.** 16 contrasts x 2 groups = 32 CRPS tests (64 with q99): E10, E25, E75, E100, L-mix,
  and F0-vs-L-Ww are the sweep; the flag ablations and TS are each contrasted with their reference.
  Minimum detectable effect: 0.164 (eight) and 0.123 (six), experiment 511's `mde.csv` from two
  seeds, not refitted.
- **Seeds.** Seeds 0 and 1 for every arm. No outcome-selected third seed.
- **Scored set.** Experiment 470's rows, five folds, "rows with no labelled event"; D15 and D18
  report only; horizon bands day 1, days 2 to 3, days 4 to 10.
- **Month-block bootstrap** with the fold count stated beside every interval, and per-fold signs.
- **Diagnostics left to the next experiment.** The ENS date range, the cold-night and Christmas
  slices, and the injected-shift leak.

## What the plan costs

- 7 new pretraining variants (F0, E10, E25, E75, E100, NF25, NF50) x 5 folds x 2 seeds = 70
  pretrains (about 32 s each) + 10 stage-2 epochs, and 10 new arms (F0, E10, E25, E75, E100, NF25,
  NF50, Z25, Z50, TS) x 25 weekly fine-tunes x 2 seeds = 500 fine-tunes (about 3.8 s each).
  About 1.2 GPU-hours in all.
