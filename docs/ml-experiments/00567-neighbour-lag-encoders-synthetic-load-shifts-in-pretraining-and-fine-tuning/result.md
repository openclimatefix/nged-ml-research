---
id: "00567"
slug: neighbour-lag-encoders-synthetic-load-shifts-in-pretraining-and-fine-tuning
title: >-
  Do the neighbouring Primaries' lags let the modular lag-encoder demand network tell a load transfer
  from a genuine level change, when it trains on synthetic load shifts?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fine-tuned weights of the final candidate arms, synthetic worlds, and forecasts in a
  private folder. More than one defensible design (how neighbours enter the network, how conservation
  is modelled in the synthetic worlds, how the auxiliary flag is labelled). Spans code whose callers
  need a search (experiments 402, 428, 486, 556, 559, 560, 566). Serving path and degradation rule:
  not touched (this repository holds none). Cost ceiling 7 GPU-h (raised by the manager); measured cost is about 2 GPU-h for
  the rate sweep, about 2 GPU-h for the other arms, and up to 3 GPU-h for the daily and real-event refits of passing arms.
hypothesis: >-
  A load transfer lowers the donor Primary's lags and raises a neighbour's lags by the same amount,
  while a genuine level change shows at one Primary only. A network that reads its neighbours' lags as
  well as its own can check whether the sum is conserved, so it can discount a transferred lag and still
  follow a genuine level change. Training it on synthetic load transfers, in both pretraining and
  fine-tuning, lowers its CRPS inside real labelled load-shift events whose recipient is another
  Primary, at no cost outside events.
prediction: >-
  Written before any fit. P1: N1 (shared neighbour lag encoders, synthetic load shifts at the best
  rate) beats N0 (own lags only, same synthetic worlds) inside real events whose recipient is visible
  by at least 0.20 CRPS points of capacity against NRA-corrected actuals; P1 fails if the gain is under
  0.10. P2: N1 minus N1-placebo (neighbours replaced by random non-neighbour Primaries) on held-out
  synthetic events is at most -0.10; fails if the placebo is within 0.05 of N1. P3: N1 costs at most 0.03
  CRPS against N0 on rows with no labelled event. P4: N2 (unshared per-Primary neighbour encoders) is
  no better than N1; fails if N2 beats N1 by at least 0.05. P5: N1-aux beats N1 inside real events by at
  least 0.05 CRPS (fails if within 0.03), and the flag's recall on real events is at least 0.5 at 5%
  false alarms. P6: the real-event leak of N1, and its injected 7-day shift (experiment 511, a flag, not
  a gate), is no higher than N0's. P6 is only measured for arms that pass the coarse screen.
decision_rule: >-
  Coarse screen (one fine-tune per fold at the month's first cutoff, scored over the whole month): an
  arm passes if its CRPS inside real events whose recipient is visible beats N0's at the same synthetic
  rate by at least 0.10 points of capacity, the interval excludes zero, and its cost on rows with no
  labelled event is at most 0.03. At most 3 passing arms are re-run with the daily refit and the
  real-event refits of experiment 556's harness (as ported in experiment 566); the leak flag level is
  experiment 566's, a mean without the largest event above 0.03 of the labelled depth. The injected
  7-day limit of 6.0 points is a flag, not a gate. Contrasts are paired by row; intervals are 5-block
  month bootstraps (near the minimum to maximum of the five folds) with per-fold signs and the minimum
  detectable effect, from the 2-seed spread, beside each. Counted contrasts: N1-N0 inside real events
  with a visible recipient at each of the four rates (4), N1-placebo on held-out synthetic events (1),
  N1-N0 on rows with no labelled event at the best rate (1), N2-N1 (1), N4-N1 (1), N1-corrected-lags
  minus N1 (1), N1-aux minus N1 inside real events (1); Bonferroni count 10. Porting threshold: the
  network gains a component (a neighbour encoder), so the gain must reach 0.20 inside events, at a cost
  of at most 0.03 outside. Wording is "interval excludes zero" or "not shown".
benchmark: >-
  N0, the lag-encoder network of experiment 560 (variant 3) trained on the same synthetic worlds with
  no neighbours, fitted at the same coarse cadence; experiment 566's C0 (lag-free) and C1 (lag encoder
  without synthetic worlds) as report-only context; the lag-free network's real-event leak from
  experiment 556.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by the seven lead bands of experiment 521 and by 24-120 h, at
  the eight MVA-only Primaries and the six signed-MW Primaries (D15 and D18 report-only), on rows with
  no labelled event, inside labelled events against NRA-corrected actuals (split by whether the
  recipient is another Primary), on held-out synthetic events, cold nights, and Christmas week; q01,
  q05, q95, q99 coverage and pinball loss; for N1-aux, the flag's area under the ROC curve, recall, and
  false-alarm rate.
---

# Neighbour lag encoders trained on synthetic load shifts

**This page is a plan; the results follow when the experiment is scored.** Every number is interim and
unaudited until the reviews have run.

## Question

The modular lag-encoder network of experiment 560 reads a Primary's own day-ago and week-ago power. A
load transfer between two Primaries makes those lags wrong, and a network that sees only its own lags
cannot tell a transfer from a genuine level change (the best achievable gap in experiments 446 and 462
is 0.054 to 0.075 CRPS points). This experiment tests whether also reading the lags of the Primary's
neighbours, which carry the mirror-image shift, lets the network discount a transferred lag.

## Method

- **Neighbours (step 0).** Neighbours are other Primaries only (the maintainer's decision for the first
  round). The graph is `Primary Substation Interconnections.csv`, read by experiment 400's
  `nged_neighbours()` and restricted to the 16 Primaries; the 16 Primaries have 1 to 6 neighbours
  (mean 3.6), and the network reads at most 4, taking the first 4 in a fixed order. The BSP and GSP
  series wait for a second round.
- **Synthetic load shifts.** Experiment 402's `inject()` (copied into this folder with one added
  parameter, the share of the donor's loss that the recipient gains, so that the 60% to 100% variant of
  experiment 459 is available) moves load from a donor to one or two neighbouring Primaries, with
  depth and duration resampled from the corrected label table (112 events with a Primary named) and
  spikes as in experiment 402. Worlds are built for the ERA5/CAMS pretraining rows (from 2019-09) and
  the ensemble fine-tuning rows alike, each ending before the fold's refit cut, on the clean base
  (no real labelled event). The training target is the original series; only the lags carry the
  shift. The held-out pairs (one third of the interconnection pairs, never used for a training world)
  and the stress world use seeds that no training world uses. Real events are used only to score.
- **Rate sweep.** The share of Primary-days touched by a synthetic load shift in the 730 days before
  the cutoff is 0%, 3.5%, 7%, and 21%. Real labelled events touch 3.6% to 4.3% of Primary-days in the
  same window (6.3% to 7.7% when padded by 2 days as experiment 510's row drop pads them), so the
  levels are 0, about half, about equal to, and about three times the padded real share. The manager
  accepted these levels. Training rows never see a held-out pair as neighbours.
- **Arms.** N0: LAG-v3 with synthetic worlds. N1: N0 plus a shared-weight neighbour lag encoder (each
  neighbour contributes its own day-ago and week-ago value, missing flag, age, the expected weather and
  datetime embeddings at that lag's own time and place, and its Primary embedding; pooled by mean and
  by attention over the target's embedding, masked for absent neighbours). N0 and N1 at all four rates
  (8 arms; N1 at 0% is also the "N1-noARA" control). At the rate that suits N1 best (chosen on the
  scored folds' held-out synthetic events from the two months before each cutoff, never the scored
  month): N2 (unshared per-Primary neighbour encoders), N4 (neighbours from the event pairs), N1-placebo
  (random non-neighbour Primaries, same count), N1-corrected-lags (corrected lags and corrected target,
  which shows how much a perfect repair would add), and N1-aux (below). 13 arms in all.
- **N1-aux.** A linear head on the mixer's hidden state gives one logit and one signed fraction per lag
  slot (2 own, 2 per neighbour). A lag is positive when its source half-hour lies inside a synthetic
  load shift of its own substation with a transfer of at least 2% of capacity; a smaller transfer is
  excluded from the loss, as is a lag that is missing or dropped. The positive class is weighted by
  the inverse event share, and the loss weight is chosen once from {0.05, 0.1, 0.3} on synthetic events
  of the two months before each cutoff. Real events are used only for scoring.
- **Cadence.** Coarse refits while screening: one pretrain and one fine-tune per fold (the month's
  first cutoff), two seeds, five folds of 511 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06); July 2026
  unused. Passing arms (at most 3) are re-run with experiment 560's daily refit and with experiment
  566's real-event refits.
- **Scoring.** Experiment 560's rows and scorer, with the rows split by whether a real labelled event
  covers either lag and whether the recipient is another Primary, plus a held-out-synthetic-event slice,
  lags removed and aged by 1, 2, and 3 extra days, and neighbour lags removed alone. Leave-D8-out beside
  every contrast. Inside-event contrasts rest on at most 74 events, clustered by Primary and month.
- **Assertions.** Every weather input comes from the run issued at or before the forecast origin, and
  every lag, own and neighbour, is at least 24 hours older than the origin; both are asserted row by row
  and one example row is printed (experiment 560's assertions, extended to the neighbour lags).
- **Reuse.** Experiment 566's harness (rows, three-step and real-event refits) is imported from its
  merged folder, not edited.

## Standing practices

Cleaned data everywhere; D15 and D18 rows before 2025-03-01 cut; saved weights for the final arms;
stored predictions reused as controls; tails and cost reported; examples plotted; NGED's event labels
are incomplete, so rows with no labelled event may hold unlabelled load shifts, the real-event counts
are lower bounds, and false-alarm rates are upper bounds. Exact conservation is a shortcut the synthetic
world offers and the real world does not; N1 is reported on both the exact and the 60% to 100% worlds.
