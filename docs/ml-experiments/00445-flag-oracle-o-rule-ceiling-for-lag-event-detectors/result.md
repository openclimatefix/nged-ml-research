---
id: "00445"
slug: flag-oracle-o-rule-ceiling-for-lag-event-detectors
title: >-
  If a detector knew exactly which recent power readings lie inside a load-shift event, how much
  would a lagged and lagless forecast mixture gain?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: one table of forecasts per rule, month, and world in this experiment's private
  folder. No fit, only forward passes of saved experiment 404 networks on one CPU core. One design
  (the lagged head is experiment 404's A0-inj network and the lagless head is its frozen cross-fitted
  table, the two heads of its B-mix arm). Production serving path: not touched (this repository
  holds none). Degradation rule: not touched. Callers: the scripts import experiment 404's modules.
hypothesis: >-
  A mixture of the lagged forecast and the lagless forecast whose weight is set by the exact record
  of which of the 3 lags lie inside a load-shift event (an oracle detector, the O-rule) scores
  lower than B-mix (a fixed 50/50 mixture of the same two forecasts) on rows whose time being
  forecast lies inside a real event.
prediction: >-
  Written before any score is read. CRPS in points of capacity, mass-weighted pinball loss, all
  horizons of experiment 404's rows, no widening beyond what the stored forecasts carry. P1. On the
  injected scoring worlds the hard and soft O-rule beat B-mix on injected harm rows at the
  signed-MW six (the flags are exact and the lagless head is known to be better there). P2. Inside
  real events at the signed-MW six the hard O-rule minus B-mix lies between -0.4 and +0.2, so the
  event-cluster interval includes zero. P3. On clean rows the hard O-rule equals the lagged head
  (A0-inj) exactly, so its contrast with B-mix there is A0-inj minus B-mix (+0.13 on real clean
  rows at the six in experiment 404's interim table).
decision_rule: >-
  Contrast = O-rule arm minus B-mix on the same rows, mean mass-weighted pinball loss, with a 95%
  event-cluster bootstrap interval (real events for real rows; world, Primary, and origin day for
  injected rows), with and without the largest event. Primary row set: rows whose target lies inside
  a padded real event ("inside real events"), signed-MW six, pooled over the five months
  2025-10, 2025-12, 2026-01, 2026-03, 2026-06 at fit seed 0 (2026-06 seeds 1 and 2 are a robustness
  check, not pooled). Reading rule. If the hard or the soft O-rule beats B-mix inside real events at
  the six, the interval excluding zero both with and without the largest event, detection is a
  bottleneck for the mixture rule: a better detector would help, and its ceiling is that contrast.
  If neither does, the limit is the lagged and lagless heads or the labelling, because the exact
  flags cannot make this pair of forecasts score better inside events. Labelling means rows inside
  events whose actuals are not normal running arrangement (NRA, the usual network configuration)
  and events missing from the logs. The smallest gain worth reporting is zero, because the O-rule is a diagnostic
  ceiling and is not ported. Interim; no 99% claim.
benchmark: >-
  B-mix, experiment 404's fixed 50/50 mixture of A0-inj and the lagless table. S1, B1, B3, and B6
  (experiment 404's stored final forecasts) are reference rows, where stored for the cell.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, by row class,
  group, and world, as in experiment 404's report.
code: ml-experiments/00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors/
---

**Plan only. Nothing has been scored. All numbers will be interim and unaudited.**

Experiments 404 (arm B1o) and 438 (the positive control) tried to test whether telling a neural
network exactly which recent lags lie inside a load-shift event helps inside real events. The extra
flag input was ignored in 438, so those experiments could not separate "detection is the bottleneck"
from "the network ignores the flag". This experiment removes the network from the question: a rule
that cannot ignore the flag picks between two forecasts that already exist.

- **Heads.** The lagged head is experiment 404's A0-inj network (three lags, trained on injected
  events). The lagless head is its frozen cross-fitted table (no lagged power). B-mix is their
  fixed 50/50 density mixture. Experiment 404 stored only final quantiles, so the two mixtures are
  recomputed with forward passes of the saved A0-inj network on one CPU core, and the recomputed
  50/50 mixture is checked against the stored B-mix file before any O-rule score is read.
- **Hard O-rule.** The lagless forecast when any of the 3 lags is flagged, the lagged forecast
  otherwise.
- **Soft O-rule.** A density mixture with weight s on the lagless forecast and 1 - s on the lagged
  forecast, where s is the share of the 3 lags that are flagged (0, 1/3, 2/3, or 1).
- **Flags.** Exact event codes on the injected worlds (scoring seeds 200 to 202). On the clean
  (real) world, the real-event mask of `events_corrected.parquet` padded by 48 half-hours each side,
  as experiment 404 scores. Each lag is flagged per lag, not as an any-of-3 row flag.
- **Rows and scoring.** Experiment 404's row classes (real harm, real clean, inside real events,
  injected harm, injected clean), signed-MW six and MVA-only eight separately, with and without the
  largest event, event-cluster intervals from `e00404_report`. Reference rows: B-mix, S1, B1, B3,
  B6.
- **Event counts.** The write-up reconciles 184 corrected events, 170 padded runs, and the earlier
  brief's 112, by group and by scored month.
- **What O-mask adds.** Experiment O-mask (S1 trained and scored with each flagged lag replaced by
  "missing") tests the other half of the question: whether a network that can no longer see the
  flagged lag does better than a rule that switches heads. If the O-rule fails to beat B-mix inside
  events, O-mask can still show that the lagged head itself can improve when its flagged inputs are
  hidden. If the O-rule succeeds, O-mask shows whether a single network recovers the gain.
- **Dependency.** The scripts import experiment 404's modules, which are still on the draft branch
  of PR 58. This PR cannot merge before PR 58 does.
- **Compute.** About 7 cells x 4 worlds forward passes of one small network on one CPU core, no
  GPU. A timing on one cell is in the write-up.
- **Reviews.** Sonnet reviews the code before scoring, the printed numbers after scoring, and the
  write-up; the PR body records each.
