---
id: "00472"
slug: anchor-plus-correction-with-a-strong-penalty-and-a-seed-averaged-anchor
title: >-
  Can a penalty of 100, or an anchor averaged over three lagless seeds, keep the gain of the
  lagless-anchor-plus-lag-correction forecaster on rows with no labelled event and remove its loss
  inside real events?
theme: lagged-power-and-error-correction
also_relevant_to:
    - lsc-detectors
    - calibration-and-uncertainty
status: planned
date: 2026-11-01
kind: other
protocol: demand-v2
size: >-
  medium. Stored: 30 small saved networks and their forecast tables in this experiment's private
  folder. More than one design would defensibly test the hypothesis (the penalty, and how the
  seed-averaged anchor enters), so the plan fixes both. The scripts import experiment 404's and 454's
  modules. Production serving path: not touched (this repository holds none). Degradation rule: not
  touched.
hypothesis: >-
  Experiment 467's correction (a 32-unit network on the frozen lagless forecast) gains 0.50 points of
  capacity over the lagless head on rows with no labelled event at the signed-MW six, but its L2
  penalty from 1e-3 to 1 never limited the outputs, and it is no better than the lagless head
  inside real events. A penalty of 100 limits the outputs, so it keeps most of the gain and removes
  the loss inside events; and an anchor that is the equal-weight mixture of the lagless heads of
  seeds 0 to 2 carries much of K2-S1's gain with a correction on top.
prediction: >-
  Written before any fit. Scores are CRPS in points of effective capacity at the signed-MW six on
  rows with no labelled event. P1. P100 keeps at least half of experiment 467's gain over the
  lagless head (3.71 to 3.21 for C1e-2, so at least 0.25 points). P2. Avg-anchor scores within 0.05
  of K2-S1 at the six on rows with no labelled event. Either prediction can fail. P3. P100 shrinks
  the share of rows with a shift below 0.01 of capacity towards 1, because the penalty now binds.
decision_rule: >-
  Gate 1 (gain over B-mix with D14 removed) is a reported diagnostic, not a gate, because
  experiment 467's gain with D14 removed was 0.11 against a minimum detectable effect of 0.24. The
  contrast is reported with D14 removed, with D11 removed, and by Primary, and a contrast inside the
  minimum detectable effect is inconclusive. Gate 2 (event robustness) is passed if the arm is no
  worse than its anchor by more than 0.03 on held-out injected load-shift rows (scoring worlds 200
  to 202) at the signed-MW six; real-event rows (NRA-corrected actuals, headline, and metered) are
  reported with their intervals and called inconclusive when the interval is wider than 0.03, as
  experiment 467's was (resolution 0.38). Gate 3 (tails) is passed if q99 pinball loss is no more
  than 5% worse than the anchor's and q99 and q95 coverage are within 0.01 of the anchor's. An arm is
  a candidate for the demand assembly if it beats its anchor (LL-ens3 for Avg-anchor, the same-seed
  lagless head for P100) on rows with no labelled event at the six by more than the minimum
  detectable effect (0.24, experiment 454's formula, computed before any contrast is read), passes
  gates 2 and 3, and serves fewer networks than K2-S1 (4) or ties it within 0.03. The smallest gain
  worth porting is the minimum detectable effect, because the correction adds a network. Stop rule:
  if P100 loses more than half of the gain over the lagless head, a penalty cannot keep the gain and
  the anchor line closes except for the averaged anchor, which is judged against K2-S1. The
  signed-MW six and the MVA-only eight are scored separately and never pooled. The Bonferroni count
  is 27 (3 arms among the references times 3 row classes times 3 references), stated as a number.
benchmark: >-
  Experiment 454's stored forecasts at fit seeds 0 to 2: LL-ens3 (the fair reference for the
  averaged anchor), the single-seed lagless head LL, B-mix, and K2-S1. Experiment 467's stored arms
  C1e-2 (seeds 0 and 1 only), its lag-zeroed control Z1e-2, and C1 (penalty 1) show how much of any
  change comes from the spread recalibration.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, by row class and
  group, as in experiment 404's report; tail coverage and pinball loss at q01, q05, q95, and q99.
code: >-
  ml-experiments/00472-anchor-plus-correction-with-a-strong-penalty-and-a-seed-averaged-anchor/
---

Planned. The plan is the front matter above; the write-up replaces this paragraph.

## Plan

- **Arms (3 fit seeds, 5 months, 15 fits each, 30 in all).** P100 is experiment 467's forecaster
  with an L2 penalty of 100 on the correction output. Avg-anchor is the same forecaster (penalty
  1e-2, 467's best) whose anchor is the 9-component equal-weight mixture of the cross-fitted lagless
  heads of seeds 0 to 2, built as experiment 454's `LL-ens3` is; the surprise at each lag is the lag
  minus that mixture's median. P10 is dropped: penalties from 1e-3 to 1 never bound the outputs, so
  100 tests whether any penalty can.
- **References (stored, not refitted).** From experiment 454 at seeds 0 to 2: LL-ens3, LL, B-mix,
  K2-S1. From experiment 467: C1e-2 (seeds 0 and 1 only), Z1e-2, and C1.
- **Serving networks.** P100: 2 (a lagless network and a correction), 7 fits per monthly refit (6
  cross-fit modules and 1 further module). Avg-anchor: 4 (3 lagless seeds and a correction), 21 fits
  per refit for the lagless seeds. K2-S1 also serves 4.
- **Standing practices followed.** Monthly refits; cleaned data (flagged exact-zero half-hours
  dropped); D15 and D18 are report-only, so the start cut does not matter; saved models; stored
  predictions reused; signed-MW six and MVA-only eight scored apart; rows with no labelled event;
  NRA-corrected actuals headline for real events, with and without the largest event; cold winter
  nights; the 12 to 36 hour band; tails raw and after table P; curtailment at f = 0.9, 0.75, 0.5 at
  the six. NGED's event labels are incomplete.
- **Evidence read before planning.** Experiment 467 (merged), 454, 455, and the queue-order check.
  The brief's gate 1, gate 2, and gate 3 limits were rewritten by the manager on 2026-11-01 and are
  used as written in the decision rule above.
- **Compute.** At most 2 GPU processes, each under 8 GB (`MemoryMax=8G`, `OMP_NUM_THREADS=1`, one
  feeder core). The first fit is timed and reported to the manager before the rest start.
- **Code.** Copied from experiment 467's `e00467_arm.py` and `e00467_score.py` into this folder
  (experiment 467 is merged, so its scripts are not edited), with the averaged anchor using
  experiment 454's mixture and quantile code.
