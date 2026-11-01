---
id: "00467"
slug: anchor-plus-penalised-lag-correction-network-for-the-lagless-forecast
title: >-
  Does a small lag-correction network, penalised towards zero on top of the frozen lagless
  forecast, match the lagless forecast where the lags are wrong and improve on it where they are
  fresh?
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
  folder. More than one design would defensibly test the hypothesis (what the correction reads, how
  it enters the mixture, how strongly it is penalised), so the plan fixes them in advance. The
  scripts import experiment 404's modules. Production serving path: not touched (this repository
  holds none). Degradation rule: not touched.
hypothesis: >-
  A forecast made of the frozen lagless forecast plus a small network that reads the three origin
  lags, the surprise at each lag, and the lead time, and shifts and rescales the lagless mixture
  with an L2 penalty pulling its outputs to zero, scores as well as the lagless forecast on rows
  where the lags are corrupted, and better on rows with no labelled event, so one anchor plus one
  small correction replaces the two networks of B-mix.
prediction: >-
  Written before any score is read. Scores are CRPS in points of effective capacity. P1. The
  correction is not equivalent to A0-inj: A0-inj re-learns the weather response and has no lagless
  input, whereas the correction cannot overwrite the anchor. P2. At the signed-MW six on rows with
  no labelled event, the penalty-1e-2 arm beats the anchor alone by 0.2 to 0.5 points (the lagless
  head trails B-mix by about 0.10 to 0.3 there, experiment 458) and stays 0.1 to 0.5 behind B-mix.
  P3. Inside real events, the arms with a penalty of 1e-2 or 1e-1 are no more than 0.03 worse than
  the anchor, and the arm with 1e-3 is worse than the anchor by more than 0.03 because it follows
  the corrupted lags. P4. At penalty 1e-1, the shift is below 0.01 of capacity on at least 95% of
  rows, so the stop rule fires for that penalty only.
decision_rule: >-
  A penalty is a candidate if, against the anchor alone on the same rows at the signed-MW six, its
  CRPS on rows with no labelled event is lower by more than the larger of 0.03 and the minimum
  detectable effect, it is no more than 0.03 worse inside real events (NRA-corrected actuals, depth
  factor 1.0), and it is no more than 0.03 worse than B-mix inside real events. The smallest gain
  worth porting is 0.03 points of capacity: one network plus a 32-unit correction replaces two
  networks, so any non-negative gain over B-mix at equal quality would be worth taking, but a gain
  over the anchor alone must pay for the added lag feed. Stop rule: if the shift is below 0.01 of
  capacity on 95% of rows for every penalty, the lags add nothing over the anchor and the idea is
  closed. The minimum detectable effect is 2.8 times the standard error of the monthly differences
  between B-mix and A0-inj (experiment 455's formula), computed before any contrast is read. The
  Bonferroni count is 6 (3 penalties, each against the anchor and against B-mix). The MVA-only eight
  are scored separately and never pooled with the six. Every number is interim: only 5 scored
  months exist.
benchmark: >-
  The anchor alone (experiment 404's stored frozen cross-fitted lagless table, scored from the same
  forecast rows) and B-mix (experiment 454's stored fixed 50/50 mixture of A0-inj and the lagless
  table, per fit seed). Reference rows: A0-inj and S1.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, by row class and
  group, as in experiment 404's report; tail coverage and pinball loss at q01, q05, q95, and q99.
verdict: planned
production_status: candidate
adopted_in: none
take_home_message: >-
  Planned: not yet run. The experiment tests whether one frozen lagless forecast plus a small
  penalised lag correction can replace the two networks of B-mix.
code: >-
  ml-experiments/00467-anchor-plus-penalised-lag-correction-network-for-the-lagless-forecast/
---

**Planned.** The correction is a hidden layer of 32 units that reads the three origin lags, the
surprise at each lag, and the lead time, and outputs a shift of every component mean and a change
of every component log standard deviation of the frozen lagless mixture. Its last layer starts at
zero, and an L2 penalty of 1e-3, 1e-2, or 1e-1 on its outputs pulls it back to zero. Training uses
the injected-event worlds of experiment 404, so the correction sees corrupted lags.

- **Relation to A0-inj.** A0-inj is a plain pooled network with the lags and no lagless input, so
  it re-learns the weather response. The correction here sits on the frozen lagless mixture and
  cannot overwrite it, which is the only new part.
- **Harness.** Experiment 404's: five months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), six
  epochs, fit seeds 0 and 1, injected worlds 100 and 101, scoring worlds 200 to 202 and the real
  world. Retraining is monthly. The start cut for D15 and D18 applies, and flagged exact-zero
  half-hours are dropped, as in experiment 404.
- **Fits.** 30 (3 penalties, 2 seeds, 5 months), 28 s each, about 15 to 20 minutes on one GPU
  process under 8 GB.
- **Stored forecasts reused.** The anchor (the lagless table), and B-mix and A0-inj at fit seeds 0
  and 1 from experiment 454. S1 is stored only at seed 0 (all five months) in experiment 404, so it
  is a seed-0 reference only. Nothing is refitted.
- **Loss.** The negative log-likelihood with experiment 404's recency weights, plus the penalty
  times the unweighted batch mean of the squared outputs (the shift in standardised units and the
  log-scale change together). Like A0-inj, the correction is trained without lag dropout.
