---
id: "00438"
slug: lag-event-flag-oracle-in-the-day-window-encoder
title: >-
  If the day-window network is told exactly which of its recent half-hours lie inside a load-shift
  event, does it forecast better inside real events?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: 12 fitted networks and their forecasts in this experiment's private folder only;
  experiment 428's folders are read and never written. One design (experiment 428's E3 network plus
  one input channel), so only one defensible design tests the hypothesis. Production serving path:
  not touched (this repository holds none). Degradation rule: not touched. Callers: the scripts
  import experiment 428's modules and the `studies` evaluation code.
hypothesis: >-
  A positive control. Experiment 428's three-day network (E3) given, at every window position, the
  true flag "this half-hour is inside a load-shift event" (E-oracle) forecasts better than E3, than
  S1 (the three lags plus their surprise), and than R-inj (the three lags, trained on injected
  events) on rows whose recent days contain an event. If the flag does not help inside real events,
  the limit on trusting the recent power is not detecting events.
prediction: >-
  Written before any fit and before any score is read. All numbers are CRPS in points of effective
  capacity at days 1 to 10, at the 1.15 widening, signed-MW six. P1. On injected harm rows (the flag
  is exact), E-oracle minus E3 lies between -0.60 and -0.15. P2. On clean rows, E-oracle minus E3
  lies within +-0.05. P3. On real event rows, E-oracle minus S1 lies between -0.40 and 0.00 (an
  exact flag at the lags can only help, but the real-event flag is the padded label table, and the
  network sees only 3 of 4 training worlds with real flags). P4. On real harm rows, E-oracle minus E3
  lies between -0.20 and 0.00. P5. The MVA-only eight show the same signs with smaller sizes.
decision_rule: >-
  The four folds, three seeds, and monthly refits of experiment 428 (test months 2026-06, 2026-01,
  2025-10, 2026-03; seeds 0 to 2). Interim; no 99% claim. Contrasts, each per group (signed-MW six,
  MVA-only eight, D15 and D18 report-only) on CRPS at days 1 to 10 at the 1.15 widening, with a
  month-and-day block bootstrap interval: O1 E-oracle minus E3, O2 E-oracle minus S1, O3 E-oracle
  minus R-inj, on clean, injected harm, real event, and real harm rows, and the last two with and
  without the largest real event. The minimum detectable effect is experiment 428's (1.96 times the
  square root of 2, times the seed standard deviation of R-inj, divided by the square root of 3). The
  smallest gain worth reading as "detection is the limit" is the minimum detectable effect on real
  event rows at the signed-MW six, because the control costs nothing to port (it is never ported).
  Reading rule: if O2 on real event rows is below minus the minimum detectable effect, detection is
  the bottleneck and better inputs (the surprise, window statistics, neighbours) are the next
  experiments. If O2 is within plus or minus the minimum detectable effect, or above it, the limit
  is downstream of detection (the network, the lagless forecast, or the training), and more
  detection work will not help. The result counts as a failed control, and not as a finding, if
  O1 on injected harm rows is above minus the minimum detectable effect, because the flag is exact
  there. The arm never advances.
benchmark: >-
  E3, S1, and R-inj of experiment 428, read from its stored forecasts (same rows, same widening,
  seeds 0 to 2).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, raw and at the 1.15 widening.
verdict: not-yet-run
best_arm: none yet
production_status: not-for-production
adopted_in: none
code: ml-experiments/00438-lag-event-flag-oracle-in-the-day-window-encoder/
---

**Take-home message: not yet run.**

## Question

CRPS is the continuous ranked probability score. The signed-MW six and the MVA-only eight are the two
groups of Primary substations by metering.

**This control asks whether the limit on trusting recent power is detection.** Experiments 404 and
428 never gave the network an event label, so a failure to follow fewer event steps might mean the
network cannot find events or might mean that, even told where the events are, it cannot use the
information. E-oracle tells it exactly. The idea is the maintainer's (2026-10-31). The oracle follows
the flag-oracle controls of experiment 402, which set XGBoost mixture weights from event flags.

## Design

- **Network.** Experiment 428's E3 (three windows of 48 half-hours, a convolution, one attention head,
  the unchanged mixture head), with a sixth channel on every window position: 1 where the half-hour
  is inside an event, else 0, and 0 where the half-hour is unobserved.
- **Flag.** On the clean series, a half-hour of the padded real-event mask of the corrected label
  table. On an injected world, a half-hour with a non-zero injected-event code (donor, recipient, or
  spike), or inside the padded real-event mask. The flag is exact on injected worlds. The padded
  real-event mask is hindsight: it covers each logged event with the padding of experiment 428, so
  the real-event result carries the unlogged-event caveat (NGED may have events missing from its logs).
- **Training and rows.** Experiment 428's: injected worlds 1 to 3 mixed with the clean series,
  cleaned data with the D15 and D18 start cut, monthly refits on the 24 months before the test month,
  scored on the held-out injected worlds 4 to 6 and the real series. The 1.15 widening is applied
  at scoring, as in 428.
- **Compute.** 12 fits (4 folds, 3 seeds), about 15 GPU-minutes each (a one-epoch timing took 27 s
  including start-up), so about 3 GPU-hours, one process under 8 GB. E3, S1, and R-inj are not
  refitted: their stored forecasts are read.
- **Reported.** The real event count and definition, the absolute score of each arm beside the
  contrasts, tails, and the plots of example forecasts and of six labelled real events.
