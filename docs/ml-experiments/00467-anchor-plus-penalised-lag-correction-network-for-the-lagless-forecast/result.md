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
status: finished
date: 2026-11-01
kind: other
protocol: demand-v2
size: >-
  medium. Stored: 50 small saved networks (30 pre-registered, 20 post hoc) and their forecast tables
  in this experiment's private
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
  Bonferroni count is 9 per reference (3 penalties times 3 row classes); the tables use 15 because
  they also hold C1, Z1e-2, and the references. The MVA-only eight
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
verdict: partly supported
production_status: not-for-production
adopted_in: none
best_arm: >-
  C1e-2 (penalty 1e-2). At the signed-MW six on rows with no labelled event it scores 3.212 points
  of capacity against 3.708 for the lagless anchor (-0.496, 95% interval -0.865 to -0.254) and 3.498
  for B-mix (-0.286, interval -0.807 to +0.195). Penalties 1e-3, 1e-1, and 1 score within 0.01 of
  it.
headline: >-
  A 32-unit correction on the frozen lagless forecast, reading the three lags, beats the anchor by
  0.50 points at the signed-MW six on rows with no labelled event (5 of 5 months), and beats B-mix
  by
  0.29 points (4 of 5 months, interval includes zero). Inside real events, scored on
  NRA-corrected actuals, it is no better than the anchor and 0.22 points worse than B-mix. Interim
  and unaudited.
lessons: >-
  The L2 penalty never binds at 1e-3 to 1, so the "penalised towards zero" label overstates the
  regularisation. About a fifth of the gain over the anchor comes from the lead-time recalibration
  of the anchor's spread, and the rest comes from the lags. The same network with its lag inputs set
  to zero scores 3.611, so the lags carry 0.40 of the 0.50 points.
take_home_message: >-
  A frozen forecast made without recent power readings (the lagless forecast), plus a 32-unit
  correction that reads the recent readings, is the best design with one lagless neural network
  and a small correction that we have tried on rows with no labelled event. At the six
  Primaries metered in signed megawatts it scores 3.21 points of capacity, against 3.71 for the
  lagless forecast alone and 3.50 for B-mix (a fixed 50/50 mixture of the lagless forecast and a
  neural network that reads the readings). The readings carry 0.40 of the 0.50-point gain over the
  lagless forecast. The design fails the rule we wrote before reading the scores, which required it
  to be no worse than the lagless forecast inside real load-shift events. On actuals corrected to
  the normal running arrangement (NRA), it is no better than the lagless forecast there and 0.22
  points worse than B-mix (not significant). The penalty on the correction never limited it, so the
  next step is a penalty that does, or a correction that ignores readings inside events. Interim
  and unaudited, five scored months.
code: >-
  ml-experiments/00467-anchor-plus-penalised-lag-correction-network-for-the-lagless-forecast/
---

**A forecast made without recent power readings (the lagless forecast), plus a small correction that
reads those readings, beats the lagless forecast alone and B-mix on rows with no labelled event.
It does not help inside real events.** A lag is a power reading available at the forecast origin:
the newest one, and the readings at the same time on an earlier day and in an earlier week. B-mix
is the fixed 50/50 mixture of the lagless forecast and a neural network that reads the lags. The
correction is a neural network with one hidden layer of 32 units. It reads the three lags, the
surprise at each lag (the lag minus the lagless median), and the lead time. It shifts and rescales
the lagless mixture, and its last layer starts at zero, so it begins as the lagless forecast. The
penalty on its outputs (1e-3, 1e-2, 1e-1, and 1) never limited them, so all four penalties give
nearly the same forecast. NGED's event labels are incomplete, so "rows with no labelled event" may
hold real events that nobody logged. NRA means normal running arrangement. Every number is interim
and unaudited. The scores on rows with no labelled event measure forecast quality, and the scores
inside real events measure robustness to wrong lags.

![Absolute CRPS by arm](figure_1_scores.svg)

![Day-ahead forecast at D5, B-mix and C1e-2](figure_2_example_day_ahead.svg)

![Ten-day forecast at D5, B-mix and C1e-2](figure_3_example_14_day.svg)

![Six real events, B-mix, C1e-2, and the anchor](figure_4_real_events.svg)

## Results

Scores are points of effective capacity (CRPS, lower is better), raw quantiles, metered actuals
unless stated, and the mean over 5 scored months and fit seeds 0 and 1. Intervals are 95%
month-block intervals. Experiment 454's minimum detectable effect at the signed-MW six is 0.24 on
rows with no labelled event and 0.38 inside events.

- **Arms.** LL is the lagless forecast alone (the anchor). C1e-3, C1e-2, C1e-1 (penalty on the
  squared outputs), C1 (penalty 1, post hoc), and Z1e-2
  (post hoc control: the same network with the lag and surprise inputs set to zero, leaving the
  lead features). References: LL, B-mix, S1 (a neural network that reads the lags and the surprise
  at each lag), S1-mix (S1 mixed 50/50 with LL), and A0-inj (a neural network that reads the lags
  only).
- **Relation to A0-inj.** A0-inj has no lagless input and re-learns the weather response. The
  correction sits on the frozen anchor, and the correction is the only new part.
- **Signed-MW six, rows with no labelled event.**
    - C1e-2 3.212, C1 3.206, C1e-1 3.205, C1e-3 3.212, Z1e-2 3.611, LL 3.708, B-mix 3.498, S1-mix
      3.435, S1 3.736, A0-inj 3.851.
    - C1e-2 minus LL -0.496 (-0.865 to -0.254, 5 of 5 months); minus B-mix -0.286 (-0.807 to +0.195,
      4 of 5 months). By month against B-mix: -0.01, +0.25, -0.87, -0.55, -0.24 (2025-10, 2025-12,
      2026-01, 2026-03, 2026-06). Leaving out 2026-01 leaves -0.14; leaving out 2025-12 leaves
      -0.42.
    - Z1e-2 minus LL -0.097 (-0.200 to -0.019): recalibrating the spread by lead time gains 0.10,
      and
      the lags gain a further 0.40.
- **Signed-MW six, inside real events.** On metered actuals C1e-2 is 5.440 against 6.158 for LL and
  5.932 for B-mix. On NRA-corrected actuals (the headline for events) C1e-2 is 14.864 against 14.822
  for LL (+0.042, -0.509 to +0.911), 14.641 for B-mix (+0.223, -0.259 to +1.127), and 14.543 for
  Z1e-2. The lags carry the level shift of the temporary running arrangement, so they stop helping
  once
  the target is the normal running arrangement.
- **MVA-only eight.** Rows with no labelled event: C1e-2 4.179, LL 4.523, B-mix 4.349, Z1e-2 4.498.
  Metered inside events: C1e-2 7.800, B-mix 7.607 (+0.194, -0.547 to +0.793). NRA events: C1e-2
  10.101, B-mix 10.603 (-0.502, -0.666 to -0.332).
- **Lead band 12 to 36 hours.** Signed-MW six, no labelled event: C1e-2 2.995, LL 3.654, B-mix 3.392
  (-0.397, -0.812 to -0.139). NRA events: C1e-2 15.423 against LL 15.189 and B-mix 15.081.
- **Tail calibration (signed-MW six, no labelled event, raw).** Coverage at q95 and q99: C1e-2
  0.946 and 0.989, LL 0.839 and 0.943, B-mix 0.858 and 0.951, Z1e-2 0.926 and 0.986. Pinball loss at
  q95 and q99: C1e-2 0.00698 and 0.00204, B-mix 0.00842 and 0.00254. After the table-P recalibration
  (4 months, because it is fitted on earlier months) the CRPS is
  3.327 (C1e-2), 3.674 (B-mix), 3.878 (LL), 3.769 (Z1e-2). Much of the gain over B-mix is the upper
  tail of the anchor being under-dispersed, and Z1e-2 also fixes part of this under-dispersion.
- **Stop rule.** The stop rule did not fire: the share of rows with a shift below 0.01 of capacity
  is 0.29 at the signed-MW six for penalties 1e-3 to 1e-1 and 0.32 at penalty 1. The mean log-scale
  change is +0.16 to +0.26, so the correction widens the anchor.
- **Avoidable-curtailment cost** (experiment 449's cost, millions of pounds, signed-MW six, fit
  seeds
  0 and 1, widened by 1.15; interval from resampling month-by-Primary cells). At f = 0.9, 0.75, and
  0.5: C1e-2 2.23, 8.08, 24.4; B-mix 2.08, 9.20, 28.7; LL 2.34, 10.15, 30.2; manual_heuristic 2.02,
  12.9, 37.8; BG2 1.37, 8.24, 25.9. C1e-2 minus manual_heuristic is -13.4 (-26.6 to -3.0) at
  f = 0.5, and C1e-2 minus Z1e-2 is -3.3 (-5.8 to -1.5) at f = 0.5. The other contrasts at f = 0.9,
  0.75, and 0.5 include zero.
- **Decision rule.** C1e-2 passes the first part at the signed-MW six: it beats the anchor on
  rows with no labelled event by more than the larger of 0.03 and the minimum detectable effect
  (0.24). It fails the second and third parts, the real-event gate on NRA-corrected actuals: +0.042
  against LL (limit 0.03) and +0.223 against B-mix (limit 0.03). Verdict: partly supported.
- **Fits.** 50 correction fits of 16 to 28 s each (30 pre-registered, 20 post hoc), one GPU process
  under 8 GB, monthly retraining, about 25 minutes in total. The networks are saved under
  `data/ml-experiments/00467-*/work/models/`.
- **Rows and reuse.** Experiment 404's rows and injected worlds (training worlds 100 and 101);
  scoring on the real world only (stored B-mix, LL, S1, S1-mix, and A0-inj of experiment 454 exist
  for it). Flagged exact-zero half-hours are dropped; the start cut applies to D15 and D18, which
  are report-only.
- **Reviews.** A fresh Sonnet reviewer found no leakage in the code and corrected the plan's seed
  notes. The second reviewer reproduced the headline numbers independently and asked for
  the control with the lag inputs zeroed (Z1e-2) and for a penalty that binds (C1), both added and
  labelled post hoc.

## Checks

- **Triggers.** Trigger 2 fired: the result contradicts the prediction that C1e-2 stays behind
  B-mix on rows with no labelled event. Triggers 3 and 4 did not fire. An audit by a fresh Sonnet
  agent re-derived the headline from the stored forecasts, and each finding is triaged below.
- **Median and mean forecast error beside the quantile score** (signed-MW six, rows with no
  labelled event, C1e-2 minus reference, points of capacity; the audit averaged all 3 stored seeds
  of LL and B-mix, so its CRPS differs from this page's by up to 0.05). Against LL: CRPS -0.55,
  median error -0.77, mean error -0.74. Against B-mix: CRPS -0.30, median error -0.35, mean error
  -0.43. The signs agree in every month against LL and in every month except 2025-10 and 2025-12
  against B-mix, where C1e-2 trails slightly on all three measures.
- **Median minus mean forecast per Primary.** C1e-2 tracks LL within 0.06 points at every one of
  the six Primaries, so the correction does not shift the median away from the mean. B-mix differs
  at D11 (-0.75 against +0.14 to +0.17) and D5 (-0.38 against -0.16 to -0.21).
- **Bias.** The mean of actual minus median is positive at five of the six Primaries for every arm
  (the forecasts are low). C1e-2 roughly halves the mean absolute bias (1.30 points against 2.55 for
  LL and 2.50 for B-mix), and Z1e-2 only reduces it to 2.17. Much of the lag gain is therefore a
  level correction.
- **Leave one Primary out** keeps the sign: C1e-2 minus LL stays between -0.38 and -0.66, and C1e-2
  minus B-mix between -0.11 (D14 left out) and -0.46. D14 carries most of the gain over B-mix, and
  without it the margin falls below the minimum detectable effect of 0.24. D11 is the one Primary
  where C1e-2 is worse than B-mix (+0.40), and the gap grows with lead time.
- **By lead band and hour.** C1e-2 beats B-mix in each lead band pooled (-0.25 to -0.39) and in all
  24 hours of the day. It trails B-mix in three month-by-band cells: 2025-12 at 36 to 120 hours
  (+0.06) and at 120 to 240 hours (+0.36), and 2025-10 at 120 to 240 hours (+0.14).

## Caveats and next steps

- **The penalty sweep is uninformative.** At 1e-3 to 1 the L2 penalty shrinks the outputs by under
  5%, so it does not separate the arms. A penalty of 10 or more, or a per-row penalty that is larger
  inside events, is the next test.
- **The Bonferroni count is 15** per reference in the contrast tables (5 arms times 3 row classes);
  the pre-registered count was 9 (3 penalties times 3 row classes). No gain in this page is
  claimed at the 99% level.
- **Not done:** the drop-largest-event interval, leave-one-Primary-out (infeasible with 21 events
  at the six), and the wind and solar versions (they need the stored `member_plain` and `g0_onepass`
  heads in their own harnesses). The demand version passes the no-labelled-event gate only, so the
  manager decides whether to run them.
- **Porting.** It would need one lagless neural network plus a 32-unit correction, instead of the
  two neural networks that B-mix needs.
