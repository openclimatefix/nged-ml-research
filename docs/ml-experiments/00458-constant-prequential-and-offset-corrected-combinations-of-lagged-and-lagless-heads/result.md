---
id: "00458"
slug: constant-prequential-and-offset-corrected-combinations-of-lagged-and-lagless-heads
title: >-
  Can a fitted weight, a prequential weight, or a shrunk level offset on the lagless forecast beat
  the fixed 50/50 mixture of the lagged and lagless forecasts?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one folder of per-row mixture integrals, weight tables, and scores in this
  experiment's private data folder; no fit and no new forecast. Production serving path: not
  touched (this repository holds none). Degradation rule: not touched. Designs: three families of
  combination are named in the brief, each fixed here before any score is read. Callers: the
  scripts import experiment 404's modules, experiment 402's mixture algebra, and experiment 435's
  tail helpers.
hypothesis: >-
  A better combination of the stored lagged head and the stored lagless head than the fixed 50/50
  mixture B-mix (a weight per lead band, a weight from the heads' recent losses, or the lagless
  head plus a shrunk recent level offset) matches B-mix inside real load-shift events and loses
  nothing on rows with no labelled event, so an explicit event detector is unnecessary for the
  forecast of the normal running arrangement (NRA).
prediction: >-
  Written before any score is read. Scores are CRPS in points of effective capacity. P1. At the
  signed-MW six, the fitted lagged weight of arm (b) is above 0.5 in the 0-12 h band and below 0.5
  in days 6-10, and (b) beats B-mix on rows with no labelled event by at most 0.05 points
  (experiment 402 step 5 found a fitted constant scoring 4.637 against 4.605 for 50/50, so the
  stop rule per band may still fire if a band weight lies within 0.05 of 0.5). P2. The prequential weights of arm (c) stay within 0.05 of
  0.5 on at least 90% of rows, because the two heads' 7-day losses differ little. P3. The shrunk
  offset k of arm (d) lies below 0.3, and (d) scores worse than B-mix on rows with no labelled event
  by more than 0.1, because a lagless forecast alone trails B-mix by about 0.3 there. P4. No
  combination meets the candidate rule at the signed-MW six or at the MVA-only eight.
decision_rule: >-
  A combination is a candidate if, against B-mix on the same rows, its CRPS on real-event rows at
  the signed-MW six (NRA-corrected actuals at depth factor 1.0) is lower by more than the larger of
  0.03 and the minimum detectable effect, it is no more than 0.03 worse on rows with no labelled
  event, and its q99 pinball loss is no more than 0.02 worse. Stop rule: if the weights fitted in
  (b) lie within 0.05 of 0.5 and the weights in (c) stay within 0.05 of 0.5 on 90% of rows, B-mix's
  50/50 is near-optimal; if k in (d) is within 0.05 of 0, the lagless forecast already does the
  work. The minimum detectable effect is 2.8 times the standard error of the monthly differences
  between B-mix and the lagged head (experiment 455's formula; with only four scored months it is
  noisy, so the seed-to-seed spread of B-mix in 2026-06 is reported beside it), computed before any
  candidate contrast is read. The stop rule is applied per lead band. A candidate must also have a
  95% month-block interval that excludes zero on real-event rows (Bonferroni interval reported
  too). The 0.03 floor is the smallest gain worth porting for every arm: (b) adds only four numbers
  per group, but (c) and (d) add a recent-actuals feed, and 0.03 points is the margin the brief
  names for all three. Say plainly if a combination matches B-mix at lower
  production cost than two neural networks, as (d) would.
benchmark: >-
  B-mix, experiment 404's stored fixed 50/50 mixture of A0-inj and the lagless table, on identical
  rows. S1 with the same lagless table at 50/50 is a second reference for the S1 pair.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from the 11 quantiles of each
  mixture, as in experiment 404's report; weights are fitted on the exact CRPS of the piecewise
  linear mixture (experiment 402's algebra).
code: ml-experiments/00458-constant-prequential-and-offset-corrected-combinations-of-lagged-and-lagless-heads/
---

**Plan only: no score has been read.** The question is whether anything cheaper than two networks
and a fixed 50/50 split reaches B-mix's score inside real load-shift events. NGED's event labels
are incomplete, so "rows with no labelled event" may hold real events that nobody logged, and
every real-event slice undercounts.

## Design

- **Stored inputs.** Experiment 404's lagged head A0-inj (the harness's stand-in for R-inj), its
  S1 head, the lagless table of experiment 445 (`O-lagless`, the same frozen cross-fitted table
  that B-mix uses), and B-mix. Five months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06; seeds 0
  to 2 in 2026-06), 16 Primaries, one origin per day, leads 0 to 240 hours. Each head is a
  cumulative distribution function that is linear between its 11 stored quantiles; mixtures use
  experiment 402's `mixture_of` algebra. A fidelity check mixes A0-inj and the lagless table 50/50
  and must reproduce the stored B-mix quantiles.
- **Arms.** (a) B-mix. (b) One lagged weight per group and lead band (the five bands of table P),
  fitted on the months before the scored month by minimising the exact mixture CRPS, which is
  quadratic in the weight. (b2), a control: one weight per group over all leads (experiment 402's
  constant). (c) Prequential weight at each origin and Primary: the softmax of the negative mean
  pinball loss of each head over the previous 7 days of actuals delivered at the issue time (6-hour
  delay; each target is scored by the forecast from the latest earlier origin), temperature fitted
  per group on earlier months, each weight floored at 0.2. (d) The lagless forecast plus k times
  the median residual of the lagless forecast over the last 24 hours of delivered actuals, added
  to all 11 quantiles, k fitted per group and lead band on earlier months. Arms (b) and (c) run
  with A0-inj and again with S1 as the lagged head. Arm (d) does not use the lagged head, so one
  run serves both pairs.
- **Variants and rows.** Only the real-world files (`w-1`) are used; the injected worlds are not
  read, so no injected window enters a fit. Scoring uses leads of 0 to 240 hours (days 1 to 10; the
  stored files run to 351 hours), with the 12-36 h band reported apart. Table P's five lead bands
  are 0-12 h, 12-36 h, 36-72 h, 72-120 h (days 4-5), and 120-240 h (days 6-10). The stored rows
  already omit every half-hour that the cleaned power table lacks (checked: no scored row of
  2025-10, 2026-01, or 2026-06 is missing from it and none is an exact zero), so flagged zeros are
  dropped; experiment 399's start cut for D15 and D18 falls before every scored month.
- **Delivery rule for (c) and (d).** An actual is usable at origin o if its target is at or before
  o minus 6 hours. Each usable target is scored by the forecast of the latest origin at or before
  the target (leads up to 24 hours), against the metered actual, never the NRA-corrected one. The
  windows read the stored forecasts of all months as one series, so 2025-12 to 2026-01 are
  adjacent while the first days of 2025-12, 2026-03, and 2026-06 have no earlier forecasts. A
  Primary-origin with fewer than 48 usable targets in the window gets weight 0.5 in (c) and offset
  0 in (d); the share of such rows is reported.
- **Fitting objectives and thin training.** Weights in (b) and the temperature in (c) minimise the
  mean exact mixture CRPS; k in (d) minimises the mean pinball loss in points. This replaces the
  brief's maximum likelihood because the stored heads are quantile forecasts with no density; the
  CRPS is quadratic in the weight. The lagless table is cross-fitted out of fold, so it holds no
  scored-month target. 2025-12 is fitted on 2025-10 alone and 2026-03 on three months, so the
  first scored months have thin training and are flagged in the weight table.
- **Months.** The first month (2025-10) has no earlier month: it is scored at 50/50 as a warm-up
  and left out of every pooled mean, for B-mix too, so all arms share rows. Training rows for
  month m exclude every target on or after the first day of m. 2026-06 weights are fitted on seed 0
  of the earlier months and applied to each seed. Months are never pooled across groups: the
  signed-MW six and the MVA-only eight are separate fits and separate headlines. D15 and D18 are
  report-only.
- **Rows.** Rows with no labelled event (not inside, not within 96 half-hours after, and no lag
  inside a padded real event), harm rows, rows inside real events (scored against NRA-corrected
  actuals at depth factors 0.75, 1.0 and 1.25, headline 1.0, and against metered actuals), and rows
  after events. The rows that the label-free step scan (experiment 456) finds step-free are a
  second row set once that scan has reported. Cold-night rows are reported apart.
- **Scores.** CRPS; coverage and pinball loss at q01, q05, q95, and q99, raw and after table P
  (experiment 435's per-group, per-lead-band factors refitted on earlier months for each arm);
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six, beside the
  differences against B-mix. Intervals: month-block bootstrap plus the sign count over months,
  events resampled, and the largest event of each month dropped. Bonferroni family: 5 arms (b) and
  (c) with two lagged heads and (d), against B-mix, on two row sets (real-event rows, rows with no
  labelled event), at two groups = 20 contrasts. Contrasts against the lagged head alone and the
  lagless head alone, the control (b2), and the pooled-group control (b3) are report-only and not
  counted. Leave-one-month-out estimates and absolute scores accompany every contrast, and months
  with a single real event are shown apart.
- **Stability.** The table of fitted weights by month, lead band, and group, with the range across
  months; a weight that swings by more than 0.2 between months is not usable.
- **Compute.** Stored forecasts only, one to two CPU cores, no GPU, 0 fits. The time of the first
  arm and the total are reported in the write-up, and the manager is messaged at plan, code
  review, results, and merge. Plots: a day-ahead and a 14-day probabilistic series of (a) and the
  best other arm at one real labelled event at the six and one stretch with no labelled event.
- **Deviation from the addendum.** The XGBoost R0 pair of experiments 392 and 402 and the
  encroachment sweep are not part of this plan, which cuts the addendum's 12 contrasts to the
  neural pair's 20 above (a different family); they need the actuals and event masks joined to those forecasts and are a
  follow-up if this plan finds headroom.

## Results

To be written.
