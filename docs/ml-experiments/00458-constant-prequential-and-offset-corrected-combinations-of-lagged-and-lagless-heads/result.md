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
status: finished
date: 2026-11-01
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
  stop rule per band may still fire if a band weight lies within 0.05 of 0.5). P2. The prequential
  weights of arm (c) stay within 0.05 of
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
verdict: not supported
best_arm: >-
  No combination beat the 50/50 mixture. At the signed-MW six inside real events (NRA-corrected
  actuals, depth factor 1.0) arm (b) with A0-inj scores 14.268 against 14.262 for 50/50 (+0.006,
  95% interval -0.005 to +0.023, minimum detectable effect 0.51).
headline: >-
  A weight per lead band, a weight from the heads' last 7 days of losses, and a shrunk offset on the
  lagless forecast all fail to beat the fixed 50/50 mixture of the lagged and lagless forecasts. At
  the six Primaries metered in signed megawatts, the two weight rules differ from 50/50 by no more
  than +0.09 points of capacity inside real events, and the offset rule is 1.21 points worse there.
  Interim and unaudited.
lessons: >-
  The best weight on the lagged head falls with lead time (about 0.75 at 0 to 12 hours, about 0.5 at
  days 6 to 10), but it swings by 0.2 or more between months, so a weight fitted on earlier months
  scores no better on a new month than 50/50. The offset rule is the one cheap rule that helps on
  rows with no labelled event (-0.23, interval -0.58 to +0.09), and it loses 1.21 points inside real
  events, because it follows the level shift.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  No simple rule for combining the lagged and lagless forecasts beats the fixed 50/50 mixture, so
  keep 50/50 and do not spend 50 network fits on a weight per lead band. The weight that fits best
  falls from about 0.75 on the lagged forecast at short leads to about 0.5 at days 6 to 10, but it
  moves by 0.2 or more from month to month. A shrunk recent-level offset on the lagless forecast
  helps on rows with no labelled event at the six (-0.23 points, not significant) and costs 1.21
  points inside real events. Interim and unaudited, four scored months.
advice:
  - >-
    Keep the fixed 50/50 mixture as the combination; a fitted or prequential weight adds nothing
    that four scored months can detect (experiment 458).
  - >-
    Do not run the 50-fit weight-per-lead-band experiment: the fitted weights are unstable across
    months and score no better held out (experiment 458).
code: ml-experiments/00458-constant-prequential-and-offset-corrected-combinations-of-lagged-and-lagless-heads/
---

**No combination rule beat the fixed 50/50 mixture of the lagged and lagless forecasts (B-mix).**
The lagged weight that fits best falls with lead time, but the fitted weight moves by 0.2 or more
between months, so weights fitted on earlier months do not generalise. A shrunk recent-level offset
on the lagless forecast (arm d) is better on rows with no labelled event and much worse inside real
events. NGED's event labels are incomplete, so "rows with no labelled event" may hold real events
that nobody logged, and every real-event slice undercounts. Every number is interim and unaudited.
It bears on the forecast scores and, through the inside-event rows, on robustness to bad inputs.

![Each combination minus 50/50 at both groups](figure_1_contrasts.svg)

![Fitted weight of arm (b) by lead band and month](figure_2_fitted_weights.svg)

![Day-ahead forecast across a real event at D5](figure_3_day_ahead_event.svg)

![Ten-day forecast across the same event](figure_4_ten_day_event.svg)

![Day-ahead forecast at D5 with no labelled event](figure_5_day_ahead_no_label.svg)

![Ten-day forecast at D5 with no labelled event](figure_6_ten_day_no_label.svg)

## Results

Scores are points of effective capacity; a contrast is the arm minus the 50/50 mixture of the same
lagged head and the lagless head, so negative is better. Months are the resampling unit (4 scored
months; seeds averaged within 2026-06; the warm-up month 2025-10 is excluded for every arm).
Intervals are 95% month-block intervals; the Bonferroni interval for 20 contrasts is in the tables
of the experiment's private folder and in the figure.

- **Arms.** (a) B-mix. (b) one lagged weight per group and lead band, fitted by minimum exact CRPS on
  earlier months. (c) prequential weight from each head's 7-day loss, floor 0.2, softmax temperature
  fitted on earlier months. (d) lagless forecast plus k times its median residual over the last 24
  hours of delivered actuals. Lagged heads: A0-inj and S1 of experiment 404; lagless head: the
  frozen table of experiment 445. Arm (d) uses no lagged head.
- **Reference.** The stored B-mix mixes Gaussian-mixture heads; the arms mix the stored 11-quantile
  heads. A recomputed 50/50 mixture ("half") scores within 0.001 to 0.02 points of the stored
  B-mix, so contrasts use the recomputed 50/50 (same algebra). Contrasts against the stored B-mix
  are in `contrasts.csv`.
- **Signed-MW six, inside real events (NRA depth 1.0; 50/50 scores 14.262 with A0-inj, 14.267 with
  S1; minimum detectable effect 0.51 and 0.63).** (b) A0-inj +0.006 [-0.005, +0.023]; (b) S1
  +0.043 [-0.000, +0.119]; (c) A0-inj +0.085 [-0.009, +0.267]; (c) S1 +0.069 [-0.004, +0.142]; (d)
  +1.210 [+0.145, +2.276]. A0-inj alone is +0.396 and the lagless head alone +0.120.
- **Signed-MW six, rows with no labelled event (50/50 scores 3.680 and 3.638; minimum detectable
  effect 0.40 and 0.36).** (b) A0-inj +0.054 [+0.010, +0.099]; (b) S1 +0.072 [+0.000, +0.144]; (c)
  A0-inj -0.019 [-0.052, +0.006]; (c) S1 -0.037 [-0.117, +0.013]; (d) -0.234 [-0.584, +0.091], with
  2 of 4 months negative. The lagless head alone is +0.306.
- **MVA-only eight.** Inside real events (3 months with events): (b) +0.031 and +0.070, (c) +0.069
  and +0.066, (d) +0.251 (all intervals include zero except (b) S1, +0.024 to +0.118). Rows with no
  labelled event: every arm within 0.065 of 50/50 and every interval of (b), (c), and (d) includes
  zero.
- **Candidate rule.** No arm is a candidate: none lowers the inside-event score by more than the
  larger of 0.03 and the minimum detectable effect (0.51 at the six). (b) is also 0.054 and 0.072
  worse than 50/50 on rows with no labelled event, over the 0.03 allowance.
- **Stop rules.** The fitted weights of (b) at the six lie between 0.36 and 0.95 (within 0.05 of 0.5
  in none of the lead bands in every month), so the "50/50 is near-optimal" rule does not fire
  literally. Held out, however, they score no better than 50/50. The prequential weights stay
  within 0.05 of 0.5 on 18% to 82% of rows at the six (below 90%). k in (d) is 0.5 to 0.95 at the
  six (not near 0), so the lagless forecast does not already do the work.
- **Weight stability.** Arm (b) at the six: 0-12 h weight 0.68 to 0.81 and days 6-10 weight 0.43 to
  0.62 across the four fitted months; the same lead band moves by 0.2 or more between 2025-12 and
  2026-01 (figure 2). Rows in 2025-12 are fitted on 2025-10 alone, so that fit is thin.
- **Tails at the six, rows with no labelled event.** q95 coverage is 0.860 for B-mix and 0.863 and
  0.863 for (b) and (c); table P lifts it to about 0.944. q99 coverage 0.952, then 0.985 to 0.986
  after table P. Arm (d) covers q01 at 0.017 and q05 at 0.060 raw, above the nominal 0.01 and 0.05.
  Inside real events q99 coverage is 0.84 for 50/50, (b), and (c), and 0.866 for (d); q99 pinball
  loss differs from 50/50 by under 0.001 for (b) and (c).
- **Production cost.** Arm (d) needs only the lagless head and 24 hours of delivered actuals, so it
  costs less than two neural networks; it does not match 50/50 inside real events.
- **Checks.** A recomputed 50/50 reproduces the stored B-mix score to 0.001 (2026-06) and about 0.01
  elsewhere. A fresh reviewer re-derived the 2026-01 B-mix score (3.9628, 59,530 rows) and found no
  leakage in (d): moving its offset window back 0, 2, 4, and 14 days raises the score 3.160, 3.308,
  3.484, and 4.100, so the gain is a persistent bias of the lagless table, and it is not uniform
  (worse than B-mix in 2025-12 and 2026-06). Rows already omit half-hours that the cleaned power
  table lacks, so flagged zeros are dropped; experiment 399's start cut falls before every scored
  month. No new forecast is a trained model, so the median-minus-mean check does not apply.
- **Predictions.** P1 partly holds: the weight falls from about 0.75 at 0-12 h to about 0.5 at days
  6-10, but (b) is 0.054 worse than 50/50 on rows with no labelled event, not better. P2 fails (18%
  to 82% of rows within 0.05). P3 fails: k is 0.5 to 0.95, and (d) is better, not worse, on rows
  with no labelled event. P4 holds.

## Caveats

- **Four scored months and 3 to 4 months with events**, so the minimum detectable effect is 0.4 to
  0.6 points at the six and the Bonferroni intervals are nearly the range of the monthly values.
- **Unlogged events.** Rows with no labelled event may hold unlabelled real events, so the (d) gain
  on those rows may be partly a level offset that follows an unlogged event.
- **Not run.** Avoidable-curtailment cost at f = 0.9, 0.75, and 0.5, cold-night rows, the rows that
  the label-free step scan (experiment 456) finds step-free, months with a single real event shown
  apart, the XGBoost R0 pair, and the encroachment sweep. The curtailment and cold-night code reads
  frames of experiments 448 and 449's harness, which these stored forecasts do not share.
- **Seeds.** Seeds 0 to 2 only in 2026-06 and seed 0 in the other months; experiment 454's new
  seeds are not yet scored.
- **Fit objective.** Weights minimise the exact CRPS of the piecewise-linear mixture, not a
  likelihood, because the stored heads are 11 quantiles.

## What to try next

- Re-score on experiment 454's seeds 0 to 4 when its heads land, and add the curtailment cost.
- If a combination is wanted, test arm (d)'s offset on the lagged head's forecast as well; a
  persistent bias in the lagless table is what it corrects.
