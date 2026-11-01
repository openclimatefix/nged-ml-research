---
id: "00463"
slug: lag-trust-cross-fitted-trust-fit-seed-zero
title: >-
  Does fitting the trust weight on out-of-fold lagged heads close the gap between a learned trust
  weight and the fixed 50/50 mixture?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: finished
date: 2026-11-01
kind: other
protocol: demand-v2
size: >-
  complex. Stored: 30 out-of-fold lagged-head networks (not kept), 10 trust networks, and 5 months
  of forecasts and learned weights in this experiment's private data folder; every earlier
  experiment is read and never written. More than one design would test the hypothesis (which head
  is the lagged head, the number of folds, whether the trust head or a constant weight is fitted),
  so each choice is fixed here before any score is read. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Callers: the scripts import the modules of
  experiments 87, 48, 397, and 404; the scoring and weight-reading helpers are written new for this
  experiment, adapted from experiments 455 and 458 (458 is unmerged, so nothing is imported).
hypothesis: >-
  Experiment 404's learned trust weight (B1) loses to the fixed 50/50 mixture (B-mix) partly
  because the lagged head is fitted in sample on the same rows where the trust weight is fitted,
  while the lagless table is out of sample on those rows (mechanism M11). A trust weight fitted on
  out-of-fold lagged heads and the cross-fitted lagless table, so that both heads are out of sample
  on the rows where the trust is fitted, matches B-mix on rows with no labelled event and beats it
  inside real load-shift events.
prediction: >-
  Written before any score is read. P1. The learned lagged weight (the product of the three
  per-lag trust weights) falls from about 0.75 at 0-12 h to about 0.5 by days 6-10, as experiment
  458's fitted weights did, and swings by 0.2 or more between months. P2. G1-trust does not beat
  B-mix: G1-trust minus B-mix on rows with no labelled event at the signed-MW six lies between 0.00
  and +0.10, and on real-event rows it lies within the minimum detectable effect of zero. P3.
  G1-trust is better than G1-insample (the same frozen-head trust fit on the in-sample lagged head)
  on rows with no labelled event, by 0.05 to 0.3 points at the signed-MW six, because the
  out-of-fold lagged head is less over-confident; this contrast is the direct test of M11. P4.
  Seed 1 is not run, because the weight swings by more than 0.1 between months.
decision_rule: >-
  Adoption verdict. G1 fails (close learned trust in the 404 harness and keep B-mix as the
  reference) if G1-trust minus B-mix on rows with no labelled event is above +0.03 points at the
  signed-MW six, or if its CRPS on real-event rows (NRA-corrected actuals, depth factor 1.0,
  headline; metered actuals reported beside it) is not lower than B-mix's by more than the minimum
  detectable effect. G1 passes only if neither fail condition holds and its leave-one-month-out
  interval on real-event rows excludes zero; nothing is adopted unless G1 passes at the signed-MW
  six and is no worse than +0.03 at the MVA-only eight. Any other outcome is inconclusive and
  nothing is adopted. Mechanism verdict, separate from adoption. M11 is supported if G1-trust beats
  G1-insample on rows with no labelled event by more than 0.03 points at the signed-MW six, and the
  learned lagged weight of G1-trust is nearer 0.5 than that of G1-insample; a G1-trust that beats
  G1-insample but still loses to B-mix supports M11 and still fails adoption. Intervals. Months are
  the resampling unit (5 months, so about 4 degrees of freedom). The 95% interval is reported, and
  the Bonferroni interval at level 1 - 0.05/20 (20 pre-registered contrasts) beside it. The
  minimum detectable effect is 2.8 times the standard error of the monthly differences between
  B-mix and A0-inj on the same rows, computed before any G1 contrast is read; it approximates the
  noise of a G1 contrast (the t quantile at 4 degrees of freedom is 2.78, so 2.8 is kept), and the
  standard error of each contrast's own monthly differences is reported beside it. The weight curve
  alone is a diagnostic and decides nothing. Seed 1 runs only if the learned weight swings by less
  than 0.1 between months.
benchmark: >-
  B-mix, experiment 404's stored fixed 50/50 mixture of A0-inj and the cross-fitted lagless table,
  on identical rows. B1, experiment 404's stored in-sample learned trust, is the second reference
  (the in-sample twin of G1-trust).
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from the 11 quantiles of
  each forecast, as in experiment 404's report, on the real-data world (world -1) with leads of 0
  to 240 hours (lead days 1 to 10).
verdict: not supported
best_arm: >-
  No learned trust beat the 50/50 mixture. At the six substations metered in signed megawatts, on
  rows with no labelled event, G1-trust scores 3.492 against 3.513 for 50/50 (-0.021, 95% interval
  -0.061 to +0.015); inside real events (actuals corrected to the normal running arrangement, depth
  factor 1.0) it scores 13.198 against 13.139 (+0.059, 95% interval +0.027 to +0.102, minimum
  detectable effect 0.38).
headline: >-
  Fitting the trust weight on lagged forecasts from neural networks that did not see the fitting
  rows closes the gap between a learned trust weight and the fixed 50/50 mixture on rows with no
  labelled event (+0.14 points for the same weight fitted in sample, -0.02 out of fold), but the
  learned weight never clearly beats 50/50: it is 0.06 points worse inside real events at the six
  signed-megawatt substations and 0.05 to 0.13 worse at the eight MVA-only substations (the metered
  event rows there are +0.04, interval spanning zero). Interim: five scored months, one seed.
lessons: >-
  The experiment 404 learned trust lost to 50/50 at least partly because the lagged forecast was
  fitted on the rows where the trust was fitted, so the trust saw a lagged forecast far better on
  those rows than on new months. Removing that mismatch removes the loss on rows with no labelled
  event (the gap of experiment 404's own trust to 50/50 had an interval spanning zero) and leaves a
  learned weight (about 0.4) that is no better than the fixed 0.5. The remaining gap inside events
  is small, and the stored data cannot say whether a better-matched lagged forecast would close it.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  The learned trust weight of experiment 404 lost to the fixed 50/50 mixture at least partly because
  the lagged forecast was fitted on the same rows as the weight. When the weight is fitted on
  lagged forecasts from neural networks that did not see those rows, the learned weight matches the
  50/50 mixture on rows with no labelled event (-0.02 points of effective capacity at the six
  signed-megawatt substations) but does not beat the 50/50 mixture inside real events (+0.06
  points), so keep the fixed 50/50 mixture. Interim: five scored months, one seed, and National
  Grid Electricity Distribution's (NGED's) event labels are incomplete, so rows with no labelled
  event may hold unlogged events.
advice:
  - >-
    Keep the fixed 50/50 mixture as the reference for combining lagged and lagless forecasts; a
    trust weight fitted out of fold only matches it (experiment 463).
  - >-
    Fit any learned combination weight on out-of-fold forecasts of both forecasts, never on a
    lagged forecast fitted on the same rows: the in-sample version scored 0.16 to 0.19 points worse
    on rows with no labelled event (experiment 463, G1-trust against G1-insample).
  - >-
    Do not run seed 1 of the cross-fitted trust fit: the learned weight swings by 0.18 to 0.24
    between months, above the 0.1 rule (experiment 463).
  - >-
    A lagged forecast from a network trained on 5/6 of the rows is much weaker than the full-data
    forecast used at test time (negative log-likelihood 0.9 against 0.2 in sample), so the learned
    weight sits below the best test-month constant; test a finer fold split before trusting any
    weight curve (experiment 463 audit).
code: ml-experiments/00463-lag-trust-cross-fitted-trust-fit-seed-zero/
---

**Fitting the trust weight on lagged forecasts that did not see its fitting rows removes the learned
weight's loss to the fixed 50/50 mixture on rows with no labelled event, but the learned weight does
not beat 50/50 inside real events.** Experiment 404 combined a lagged forecast (one that reads the
recent power readings, the lags) with a lagless forecast (one that does not), either by a fixed
50/50 average or by a learned trust weight that sets how far to trust the lagged forecast. The
learned weight lost to 50/50. We asked whether it lost because the lagged forecast had been fitted
on the same rows as the weight, so that the lagged forecast looked better on those rows than it is
on new months. For each of five scored months, we split the two-year training window into six
consecutive blocks and trained six lagged-forecast neural networks, each leaving out one block.
Every training row then has a lagged forecast from a neural network that did not see it. We fitted
the trust weight on those forecasts, beside the lagless forecasts that were also made without the
row, and scored it on the scored month. A control fits the same weight on lagged forecasts that did
see the rows (G1-insample), which shows how much worse the score is because of the mismatch.
NGED's event labels (the load shifts it has logged) are incomplete, so rows with no labelled event
may hold unlogged events and the real-event rows undercount. The numbers are interim: five scored
months and one seed, and the choice of fold split is untested.

**The experiment fails its adoption rule, and the in-sample mismatch accounts for part of the earlier
loss.** Inside real events the learned weight is worse than 50/50, not better. Fitting out of fold
gains 0.16 points over fitting in sample on rows with no labelled event at the six signed-megawatt
substations, and 0.19 at the eight MVA-only substations. The 95% interval of the first gain barely
excludes zero and its Bonferroni interval spans zero, so the mechanism rests on the pre-registered
point-estimate rule and on the eight-substation result. The search narrows to a better-matched
lagged forecast (a finer fold split, so the out-of-fold neural network is nearer the full-data one),
or to keeping 50/50.

![Figure 1: G1-trust minus each reference, by group and row set; left of the dashed line favours G1-trust](figure_1_contrasts.svg)

![Figure 2: learned weight on the lagged forecast by lead band; G1-trust sits near 0.4](figure_2_learned_weights.svg)

Figures 3 and 4 show a day-ahead and a ten-day forecast from one origin across a real event at the
primary substation D5 (the grey band marks event rows). Figures 7 to 10 repeat the ten-day forecast
across real events at two more signed-megawatt substations and two MVA-only substations. The two
forecasts are close everywhere, as the small contrasts suggest.

![Figure 3: day-ahead forecast across a real event at D5](figure_3_day_ahead_event.svg)

![Figure 4: ten-day forecast, same event and origin](figure_4_ten_day_event.svg)

![Figure 7: ten-day forecast across a real event, second signed-megawatt substation](figure_7_ten_day_event_signed_b.svg)

![Figure 8: ten-day forecast across a real event, third signed-megawatt substation](figure_8_ten_day_event_signed_c.svg)

![Figure 9: ten-day forecast across a real event, first MVA-only substation](figure_9_ten_day_event_mva_a.svg)

![Figure 10: ten-day forecast across a real event, second MVA-only substation](figure_10_ten_day_event_mva_b.svg)

Figures 5 and 6 show the same forecasts at D5 with no labelled event nearby.

![Figure 5: day-ahead forecast at D5 with no labelled event nearby](figure_5_day_ahead_no_label.svg)

![Figure 6: ten-day forecast at D5, same origin](figure_6_ten_day_no_label.svg)

## Results

**Scores are in points of effective capacity, and a negative contrast favours the first arm.** A
point is 200 times the mass-weighted pinball loss over 11 quantiles. Leads are 0 to 240 hours (lead
days 1 to 10) on the real-data world. The unit of resampling is the month: five scored months
(2025-10, 2025-12, 2026-01, 2026-03, 2026-06), four where a group has no event. Intervals are 95%
month-resampled intervals; the Bonferroni interval for 20 contrasts is in the experiment's private
folder and in figure 1. Four row sets are used. No-label rows have no labelled event in the day
windows, the lags, or the target. Harm rows have a labelled event in a day window or lag and the
target outside every event. Real-event rows have the target inside a padded event or within 96
half-hours after one. NRA-corrected actuals subtract the estimated load moved by the event, so
the actual is the demand under the normal running arrangement (NRA); depth factor 1.0 is the
headline.

- **B-mix.** The 50/50 mixture of the lagged forecast A0-inj (experiment 404's plain network trained
  on injected events) and the lagless forecast; it is the reference.
- **B1.** Experiment 404's trust weight, learned jointly with the lagged forecast, in sample.
- **G1-insample.** The trust head fitted with both forecasts frozen, the lagged forecast being the
  full-data A0-inj network evaluated on its own training rows.
- **G1-trust.** The same frozen-head fit on the out-of-fold lagged forecasts.
- **Why G1-insample exists.** B1 differs from G1-trust in three ways: its lagged forecast is trained
  jointly with the trust, its lagged forecast also reads the lagless forecast, and it is fitted in
  sample. G1-trust against G1-insample isolates the third. The control was not in the brief; the
  neural Discuss session is informed here.

**Absolute scores at the six signed-megawatt substations.**

| Arm | No label | Harm | Real event, NRA 1.0 | Real event, metered |
|---|---|---|---|---|
| B-mix | 3.513 | 3.680 | 13.139 | 5.955 |
| B1 | 3.788 | 3.780 | 12.507 | 6.470 |
| G1-insample | 3.650 | 3.651 | 13.351 | 5.962 |
| G1-trust | 3.492 | 3.614 | 13.198 | 6.006 |
| A0-inj alone | 3.824 | 3.811 | 13.487 | 6.078 |
| Lagless alone | 3.763 | 3.933 | 13.243 | 6.284 |

**Contrasts at the six signed-megawatt substations, with 95% intervals.** The minimum detectable
effect is 0.32, 0.42, 0.38, and 0.26 for the four row sets in the table order.

| Contrast | No label | Harm | Real event, NRA 1.0 | Real event, metered |
|---|---|---|---|---|
| G1-trust minus B-mix | -0.021 [-0.061, +0.015] | -0.066 [-0.136, +0.003] | +0.059 [+0.027, +0.102] | +0.051 [-0.004, +0.106] |
| G1-trust minus G1-insample | -0.158 [-0.332, -0.004] | -0.037 [-0.212, +0.130] | -0.153 [-0.420, +0.008] | +0.044 [-0.067, +0.177] |
| G1-trust minus B1 | -0.296 [-0.781, +0.160] | -0.166 [-0.455, +0.064] | +0.691 [-0.311, +2.336] | -0.464 [-1.055, +0.040] |

- **Sensitivity of the headline.** Real event, NRA depth 0.75: +0.061; depth 1.25: +0.060; with the
  largest event dropped: +0.015. Leaving D5 out lowers the rows-pooled real-event contrast from
  +0.076 to +0.011.
- **B1 inside events.** B1 is the only one of the six arms better than 50/50 on real-event rows
  with NRA-corrected actuals (-0.632 [-2.281, +0.359], 3 of 5 months negative); its interval spans
  zero.
- **Eight MVA-only substations, G1-trust minus B-mix.** No label +0.051 [+0.027, +0.071]. Harm
  +0.125 [+0.056, +0.195]. Real event with NRA depth 1.0 +0.105 [+0.029, +0.181] (four months), and
  with metered actuals +0.040 [-0.044, +0.134].
- **Eight MVA-only substations, G1-trust minus G1-insample.** No label -0.193 [-0.277, -0.118].
  Harm -0.380 [-0.787, -0.083]. Real event with NRA depth 1.0 -0.788 [-1.003, -0.573].
- **Learned weight on the lagged forecast (mean over scored months).** At the six signed-megawatt
  substations, G1-trust is 0.41 at 0-12 h, 0.43 at 12-36 h, 0.43 at 36-72 h, 0.40 at days 4-5, and
  0.36 at days 6-10. G1-insample falls from 0.93 to 0.82, and B1 rises from 0.73 to 0.80. The
  range over months within a lead band is 0.18 to 0.24 for G1-trust (0.31 to 0.55 at 0-12 h),
  above the 0.1 stability rule, so seed 1 was not run. At the eight MVA-only substations, G1-trust
  falls from 0.52 to 0.45.
- **Training-row diagnostic.** The weighted negative log-likelihood of the lagged forecast on the
  training rows is 0.83 to 0.92 out of fold and 0.19 to 0.24 in sample, against 0.71 to 0.97 for
  the cross-fitted lagless forecast. An out-of-fold neural network is therefore a much weaker
  forecast than the full-data one that scores the scored month.
- **Tails at the six signed-megawatt substations, no label.** Coverage at q01, q05, q95, and q99
  is 0.006, 0.029, 0.845, and 0.943 for G1-trust, and 0.006, 0.028, 0.855, and 0.949 for B-mix. The
  q99 pinball loss is 0.0027 and 0.0026. Inside real events, q99 coverage is 0.805 for G1-trust and
  0.807 for B-mix. Every arm under-covers the upper tail (the nominal values are 0.95 and 0.99), and
  G1-trust is no different from B-mix. Curtailment cost does not apply in the 404 harness.
- **Predictions.** P1 failed on level: the weight is 0.41 falling to 0.36, not 0.75 falling to 0.5;
  the monthly swing is 0.18 to 0.24 (at least 0.2 in 2 of 5 lead bands). P2 half held: the real-event
  contrast is inside the minimum detectable effect, but its interval excludes zero, and the
  no-label contrast (-0.021) is just outside the predicted 0.00 to +0.10. P3 held (-0.158 within
  -0.05 to -0.30). P4 held.
- **Fit count and time.** 40 fits: 30 out-of-fold lagged-forecast fits at about 14 s each and 10
  trust fits at about 7 to 12 minutes each, about 1 hour 50 minutes on one GPU process under 8 GB.
- **Where the code and outputs are.** The code is in
  `ml-experiments/00463-lag-trust-cross-fitted-trust-fit-seed-zero/`. It reuses the modules of
  experiment 404 and of experiments 87, 48, and 397, and the scoring pattern of experiments 455 and
  458. Forecasts and trust networks are in the experiment's private data folder. The 30 out-of-fold
  networks are not kept; their out-of-fold forecasts are, as `oof_<month>_s0.npy`. Retraining is
  monthly. The cleaned data and the D15 and D18 cut are inherited from experiment 404's rows.

## Checks

- **Median forecast minus mean forecast per site.** This check is not available because the 404
  harness stores 11 quantiles and no mean forecast. The error of the median forecast is reported
  instead.
- **Error of the median beside the score contrasts.** G1-trust minus B-mix, mean absolute error of
  the q50 forecast (fraction of capacity): +0.0002 (no label), -0.0011 (harm), and +0.0005 (real
  event) at the six signed-megawatt substations, and +0.0008, +0.0019, and -0.0008 at the eight
  MVA-only substations. One point of score is 0.005 of capacity at q50 (the pinball loss at q50
  is half the absolute error), so +0.0002 is about +0.02 points. The no-label score contrast is
  slightly negative (-0.021) while the median error is slightly positive; both lie inside their
  noise.
- **Trigger that fired.** Audit trigger 2 (a result that contradicts its prediction and an earlier
  experiment) fired: the learned weight of 0.41 falling to 0.36 contradicts the prediction (0.75
  falling to 0.5) and experiment 458's fitted weights. Triggers 1, 3, 4, 5, 6, and 7 did not fire.
- **What the audit confirmed.** A fresh Sonnet audit re-derived every headline contrast from the
  stored forecasts and matched the printed values to three decimals. It found no leakage, and found
  the low weight consistent with an out-of-fold lagged forecast weaker than the full-data one.
- **What the audit qualified.** The best constant weight on the scored months is 0.6 falling to 0.4
  at the six signed-megawatt substations, above the learned weight; that gap lowers the score by
  0.03 to 0.08 points. Training on five sixths of the rows is one of three candidate causes: the
  fold also removes a four-month season, and the full network memorises its training rows.
- **Caveats the audit adds.** The lagless forecast has the same mismatch between its cross-fitted
  and full-data versions; this has not been tested. The trust trains on the injected world (4.6% of
  half-hours carry an injected event) and is scored on real power. The weight is fitted by negative
  log-likelihood and scored by pinball loss, so the fitted weight need not be the weight that
  minimises the score.

## What it rules out and what to try next

- **What it rules out.** Experiment 404's in-sample fit is no longer a reason for a learned trust to
  lose to 50/50 on rows with no labelled event, because fitting out of fold removes that loss. A
  learned combination weight is not an improvement over 50/50 in this harness.
- **What could be tested next.** A finer fold split (12 folds or interleaved weeks) and a lagged
  forecast evaluated on the scored month would show whether the remaining gap closes. B1's
  NRA-corrected result inside events (-0.63, interval spanning zero) is the only sign we found of a
  gain from a learned weight there. Otherwise the fixed 50/50 mixture stays.
