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
  No learned trust beat the 50/50 mixture. At the signed-MW six on rows with no labelled event,
  G1-trust scores 3.492 against 3.513 for 50/50 (-0.021, 95% interval -0.061 to +0.015); inside real
  events (NRA-corrected actuals, depth factor 1.0) it scores 13.198 against 13.139 (+0.059, 95%
  interval +0.027 to +0.102, minimum detectable effect 0.38).
headline: >-
  Fitting the trust weight on out-of-fold lagged forecasts closes the gap between a learned trust
  weight and the fixed 50/50 mixture on rows with no labelled event (+0.14 for the same trust fitted
  in sample, -0.02 out of fold), but the learned weight never beats 50/50: it is 0.06 points worse
  inside real events at the signed-MW six and 0.05 to 0.13 worse at the MVA-only eight. Interim and
  unaudited.
lessons: >-
  The experiment 404 learned trust lost to 50/50 mainly because the lagged forecast was fitted on
  the rows where the trust was fitted, so the trust saw a lagged forecast that was far better on
  those rows than on new months. Removing that mismatch removes the loss on rows with no labelled
  event and leaves a learned weight (about 0.4) that is no better than the fixed 0.5. The remaining
  gap inside events is small, and the stored data cannot say whether a better-matched lagged head
  would close it.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  The learned trust weight of experiment 404 lost to the fixed 50/50 mixture largely because the
  lagged forecast was fitted on the same rows as the weight. When the weight is fitted on lagged
  forecasts from networks that did not see those rows, it matches 50/50 on rows with no labelled
  event (-0.02 points at the signed-MW six) but still does not beat it inside real events (+0.06),
  so keep the fixed 50/50 mixture. Interim and unaudited: five scored months, one seed, and NGED's
  event labels are incomplete.
advice:
  - >-
    Keep the fixed 50/50 mixture as the reference for combining lagged and lagless forecasts; a
    trust weight fitted out of fold only matches it (experiment 463).
  - >-
    Fit any learned combination weight on out-of-fold forecasts of both heads, never on a lagged
    head fitted on the same rows: the in-sample version scored 0.14 to 0.19 points worse on rows
    with no labelled event (experiment 463, G1-trust against G1-insample).
  - >-
    Do not run seed 1 of the cross-fitted trust fit: the learned weight swings by 0.18 to 0.24
    between months, above the 0.1 rule (experiment 463).
  - >-
    A lagged head trained on 5/6 of the rows is much weaker than the full-data head used at test
    time (negative log-likelihood 0.9 against 0.2 in sample), so the learned weight sits below the
    best test-month constant; test a finer fold split before trusting any weight curve (experiment
    463 audit).
code: ml-experiments/00463-lag-trust-cross-fitted-trust-fit-seed-zero/
---

**The fixed 50/50 mixture of the lagged and lagless forecasts stays the reference: a trust weight
fitted on out-of-fold lagged forecasts matches it and does not beat it.** We asked whether the
learned trust weight of experiment 404 lost to 50/50 because the weight was fitted on the same rows
as the lagged forecast it weighs. We trained 6 lag networks per scored month, each leaving out one
block of the training window, so that every training row has a lagged forecast from a network that
did not see it. We then fitted the trust weight on those forecasts and on the lagless table's own
out-of-block forecasts, and scored it on five new months. A copy of the fit on in-sample lagged
forecasts (G1-insample) shows what the mismatch costs. The mismatch explains the earlier loss. It
does not turn the learned weight into an improvement. NGED's event labels are incomplete, so rows
with no labelled event may hold unlogged events and the real-event rows undercount. Every number is
interim and unaudited: five scored months, one seed.

**Verdict, then what it rules out.** The experiment fails its adoption rule: inside real events the
learned weight is worse than 50/50, not better. The mechanism holds: fitting out of fold gains 0.16
points over fitting in sample on rows with no labelled event at the signed-MW six, and 0.19 at the
MVA-only eight. The search narrows to a better-matched lagged head (a finer fold split, so the
out-of-fold network is nearer the full-data network) or to keeping 50/50.

![G1-trust minus each reference, by group and row set](figure_1_contrasts.svg)

![Learned lagged weight by lead band, month, and arm](figure_2_learned_weights.svg)

![Day-ahead forecast across a real event at D5](figure_3_day_ahead_event.svg)

![Ten-day forecast, same event and origin](figure_4_ten_day_event.svg)

![Ten-day forecast across a real event, second signed-MW Primary](figure_7_ten_day_event_signed_b.svg)

![Ten-day forecast across a real event, third signed-MW Primary](figure_8_ten_day_event_signed_c.svg)

![Ten-day forecast across a real event, first MVA-only Primary](figure_9_ten_day_event_mva_a.svg)

![Ten-day forecast across a real event, second MVA-only Primary](figure_10_ten_day_event_mva_b.svg)

![Day-ahead forecast at D5 with no labelled event nearby](figure_5_day_ahead_no_label.svg)

![Ten-day forecast at D5, same origin](figure_6_ten_day_no_label.svg)

## Results

Scores are points of effective capacity (200 times the mass-weighted pinball loss over 11 quantiles;
leads of 0 to 240 hours, lead days 1 to 10; real-data world). A contrast is the first arm minus the
second on the same rows, so negative is better. The unit of resampling is the month: 5 scored months
(2025-10, 2025-12, 2026-01, 2026-03, 2026-06), 4 where a group has no event. Intervals are 95%
month-resampled intervals; the Bonferroni interval for 20 contrasts is in the experiment's private
folder and in figure 1.

- **Arms.** B-mix (50/50 of A0-inj and the lagless table); B1 (experiment 404's trust learned
  jointly with the lagged head, in sample); G1-insample (the trust head fitted with both heads
  frozen, the lagged head being experiment 404's full-data A0-inj network evaluated on its own
  training rows); G1-trust (the same frozen-head fit on out-of-fold lagged forecasts). The
  G1-insample control was added to the brief's design because B1 differs from G1-trust in three
  ways at once; the neural Discuss session is informed here.
- **Signed-MW six, absolute scores (no label / harm / real event with NRA depth 1.0 / real event
  metered).** B-mix 3.513 / 3.680 / 13.139 / 5.955; B1 3.788 / 3.780 / 12.507 / 6.470; G1-insample
  3.650 / 3.651 / 13.351 / 5.962; G1-trust 3.492 / 3.614 / 13.198 / 6.006; A0-inj alone 3.824 /
  3.811 / 13.487 / 6.078; lagless alone 3.763 / 3.933 / 13.243 / 6.284.
- **Signed-MW six, G1-trust minus B-mix (minimum detectable effect 0.32 / 0.42 / 0.38 / 0.26).** No
  label: -0.021 [-0.061, +0.015]. Harm: -0.066 [-0.136, +0.003]. Real event, NRA depth 1.0: +0.059
  [+0.027, +0.102] (depth 0.75: +0.061; depth 1.25: +0.060; drop the largest event: +0.015). Real
  event, metered: +0.051 [-0.004, +0.106].
- **Signed-MW six, G1-trust minus G1-insample.** No label: -0.158 [-0.332, -0.004] (Bonferroni
  interval -0.421 to +0.045). Harm: -0.037. Real event, NRA depth 1.0: -0.153 [-0.420, +0.008].
- **Signed-MW six, G1-trust minus B1.** No label: -0.296 [-0.781, +0.160]. Real event, NRA depth
  1.0: +0.691 [-0.311, +2.336]; B1 is the only arm better than 50/50 there (-0.632 [-2.281,
  +0.359], 3 of 5 months negative), and its interval spans zero.
- **MVA-only eight, G1-trust minus B-mix.** No label +0.051 [+0.027, +0.071]; harm +0.125 [+0.056,
  +0.195]; real event with NRA depth 1.0 +0.105 [+0.029, +0.181] (4 months). G1-trust minus
  G1-insample: -0.193 [-0.277, -0.118], -0.380 [-0.787, -0.083], -0.788 [-1.003, -0.573].
- **Learned lagged weight (mean over scored months).** Signed-MW six: G1-trust 0.41 at 0-12 h, 0.43,
  0.43, 0.39, then 0.36 at days 6-10; G1-insample 0.93 falling to 0.82; B1 0.73 rising to 0.80. The
  range over months within a lead band is 0.18 to 0.24 for G1-trust (0.31 to 0.55 at 0-12 h), above
  the 0.1 stability rule, so seed 1 was not run. MVA-only eight: G1-trust 0.52 falling to 0.45.
- **Training-row diagnostic.** The lagged head's weighted negative log-likelihood on the training
  rows is 0.83 to 0.92 out of fold and 0.19 to 0.24 in sample, against 0.71 to 0.97 for the
  cross-fitted lagless head. An out-of-fold fit is therefore a much weaker head than the full-data
  head that scores the test month.
- **Tails, signed-MW six, no label (coverage at q01 / q05 / q95 / q99; q99 pinball loss).**
  G1-trust 0.006 / 0.029 / 0.845 / 0.943; 0.0027. B-mix 0.006 / 0.028 / 0.855 / 0.949; 0.0026. Inside
  real events: G1-trust 0.805 at q99 against 0.807 for B-mix. Every arm under-covers the upper tail
  (nominal 0.95 and 0.99) on these rows, and G1-trust is no different from B-mix. Curtailment cost
  does not apply in the 404 harness.
- **Fit count and time.** 40 fits (30 out-of-fold lagged-head fits at about 14 s each, 10 trust fits
  at about 7 to 12 minutes each), about 1 hour 50 minutes in total on one GPU process under 8 GB.
- **Where things are.** Code in `ml-experiments/00463-lag-trust-cross-fitted-trust-fit-seed-zero/`,
  reusing experiment 404's modules (and 87, 48, 397's) and 455's and 458's scoring pattern.
  Forecasts and trust networks in the experiment's private data folder; the 30 out-of-fold networks
  are not kept (their out-of-fold forecasts are, as `oof_<month>_s0.npy`). Retraining: monthly.
  Cleaned data and the D15 and D18 cut: inherited from experiment 404's rows.

## Checks

- **Median forecast minus mean forecast per site.** Not available: the 404 harness stores 11
  quantiles and no mean forecast. The error of the median forecast is reported instead (below).
- **Error of the median beside the score contrasts.** G1-trust minus B-mix, mean absolute error of
  the q50 forecast (fraction of capacity): +0.0002 (no label), -0.0011 (harm), +0.0005 (real event)
  at the signed-MW six, and +0.0008, +0.0019, -0.0008 at the MVA-only eight; all far below the
  score differences' scale. The no-label score contrast is negative (-0.021) while the median error
  is slightly positive (+0.02 points); both lie inside their noise.
- **Triggers.** Trigger 2 fired: the learned weight (0.41 falling to 0.36) contradicts the
  prediction (0.75 falling to 0.5) and experiment 458's fitted weights. A fresh Sonnet audit
  re-derived every headline contrast from the stored forecasts and found no difference to three
  decimals, found no leakage, and found the low weight consistent with a weaker out-of-fold lagged
  head. It qualified the explanation: the best constant weight on the test months is 0.6 falling to
  0.4 at the six, above the learned weights, and the cost of that gap is 0.03 to 0.08 points; "5/6
  of the rows" is only one of three candidate causes (the fold also removes a four-month season;
  the full network memorises its training rows). Triggers 1, 3, 4, 5, 6, and 7 did not fire.
  Leaving D5 out of the signed-MW six real-event contrast reduces it from +0.076 to +0.011 (rows
  pooled) and does not change its sign or the verdict.
- **Caveats the audit adds.** The lagless head has the same cross-fitted-against-full mismatch,
  untested; the trust trains on the injected world (4.6% of half-hours carry an injected event) and
  is scored on real power; the weight is fitted by negative log-likelihood and scored by pinball
  loss, so it need not be the score-optimal weight.

## What it rules out and what to try next

- **Ruled out.** Fitting the trust on in-sample lagged forecasts (the experiment 404 design) as the
  reason a learned trust cannot reach 50/50 on rows with no labelled event; and a learned
  combination weight as an improvement over 50/50 at this harness.
- **Next.** If a learned weight is still wanted, fit it with a finer fold split (12 folds or
  interleaved weeks, so each fold network is nearer the full-data network) and a lagged head
  evaluated on the test month, and pre-register whether the aim is a gain inside events (B1's
  NRA-corrected result, -0.63 with a wide interval, is the only sign of one). Otherwise keep 50/50.
