---
id: "00509"
slug: event-band-coverage-paired-injection-on-and-off
title: >-
  Are the forecast bands appropriately confident when a power lag sits inside a load-shift event?
theme: calibration-and-uncertainty
also_relevant_to:
  - lagged-power-and-error-correction
status: finished
kind: other
date: 2026-11-02
size: >-
  medium. Stored forecasts only (experiments 454, 470, and 482), no fit and no GPU. Designs: one,
  because the brief fixes the paired comparison. Callers: this folder's scripts, which read the
  stored outputs of experiments 454, 470, and 482. Production serving path and degradation rule: not
  touched. One Sonnet plan review, one Sonnet code review before the first scoring run, one Sonnet
  results review, and two Sonnet write-up reviews.
hypothesis: >-
  A forecast that ignores a bad lag is about as wide as the same forecast without the lag, so the
  bands of the forecasts trained on injected shifts (S1, B-mix, K2-S1) are already appropriately
  confident inside a load-shift event. The pooled network that never saw an injected shift (A-pool)
  is too confident.
prediction: >-
  Written before any score is read. P1: K2-S1's q95 coverage on rows whose newest lag is inside the
  event is within 0.02 of its coverage on the same rows with the injection off in each of scenarios
  1 to 7, and its 90% band is under 3% wider in scenarios 1 to 4, 6, and 7 (P5 expects scenario 5 to
  fail the width part). A fall of more than 0.03 would
  mean the band is too confident. P2: A-pool's q95 coverage on negative unmirrored shifts
  (scenarios 2, 4, and 6) is at least 0.10 below its injection-off value. P3: the factor on the
  distance from the median that restores nominal coverage is the same within 0.05 inside the event
  and outside it, for K2-S1, B-mix, and the lagless network. P4: in scenario 8 (a genuine level change)
  the bands are too narrow for every forecast that follows less than the full change. P5: the 3%
  width threshold fails in scenarios 5 and 8. The early check on the same stored rows already found
  +4.0% and +3.1%, so P5 confirms a known number and is labelled that way.
decision_rule: >-
  Descriptive, so no arm is adopted. P1 holds when every K2-S1 coverage difference is under 0.02
  and every width ratio is under 1.03 in the stated scenarios. The 0.02 and 3% thresholds sit at
  about 2 to 4 bootstrap standard deviations at 15 clusters, so a difference is reported with its
  bootstrap standard deviation and the coverage minimum detectable effect from the 3-seed spread. A
  difference counts as real when its cluster-bootstrap interval, at the Bonferroni level for 48
  contrasts (3 confirmatory forecasts, K2-S1, B-mix, and A-pool, x 8 scenarios x 2 measures,
  coverage and width, of `newest_in` rows over days 1 to 10), excludes zero. Every other contrast
  (LL, S1, other classes, the 12 to 36 hour band, the widening factors, the real events) is
  descriptive and carries its bootstrap standard deviation only.
benchmark: >-
  The same rows with the injection off: the clean forecasts stored by experiments 454 (LL, S1,
  B-mix, K2-S1; three seed cells averaged) and 470 (A-pool).
row_set_id: none
metric: >-
  Coverage of q01, q05, q95, and q99 and of the central 90% interval, mean width of the 90% interval
  in points of effective capacity, probability integral transform histogram, and the widening factor
  that restores nominal coverage.
code: ml-experiments/00509-event-band-coverage-paired-injection-on-and-off/
verdict: partly supported
best_arm: >-
  Not an arm contest. K2-S1 changes its q95 coverage by at most 0.025 and its 90% band by -0.7% to
  +7.5% when its lags sit inside an injected event, and B-mix is similar. A-pool loses 0.24 of q95
  coverage in scenario 4.
headline: >-
  Injecting a load shift into the lags changed K2-S1's q95 coverage by -0.024 to +0.025 on rows
  whose newest lag is inside the event (bootstrap standard deviation 0.004 to 0.009), against a
  change of -0.236 for A-pool in the -3x, 14-day, unmirrored scenario. K2-S1's band under-covers
  with or without the injection (q95 coverage 0.78 to 0.83 at these three donors).
lessons: >-
  A lagged forecast that learned to ignore bad lags keeps about the same band inside an event as
  without one, so the bands are not made worse by events. They are too narrow at these three donors
  everywhere, so the next step is general calibration, not an event-specific widening.
take_home_message: >-
  When a power lag sits inside an injected load shift, the forecast that mixes lagged and lagless
  heads (K2-S1) keeps almost the same band and almost the same coverage as on the same rows with no
  injection, so the band is about as confident as it already was, not more. The pooled network that
  never saw an injected shift (A-pool) is the exception: its q95 coverage falls by 0.24 in the worst
  scenario. At the three donors we could check, every forecast under-covers its upper tail with or
  without the event, so widening the band is a general calibration question and not an event rule.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Score a forecast's coverage inside a load-shift event against the same rows with the injection
    off, never against other rows, because the three stored donors under-cover by 0.12 to 0.19 at
    q95 even without an event (figure 1 and the table below).
  - >-
    Do not add an event-specific widening to K2-S1 or B-mix: the factor that restores q95 coverage
    is within 0.05 inside and outside the event in 6 of 8 scenarios (figure 4).
  - >-
    Keep A-pool, or any network trained only on real lags, away from lagged inputs that can sit
    inside a load-shift event: its q95 coverage fell by 0.09 to 0.24 in the three unmirrored
    negative scenarios (figure 1).
---

**When a power lag sits inside a load-shift event, the forecast that mixes lagged and lagless heads
(K2-S1) keeps almost the same band and almost the same coverage as on the same rows with no event,
so its band is about as confident as before.** The pooled network that never saw an injected shift
(A-pool) is the exception. At the three donors we could check, every forecast under-covers its upper
tail with or without an event, so the widening needed is a general calibration question and not an
event rule. The verdict is partly supported, because two of the planned thresholds failed.

**We asked whether the uncertainty band of a demand forecast is too narrow, too wide, or about right
when one of its three power lags falls inside a load-shift event.** Experiment 482 injected shifts
into the lags of stored forecasts and found that K2-S1's central 90% band covered 85% of actuals
inside the event. It did not score the same rows with the injection switched off, so that figure
could not be told apart from the networks' ordinary under-coverage. This experiment scores the
forecast with the injected lags (stored by experiment 482) and the forecast with no injection (stored
by experiments 454 and 470) on identical rows, and reports the difference. Nothing was fitted.

**The comparison covers 15 of experiment 482's 80 (month, donor) clusters.** Experiment 482 stored
row-level quantiles for three donor Primaries (D5, D12, and D14) in five test months. The 13 other
donors keep only summary tables, so they cannot be paired. The five forecasts are the lagless network
(LL), S1, B-mix, K2-S1, and A-pool, as defined in experiment 482. The signed-MW six are the six
Primaries whose power is signed in MW, and the MVA-only eight are the eight recorded only in MVA. D5
and D14 are in the first group and D12 is in the second. The headline rows are experiment 482's
`newest_in` class (the newest lag is inside the event) over days 1 to 10 (24 to 240 hours ahead),
which is 6,495 rows for the 1-day scenarios, 90,915 for the 14-day scenarios, and 19,485 for
scenario 8.

## Injected against injection off

![q95 coverage difference](figure_1_q95_coverage_difference.svg)

![Band width change](figure_2_band_width_change.svg)

- **K2-S1's q95 coverage moves by -0.024 to +0.025.** The differences are +0.011, -0.014, +0.016,
  -0.015, +0.018, -0.024, +0.019, and +0.025 in scenarios 1 to 8, with bootstrap standard
  deviations of 0.004 to 0.009. Its injected q95 coverage is 0.78 to 0.85, and 0.78 to 0.83 with no
  injection.
  Intervals at the Bonferroni level for 48 contrasts exclude zero in scenarios 1, 3, 5, 6, and 8.
- **K2-S1's 90% band changes by -0.7% to +7.5%.** The changes are +1.5%, -0.7%, +3.0%, +1.3%,
  +7.5%, -0.7%, +4.0%, and +4.1% in scenarios 1 to 8. The band is 30 points of capacity wide.
- **A-pool's q95 coverage falls by 0.09 in scenario 2, 0.24 in scenario 4, and 0.13 in scenario 6.**
  These are the negative unmirrored shifts. Standard deviations are 0.025, 0.039, and 0.025. It rises
  by 0.04 to 0.15 in the other scenarios, because a positive shift lifts its median above the actual.
  Its q05 coverage in scenario 4 falls to 0.007 from 0.042.
- **S1 loses 0.05 to 0.07 of q95 coverage in the negative unmirrored scenarios, and B-mix loses
  0.01 to 0.03.** The lagless network changes by exactly zero in all 8 scenarios (the code check).
- **The ranking holds in the 12 to 36 hour band** (735 and 10,289 rows, so fewer). K2-S1's q95
  coverage changes by -0.031 to +0.029, and A-pool's by -0.304 in scenario 4.
- **Tail calibration, scenario 4, `newest_in`, days 1 to 10.** The table shows q05 and q95 coverage
  (nominal 0.05 and 0.95), injected and with the injection off, and the pinball loss at q95 in points
  of capacity.

| Forecast | q05 injected | q05 off | q95 injected | q95 off | q95 pinball injected | off |
|---|---|---|---|---|---|---|
| Lagless network | 0.061 | 0.061 | 0.771 | 0.771 | 1.789 | 1.789 |
| S1 | 0.042 | 0.062 | 0.693 | 0.759 | 2.231 | 1.790 |
| B-mix | 0.044 | 0.054 | 0.762 | 0.790 | 1.725 | 1.666 |
| K2-S1 | 0.038 | 0.049 | 0.786 | 0.802 | 1.601 | 1.594 |
| A-pool | 0.007 | 0.042 | 0.574 | 0.811 | 3.293 | 1.598 |

## Probability integral transform and widening factor

![PIT histograms, scenario 4](figure_3_pit_scenario_4.svg)

![Widening factor](figure_4_widening_factor.svg)

- **The probability integral transform of every forecast piles up in the top bin.** In scenario 4,
  27% of actuals fall above the forecast's 90th percentile for K2-S1 and A-pool with no injection
  (nominal 10%). The injection raises that share to 30% for K2-S1 and to 54% for A-pool.
- **The factor on the q95 distance from the median is the same inside and outside the event for
  K2-S1 in 6 of 8 scenarios.** Injected minus injection-off is -0.03, +0.01, -0.05, -0.03, -0.11,
  +0.03, -0.04, and -0.06 in scenarios 1 to 8. The two exceptions (scenarios 5 and 8) are lower inside
  the event, so the band was already wider. B-mix behaves the same way. The factor with no injection
  is 1.8 to 1.9, and the stored widening is 1.15, so most of the gap is general, not event-specific.
- **A-pool's factor falls to 0.89 in scenario 5 and rises to 2.33 in scenario 4**, which says its
  band is too wide when a mirrored positive shift lifts its median and too narrow when a negative
  one pulls it down.
- **The lower tail needs less widening.** The q05 factor for K2-S1 is 0.91 to 1.27 inside the event
  and 1.00 to 1.27 with no injection.

## Real events (sensitivity)

![Real events](figure_5_real_events.svg)

- **This is a sensitivity, not a paired result.** It compares rows with any of the three lags inside
  a labelled event (58,638 rows at 12 of 30 (month, Primary) clusters in the signed-MW six, 40,819 rows
  at 12 of 40 clusters in the MVA-only eight) with rows that have no label, over days 1 to 10. The
  rows come from different Primaries and months, so the contrast mixes the event with the Primary.
- **Against the NRA-corrected actual, K2-S1's q95 coverage is 0.92 on event rows against 0.88 on
  rows with no label** in the signed-MW six (difference +0.044, bootstrap standard deviation 0.040,
  minimum detectable effect 0.113), and 0.92 against 0.92 in the MVA-only eight (-0.005, 0.025).
- **Against the metered actual the signed-MW six drop to 0.70** (difference -0.176, standard
  deviation 0.080). The band does not follow the metered power during the event, which is the
  event itself, so the NRA-corrected actual is the headline.
- **K2-S1's band is 1.2 points narrower on event rows in the signed-MW six** (standard deviation 1.5).
  In the MVA-only eight event rows are 15 points wider than rows with no label, for every forecast
  including the lagless network, so that is a Primary effect.
- **The labels are incomplete.** NGED says load-shift events may be missing from its logs, so rows
  with no label may hold events no label covers, and the real-event slices undercount.

## Other 13 donors (summary level, not the same rows)

- **The three stored donors are the worst-covered.** K2-S1's raw q95 coverage on `newest_in` rows is
  0.790 at the three stored donors and 0.902 at the other 13 in scenario 4, 0.828 and 0.929 in
  scenario 1, and 0.774 and 0.855 in scenario 8. With the 1.15 widening the figures are 0.836 and
  0.933 (scenario 4). The 13 donors have no injection-off scores, so these figures do not form a pair.

## Predictions

- **P1 held in part.** K2-S1's q95 coverage changed by less than 0.02 in scenarios 1 to 5 and 7, but
  by -0.024 in scenario 6, which is under the 0.03 line for too confident. The band was under 3%
  wider in scenarios 1, 2, 4, and 6, was 2.98% wider in scenario 3, and 4.0% wider in scenario 7.
- **P2 held in two of three.** A-pool's q95 coverage fell by 0.24 and 0.13 in scenarios 4 and 6, and
  by 0.09 in scenario 2 (under the 0.10 line, interval -0.17 to -0.01).
- **P3 held in 6 of 8 scenarios for K2-S1** (tolerance 0.05); it failed in scenarios 5 and 8 by
  -0.11 and -0.06, where the factor was lower inside the event.
- **P4 held trivially.** In scenario 8 every forecast's q95 coverage is 0.76 to 0.87, below 0.95, but
  the same forecasts under-cover in the other scenarios too, so scenario 8 is not worse than them.
- **P5 held.** The width threshold failed in scenario 5 (+7.5%) and scenario 8 (+4.1%). The early check
  on the same rows had found +4.0% and +3.1%; the numbers here use the days 1 to 10 band.

## Checks and caveats

- **Resolution.** A difference of 0.02 is 2 to 5 bootstrap standard deviations at 15 clusters. The
  coverage minimum detectable effect from the seed cells (2.8 times the seed standard deviation of
  the pooled coverage) is 0.011 at q95 for K2-S1 (0.008 to 0.012 at q05) and 0.022 to 0.028 at q95
  for A-pool. The seed standard deviation comes from 3 seed cells (5 for A-pool), so it is noisy, and it measures seed variation
  and not the clustering of rows. The Bonferroni count is 48 for the confirmatory contrasts (3
  forecasts, 8 scenarios, coverage and width); every other contrast is descriptive.
- **Scenario 8 is a different contrast.** Both forecasts are scored against the shifted actual, so
  the contrast is between a forecast that saw the shifted lags and one that did not. In scenario 8 the
  `before` and `after` classes mix shifted and unshifted targets.
- **Reproduction.** The stored forecasts of experiments 454 and 470 match experiment 482's no-shift
  rows to 2.4e-06 on `before` rows for every forecast, and the pooled coverage of K2-S1's scenario 4
  `newest_in` rows over all lead hours (0.790) matches experiment 482's summary tables.
- **Three donors only.** They under-cover more than the other 13, so the absolute coverage figures are
  not representative of the 16 donors. The 80-cluster re-run was not done (it needs a GPU stage that
  has not been timed). The result is not borderline, so the re-run was not requested.
- **Median minus mean forecast:** not computed, because no mean forecast is scored. Triggers fired:
  none. The planned predictions P1 and P2 failed in one scenario each, and neither contradicts an
  earlier finding.
- **Example forecasts and event plots.** This page re-scores stored quantiles and draws no new fan
  chart. Experiments 468 and 482 hold 6 injected-scenario fans and 4 real-event plots on the same
  forecasts.
- **Not done.** The avoidable-curtailment cost, the price of widening on rows with no labelled event
  (experiments 446 and 483 hold it), and a table P comparison for LL, S1, and B-mix (it exists only
  for K2-S1 and A-pool). Table P like for like was not re-run here.

**What the next experiment should answer.** The three stored donors under-cover at q95 by 0.12 to
0.19 without any event. Whether the other 13 donors also need a larger factor than 1.15 on rows with
no event is a question for experiment 483's widening results, not for an event rule.
