---
id: "00509"
slug: event-band-coverage-paired-injection-on-and-off
title: >-
  Are the forecast bands appropriately confident when a power lag sits inside a load-shift event?
theme: calibration-and-uncertainty
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
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
---

**Plan, written before any score is read.** We ask whether a forecast's uncertainty band is too
narrow, too wide, or about right when one of its power lags sits inside a load-shift event. Experiment
482 injected load shifts into the lags of stored demand networks and reported coverage inside the
event. It did not report the same rows with the injection switched off, so a coverage of 85% inside
the event (nominal 90%) could not be told apart from the networks' ordinary under-coverage. This
experiment scores both forecasts of the same rows and reports the difference.

## Design

- **Rows.** Experiment 482 stored row-level injected quantiles (all 11 levels, plus the shifted and
  the clean actual) for 3 of its 16 donor Primaries (D5, D12, and D14) in 5 test months, which is 15
  of its 80 (month, donor) clusters. The paired analysis uses those 15 clusters and the donor's rows
  only. The row classes are experiment 482's four: `before`, `newest_in`, `older_in`, and `after`.
  In scenario 8 the `before` and `after` rows mix shifted and unshifted targets.
- **Columns and targets.** Experiment 482's rows files hold `primary`, `origin`, `target`,
  `lead_hours`, `y_world`, `y_clean`, `cls`, and 11 quantiles per forecast; the `A0-inj` forecast is
  ignored. Both the injected and the injection-off forecast are scored against `y_world`, the actual
  of the world that occurred. In scenarios 1 to 7 `y_world` equals `y_clean`. In scenario 8 the
  target moves with the lags, so the contrast there compares two forecasts of the same shifted
  target that differ only in whether they saw the shifted lags. Scenario 8 is a different kind of
  contrast from scenarios 1 to 7 and is reported apart.
- **Injection-off forecasts.** Experiments 454 (LL, S1, B-mix, and K2-S1, three seed cells averaged)
  and 470 (A-pool) stored the forecasts without injection. Joined on (Primary, origin, target), they
  reproduce experiment 482's own no-shift baselines to within 3e-6. The script asserts this on
  `before` rows, for every forecast, to a tolerance of 1e-5 of capacity, and stops if it fails. It
  also asserts that the join has no null. Quantiles are averaged over the three seed cells level by
  level, as experiment 482 does. Rows come from experiment 482 after the cleaning it inherited, so
  flagged zero half-hours are already dropped.
- **Forecasts.** The lagless network (LL), S1, B-mix, K2-S1, and A-pool, raw and with the 1.15
  widening. A-pool and K2-S1 also carry table P (experiment 435) like for like; LL, S1, and B-mix
  have no table P.
- **Other 13 donors.** Reported only as summary-level coverage and width from experiment 482's
  summary tables, labelled as not the same rows.
- **Real events.** The labelled events at the signed-MW six, scored from experiment 454's stored
  forecasts on rows with any of the three lags inside a labelled event (`lag_real`; the rows are labelled
  "any lag inside", not "newest lag inside"), against the NRA-corrected actual (headline, experiment 435's correction) and the metered
  actual (secondary), beside the same Primaries' rows with `lag_real`, `tgt_real`, and `win_real` all false. This is a sensitivity:
  it reports event counts and the minimum detectable effect, and NGED's labels are incomplete, so
  rows with no labelled event may hold unlogged events.

## Measurements

- Per forecast, scenario, and row class: coverage of q01, q05, q95, and q99; coverage and mean width
  of the central 90% interval; a 10-bin probability integral transform (PIT) histogram, taken from
  the 11 stored quantiles; and the pinball loss at q05 and q95.
- Paired contrast (the main result): injected minus injection-off on identical rows, for coverage
  and for width, with a cluster bootstrap over the 15 (month, donor) clusters. LL is the check: the
  injection cannot change it, so its difference should be zero.
- The widening factor f scales each tail's distance from the median: q50 + f (q - q50), after
  sorting the 11 quantiles. For the upper tail, f is the 95th percentile of (y - q50) / (q95 - q50);
  for the lower tail, of (q50 - y) / (q50 - q05). The two tails get separate factors, and the
  bootstrap re-solves f within each resample.
- The widening factor that restores nominal q05 and q95 coverage, inside the event and on the same
  rows with the injection off, per forecast. A factor needed inside the event but not outside is an
  event effect; one needed in both is the general calibration gap.
- Splits: sign, size (1x to 3x), duration (1, 3, and 14 days), mirror, and scenario 8. Two lead
  bands by `lead_hours`: 12 to 36 hours, and days 1 to 10 (24 to 240 hours). Widths are in points
  of effective capacity (fractions of capacity times 100). The signed-MW six and the MVA-only eight are separate;
  D15 and D18 are report-only.
- Statistics: Bonferroni count 48. The coverage minimum detectable effect comes from the spread of
  the 3 seed cells. July 2026 stays unused.

## Standing practices followed

Reuse of stored forecasts (no fit, no GPU); cleaned data and the D15 and D18 cut come with the stored
forecasts; "load-shift", never "switching", in titles; unlogged-event caveat; OCF palette with no
mark in the background colour (`#fffbf5`). Not followed because they do not apply: monthly refits and
saving fitted models (nothing is fitted), few-fold screen, curtailment-cost metric.
