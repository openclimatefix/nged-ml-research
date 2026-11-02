---
id: "00499"
slug: two-constant-tail-widening-in-place-of-table-p-for-wind-and-solar
title: >-
  Can two constants per farm replace table P for the wind and solar forecasts?
theme: calibration-and-uncertainty
also_relevant_to:
  - generator-forecasting
status: planned
kind: other
date: 2026-11-02
size: >-
  medium. Stored: tables under the experiment's data folder only; no model is fitted. Serving path:
  not touched. Degradation rule: not touched. Designs: one, because the fit rules and the limits
  come from the brief. Callers: this folder's scripts, which copy experiment 461's table P and clip
  code. One Sonnet plan review, one Sonnet code review before the first run, one Sonnet results
  review, and two Sonnet write-up reviews.
hypothesis: >-
  Table P (a factor per farm, lead band, and tail quantile, plus experiment 451's clip) can be
  replaced for wind and for solar by two constants (one scaling q01 and q05, one scaling q95 and
  q99), fitted on earlier months only, with no loss of tail calibration.
prediction: >-
  Written before any score is read. P1: for wind, two constants per farm (rule e) are within the limits of table P on all six
  criteria, because experiment 481 found a constant widening of 1.10 best at every seed count; where
  the minimum detectable effect exceeds a limit the result is reported as inconclusive. P2: for
  solar, no two-constant rule matches table P, because the forecast spread at low-sun hours (05 and
  19 UTC) is about one seventh of the midday spread, which one constant per farm serves badly. P3: the lead-band-group
  rule (f) is no better than the per-farm rule (e) by more than the minimum detectable effect.
  P4: most per-farm cells are inconclusive, and so is every wind q99 criterion in the first lead band (about 30 expected events).
decision_rule: >-
  A two-constant rule (d, e, or f) matches table P for a family if, on the same scored rows, the
  differences from table P are inside these limits: each of q01, q05, q95, and q99 coverage within
  0.01 (two-sided); the share of actuals above q99 higher by at most 0.2 percentage points
  (one-sided; this is 1 minus q99 coverage, so it binds before the q99 coverage limit); and the mean
  pinball loss over the four tail quantiles at most 5% higher (one-sided; per-quantile pinball loss
  is reported but does not decide). These are the six criteria. The decision is the family-level
  verdict on rows pooled in MW; per-farm verdicts are descriptive, at the same interval level. "Matches" needs the whole paired
  interval inside the limit, at the Bonferroni level for 36 tests (3 rules x 2 families x 6
  criteria). An interval that crosses a limit is inconclusive; an interval wholly beyond a limit is
  not equivalent. If the minimum detectable effect (printed first) exceeds a limit, a null at that
  limit is inconclusive and never called equivalent. Stop rule: keep table P for each family and
  farm where no two-constant rule matches it, and report per farm. Serving cost: table P is one
  factor per farm, lead band, and tail quantile (16 per farm) plus a clip; rule d is two numbers
  per family, rule e two per farm, rule f six per farm. Any match is worth porting even with no
  score gain.
benchmark: >-
  Table P, the raw forecast, and the symmetric 1.15 widening, all fitted or applied on the same
  scored rows with experiment 451's clip (the raw forecast unclipped, as experiment 461 scores it).
row_set_id: >-
  Experiment 461's stored rows (wind `member_plain` at p99.9 capacity, 3 farms, 52 Tuesday runs,
  months 2025-07 to 2026-06; solar `a2_member` at p99.9 capacity, 6 series, daily initialisations,
  months 2025-09 to 2026-06), true days 1 to 10, July 2026 dropped.
metric: >-
  Coverage and pinball loss at q01, q05, q95, and q99; the share above q99; the quantile score in
  MW (mean over the 11 levels).
---

**Plan: re-score experiment 461's stored wind and solar forecasts with two-constant tail rules and
table P on the same rows; no model is fitted.** This is the wind and solar half of [experiment
492](../00492-two-constant-tail-widening-in-place-of-table-p-for-demand/result.md), which found that
two constants match table P at the eight MVA-only demand Primaries and not at the six signed-MW
Primaries.

## Plan

- **Forecasts reused.** Experiment 461's `rows_member_plain_p999.parquet` (wind, seeds 0 to 4
  already averaged quantile by quantile, 59,861 rows) and `rows_a2_member_p999.parquet` (solar, 3
  seeds averaged, rows from 2026-07 dropped). Stored forecasts are raw quantiles in MW; table P is
  recomputed here with experiment 461's code (copied, since the experiment 461 table P results are
  not stored per row). Gate: the table P pooled share above q99 must reproduce experiment 461's
  (wind 0.88%) before any contrast is read.
- **Lead bands.** The stored minimum lead is 24 h for wind and 29.5 h for solar, so the first band
  is 24 to 36 h (wind) and 29.5 to 36 h (solar); the experiment makes no claim about 12 to 24 h.
  Rule f uses three groups cut from the stored bands: the first band, 36 to 72 h, and 72 to 240 h.
- **Fit window.** A fitting row must have its valid time (initialisation plus lead) before the first
  initialisation of the scored month, so no actual unknown at the first scored run is fitted. The
  wind file keys `month` by initialisation month and the solar file by valid month, so the plain
  "earlier months" rule of experiment 461 would leak up to 10 days for wind. A gate re-runs table P
  with experiment 461's plain window and must reproduce its pooled wind share above q99 (0.88%).
  Scored months: wind 2025-08 to 2026-06, solar 2025-10 to 2026-06 (July 2026 and later dropped).
- **Rules**, each fitted on fitting rows only (the first month is never scored):
  (a) raw; (b) symmetric 1.15 widening; (c) table P by farm and lead band; (d) two constants per
  family; (e) two constants per farm; (f) two constants per farm and lead-band group. Constants are
  the 0.50 to 1.80 grid value (step 0.01) minimising summed pinball loss in fractions of the
  capacity (so each farm weighs equally); the lower constant scales q01 and q05, the upper q95 and
  q99. A cell with fewer than 200 fitting rows (rule f) falls back to rule e, and the count is
  reported, as is the number of table P cells below its 200-row minimum (factor 1). Rules b to f
  use experiment 451's clip: every tail at least 0, every level at most the larger of the capacity
  in use and the series' largest earlier actual (MW). Coverage counts `actual <= q`. The constants
  are fitted by pinball loss and table P by matching coverage, so the comparison is partly one of
  fitting method. The constants are fixed in each resample, so fitting uncertainty is not in the
  intervals.
- **Scored rows.** The months after the first, for both families, on the same rows for every rule.
- **Intervals.** Paired differences from table P, with a run-block bootstrap (one block per
  Tuesday run for wind; one initialisation week for solar, because daily solar runs share their
  weather; no farm blocks, because wind has only 3 farms; a sensitivity run uses daily solar
  blocks), 20,000 resamples, at the
  Bonferroni level for 36 tests. Per-farm intervals use the same level and are descriptive.
- **Minimum detectable effect, printed before any contrast.** Computed by the script from the
  run-block spread of table P's own hit rate and pinball loss (an unpaired upper bound on the
  paired difference's standard error), at 80% power and the Bonferroni level.
- **Slices.** All scored rows (true days 1 to 10) and the stored first lead band. Cold winter
  nights and the avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 belong to the demand
  protocol and are not applicable to wind and solar here.
- **Not applicable.** The cut of D15 and D18 and the flagged exact-zero half-hours concern demand
  Primaries; the wind and solar rows are experiment 461's.
- **Weather.** The stored forecasts use ECMWF ensemble weather at forecast time; no past weather
  enters this experiment.
- **Compute.** CPU only, one to two cores, under 8 GB; no fits, no saved models.
- **Reviews.** Plan, code, results, and two write-up reviews, all fresh Sonnet.
- **Earlier advice.** Follows experiment 492's rule, limits, and bootstrap; the stored-lead
  correction and the inconclusive-null wording come from the queue-order check of 2026-11-02.
- **Serving table.** Table P: 16 factors per farm (4 lead bands, 4 tails) plus the clip, refitted
  as often as the factors are refreshed. Rule d: 2 numbers per family. Rule e: 2 per farm. Rule f:
  6 per farm. Each is refitted when the forecasts are, with the same clip.
- **Stored rows.** Wind: 52 Tuesday runs, 3 farms, seeds 0 to 4 averaged. Solar: 6 series, daily
  runs (Solar E has fewer), seeds averaged.
