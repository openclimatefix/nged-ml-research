---
id: "00399"
slug: xgboost-short-history-start-cut-d15-d18
theme: retraining-and-training-history
also_relevant_to:
  - demand-inputs
status: finished
kind: change
protocol: demand-v1
size: >-
  medium. Stored: 60 single-series prediction files (private) and one score table. Designs: more
  than one defensible (how many months to cut, and whether to remove near-zero readings by value
  or by run), so the experiment sweeps them. Callers: imports experiment 87's fit modules and
  experiment 381's fit script, and no other folder. Production serving path and degradation rule:
  not touched.
hypothesis: >-
  Dropping the first months of the two short-history Primaries, D15 and D18, from the monthly
  XGBoost demand model's training rows lowers its error at those Primaries, because the readings
  taken while the meters were being commissioned teach the model a wrong level.
prediction: >-
  Written before any score is read. P1. At D18, cutting to 2025-03-01 (NC) lowers the CRPS in days
  1 to 10 by 3% or more against N0, and cutting 3 or 6 months (N3, N6) lowers it by a similar
  amount, because D18 looks commissioned about 7.5 weeks after its first reading. P2. At D15, no
  cut lowers the CRPS by more than 2%, because its level stepped up in July 2025, September 2025,
  and March 2026, and a start cut can remove only part of that bias. P3. Removing near-zero
  readings (NCZ, NCR) changes the CRPS by less than 1% against NC at both Primaries, and NCR
  differs from NCZ by less than NCZ differs from NC.
decision_rule: >-
  Twelve planned contrasts (each of N3, N6, NS, NC, NCZ, and NCR against N0, at each of D15 and
  D18), read on days 1 to 10 under `demand-v1`, report-only. D18 has 6 scored months and D15 has 4,
  so every figure is descriptive and no interval is claimed to establish a gain. Resampling is by
  whole months where a slice holds 3 or more months. An arm counts as helping at a Primary when its
  days 1 to 10 CRPS is lower than N0's in at least two thirds of the scored months and lower in
  the pooled figure, raw and at the 1.15 widening. The porting threshold is zero, because a cut
  date per Primary is a number production already holds as the settling rule. The result says
  nothing about the other 14 Primaries.
benchmark: >-
  N0, the stored monthly control (`monthly/control` of experiment 381: the plain recipe refitted
  monthly, no 1.15 widening). References: the GAM stack and BG2 (experiment 379's scores), the
  `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS estimated from the 11 quantiles, signed median bias, coverage at q01, q05, q95, and q99, and
  avoidable-curtailment cost, from `studies.eval_protocol`, in points of effective capacity; days 1
  to 10 and by lead band; by calendar month since the first reading.
code: ml-experiments/00399-xgboost-short-history-start-cut-d15-d18/
verdict: partly supported
best_arm: >-
  NC (every row before 2025-03-01 dropped): CRPS at days 1 to 10 of 15.40 at D15 (N0 15.61) and
  2.665 at D18 (N0 2.696), in points of capacity, seed 0.
headline: >-
  Cutting both Primaries' training start to 2025-03-01 (NC) lowered the monthly XGBoost model's
  CRPS by 1.3% at D15 (lower in all 4 scored months) and 1.2% at D18 (lower in 3 of 6 months). A
  6-month cut (N6) lowered D15's CRPS by 4.4% in 2 of 4 months and raised D18's by 7.2%. Removing
  near-zero readings made no difference at D18; at D15 it left the error unchanged (NCR) or raised
  it by 1.3% against NC (NCZ).
take_home_message: >-
  Dropping the noisy first weeks of D15 and D18 from the training rows helps the monthly XGBoost
  forecast only a little: about 1% lower error at both, mostly in the first scored months, when
  the history was shortest. Cutting 6 months is not safe, because it made D18 worse. The larger
  problem at D15 is a level error that no start cut fixes: the median forecast is about 21 points
  of capacity too low, against 3 for the GAM stack, so the next experiment should look at D15's
  level, not its start.
lessons:
  - >-
    The cut to 2025-03-01 (NC) and NCR were the only arms lower than N0 in every D15 month, and NC
    tied the best arm at D18; NC is the simpler cut and a safe default, with a gain of about 1%.
  - >-
    A 6-month cut (N6) lowers D15's median bias from -20.8 to -15.9 points but raises D18's CRPS
    by 7.2%, so the cut length cannot be one number for both Primaries.
  - >-
    D15's median forecast stays 16 to 21 points of capacity too low under every cut (GAM stack:
    -3.0), so D15's error is a level problem and not a commissioning-noise problem.
  - >-
    Removing every reading within 0.009 MW of zero (NCZ) is 1.3% worse than NC at D15, because most
    of those readings are real crossings of zero; removing only runs of 2 or more (NCR) is
    neutral.
production_status: candidate
adopted_in: none
advice:
  - >-
    Drop the rows before 2025-03-01 for D15 and D18 in XGBoost demand experiments, and do not cut
    further than that at D18 (Result, N6).
  - >-
    Look at D15's level, not its first months, for its large error: every start cut leaves a
    median bias of at least -15.9 points of capacity (Figure 3).
  - >-
    Do not remove near-zero readings from D15 by value; they are mostly real zero crossings
    (Checks, NCZ).
  - >-
    Refit the control in the same environment as the arms: the stored control differed from its
    refit by a mean of 0.001 (maximum 0.021) of capacity at the median, which is small but not
    zero (Checks).
date: 2026-10-30
---

# Does cutting the first months of D15 and D18 help the monthly XGBoost demand forecast?

**Cutting the noisy start of D15 and D18 lowers the monthly XGBoost demand forecast's error by
about 1%, and cutting 6 months is not safe.** We asked whether leaving out the first months of
readings, taken while a new substation's meter was being commissioned, gives a better forecast at
D15 and D18, the two Primary substations whose history starts after the weather archive does. We
refitted the XGBoost model of [experiment 381](../00381-simple-weekly-xgboost-demand-against-gam-stack/result.md)
(refitted monthly, with the first months of each series dropped in six different ways) and scored
the forecasts for the months the protocol scores: January to June 2026 at D18, and March to June
2026 at D15. The cut to 2025-03-01 (NC) lowered the error at both Primaries by about 1%. A 6-month
cut lowered D15's error by 4.4% and raised D18's by 7.2%. The result says nothing about the other
14 Primaries, which have history from before the weather archive starts.

**Verdict: partly supported.** The prediction that D18 gains 3% or more from NC failed (it gained
1.2%). The prediction that no cut gains more than 2% at D15 also failed (N3 gained 2.4%, N6 4.4%). P3
failed for NCZ at D15, which was 1.28% worse than NC (15.594 against 15.397); NCR (+0.34%) and both
removals at D18 (within 0.2%) held.
The decision rule asks for a lower CRPS in two thirds of the scored months and in the pooled
figure. At D15 the arms N3, NS, NC, and NCR met it, and N6 and NCZ did not. At D18 no arm met it.
The porting threshold is zero, because production already holds a cut date per Primary as the
settling rule. Every figure below is a point estimate, because 4 and 6 scored months are too few
for an interval.

**Figure 1.** Each cut's days 1 to 10 CRPS, relative to the control N0 refitted in the same
environment (percentages of N0, negative is better).

![Dot chart of each cut's CRPS change against N0 at D15 and D18](change.svg)

| Arm | D15 CRPS | D15 months lower | D18 CRPS | D18 months lower |
|---|---|---|---|---|
| N0, all rows (refitted) | 15.606 | | 2.696 | |
| N0, stored (experiment 381) | 15.599 | 3 of 4 | 2.696 | 3 of 6 |
| N3, first 3 months cut | 15.227 | 3 of 4 | 2.676 | 3 of 6 |
| N6, first 6 months cut | 14.915 | 2 of 4 | 2.891 | 1 of 6 |
| NS, settling rule | 15.439 | 3 of 4 | 2.666 | 3 of 6 |
| NC, cut to 2025-03-01 | 15.397 | 4 of 4 | 2.665 | 3 of 6 |
| NCZ, NC and near-zero removed | 15.594 | 2 of 4 | 2.669 | 3 of 6 |
| NCR, NC and near-zero runs removed | 15.450 | 4 of 4 | 2.665 | 3 of 6 |
| GAM stack | 12.848 | | 2.499 | |
| BG2 | 13.354 | | 2.646 | |
| `manual_heuristic` | 22.590 | | 4.015 | |
| `climatology` | 46.195 | | 7.114 | |

CRPS is in points of capacity at days 1 to 10 under `demand-v1` (report only), raw (no 1.15
widening). The columns "months lower" count scored valid-time months in which the arm beat the
refitted N0. Every arm is the plain monthly control recipe (`colsample_*` fixed at 1.0), seed 0.
The widened rows (1.15) are in the scoring log: N0 15.464 and NC 15.274 at D15, N0 2.753 and NC
2.708 at D18. Widening lowers D15's CRPS and raises D18's, and it reorders arms that are within 0.1
points of each other.

**Figure 2.** The gain depends on the month, which is why the months-lower counts are low.

![CRPS of three cuts minus N0 by month, at D15 and D18](monthly.svg)

At D18 the gain comes mainly from January 2026, the month served by the refit with the shortest
history (2.82 for N0, 2.65 for NC). From February on, NC is within 0.1 points of N0 and sometimes
worse. At D15 the cuts help most in March 2026 (23.40 for N0, 22.88 for NC and 20.78 for N6) and
little afterwards. N6 raises D18's CRPS by 0.45 points in March 2026 and 0.65 in April 2026.

**Figure 3.** D15's error is a level error that no start cut removes.

![Signed median bias at D15 for N0, three cuts, and the references](bias.svg)

At D15 the median forecast of every XGBoost arm is 15.9 to 20.9 points of capacity below the
actual, averaged over days 1 to 10 (N0 -20.8, N3 -20.2, N6 -15.9, NC -20.5). The GAM stack's bias
is -3.0 and BG2's is -13.1. At D18 the biases are small (N0 -0.66, N6 +1.00). Cutting 6 months
removes about a quarter of D15's bias. D15's level stepped up in July 2025,
September 2025, and March 2026, so a model trained on the earlier months learns a level that no
longer holds, and a cut at one date cannot remove that.

## What each cut costs in production

Every arm needs one cut date per Primary. NS needs the settling rule, which production already
holds (D15 to 2025-03-20, D18 to 2025-03-04). N3 and N6 need the first reading's date plus a
number of months. NC needs the single date 2025-03-01. NCZ and NCR also need a row filter on the
power data and on the three origin lags, so they cost more and gain nothing.

## Checks

- **How many near-zero readings were removed.** NCZ removed 104 half-hours at D15 and 10 at D18
  (readings of 0.009 MW or less, 2025-03-01 to 2026-06-30), all inside the scored year. NCR removed
  16 and 2. At D15 most removed readings are single half-hours, so they are real crossings of zero
  by a series that swings by about 2 MW and exports; this is why NCZ is 1.3% worse than NC at D15.
  D18's July 2025 dip (1 to 3 July) was left in, as planned.
- **The same rows.** Scoring on the rows NCZ keeps (`ncz_common`) changes each XGBoost arm's CRPS
  by 0.003 points or less, except N6 at D15 (0.017). D18's values are identical. The two row sets
  rank the raw XGBoost arms identically.
- **Reproduction.** The stored N0 of experiment 381 and the N0 refitted here differ by 0.007
  points of CRPS at D15 and 0.0005 at D18. The forecasts differ by a mean of 0.001 (maximum 0.021)
  of capacity at the median; the `uv.lock` library versions (xgboost 3.4.1, numpy 2.5.4, polars
  2.1.0) are the same, and we did not find the cause. All contrasts use the refitted N0.
- **The settling cut uses later data.** `new_meter_settling` takes each series' capacity from
  all readings up to 2026-06-30, not up to each refit, so NS's cut dates (D15 2025-03-20, D18
  2025-03-04) could move by days if recomputed at each refit. They match the dates the plan quotes.
- **Power source.** The training power came from `studies.power.scan_power`, not from the stored
  `observed.parquet`. The two agree on every shared half-hour at D15 and D18, and the training
  labels agree to 5e-7 MW.
- **Median minus mean forecast, per site.** N0: -0.116 MW at D15 and -0.019 MW at D18. NC: -0.124
  and -0.030. A cut does not shift the gap.
- **MAE of the median.** D15: N0 28.05, N6 26.64, NC 27.62. D18: N0 4.738, N6 5.068, NC 4.732.
  It points the same way as the CRPS contrast for N6, NC, and NS. For N3 and NCZ at D18 and NCZ at
  D15 the MAE is within 0.02 of N0, on the other side of it.
- **Audit triggers.** Trigger 2 fired: the results contradict predictions P1, P2, and P3. The
  explanation (the refit with the shortest history gains most, and D15's error is a level error)
  is in Figures 2 and 3 and in the take-home message. Trigger 7 may apply, because the research
  default for the other experiments now drops rows before 2025-03-01 (NC); that default rests only
  on NC being no worse than N0 and on production's existing cut date, and the result is not
  replicated. No audit has run.
- **Curtailment cost.** The scorer returns no cost for D15 and D18, because neither is one of the
  signed-MW Primaries it defines the cost for.

## Limits

The scored months are 4 at D15 and 6 at D18, one seed, one fit per refit. A gain of 1% is larger
than the refit noise (0.007 and 0.0005 points) but is not shown to exceed the month-to-month
variation. NGED's horizon is 1 to 10 days and the headline band covers it. D18 steps up by about
1.5 MW around 2026-07, after the scored year, and no claim is made about that step.

## What the next experiment should answer

Why D15's median forecast is 16 to 21 points of capacity too low while the GAM stack's is 3 points
too low: a level input, a shorter training window anchored on recent months, or the stack's own
level term are the candidates. The level anchor of the sister experiment on stored forecasts has
not merged and was not added.
