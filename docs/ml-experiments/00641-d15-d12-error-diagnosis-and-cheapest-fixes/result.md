---
id: "00641"
slug: d15-d12-error-diagnosis-and-cheapest-fixes
title: >-
  What drives the forecast error at the Primaries D15 and D12, and which cheap fix lowers it
  without harming the other 14 Primaries?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  Medium. (1) Stores new tables and, for a fix that refits, new network weights and forecasts in
  this experiment's private folder only. (2) More than one defensible design: the fix list is
  chosen after the diagnosis. (3) Spans code from experiments 612 to 614 whose callers cannot be
  named without searching, only if a fix refits; a fix that acts on stored forecasts has no such
  callers. (4) Touches no serving path and no degradation rule. (5) Changes no contract. One Sonnet
  plan review, one Sonnet code review before the first scored run, one Sonnet prose review.
hypothesis: >-
  Most of K1's excess error at D15 and D12 sits in a small number of multi-day level shifts
  (unlabelled transfers) at D15 and in a midday peak at D12, so a per-Primary level correction at
  D15 and a midday-aware correction at D12 lower their CRPS, rather than the error being spread
  evenly as noise no forecast could remove.
prediction: >-
  Written before any new score is read; the first look at the stored forecasts (below) motivated
  it. P1: at D15, at least half of K1's development-month absolute error sits on at most 20% of
  days, and those days have a one-sided bias of the median that is flat across the hours of the
  day. P2: at D12, absolute error at 10:00 to 15:00 local time is at least twice the night error,
  and the midday bias is larger in winter than in June. P3: the best of the tested fixes lowers
  pooled CRPS at its target Primary by at least 5%. P4: no fix raises the pooled CRPS of the other
  14 Primaries by more than 1%.
decision_rule: >-
  Diagnosis is descriptive. For the fixes: a fix is `supported` when the 95% day-block bootstrap
  interval of its CRPS difference to K1's stored forecast, pooled over D15 and D12 on the
  development months at leads 1 to 10 days, excludes zero in the predicted direction and the point
  estimate is at least 3% of the pooled D15 and D12 CRPS, and the pooled CRPS of the other 14
  Primaries does not rise by more than 1%. A fix with a smaller gain is `small gain`. No rescue
  tuning after a failed bar. Each fix has a stop rule recorded in the amendment that names it.
benchmark: >-
  K1's stored forecasts (the control), the manual heuristic and the Candidate and K3 from the same
  export, on the same rows.
row_set_id: backtest_export_roster_2026-11-07_development_months
metric: >-
  CRPS, MAE of the median, q05-q95 coverage, and q01, q05, q95, and q99 pinball loss and coverage,
  in percent of capacity, at leads 1 to 10 days.
---

# What drives the forecast error at D15 and D12, and which cheap fix lowers it?

**Status.** Planned. This page is the plan; the diagnosis, the ranked fix list, and the results are
added in later commits.

## Question

D15, D8, and D12 carry 44% of K1's development-month CRPS
([the headroom answer](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries/result/)
gives the shares: 21.9%, 11.7%, and 10.5%). D8 has its own series of experiments. This experiment
asks what drives the error at D15 and D12, and whether a cheap fix lowers it.

## First look at the stored forecasts (before this plan, descriptive)

- D15's mean absolute error of K1's median is 24.8% of capacity at leads 1 to 10 days, against 2.6%
  to 14.1% at the other 15 Primaries. Its median is too low on average (bias +15.4 points).
- D15's error is flat across the hours of the day (23.4 to 26.9) but concentrated in a few days: the
  worst days (2025-10-21 and nine days in 2026-03) carry mean absolute errors of 60 to 127 points, all
  one-sided. March 2026 (bias +50.7, 55% of readings above q95) looks like a level shift that the lags
  have not caught.
- D12's absolute error peaks at 10:00 to 14:00 (18 to 20 points, against 8 to 9 at night). December
  2025 has a positive bias of 8 points and the worst days are 8 to 18 December.

## Diagnosis (CPU only, no fit)

Script `e00641_diagnose.py`, reading the stored export of experiments 612 and 613 (the K1, Candidate,
and K3 files, five development months), with the cleaned `observed_power`, the 2025-03-01 cut at D15
and D18, and the heuristic. Rows with a null observed power are dropped, and the count per month is
reported. For D15, D12, and their neighbours (all 16 Primaries for the comparison):

1. Error by hour of day (local), month, lead band, and a weather regime (cold night, warm midday,
   from the ensemble temperature that the export holds, or from ERA5 if it does not).
2. Residual autocorrelation of the daily mean error (lags 1 to 14 days), and the share of absolute
   error carried by the worst 5%, 10%, and 20% of days.
3. Level shifts: the daily median of observed power minus the previous 28 days' median, flagged
   when above 3 robust standard deviations for at least 3 days; the number, depth, and duration of
   shifts, overlap with the labelled and verified events in the export, and how much of K1's error
   falls inside them.
4. Midday error against PV: the midday-minus-night error at D12 against the sunshine in the ERA5
   shortwave column.
5. Step changes and their sizes (the daily-mean first difference), as in experiment 361.
6. Metered generators: from the metadata tables, whether a metered generator sits behind D15 or D12,
   and where its output exists, its output on the worst days. Any generator is shown anonymised.
7. The same panels for the two neighbours of each Primary with the most similar capacity, as a
   comparison.

Figures (at most two panels each, per the `dataviz` skill): error by hour, error by month, the worst
days as a share of error, the D15 level-shift plot with K1's forecast, the D12 midday plot against
the sunshine.

## Fixes (ranked after the diagnosis)

At most four fixes, each with a prediction that could fail, a cost, and a stop rule, are written into
an amendment commit to this page before any fix is scored. Candidates the diagnosis may support:

- A per-Primary level term: shift the stored median toward the median error of the last 7 to 28 days
  (acts on stored forecasts; no fit).
- A per-Primary, per-hour calibration of the median and the band, fitted on earlier months (acts on
  stored forecasts).
- A transfer-aware level term driven by the level-shift flag of step 3 of the diagnosis.
- A generator-aware input, only if step 6 finds a metered generator behind either Primary (refits
  K1).

The top two are run on the development months with K1's stored forecast as the control (a fix that
refits uses K1's recipe, monthly refits as in experiment 613). Scoring covers D15 and D12 and the
other 14 Primaries. The compute ceiling is 4 GPU hours; stored-forecast fixes use none.

## Standing practices

- Follows: cleaned data, the 2025-03-01 cut at D15 and D18, monthly refits, reuse of stored
  predictions (K1's are the control), tails and cost reporting, example forecast plots, draft pull
  request. Departs from: screening on four folds, because the stored forecasts already cover the five
  development months and the fixes mostly act on them.
- Weather at lag times: any weather column the diagnosis uses is ERA5, which is past weather and is
  not available at forecast time; it is used only to describe, never to fit a fix.
- The July to October 2026 window is not used.
- Generators are anonymised; D15 and D12 are named Primaries.
