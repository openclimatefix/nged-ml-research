---
id: "00645"
slug: k1-vs-stored-bg2-on-the-development-months
title: >-
  Did K1 beat BG2 on the five development months, when both were scored on the same rows?
theme: production-reference-and-best-guess
also_relevant_to: []
status: finished
kind: other
protocol: demand-v2
size: >-
  Simple. (1) It stores one scored table per cut and no model; (2) one design, a free re-score of
  stored forecasts on the rows they share; (3) it reuses the scorer of experiment 620 and the
  detector of experiment 637, whose callers are known; (4) it touches no serving path and no
  degradation rule; (5) it changes no contract.
hypothesis: >-
  Experiment 633's BG2 lead over K1 on July to October 2026 is a feature of that window and of
  BG2's fresher power lags, not a lead BG2 already held on the development months. On the five
  development months, K1 scores no worse than the stored BG2 of experiment 87 on event-free rows at
  the 13 Primaries other than D14, D15, and D18.
prediction: >-
  Written before any BG2 row is joined to the K1 rows. On the five development months K1's CRPS
  minus BG2's is below +0.05 points of capacity on event-free rows at those 13 Primaries, pooled over
  the four lead bands. On all rows at 16 Primaries it lies between -0.30 and +0.30, against the
  window's -0.71 (K1 minus BG2 +0.71 in 633's sign).
decision_rule: >-
  Descriptive. The development months were already read by experiments 87, 425, 449, 470 and 613, so
  the result decides nothing and ranks nothing. The reading counts as "K1 held its own on the
  development months" when the pooled CRPS contrast (K1 minus BG2) on event-free rows without D14,
  D15, and D18 has a month-block 95% interval whose upper end is below +0.10; as "BG2 led" when the
  interval lies wholly above +0.10. Anything else is reported as unclear. The primary reading is experiment 637's frozen detector;
  the export's own `event_free` flag is a second reading and cannot change the verdict. Threshold 0.10 is one seventh of the window's lead of 0.71,
  a gap the page would still call material; no port is at stake.
benchmark: >-
  Experiment 87's stored BG2 (arm `p_s6_bg2`), the manual heuristic, and experiment 633's window
  numbers.
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  CRPS (twice the mean pinball loss over 11 levels) and MAE of the median in points of capacity,
  with 90% coverage and q01, q05, q95, and q99 pinball loss and coverage.
verdict: supported
best_arm: >-
  K1 on the 13 Primaries other than D14, D15, and D18, event-free rows, days 1 to 10: CRPS 3.31
  points of capacity against BG2's 3.63 (K1 minus BG2 -0.32, month-block 95% interval -0.42 to
  -0.22).
headline: >-
  On the five development months K1 scored 0.32 points of capacity below the stored BG2 (interval
  -0.42 to -0.22) on event-free rows at 13 Primaries, and the sign held in every month and every
  lead band. The window reading of experiment 633 was the opposite sign at 16 Primaries (+0.71).
lessons: >-
  BG2 did not lead K1 when both were fresh, even though BG2 reads power only 3.5 hours old and K1
  reads power at least 24 hours old. BG2's lead in the window therefore came from the window (a
  load transfer at D14 and D18, and K1 fitted once and left unrefitted), not from a standing
  advantage. BG2 leads at D15 only, the Primary with the short history.
take_home_message: >-
  On the five development months the neural network K1 forecast demand better than BG2, by 0.32
  points of capacity at 13 Primaries away from the load transfers, although BG2 sees fresher power.
  BG2's lead over K1 in July to October 2026 is therefore a property of that window, and the
  question to chase is why K1 lost ground there after one fit.
advice:
  - >-
    Quote BG2 beside K1 on the development months (K1 ahead by 0.32 at 13 Primaries) whenever the
    window result of experiment 633 is quoted, so the +0.71 window gap is not read as BG2's usual
    lead (headline table).
  - >-
    Look for what K1's single unrefitted fit lost in July to October 2026, because BG2, refitted
    weekly, held its development-month margin over the manual heuristic (about 0.55 to 0.6 points)
    while K1 did not (comparison table).
  - >-
    Treat D15 as the Primary where BG2 is ahead (K1 minus BG2 +1.41 points) and do not pool it into
    a claim about BG2 in general (per-Primary figure).
production_status: not-for-production
adopted_in: none
---

**On the five development months, K1 forecast demand better than BG2, so BG2's lead in July to
October 2026 does not come from BG2 being better in general.** We asked whether K1, the neural
network that reads power at least 24 hours old, beat the weekly-refitted BG2 of experiment 87, which
reads power 3.5 hours old, on the five test months (origins in October 2025, December 2025, January,
March, and June 2026) on which both had forecasts. We took stored forecasts of both, matched them
row by row, and scored them with the same code. At the 13 Primary substations that are not D14, D15,
or D18, K1's continuous ranked probability score (CRPS, the score of a probabilistic forecast, in
points of capacity, lower is better) was 0.32 below BG2's. Experiment 633 found the opposite
(BG2 ahead by 0.71) at 16 Primaries on July to October 2026. This is a descriptive second reading
of months that earlier experiments already used, so it ranks nothing.

![K1's CRPS minus BG2's, by lead band, on the development months and in experiment 633's window](k1_minus_bg2_by_lead_band.svg)

![K1's CRPS minus BG2's at each Primary, development months, event-free rows](k1_minus_bg2_by_primary.svg)

![Ten-day forecasts of D5 from K1 and from BG2, issued on 10 December 2025](example_d5.svg)

## What was compared

- **K1, Candidate, K3:** the stored forecasts of the roster export (experiments 612 and 613, five
  seeds averaged), read from `backtest_export_roster_2026-11-07/`. Nothing was refitted.
- **BG2:** arm `p_s6_bg2` of [experiment
  87](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations/result/):
  the full five-member blend with the solar offset, refitted weekly, spread factor 1.15 already
  applied. It reads the three power lags of `origin_lags`, whose newest reading is 3.5 hours before
  the origin (`NEWEST_AGE` in `e00087_common.py`). K1 reads lags at least 24 hours old.
- **Rows:** 915,135 rows on which all six systems have a forecast and the cleaned reading is finite,
  out of 1,140,221 exported rows. BG2 has no forecast on the other 225,086 rows (experiment 87 stored
  292 of the daily runs; `row_audit.csv` lists each month). No reading was missing, no valid time is
  on or after 2026-07-01, and the climatology was fitted before each month's first day.
- **Event-free:** days flagged by [experiment
  637](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00637-ara-detector-rescore-of-the-bakeoff-final-test/result/)'s
  frozen detector (rule file hash checked) are removed: 23,339 of the 915,135 rows (2.6%).
  The export's own `event_free` flag is a second reading and gives the same sign everywhere.
- **Scorer and intervals:** the formulas of [experiment
  620](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries/result/)'s
  scorer. The intervals resample the 5 origin months, so they are coarse; day-block intervals are in
  `contrasts.csv` and agree.

## Results

CRPS, points of capacity, days 1 to 10, event-free rows at the 13 Primaries other than D14, D15, and
D18 (722,649 rows). Month-block 95% intervals are on the contrasts.

| System | CRPS | MAE of the median | Coverage of q05 to q95 | q99 pinball loss |
|---|---|---|---|---|
| K1 | 3.31 | 5.95 | 0.865 | 0.241 |
| Candidate | 3.35 | 6.01 | 0.891 | 0.241 |
| K3 | 3.57 | 6.41 | 0.858 | 0.271 |
| BG2 | 3.63 | 6.42 | 0.891 | 0.322 |
| Manual heuristic | 4.18 | 7.31 | 0.822 | 0.495 |
| Climatology | 7.42 | 13.75 | 0.899 | 0.408 |

- **K1 minus BG2:** CRPS -0.32 (-0.42 to -0.22); MAE of the median -0.47 (-0.62 to -0.33); q95
  pinball loss -0.10 (-0.18 to -0.04); q99 pinball loss -0.08 (-0.11 to -0.06). The Candidate is
  -0.28 (-0.38 to -0.19). K3, which reads no power lags, is -0.06 (-0.14 to +0.00). The manual
  heuristic is +0.55 (+0.43 to +0.70).
- **Pre-registered reading:** the upper end of K1's interval (-0.22) is below +0.10, so "K1 held its
  own" is met, and K1's contrast is below the predicted +0.05. K1 was ahead, not level.
- **Cuts:** all rows at 16 Primaries K1 minus BG2 is -0.23 (-0.50 to +0.08), because D15 adds +1.41
  there; without D15 and D18 it is -0.31. Event-free rows at 16 Primaries: -0.24 (-0.52 to +0.08).
- **By month (13 Primaries, event-free):** K1 minus BG2 is -0.22, -0.51, -0.29, -0.36, and -0.18 for
  the five months in order. The sign holds in every month.
- **By lead band (13 Primaries, event-free):** -0.23, -0.33, -0.34, and -0.32 at day 1, day 2,
  days 3 to 5, and days 6 to 10. BG2's fresher lags did not close the day-1 gap.
- **By Primary:** K1 is ahead at 14 of 16 Primaries. BG2 leads at D15 (+1.41) and, by under 0.1, at
  D5. K1 leads most at D18 (-0.80) and D8 (-0.58).
- **Tails:** K1's coverage of q05 to q95 is 0.865 against BG2's 0.891, so K1's band is narrower and
  slightly under-covers. K1's q99 pinball loss is lower (0.241 against 0.322).
  The threshold-weighted CRPS contrast is -0.05 (-0.09 to -0.01).

## Comparison with experiment 633's window

K1 minus BG2 CRPS, points of capacity. Window values are quoted from the leakage audit of experiment
633 (sign flipped), not recomputed; their intervals resample origin days.

| Rows | Reading | Day 1 | Day 2 | Days 3-5 | Days 6-10 |
|---|---|---|---|---|---|
| 16 Primaries, all rows | Development months | -0.04 | -0.17 | -0.22 | -0.29 |
| 16 Primaries, all rows | Window | +0.78 | +0.71 | +0.69 | +0.71 |
| 13 Primaries, event-free | Development months | -0.23 | -0.33 | -0.34 | -0.32 |
| 13 Primaries, event-free | Window | +0.03 | +0.01 | +0.01 | +0.04 |

- **K1 against the manual heuristic moved too:** on the development months K1 is 0.87 below the
  heuristic and BG2 is 0.55 below it (13 Primaries, event-free). BG2 held its margin in the window
  (about 0.6), while K1 fell to +0.10 above it. The shift is in K1, not in BG2.
- **Lag age is not the explanation:** BG2 has the fresher lags in both periods, yet K1 wins here.

## Checks

- **Mean against median forecast:** the stored forecasts are quantiles, with no mean forecast. The
  median is the point forecast; the MAE of the median moves the same way as CRPS in every contrast
  above, so trigger 3 did not fire.
- **Triggers:** none fired. No control was run, the result agrees with the development-month
  evidence in experiments 425, 449, and 470 (networks ahead of BG2), and dropping D15 and D18 does
  not flip the sign. A fresh Sonnet agent reviewed the code before the run and found four items,
  fixed: the plan's sign and threshold wording, a missing filter on BG2's valid times, a guard for
  fewer than three bootstrap blocks, and a capacity-agreement assertion.
- **Dropped half-hours:** the stored forecasts exclude the flagged `substation_zero` half-hours
  (no reading was missing in the scored rows).
- **Not done:** no avoidable-curtailment cost, and no load-shift event plots (the export holds one
  verified transfer, D5 and D16 in December 2025). The thresholds of the threshold-weighted CRPS use
  the cleaned power before 2025-10-01 for every month.

## Caveats and next steps

- **BG2 has 80% of the K1 rows** (292 of the daily runs), so late-June origins are thinner; the
  rows are common to all systems, so the comparison stays paired.
- **The Candidate's band widths were set on these months** (experiment 613, in-sample), and K1 was
  selected on them too, so development-month margins are optimistic for the networks. BG2 was also
  tuned on its validation year.
- **Next:** find what K1 lost after one unrefitted fit in July to October 2026 (a refit of K1 on
  the window's early months is the direct test).
