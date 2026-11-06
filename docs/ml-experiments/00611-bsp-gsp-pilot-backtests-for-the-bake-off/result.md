---
id: "00611"
slug: bsp-gsp-pilot-backtests-for-the-bake-off
title: >-
  How well does the lag-free demand network forecast the 2 bulk supply points and 2 grid supply
  points when each group is pooled on its own?
theme: hierarchy-and-pooling
also_relevant_to: []
status: finished
kind: other
date: 2026-11-06
size: >-
  medium, by the five triggers. Stored: fitted weights, forecasts, and a Parquet export, all under
  this experiment's private data folder. More than one design is defensible, which is why the plan
  fixes two (Design 1, Design 2) from experiment 604 before any score is read. Callers: the scripts
  import experiments 87, 397, 415, 428, 448, 510, 511, 559, 564, 565, and 577. No serving path or
  degradation rule is touched (this repository holds none). One Sonnet code review before the first
  fit, one Sonnet write-up review, one Sonnet figure review.
hypothesis: >-
  The lag-free network with the steadier offset O2 (experiment 577), pooled per group, forecasts
  the 2 bulk supply points (BSPs) and the 2 grid supply points (GSPs) at least as well as a
  month-by-half-hour climatology, and at the BSPs as well as the weekly XGBoost model (BG2).
prediction: >-
  Written before any score is read. Design 1 beats the climatology at all 4 series at 24 to 120 h
  (CRPS lower by at least 1 point of effective capacity, as experiment 564 found at 24 to 120 h).
  Design 1 is within the minimum detectable effect of BG2 at the BSPs and worse than BG2 by more
  than 1 point at the GSPs, because the GSPs carry embedded generation. The q05 to q95 band covers
  between 0.85 and 0.95 of rows at every lead band. The GSP that mostly exports power has a
  coverage below the other three series.
decision_rule: >-
  This is a pilot that produces forecasts for the bake-off, so there is no adoption rule. Design 1
  against Design 2 and each design against the references are descriptive; the Bonferroni count is
  stated beside each contrast, with its minimum detectable effect. A contrast counts as a gain when
  its Bonferroni month-block interval (5 blocks) excludes zero.
benchmark: >-
  Climatology of the fold's training window; weekly XGBoost model (BG2) of experiment 415 where its
  stored forecasts cover the same rows, and otherwise stated as missing.
row_set_id: experiment_611_nodes_v1
metric: >-
  CRPS in points of effective capacity, q05 to q95 coverage, and q05 to q95 width, by lead band
  (0 to 24, 24 to 120, 120 to 240 h) and by group (BSPs, GSPs), with the GSP that mostly exports
  power reported separately.
code: ml-experiments/00611-bsp-gsp-pilot-backtests-for-the-bake-off/
verdict: partly supported
best_arm: >-
  Design 1 (lag-free network with the O2 offset) at the bulk supply points: CRPS 6.23 points of
  effective capacity at 24 to 120 h, against 7.05 for BG2 and 9.22 for the climatology. At the
  grid supply points Design 1 (17.89) and Design 2 (17.76) tie BG2 (18.00).
headline: >-
  Design 1 beats the weekly XGBoost model (BG2) at the 2 bulk supply points by 0.84 points at 24 to
  120 h (Bonferroni interval -1.63 to -0.24, minimum detectable effect 1.11) and ties it at the 2
  grid supply points (-0.05, interval -4.31 to +5.70). The q05 to q95 band covers 0.84 (bulk) and
  0.86 (grid) of rows against a target of 0.90.
lessons:
  - >-
    Adding power lags (Design 2) did not improve on the lag-free network at either group, so the
    weather and calendar inputs carry almost all the skill at these 4 series.
  - >-
    Both designs under-cover: the q05 to q95 band holds the observed power on 0.80 to 0.88 of
    rows, and the one-sided q95 share is 0.88 (bulk) and 0.92 (grid) against 0.95.
take_home_message: >-
  A neural network that reads weather and the calendar but no recent power forecasts the 2 bulk
  supply points better than the weekly XGBoost model, and the 2 grid supply points about as well.
  Giving it the previous day's and the previous week's power did not help further. The 90% band is
  too narrow everywhere, so the bake-off should score calibration beside accuracy at these 4 series.
advice:
  - >-
    Use the lag-free network with the O2 offset as the first candidate for the bake-off at the bulk
    and grid supply points, because Design 2's lags added no skill (Table 1, contrast
    "Design 2 minus Design 1").
  - >-
    Widen or recalibrate the q05 to q95 band before reporting coverage at these 4 series, because
    coverage is 0.80 to 0.88 against a target of 0.90 (Table 1).
  - >-
    Report the grid supply point D1 separately, because it mostly exports power and its CRPS
    (15.7 points) differs from the other grid supply point's (20.1 points) (Table 2).
production_status: not-for-production
adopted_in: none
---

**A neural network that reads weather and the calendar but no recent power forecasts the 2 bulk
supply points (BSPs) better than the weekly XGBoost model (BG2), and the 2 grid supply points
(GSPs) about as well.** Adding the previous day's and the previous week's power (Design 2) did not
improve the forecast. The 90% band (q05 to q95) is too narrow for both designs: it holds the
observed power on 0.84 of rows at the BSPs and 0.86 at the GSPs, against a target of 0.90.

We backtested two designs over the five development months (2025-10, 2025-12, 2026-01, 2026-03,
and 2026-06), at leads of 0 to 240 h, with seeds 0 and 1. The design comes from
[experiment 604](../00604-bsp-gsp-pilot-design-for-the-bake-off/result.md). The two BSPs are D13 and
D17 and the two GSPs are D6 and D1. **Design 1** is the lag-free network with the O2 offset of
[experiment 577](../00577-static-between-steps-offsets-and-probabilistic-offset-widening/result.md).
**Design 2** is the lag network of
[experiment 415](../00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/result.md).
Each group is one network of 2 series. BG2, a climatology, and the manual heuristic are the stored
forecasts of experiment 87. Forecasts are in the private Parquet export.

## Figures

![CRPS by lead band for each forecast](figure_1_crps_by_lead_band.svg)

*Figure 1. CRPS (smaller is better) by lead band. Look at the top panel: Design 1 (dark blue) is the
lowest dot at all 3 lead bands, and its range over the 5 months does not overlap the climatology's
at 24 to 120 h. Look at the bottom panel: every forecast's range overlaps, so the 2 grid supply
points cannot rank the forecasts.*

![Coverage of the q05 to q95 band](figure_2_coverage_by_lead_band.svg)

*Figure 2. Share of rows with the observed power inside q05 to q95. The dashed line is the target of
0.90. Look at both panels: Design 1 and Design 2 sit left of the line in every lead band, and BG2 is
the closest to it at the bulk supply points.*

![Width of the q05 to q95 band](figure_3_width_by_lead_band.svg)

*Figure 3. Mean width of the q05 to q95 band as points of effective capacity. Look at the top panel:
the 2 designs' bands are about a quarter narrower than BG2's, which is why their coverage is lower.*

![D13 day-ahead forecast, two fortnights](figure_4_day_ahead_D13.svg)

*Figure 4. D13 (bulk supply point): the forecast issued at 09:00 UTC the day before, over a smooth
fortnight of June 2026 (top) and the fortnight around its highest-power night (bottom). Look for the
observed power leaving the band at the daily minimum in June, where the median runs low.*

![D17 day-ahead forecast, two fortnights](figure_5_day_ahead_D17.svg)

*Figure 5. The same two windows for D17 (bulk supply point).*

![D6 day-ahead forecast, two fortnights](figure_6_day_ahead_D6.svg)

*Figure 6. The same two windows for D6 (grid supply point).*

![D1 day-ahead forecast, two fortnights](figure_7_day_ahead_D1.svg)

*Figure 7. The same two windows for D1 (grid supply point that mostly exports power). Look at the
June panel: the daily swing of about 300 MW is followed closely, and the band is wide.*

## Results

Table 1. CRPS in points of effective capacity (smaller is better), q05 to q95 coverage (target
0.90), and the share of rows at or below q95 (target 0.95), at 24 to 120 h.

| Group | Forecast | CRPS | Coverage | Share below q95 |
|---|---|---|---|---|
| BSPs | Design 1 | 6.23 | 0.84 | 0.88 |
| BSPs | Design 2 | 6.77 | 0.82 | 0.86 |
| BSPs | BG2 | 7.05 | 0.88 | 0.91 |
| BSPs | Manual heuristic | 8.74 | 0.81 | 0.88 |
| BSPs | Climatology | 9.22 | 0.85 | 0.90 |
| GSPs | Design 1 | 17.89 | 0.86 | 0.92 |
| GSPs | Design 2 | 17.76 | 0.85 | 0.90 |
| GSPs | BG2 | 18.00 | 0.87 | 0.92 |
| GSPs | Manual heuristic | 22.60 | 0.77 | 0.86 |
| GSPs | Climatology | 23.08 | 0.81 | 0.88 |

Table 2. The same measures for each series at 24 to 120 h. D1 mostly exports power.

| Series | Design 1 CRPS | Design 2 CRPS | BG2 CRPS | Design 1 coverage | Design 1 band width (MW) |
|---|---|---|---|---|---|
| D13 (bulk) | 5.19 | 5.69 | 5.72 | 0.80 | 14.0 |
| D17 (bulk) | 7.26 | 7.85 | 8.38 | 0.88 | 20.0 |
| D6 (grid) | 20.12 | 18.22 | 18.22 | 0.85 | 401.0 |
| D1 (grid, exporter) | 15.66 | 17.30 | 17.79 | 0.88 | 152.8 |

Table 3. Contrasts at 24 to 120 h, CRPS in points (negative favours the first forecast). The
Bonferroni count is 42: 7 pairs of forecasts (each design against the climatology, BG2, and the
manual heuristic, plus Design 2 against Design 1), times 3 lead bands, times 2 groups. Each interval
resamples the 5 months at the Bonferroni level. The 24 to 120 h rows are shown here, and the other
lead bands are in the private `scores/contrasts.csv`.

| Contrast | Group | Difference | Interval | Minimum detectable effect |
|---|---|---|---|---|
| Design 1 minus BG2 | BSPs | -0.84 | -1.63 to -0.24 | 1.11 |
| Design 1 minus BG2 | GSPs | -0.05 | -4.31 to +5.70 | 8.06 |
| Design 1 minus climatology | BSPs | -2.99 | -4.26 to -1.95 | 1.88 |
| Design 1 minus climatology | GSPs | -5.08 | -11.71 to +1.95 | 11.07 |
| Design 2 minus Design 1 | BSPs | +0.52 | -0.05 to +1.49 | 1.32 |
| Design 2 minus Design 1 | GSPs | -0.07 | -4.56 to +2.70 | 6.06 |

## Pre-registered predictions

- **Held:** Design 1 beats the climatology at all 4 series by at least 1 point at 24 to 120 h
  (Table 2 gives D13 5.19 against 7.12, D17 7.26 against 11.32, D1 15.66 against 24.65, D6 20.12
  against 21.50; the last is 1.4 points). Design 1 is within the minimum detectable effect of BG2 at
  the BSPs.
- **Failed:** Design 1 was not worse than BG2 by more than 1 point at the GSPs (difference -0.05).
  The q05 to q95 band did not cover between 0.85 and 0.95 at every lead band (the BSPs reach 0.80 to
  0.85). The exporting GSP D1 did not have the lowest coverage (0.88, against 0.80 at D13).

## Checks

- **Leakage:** a fresh Sonnet reviewer found none: fits stop at the refit cut, training rows end 3.5
  h before it, and the weather run is issued at or before the origin minus 9 h.
- **Mean forecast against median forecast, and the triggers:** not computed, because the stored
  forecasts are quantiles only. No audit trigger could fire on a contrast of the mean forecast.
  Design 2 is not a nested arm of Design 1, because the two use different fitting schedules.
- **July 2026:** never read.

## Caveats

- **Five months and 2 series per group.** Each interval resamples 5 months, so the intervals are
  wide, and the GSP contrasts cannot rank any forecast. The result is a pilot, not a ranking.
- **Design 2 is not the A1 network.** The shared lag encoder A1 sits in the three-step stack of
  experiment 574 for the Primaries, so Design 2 uses experiment 415's plain lag network, one fit per
  month, without A1.
- **BG2 and the other references** are single stored fits of experiment 87, scored on the rows they
  share with each design. The lead range is 0 to 240 h. NGED's horizon is 1 to 10 days.
- **Pretraining reached the 60-epoch cap** in the Design 1 fits, so early stopping did not trigger,
  as in experiment 577.
- **Not scored:** q01 and q99 pinball loss and curtailment cost. The export holds q01 and q99, so
  either can be scored later.
- **Labels are incomplete.** NGED's event labels miss some events, so a clean half-hour may hold an
  unlabelled load-shift.

## Reproduction

`run_d1.sh` fits Design 1, `run_d2.sh` fits Design 2, `e00611_score.py D1 D2` writes the scores and
the export, and `e00611_figures.py` draws the figures. Cost: about 15 minutes of GPU process time,
against a ceiling of 3 GPU-h. Code reused: the folders of experiments 87, 397, 415, 428, 448, 510,
511, 559, 564, 565, and 577.
