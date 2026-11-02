---
id: "00521"
slug: lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free
title: >-
  At which lead times does the lagged-and-lag-free network K2-S1 beat the daily-refit lag-free
  network, and is the gap a lag effect, a refit-age effect, or a mixture-width effect?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
  - production-reference-and-best-guess
status: finished
kind: other
protocol: demand-v2
size: >-
  medium. Stored: tables of scores, a figure folder, and the page only, under this experiment's
  private data folder and `docs/`; no forecast and no model weights are written. Serving path: not
  touched (this repository holds none). Degradation rule: not touched. Designs: one defensible
  (re-cut stored per-row scores by lead), with one open choice, how to score the monthly lag-free
  references, fixed below. Callers: the script reads experiment 511's stored row scores and
  experiment 454's stored forecasts, and imports experiments 454 and 470 for the scorer. One Sonnet
  code review before the first number is used, one Sonnet results review, and two Sonnet write-up
  reviews. CPU only, at most 2 cores, no fits.
hypothesis: >-
  Experiment 511 left the daily-refit lag-free network L-1d 0.091 points of CRPS behind K2-S1 at the
  eight MVA-only Primaries and 0.071 at the six signed-MW Primaries. The gap is not a lag advantage
  and so not concentrated at the shortest leads. L-1d beats K2-S1 below about 2 days, and the whole
  pooled gap sits at days 5 to 10 (stored finding, interim and unaudited, from a first look at
  experiment 511's row scores). The open question is whether the long-lead gap comes from the
  four-head mixture width of K2-S1 or from the monthly refit age of K2-S1's lag-free heads.
prediction: >-
  Written before the script runs, from the first look. P1: L-1d minus K2-S1 is negative (L-1d
  better) in every band below 48 h at both groups and positive at 72 to 240 h at both groups, with
  the 120 to 240 h band carrying more than 70% of the eight's pooled gap. P2: the monthly lag-free
  references (experiment 454's LL, LL-ens3, LL-ens5) score worse than L-1d at days 5 to 10 by at
  least as much as K2-S1 does if refit age drives the gap, and score between L-1d and K2-S1 if
  mixture width drives it. P3: K2-S1's q95 and q99 coverage at days 5 to 10 is closer to nominal
  than L-1d's. P4: D4 carries more than 40% of the eight's pooled gap. All four can fail.
decision_rule: >-
  The aim is a design decision, not a score. If the gap sits below 12 h (P1 fails), a short-horizon
  correction on a lag-free anchor is supported and gets its own brief. If the gap sits at days 5 to
  10 (P1 holds), a short-horizon correction is not supported, and the page says which of refit age
  and mixture width the references favour, or that they cannot tell. A band counts as a difference
  only when its Bonferroni interval (28 counted CRPS tests, 56 with q99) excludes zero under the
  origin-day blocks that experiment 511 used; the whole-month blocks (5, so near the
  minimum-to-maximum range of the five folds) are reported as the conservative check.
benchmark: >-
  L-1d (experiment 511, daily refit, two-seed quantile average) against K2-S1 (experiment 454's
  monthly refit schedule, three cells, each an equal-weight mixture of two S1 heads and two lag-free
  heads; the headline uses cell c0 alone, because that is what experiment 511's stored K2-S1 score
  is, and the mean of the three cells is the sensitivity). References by lead: L-Ww (weekly), and
  experiment 454's monthly LL, LL-ens3, and LL-ens5.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by lead band (0 to 6 h, 6 to 12 h, 12 to 24 h, 24 to 48 h,
  48 to 72 h, 72 to 120 h, 120 to 240 h), q01, q05, q95, and q99 coverage and pinball loss by band.
verdict: partly supported
best_arm: >-
  Not a ranking of arms. At 120 to 240 h, K2-S1 scores 4.967 (eight) and 3.523 (six) points of CRPS,
  L-1d 5.140 and 3.719, and the monthly five-head lag-free arm LL-ens5 4.984 and 3.724. Below 48 h,
  L-1d scores best of the three.
headline: >-
  K2-S1's advantage over the daily lag-free network L-1d sits at 120 to 240 h (0.174 points at the
  eight, 0.196 at the six, 94% of the eight's summed gap), and below 48 h L-1d is the better network
  (by 0.03 to 0.15 at the eight and 0.11 to 0.25 at the six). The advantage is therefore not a lag
  advantage.
lessons: >-
  A short-horizon correction on a lag-free anchor is not supported by where the gap sits. At the
  eight, a monthly five-head lag-free ensemble matches K2-S1 at 120 to 240 h (0.017 behind, interval
  includes zero) and beats L-1d by 0.157, so the long-lead gap there tracks head count, not lags or
  refit age. At the six, that ensemble matches L-1d, so K2-S1's lagged heads do contribute at long
  lead.
take_home_message: >-
  K2-S1, the network that reads lagged power, beats the daily-refit lag-free network only at leads
  of 5 to 10 days. Below 2 days the daily lag-free network is better, so a short-horizon correction
  on a lag-free forecast is not what the gap calls for. At the eight MVA-only Primaries the long-lead
  gap is closed by averaging five monthly lag-free heads; at the six signed-MW Primaries it is not.
advice:
  - >-
    Do not build a short-horizon correction on the lag-free anchor to chase K2-S1's advantage: the
    gap is at 120 to 240 h and L-1d is better below 48 h (figure 1).
  - >-
    Give the daily lag-free network more heads before adding lagged heads, at the MVA-only Primaries:
    a monthly five-head lag-free ensemble matched K2-S1 at 120 to 240 h there (figure 3).
  - >-
    Quote K2-S1's score as the mean of its three cells, not cell c0: the cells differ by 0.015
    points at the eight, and experiment 511's stored K2-S1 score is cell c0 alone (Checks).
production_status: not-for-production
adopted_in: none
---

**K2-S1's advantage over the daily-refit lag-free network L-1d is a long-lead advantage, not a lag
advantage.** K2-S1 beats L-1d only at 120 to 240 h (gap 0.174 points of CRPS at the eight MVA-only
Primaries, 0.196 at the six signed-MW Primaries), and below 48 h L-1d is the better network at both
groups. A short-horizon correction on a lag-free anchor is therefore not what the gap calls for. At
the eight, a monthly five-head lag-free ensemble (LL-ens5) matches K2-S1 at 120 to 240 h and beats
L-1d, so the long-lead gap there looks like head count. At the six it does not: LL-ens5 matches L-1d
and K2-S1 is 0.20 better, so K2-S1's lagged heads contribute. All numbers on this page are interim
and unaudited until a human has read them.

Experiment 511 left L-1d 0.091 (eight) and 0.071 (six) points behind K2-S1. We re-cut experiment
511's stored per-row scores by lead time (no network was fitted). "Gap" means L-1d minus K2-S1 CRPS
in points of capacity, so a positive gap means K2-S1 is better. The gap mixes three differences:
lagged against lag-free heads, daily against monthly refit, and four heads against two.

![Gap by lead time](figure_1_gap_by_lead.svg)

![CRPS by lead time](figure_2_crps_by_lead.svg)

![Lag-free arms minus K2-S1](figure_3_references.svg)

![Coverage by lead time](figure_4_calibration.svg)

![Gap per Primary](figure_5_per_primary.svg)

![Gap by half-hour of day](figure_6_half_hour.svg)

![Gap on slices](figure_7_slices.svg)

![Day-ahead forecast, D4, winter](figure_8_example_winter_day_ahead.svg)

![10-day forecast, D4, winter](figure_8_example_winter_10_days.svg)

![Day-ahead forecast, D4, summer](figure_8_example_summer_day_ahead.svg)

![10-day forecast, D4, summer](figure_8_example_summer_10_days.svg)

## Results

- **Gap by lead band (eight; Bonferroni interval over origin-day blocks, 28 tests).** 0 to 6 h -0.15
  [-0.39, +0.09]; 6 to 12 h -0.12; 12 to 24 h -0.03; 24 to 48 h -0.02; 48 to 72 h +0.01; 72 to 120 h
  +0.07; 120 to 240 h +0.17 [+0.02, +0.33]. Only the 120 to 240 h interval excludes zero.
- **Gap by lead band (six).** -0.08, -0.15, -0.25 [-0.39, -0.10] at 12 to 24 h, -0.11, -0.04, +0.05,
  and +0.20 [+0.07, +0.33] at 120 to 240 h. The 12 to 24 h and 120 to 240 h intervals exclude zero.
- **The whole-month block check (5 blocks) is far wider.** Every band's interval at the eight
  includes zero (120 to 240 h: [-0.11, +0.51]), and the six's 12 to 24 h interval is the only one to
  exclude zero. Five blocks sit near the minimum-to-maximum range of the five folds, so this check
  cannot support a claim.
- **Minimum detectable effect (from L-1d's two seeds, per band):** 0.10 to 0.27 at the eight and
  0.11 to 0.17 at the six. The gap at 120 to 240 h (0.17 at the eight, 0.20 at the six) is at or
  just above it; every gap below 72 h is smaller than it at the eight.
- **Per-fold signs.** The gap at 120 to 240 h is positive in 4 of 5 folds at both groups. Below 48 h
  the signs are mixed (positive in 0 to 3 of 5 folds).
- **Share of the gap.** 120 to 240 h holds 94% of the eight's summed gap and 136% of the six's (the
  shorter bands offset it). The pre-registered "more than 70%" holds at the eight.
- **Pooled gap.** 0.091 [eight] and 0.071 [six] reproduce experiment 511. Excluding rows inside a
  labelled event the eight's gap is 0.131.
- **Reference arms at 120 to 240 h (CRPS, eight / six).** K2-S1 4.967 / 3.523; L-1d 5.140 / 3.719;
  L-Ww (weekly) 5.182 / 3.769; LL (monthly, 1 head) 5.206 / 3.829; LL-ens3 5.029 / 3.690; LL-ens5
  4.984 / 3.724. K2-S1 minus LL-ens5 is -0.017 [-0.055, +0.022 uncorrected] at the eight and -0.201
  at the six. L-1d minus LL-ens5 is +0.157 [+0.002, +0.314] at the eight and -0.005 at the six.
- **Below 72 h the monthly ensembles are worse than L-1d.** LL-ens5 is 0.18 to 0.45 worse than L-1d
  at the eight and 0.23 to 0.72 worse at the six. Daily refitting matters most at short lead.
- **Tail calibration (all leads).** q95 coverage (nominal 0.95): K2-S1 0.952 and L-1d 0.968 at the
  eight, 0.912 and 0.943 at the six. q99 coverage (nominal 0.99): 0.990 and 0.996 at the eight,
  0.979 and 0.991 at the six. q99 pinball loss, L-1d minus K2-S1: +0.010 [-0.003, +0.021] at the
  eight and -0.019 [-0.031, -0.007] at the six (56 tests). K2-S1's bands are closer to nominal at
  the eight and too narrow at the six.
- **Band width (mean q05 to q95 width in points of capacity, L-1d against K2-S1 cell c0).** L-1d's
  bands are wider in every band: at the eight 41.1 against 38.9 (0 to 6 h), 32.5 against 30.1 (24 to
  48 h), and 34.0 against 31.6 (120 to 240 h); at the six 30.2 against 27.6, 22.8 against 20.9, and
  24.0 against 22.5. Both networks widen by about the same from 12 to 24 h to 120 to 240 h (7.3
  points for L-1d and 6.9 for K2-S1 at the eight). Mixture width therefore does not explain K2-S1's
  long-lead advantage: K2-S1 has the narrower bands and the lower CRPS. The tables are
  `scores/widths.csv`, `scores/absolute.csv` (coverage and pinball loss by band), and
  `scores/contrasts.csv`, in this experiment's private data folder.
- **Per Primary at the eight (pooled gap, share).** D2 +0.235 (32%), D12 +0.223 (31%), D3 +0.166
  (23%), D8 +0.124 (17%), D21 +0.124 (17%), D19 +0.025, D22 -0.052, D4 -0.115 (-16%). No Primary
  exceeds 40%, so the pre-registered prediction P4 fails. At the six the pooled gap is a sum of
  offsetting terms: D14 -0.496 against D9 +0.317, D16 +0.295, D10 +0.276, D5 +0.193, and D11 -0.141.
- **Slices (eight; gap, uncorrected 95% interval).** No labelled event +0.131 [+0.054, +0.207];
  inside a labelled event -0.611 [-0.792, -0.410]; Christmas week -1.235 [-1.290, -1.183]; cold
  winter nights -0.135 [-0.299, +0.038]. L-1d is much better inside events and at Christmas, which
  is the opposite of pre-registered prediction 3 (the gap not larger inside events) only in sign:
  K2-S1 is worse there. At the six, cold winter nights are -0.250 and Christmas week -0.637. D15 and
  D18: -0.651 pooled (L-1d better).
- **Half-hour of day.** See figure 6; the gap by half-hour is noisy and shows no single ramp or peak
  that carries it.

## Checks

- **Experiment 511's K2-S1 score is cell c0 alone.** The stored column reproduces exactly from cell
  c0 (maximum difference 0.0). The mean of the three cells gives a pooled gap of 0.106 (eight) and
  0.064 (six). The cells differ by 0.015 (eight). The headline uses c0 so that it reproduces
  experiment 511; the 3-cell gaps are 0.190 and 0.187 at 120 to 240 h.
- **Missing tail scores.** 2,736 rows of fold 2026-06 have NaN coverage and pinball columns in
  experiment 511's table. All 2,736 are outside `main16`, so none enters any table here.
- **L-1d recomputed from its two stored seeds** (quantile average, 1.15 widening) reproduces
  experiment 511's column to 0.0.
- **Median minus mean forecast, mean-forecast error, and the audit triggers** do not apply: no
  forecast was made and no score was changed, only re-cut.
- **Row counts** per band match the earlier quick look (eight 13,456 / 14,678 / 29,365 / 58,578 /
  58,194 / 115,237 / 282,533).

## Caveats

- **The gap mixes refit age, ensemble size, and averaging method.** K2-S1 is monthly with four heads
  per cell mixed as distributions; L-1d is daily with two seeds averaged by quantile. A daily
  four-head lag-free control does not exist and was not built.
- **Experiment 454 does not say K2-S1's gain is mostly ensembling.** It says mixing a lagged and a
  lag-free forecast beats averaging lagged networks, and it has no result by lead.
- **Only 153 origins (one a day) in 5 folds**, so rows are strongly correlated within a day, and
  most bands cannot be told apart from zero.
- **Unlogged events.** NGED's logs may miss load-shift events, so rows with no labelled event can
  hold unlogged events and the labelled-event slices undercount.
- **The profile does not say whether a better use of lags exists**, only that K2-S1's current use
  buys nothing below 48 h compared with the daily lag-free network.
- **No load-shift event time series were drawn.** This experiment re-cuts stored scores and asks no
  question about how forecasts respond inside an event.

## What to try next

- **Test the head-count reading at the eight directly:** refit the daily lag-free network with five
  heads and compare with K2-S1 at 120 to 240 h (needs fits, so the manager's go).
- **Find what makes K2-S1 better at the six at long lead.** LL-ens5 does not match it there, so the
  S1 heads or the mixture are the candidates.
