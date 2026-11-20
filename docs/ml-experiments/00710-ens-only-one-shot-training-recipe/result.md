---
id: "00710"
slug: ens-only-one-shot-training-recipe
title: >-
  Can the demand network train on ECMWF ENS alone, in one stage or with a recency-weighted final
  stage, and forecast as well as the 3-stage recipe that pre-trains on ERA5 and CAMS?
theme: retraining-and-training-history
also_relevant_to:
  - weather-products-and-blending
  - switching-labels-in-the-forecast
status: finished
kind: change
protocol: demand-v2
size: >-
  Complex. (1) It stores about 380 fitted networks in the minimal plan, with per-row forecasts and
  two injected-event forecasts per network; (2) more than one design defensibly tests the hypothesis
  (the shape, mechanism, and half-life of the recency weight, and the injection state); (3) it
  reuses the training code of experiments 425 and 707, the injection code of experiments 581 and
  699, and the scoring code of experiments 587 and 707, whose callers cannot be named without
  searching; (4) it touches no serving path or degradation rule; (5) it changes no contract. Full
  set of reviews: plan, code, results, write-up, prose.
hypothesis: >-
  A pooled mixture density network trained only on the 47 to 58 months of ECMWF ENS history (window
  F of experiment 707), with synthetic switching events injected into its lag inputs and a recency
  weight on the final stage or throughout one stage, forecasts demand as well as the 3-stage
  reference that first pre-trains on ERA5 and CAMS, within 0.02 points of CRPS.
prediction: >-
  Written before any fit, as pre-registration. C1 (the best ENS-only recipe `B*-i` minus the
  reference `M1-i`) is between -0.02 and +0.03 points of CRPS at the six signed-MW Primaries, and
  between 0 and +0.04 at the eight MVA-only Primaries, where the reference held a lead of 0.041 in
  experiment 707. The placebo contrast C5 is below zero at the eight. Injection raises CRPS on
  ordinary rows by under 0.02 (I1 to I3) and lowers the slump share (I4 to I6).
decision_rule: >-
  Non-inferiority margin 0.02 points of CRPS, the demand protocol's tolerance for a headline. C1 is
  read per group on its interval [lower, upper] at 99.17% (6 readings, family R), and the first test
  that holds gives the verdict. Lower limit above +0.02: refuted. Upper limit below +0.02:
  supported (and beats the reference where the upper limit is below 0). Lower limit above 0: not
  supported. Otherwise inconclusive. A supported verdict also needs the point estimate of C1 below
  +0.02 on seeds 0 to 2 and on seeds 3 and 4 separately, the upper limit on the eight without D8
  below +0.02, and the one-sided 95% bounds on the tails, lead bands, cold nights, and sunny
  half-hours within 0.05. The experiment is supported when both groups are, partly supported when
  one is, and otherwise the worse of the two group verdicts. A claim that a recency weight helps
  needs the upper limits of C4 and of the placebo contrast C5 both below 0.
benchmark: >-
  The reference recipe `M1`, which is arm `A_F` of experiment 707 (ERA5 and CAMS pre-training, then
  ENS, then a recency-weighted fine-tune), refitted with injection (`M1-i`) and stored without it
  (`M1-n`); the ENS-only arms `M2` (`R_F`) and `M3`; climatology, the manual heuristic, BG2, and
  `S0M` as reference rows.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS (twice the mean pinball loss over 11 quantile levels) at lead days 1 to 10 in points of
  effective capacity, with the normalised MAE of the median beside it, and, per Primary group, the
  sunny half-hours, cold nights, the slump share into injected and real switching events, and the
  half-hours after an event.
verdict: partly supported
best_arm: >-
  `FTh90d-i` (ENS-only, two stages, 90-day half-life in the fine-tune, trained with injection): CRPS
  2.826 points at the six signed-MW Primaries and 3.596 at the eight MVA-only Primaries on seeds 0
  to 4, against 2.860 and 3.606 for the 3-stage reference `M1-i`.
headline: >-
  The ENS-only recipe `FTh90d-i` minus the 3-stage reference `M1-i` (C1) is -0.034 points of CRPS
  (99.17% interval -0.084 to -0.001) at the six signed-MW Primaries, so the recipe matches and
  beats the reference there. At the eight MVA-only Primaries C1 is -0.010 (-0.039 to +0.014), within
  the 0.02 margin, but the guard "upper limit below +0.02 without D8" failed (+0.012, -0.013 to
  +0.038), so the eight is inconclusive. Giving the reference the same 90-day half-life (E1) removed
  most of the gap at the eight.
lessons: >-
  C1 at the six rests on 3 winter months (2025-12, 2026-01, 2026-02: -0.126, -0.173, -0.070) and
  is -0.021 (-0.047 to -0.002, 95%) without 2026-01. The ENS-only recipe's advantage came from the
  90-day half-life and from injection harming the reference: injection raised the reference's CRPS
  by +0.036 at the eight (I1, +0.012 to +0.060) and the ENS-only recipes' by +0.008 and -0.004 (I2,
  I3). One-stage recipes scored 0.052 to 0.070 worse than the two-stage ENS-only recipe (C3), and
  a recency weight beat equal weights and the placebo in one stage (C4, C5). Injection cut the
  slump share from about 0.22 to 0.26 down to about 0.08 to 0.13 (I4 to I6).
take_home_message: >-
  A demand neural network trained only on ECMWF ENS forecasts, with a short fine-tune that favours
  the last 90 days, forecast as well as the 3-stage recipe at the six substations that report
  signed megawatts, and within the margin at the eight that report apparent power, but the eight
  depended on one substation (D8). Giving the 3-stage recipe the same 90-day setting removed most
  of the ENS-only recipe's lead, so the experiment does not show that ERA5 and CAMS pre-training
  can be dropped. Training with synthetic load-shift events cut the forecast's slump into real-looking
  events by more than half, at a small cost on ordinary rows.
advice:
  - >-
    Fit the 3-stage reference with the 90-day fine-tune half-life before comparing it with any
    ENS-only recipe, because that one setting accounted for most of the ENS-only lead at the eight
    (E1, C1).
  - >-
    Report the leave-one-Primary-out reading of every contrast at the eight, because D8 and D22
    carry the ENS-only recipe's match there (C1 without D8: +0.012; without D22: +0.003).
  - >-
    Train with injected load-shift events where the forecast must not follow a switching event: the
    slump share fell from 0.22 to 0.26 to 0.08 to 0.13 in all three recipes, at a cost on ordinary
    rows that was clear only for the 3-stage reference at the eight (I1 to I6).
  - >-
    Use a recency weight in any single stage that trains on the long ENS history, and do not use
    that single stage alone: it beat equal weights and the age-permuted placebo, and still scored
    0.052 to 0.070 worse than a stage followed by a short fine-tune (C3, C4, C5).
  - >-
    Select a fine-tune on a recency-weighted held-out set, because the unweighted set picked the
    first epoch in 58 to 60 of 60 fits, so the half-life comparison is a comparison of one epoch
    (fit logs, result review M3).
production_status: candidate
adopted_in: none
---
**An ECMWF-ENS-only recipe forecast as well as the 3-stage reference at the six signed-megawatt
Primaries, and was within the margin at the eight MVA-only Primaries only when D8 was counted.** The
ENS-only recipe `FTh90d-i` trains the pooled mixture density network on the ECMWF ENS archive alone,
then fine-tunes it with a 90-day half-life, and the selection keeps the first epoch of that
fine-tune. Its CRPS, the continuous ranked probability score, is 2.826 points of effective capacity
at the six and 3.596 at the eight, against 2.860 and 3.606 for the 3-stage reference `M1-i`. A lower
CRPS is better. The pre-registered rule gives `partly supported`: `supported and beats the
reference` at the six, and `inconclusive` at the eight.

We asked whether the demand network needs the ERA5 and CAMS pre-training stage. The reference trains
first on ERA5 reanalysis and CAMS solar estimates of past weather, then on the archive of ENS
forecasts, then fine-tunes on the last 730 days of ENS with a half-life of 0.5 years. The ENS-only
recipes drop the first stage, and vary where and how a recency weight enters. Every recipe was also
trained with synthetic switching events injected into its lag inputs ("injection", state `-i`) or
without them (state `-n`). The six Primaries that report signed megawatts and the eight that report
only apparent power (MVA) are scored separately. The recipes, the contrasts, and the decision rule
were fixed before any fit and are kept under "Pre-registered plan" below.

**In brief, the six matched and beat the reference, and the eight is open.** At the six, C1
(`FTh90d-i` minus `M1-i`) is -0.034 points (99.17% interval -0.084 to -0.001). At the eight, C1 is
-0.010 (-0.039 to +0.014), which would be `supported`, but the guard that the upper limit stay below
+0.02 without D8 failed (+0.012, with a 99.17% interval of -0.013 to +0.038).

**Four qualifiers belong to the headline.**

- **The 90-day half-life, not the missing pre-training stage, drives most of the gap at the eight.**
  The reference was never given the 90-day half-life in the minimal plan. The extension arm `M1h90`
  (E1) gives it. `M1h90-i` minus `M1-i` is -0.023 (-0.040 to -0.005, 95%) at the eight, and
  `M1h90-i` minus `FTh90d-i` is -0.012 (-0.032 to +0.009, 97.5%).
- **Injection harmed the reference more than the ENS-only recipes at the eight.** I1 (`M1-i` minus
  `M1-n`) is +0.036 (+0.012 to +0.060) at the eight, against +0.008 for `M2` (I2) and -0.004 for
  `FTh90d` (I3).
- **The six's advantage sits in 3 winter months.** C1 is -0.126, -0.173, and -0.070 in 2025-12,
  2026-01, and 2026-02, and lies within 0.02 of zero in the other 9 months.
- **The six is not separated from experiment 707's failed placebo.** Whether ERA5 and CAMS help the
  six was already unresolved in [experiment
  707](../00707-ens-backfill-pretraining-gain-against-ens-history-length/result.md), and C2 (`M2-i`
  minus `M1-i`, -0.029, -0.073 to +0.003) repeats its sign.

The experiment adopts nothing. The result applies to the pooled network of experiment 707, not to
the modular Candidate network of the production service.

## Figures

![Contrasts C1 to C5, E1, S2, and I1 to I3 at the six and the eight, with the +0.02
margin](contrasts.svg)

![Slump share into injected events with and without injection, per recipe and
group](slump_share.svg)

![Day-ahead and 14-day forecasts of the Primary D10 for the 3-stage reference and the ENS-only
recipe](example_forecast.svg)

![Day-ahead forecasts around six injected and two real load shifts, at both metering
groups](load_shift_events.svg)

**The example forecast is a winter origin, which is a favourable month for the ENS-only recipe.**
The Primary D10 and the origin are those of experiment 425's winter example. The first choice, D5,
was replaced because D5 sits inside a real event. C1 at the six is -0.173 in fold 2026-01, the
largest of the 12 folds, so Figure 3 does not represent a typical month. The injected events in
Figure 4 are the first event in fold order of depth 0.50 for each of the lengths 3, 7, and 15 days
in each group, a rule fixed before looking. The real events are the December pair of D5 and D16
(transfer 23), the only real events that experiment 580 verified and experiment 582 left usable. The
event log from NGED may miss events, so some half-hours that look clean may sit inside an unlogged
event.

## Results

### Headline CRPS and the error of the median

**Every table gives CRPS in points of effective capacity with the normalised mean absolute error
(MAE) of the median forecast in brackets.** Lower is better for both. The rows `M1-n` and `M2-n` are
the stored forecasts of experiment 707's `A_F` and `R_F`. A seed set of `01234` means five seeds,
and `012` means three. The shared scoring rows number 2,922,247 half-hours across every arm and
reference, after dropping 785 actuals that the `substation_zero` rule flags.

| Arm | Seeds | The six | The eight |
|---|---|---|---|
| `M1-i` | 01234 | 2.860 (5.051) | 3.606 (6.537) |
| `M1-n` | 01234 | 2.838 (5.037) | 3.571 (6.470) |
| `M2-i` | 01234 | 2.831 (5.007) | 3.620 (6.571) |
| `M2-n` | 01234 | 2.819 (4.995) | 3.612 (6.565) |
| `M3-i` | 012 | 2.922 (5.171) | 3.782 (6.879) |
| `W-h1-i` | 012 | 2.864 (5.070) | 3.711 (6.750) |
| `PP-i` | 012 | 2.913 (5.158) | 3.673 (6.685) |
| `P-i` | 012 | 2.955 (5.216) | 3.810 (6.930) |
| `FTh90d-i` | 01234 | 2.826 (5.001) | 3.596 (6.520) |
| `FTh90d-n` | 012 | 2.831 (5.016) | 3.604 (6.544) |
| `M1h90-i` | 012 | 2.858 (5.049) | 3.588 (6.495) |

**The four load-shift scenarios separate the ordinary rows from the rows near a switching event.**
"Inside events" are the half-hours that NGED logged as inside a switching event, and "the 7 days
after" are the half-hours in the week after an event ends. The "outside" rows are all the others.
The event log may miss events, so the "outside" rows are not event-free. The two event slices share
no row (checked: 0 rows in both).

At the six, with CRPS and the MAE of the median in brackets:

| Arm | All rows | Outside events and 7 days after | Inside events | The 7 days after |
|---|---|---|---|---|
| `M1-i` | 2.860 (5.051) | 2.613 (4.712) | 9.104 (13.517) | 3.617 (6.216) |
| `M1-n` | 2.838 (5.037) | 2.608 (4.714) | 8.474 (12.824) | 3.764 (6.535) |
| `M2-i` | 2.831 (5.007) | 2.597 (4.682) | 8.771 (13.183) | 3.524 (6.081) |
| `M2-n` | 2.819 (4.995) | 2.592 (4.680) | 8.386 (12.592) | 3.747 (6.432) |
| `M3-i` | 2.922 (5.171) | 2.683 (4.840) | 8.877 (13.331) | 3.762 (6.455) |
| `W-h1-i` | 2.864 (5.070) | 2.629 (4.748) | 8.724 (13.108) | 3.681 (6.209) |
| `PP-i` | 2.913 (5.158) | 2.672 (4.825) | 8.943 (13.455) | 3.734 (6.355) |
| `P-i` | 2.955 (5.216) | 2.722 (4.902) | 8.790 (13.097) | 3.731 (6.246) |
| `FTh90d-i` | 2.826 (5.001) | 2.593 (4.680) | 8.796 (13.190) | 3.420 (5.898) |
| `FTh90d-n` | 2.831 (5.016) | 2.604 (4.704) | 8.438 (12.626) | 3.678 (6.309) |
| `M1h90-i` | 2.858 (5.049) | 2.618 (4.721) | 8.959 (13.289) | 3.560 (6.103) |

At the eight:

| Arm | All rows | Outside events and 7 days after | Inside events | The 7 days after |
|---|---|---|---|---|
| `M1-i` | 3.606 (6.537) | 3.616 (6.558) | 4.645 (8.156) | 2.878 (5.135) |
| `M1-n` | 3.571 (6.470) | 3.578 (6.489) | 4.711 (8.255) | 2.868 (5.113) |
| `M2-i` | 3.620 (6.571) | 3.630 (6.593) | 4.671 (8.222) | 2.885 (5.135) |
| `M2-n` | 3.612 (6.565) | 3.620 (6.586) | 4.719 (8.304) | 2.895 (5.133) |
| `M3-i` | 3.782 (6.879) | 3.796 (6.909) | 4.745 (8.387) | 2.879 (5.144) |
| `W-h1-i` | 3.711 (6.750) | 3.720 (6.772) | 4.755 (8.364) | 2.967 (5.319) |
| `PP-i` | 3.673 (6.685) | 3.681 (6.705) | 4.726 (8.343) | 2.974 (5.327) |
| `P-i` | 3.810 (6.930) | 3.824 (6.959) | 4.725 (8.238) | 2.954 (5.292) |
| `FTh90d-i` | 3.596 (6.520) | 3.605 (6.541) | 4.710 (8.261) | 2.843 (5.062) |
| `FTh90d-n` | 3.604 (6.544) | 3.612 (6.564) | 4.775 (8.371) | 2.885 (5.131) |
| `M1h90-i` | 3.588 (6.495) | 3.597 (6.515) | 4.648 (8.132) | 2.880 (5.161) |

| Subset | The six | The eight |
|---|---|---|
| All rows | 826,396 (12) | 1,103,553 (12) |
| Sunny half-hours (months) | 131,652 (12) | 175,970 (12) |
| Cold nights | 35,496 (7) | 37,704 (7) |
| Inside events | 28,120 (8) | 9,376 (7) |
| The 7 days after | 21,496 (7) | 26,864 (9) |
| Outside events and 7 days after | 776,780 (12) | 1,067,313 (12) |

**Inside logged events the injected state is worse at the six and within 0.07 of the uninjected
state at the eight; in the week after an event, injection helps at the six for all three recipes
that have both states.** `M1-i` scores 9.104 against `M1-n`'s 8.474 inside events at the six, and
3.617 against 3.764 in the 7 days after. None of these absolute differences has a paired interval,
and the 95% intervals on the absolute values overlap.

### Contrasts

**Each row of the table is treatment minus reference in CRPS, with the Bonferroni level of its
family in brackets, and the MAE of the median beside it at 95%.** The month-block bootstrap
resamples whole forecast-origin months (100,000 resamples). C1 to C5 and I1 to I3 are
pre-registered; E1 is the pre-registered extension; S2 is a secondary reading (`FTh90d-i` minus
`M2-i`, the effect of the 90-day half-life on the ENS-only recipe).

| Contrast | Group | CRPS (level) | MAE of the median (95%) |
|---|---|---|---|
| C1 FTh90d-i minus M1-i (headline) | The six | -0.034 (-0.084 to -0.001) (99.17%) | -0.050 (-0.108 to +0.001) |
| C1 FTh90d-i minus M1-i (headline) | The eight | -0.010 (-0.039 to +0.014) (99.17%) | -0.017 (-0.065 to +0.025) |
| C2 M2-i minus M1-i | The six | -0.029 (-0.073 to +0.003) (99.17%) | -0.043 (-0.091 to -0.002) |
| C2 M2-i minus M1-i | The eight | +0.014 (-0.002 to +0.032) (99.17%) | +0.034 (+0.007 to +0.063) |
| C3 PP-i minus M2-i | The six | +0.070 (+0.040 to +0.096) (99.17%) | +0.130 (+0.087 to +0.172) |
| C3 PP-i minus M2-i | The eight | +0.052 (+0.017 to +0.106) (99.17%) | +0.114 (+0.041 to +0.212) |
| C4 W-h1-i minus M3-i | The six | -0.058 (-0.092 to -0.015) (99.38%) | -0.101 (-0.156 to -0.038) |
| C4 W-h1-i minus M3-i | The eight | -0.071 (-0.122 to -0.011) (99.38%) | -0.129 (-0.209 to -0.044) |
| C5 W-h1-i minus P-i (placebo) | The six | -0.091 (-0.154 to -0.029) (99.38%) | -0.145 (-0.232 to -0.064) |
| C5 W-h1-i minus P-i (placebo) | The eight | -0.099 (-0.152 to -0.050) (99.38%) | -0.180 (-0.262 to -0.098) |
| E1 M1h90-i minus FTh90d-i | The six | +0.018 (-0.011 to +0.057) (97.5%) | +0.026 (-0.015 to +0.070) |
| E1 M1h90-i minus FTh90d-i | The eight | -0.012 (-0.032 to +0.009) (97.5%) | -0.031 (-0.063 to +0.001) |
| S2 FTh90d-i minus M2-i | The six | -0.005 (-0.017 to +0.004) (99.17%) | -0.006 (-0.025 to +0.012) |
| S2 FTh90d-i minus M2-i | The eight | -0.024 (-0.050 to -0.002) (99.17%) | -0.051 (-0.093 to -0.013) |
| I1 M1-i minus M1-n | The six | +0.022 (-0.024 to +0.106) (99.58%) | +0.013 (-0.042 to +0.081) |
| I1 M1-i minus M1-n | The eight | +0.036 (+0.012 to +0.060) (99.58%) | +0.067 (+0.031 to +0.102) |
| I2 M2-i minus M2-n | The six | +0.012 (-0.013 to +0.061) (99.58%) | +0.013 (-0.015 to +0.050) |
| I2 M2-i minus M2-n | The eight | +0.008 (-0.023 to +0.046) (99.58%) | +0.006 (-0.039 to +0.056) |
| I3 FTh90d-i minus FTh90d-n | The six | +0.009 (-0.013 to +0.045) (99.58%) | +0.008 (-0.017 to +0.034) |
| I3 FTh90d-i minus FTh90d-n | The eight | -0.004 (-0.046 to +0.041) (99.58%) | -0.018 (-0.077 to +0.042) |

**The decision rule gives these readings.**

- **C1 at the six: `supported and beats the reference`.** The upper limit is -0.001, below zero.
  Every guard held: the point estimate on seeds 0 to 2 is -0.024 (-0.062 to +0.005, 95%) and on
  seeds 3 and 4 is -0.045 (-0.088 to -0.012); `FTh90d-i` minus `M2-i` is -0.005; the q01 and q99
  tails, the five lead bands, and cold nights stayed within 0.05. The protected sunny slice
  `sunny_clear_sky_index` was not built, so the verdict has no sunny guard, as the plan allows.
- **C1 at the eight: `inconclusive`, because the guard "upper limit below +0.02 on the eight without
  D8" failed.** C1 without D8 is +0.012 (-0.013 to +0.038). The other guards held.
- **C2 (`M2-i` minus `M1-i`) is -0.029 at the six and +0.014 at the eight.** The eight repeats the
  sign of experiment 707 with a smaller size (+0.014 here against +0.041 there). The post hoc
  contrast `M2-n` minus `M1-n` (the stored forecasts) is +0.041 (+0.030 to +0.053, 95%) at the eight
  and -0.019 (-0.038 to -0.002) at the six.
- **C3 (`PP-i` minus `M2-i`) is +0.070 at the six and +0.052 at the eight,** so the best one-stage
  recipe scored worse than the two-stage ENS-only recipe. The lower limit is +0.040 at the six,
  above the margin, and +0.017 at the eight, below it.
- **C4 and C5 both lie below zero at both groups, so the rule's wording "a recency weight helps in
  one stage" holds.** C4 (`W-h1-i` minus `M3-i`, equal weights) is -0.058 and -0.071. The placebo is
  not neutral: `P-i` minus `M3-i` is +0.033 (-0.010 to +0.078, 95%) at the six and +0.028 (-0.015 to
  +0.071) at the eight, so C5 (-0.091 and -0.099) is C4 plus the placebo's own loss, and C4 gives
  the size of the gain over equal weights.
- **E1 is explanatory and changes no verdict.** At the eight `M1h90-i` is 0.012 ahead of `FTh90d-i`.
  At the six `M1h90-i` is 0.018 behind (-0.011 to +0.057). The guards of E1 failed on cold nights in
  both groups, and on lead days 2 to 7 at the six, and are read as explanatory.
- **Injection (I1 to I3) did not show "no harm" for any recipe.** The upper limit stays above +0.02
  in all six readings, so the rule's wording "injection does no harm" is not earned. The one clear
  harm is I1 at the eight (+0.036). The pre-registered prediction "under 0.02" holds as a point
  estimate for I2 and I3 and fails for I1 (+0.022 at the six, +0.036 at the eight).
- **The slump reduction (I4 to I6) is clear in every recipe and group.** The paired difference in
  slump share, `-i` minus `-n`, is -0.142 (-0.169 to -0.118) and -0.130 (-0.143 to -0.116) for `M1`,
  -0.156 and -0.133 for `M2`, and -0.154 and -0.131 for `FTh90d`, at the six and the eight. All six
  upper limits lie below zero.

**The prediction written before any fit was right on two counts, wrong on one, and mixed on one.**
C1 was predicted between -0.02 and +0.03 at the six and between 0 and +0.04 at the eight, and came
out below both ranges (-0.034 and -0.010). The placebo contrast C5 was predicted below zero at the
eight, and is (-0.099). Injection was predicted to lower the slump share, and does. Injection was
predicted to raise CRPS on ordinary rows by under 0.02, which holds for I2 and I3 in both groups and
fails for I1 in both.

### Leave-one-Primary-out, fold by fold, and seed halves

**At the eight, C1 changes sign when D8 or D22 is left out, and at the six it does not.** The table
gives C1 (95% intervals) with one Primary removed. Without D8, C1 is +0.012 (-0.007 to +0.031);
without D22, +0.003 (-0.024 to +0.030). At the six, C1 stays between -0.027 and -0.045, but the
upper limit reaches 0.000 without D5 (-0.059 to -0.000) or D10 (-0.073 to -0.000), so the "beats"
wording is fragile.

| Left out | C1, the six (95%) | Left out | C1, the eight (95%) |
|---|---|---|---|
| D5 | -0.027 (-0.059 to -0.000) | D2 | -0.010 (-0.032 to +0.011) |
| D9 | -0.045 (-0.087 to -0.011) | D3 | -0.012 (-0.036 to +0.010) |
| D10 | -0.032 (-0.073 to -0.000) | D4 | -0.021 (-0.043 to -0.002) |
| D11 | -0.041 (-0.081 to -0.008) | D8 | +0.012 (-0.007 to +0.031) |
| D14 | -0.030 (-0.061 to -0.006) | D12 | -0.022 (-0.057 to +0.006) |
| D16 | -0.031 (-0.063 to -0.005) | D19 | -0.022 (-0.041 to -0.004) |
| | | D21 | -0.009 (-0.032 to +0.013) |
| | | D22 | +0.003 (-0.024 to +0.030) |

**C1 at the six is concentrated in winter.** C1 by fold at the six (2025-07 to 2026-06) is +0.001,
-0.006, -0.011, -0.009, -0.010, -0.126, -0.173, -0.070, +0.005, -0.001, -0.019, and +0.020. At the
eight it is -0.034, +0.022, +0.007, -0.062, +0.029, -0.087, -0.015, +0.010, +0.036, +0.001, -0.004,
and -0.020 (negative in 6 of 12 folds). Without fold 2026-01, C1 is -0.021 (-0.047 to -0.002) at the
six and -0.010 (-0.033 to +0.011) at the eight. The t test on the 12 monthly differences gives p =
0.077 at the six and p = 0.371 at the eight.

**The minimum detectable effect of the design was 0.018 points.** The scorer computes it from the
seed-half differences (standard deviation 0.0253 over 12 folds).

### Sunny half-hours and cold nights

**On sunny half-hours the ENS-only recipe and the reference score alike, and on cold nights the
ENS-only recipe scores better than the injected reference but not better than the uninjected one.**
The sunny slice is `sunny_unit_shape`, the descriptive slice; the clear-sky-index slice that the
plan protects could not be built. C1 on cold nights is -0.174 (-0.205 to -0.116, 90%) at the six and
-0.126 (-0.173 to -0.069) at the eight. `FTh90d-i` scores 4.399 on cold nights at the six, against
4.573 for `M1-i` and 4.312 for `M1-n`. Cold nights cover 7 months, so the absolute intervals are
wide (for example 2.378 to 5.539 for `FTh90d-i` at the six). Sunny half-hours and cold nights, six
then eight:

| Arm | Sunny half-hours | Cold nights |
|---|---|---|
| `M1-i` | 3.707 (6.685) | 4.573 (7.167) |
| `M1-n` | 3.682 (6.634) | 4.312 (6.883) |
| `M2-i` | 3.730 (6.739) | 4.466 (7.019) |
| `M2-n` | 3.712 (6.690) | 4.270 (6.747) |
| `M3-i` | 4.151 (7.557) | 4.557 (7.178) |
| `W-h1-i` | 3.795 (6.870) | 4.452 (7.065) |
| `PP-i` | 3.913 (7.063) | 4.590 (7.292) |
| `P-i` | 4.113 (7.464) | 4.653 (7.321) |
| `FTh90d-i` | 3.706 (6.699) | 4.399 (6.887) |
| `FTh90d-n` | 3.706 (6.679) | 4.265 (6.678) |
| `M1h90-i` | 3.702 (6.682) | 4.522 (7.041) |

| Arm | Sunny half-hours | Cold nights |
|---|---|---|
| `M1-i` | 4.871 (8.977) | 3.354 (5.926) |
| `M1-n` | 4.828 (8.868) | 3.320 (5.928) |
| `M2-i` | 4.935 (9.122) | 3.284 (5.775) |
| `M2-n` | 4.902 (9.067) | 3.276 (5.815) |
| `M3-i` | 5.463 (10.200) | 3.523 (6.168) |
| `W-h1-i` | 5.164 (9.625) | 3.505 (6.228) |
| `PP-i` | 5.130 (9.516) | 3.472 (6.142) |
| `P-i` | 5.420 (10.096) | 3.617 (6.417) |
| `FTh90d-i` | 4.850 (8.934) | 3.228 (5.686) |
| `FTh90d-n` | 4.827 (8.883) | 3.238 (5.771) |
| `M1h90-i` | 4.786 (8.781) | 3.316 (5.854) |

### Slump into load-shift events

**Injection cut the slump share of every recipe from about 0.22 to 0.26 down to about 0.08 to
0.13.** The slump share is `sign(d) * (q50 with shifted lags - q50 with unshifted lags) / abs(d)`,
with `d` the depth of the injected event as a share of capacity. A forecast that keeps to the normal
running arrangement scores 0, and one that takes up the whole step scores 1. The table gives 95%
month-block intervals over events (144 events at the six, 192 at the eight).

| Arm | The six: share (95% interval) | The eight: share (95% interval) |
|---|---|---|
| `M1-i` | 0.082 (0.061 to 0.105) | 0.123 (0.095 to 0.150) |
| `M1-n` | 0.224 (0.202 to 0.247) | 0.253 (0.220 to 0.284) |
| `M2-i` | 0.087 (0.065 to 0.111) | 0.127 (0.102 to 0.151) |
| `M2-n` | 0.244 (0.215 to 0.274) | 0.261 (0.224 to 0.294) |
| `FTh90d-i` | 0.083 (0.062 to 0.105) | 0.118 (0.093 to 0.141) |
| `FTh90d-n` | 0.236 (0.208 to 0.266) | 0.249 (0.211 to 0.283) |

**The share rises through an event, and falls back in the week after.** The table gives the share by
days since onset and in the 7 days after the event ends, for seeds 0 to 4 where five exist and seeds
0 to 2 otherwise. The rows "first 4 days" and "day 4 to end" hold different event mixes (108 of the
144 events at the six hold each phase), so the two cannot be compared as one effect.

| Arm | The six: all events | First 4 days | Day 4 to end | 7 days after | The eight: all events | First 4 days | Day 4 to end | 7 days after |
|---|---|---|---|---|---|---|---|---|
| `M1-i` (01234) | 0.082 | 0.061 | 0.100 | 0.085 | 0.121 | 0.085 | 0.148 | 0.103 |
| `M2-i` (01234) | 0.086 | 0.066 | 0.103 | 0.087 | 0.126 | 0.091 | 0.154 | 0.105 |
| `M3-i` (012) | 0.114 | 0.088 | 0.137 | 0.115 | 0.167 | 0.120 | 0.202 | 0.139 |
| `W-h1-i` (012) | 0.105 | 0.085 | 0.124 | 0.106 | 0.153 | 0.115 | 0.184 | 0.125 |
| `PP-i` (012) | 0.101 | 0.083 | 0.117 | 0.106 | 0.144 | 0.110 | 0.171 | 0.123 |
| `P-i` (012) | 0.113 | 0.091 | 0.134 | 0.116 | 0.162 | 0.116 | 0.199 | 0.139 |
| `FTh90d-i` (01234) | 0.082 | 0.061 | 0.099 | 0.088 | 0.116 | 0.083 | 0.144 | 0.104 |
| `FTh90d-n` (012) | 0.236 | 0.180 | 0.286 | 0.165 | 0.249 | 0.194 | 0.293 | 0.158 |
| `M1h90-i` (012) | 0.077 | 0.054 | 0.095 | 0.087 | 0.111 | 0.075 | 0.138 | 0.101 |

**The recipes without a fine-tune hold a larger share than the two-stage recipes.** `M3-i`,
`W-h1-i`, `PP-i`, and `P-i` (seeds 0 to 2) hold 0.10 to 0.11 at the six and 0.14 to 0.17 at the
eight, against 0.08 to 0.09 and 0.12 to 0.13 for `M1-i`, `M2-i`, and `FTh90d-i`.

**The real December events show a small slump share and an error against the normal running
arrangement of 5 to 11 points of capacity in magnitude.** Two real events were verified well enough
to use: D5 and D16, both in transfer 23 of experiment 580 (a step of -0.256 of capacity at D5 and
+0.151 at D16). Only the injected state was stored for the real events, so they cannot show what
injection changed. Per event, with no interval:

| Primary | Transfer | Step (share of capacity) | Arm | Half-hours | Slump share | Error against the normal running arrangement (points) |
|---|---|---|---|---|---|---|
| D5 | 23 | -0.256 | `M1-i` | 2,106 | +0.027 | +7.63 |
| D5 | 23 | -0.256 | `M2-i` | 2,106 | +0.069 | +7.33 |
| D5 | 23 | -0.256 | `FTh90d-i` | 2,106 | +0.065 | +7.75 |
| D16 | 23 | +0.151 | `M1-i` | 1,404 | +0.103 | -7.84 |
| D16 | 23 | +0.151 | `M2-i` | 1,404 | +0.112 | -7.13 |
| D16 | 23 | +0.151 | `FTh90d-i` | 1,404 | +0.104 | -7.22 |
| D5 | 23 | -0.256 | `M1-i` | 9,126 | +0.122 | +11.31 |
| D5 | 23 | -0.256 | `M2-i` | 9,126 | +0.135 | +11.03 |
| D5 | 23 | -0.256 | `FTh90d-i` | 9,126 | +0.130 | +10.84 |
| D16 | 23 | +0.151 | `M1-i` | 9,126 | +0.015 | -5.76 |
| D16 | 23 | +0.151 | `M2-i` | 9,126 | +0.026 | -5.43 |
| D16 | 23 | +0.151 | `FTh90d-i` | 9,126 | +0.013 | -5.59 |

**The first six rows are the December fold and the last six the January fold, each with the
half-hours of that fold.** The slump share is small in both (0.01 to 0.14), and the magnitude of the
error against the normal running arrangement is 5.4 to 11.3 points of capacity. The share differs
little between `M1-i`, `M2-i`, and `FTh90d-i`. The flagged May events (transfers 24 to 26) are not
used, because their depths are unverified (experiment 582), and the lines for transfers 25 and 26 at
D12 are identical in the scorer log, which suggests one event entered twice.

### Coverage, tails, and convergence

**The lower tail is over-covered and the upper tail is near nominal, in the same way for every
arm.** The share of actuals below q05 is 0.022 to 0.038 (nominal 0.05), and below q95 is 0.930 to
0.964 (nominal 0.95). The table gives both shares.

| Arm | The six: q05 / q95 | The eight: q05 / q95 |
|---|---|---|
| `M1-i` | 0.022 / 0.932 | 0.026 / 0.959 |
| `M1-n` | 0.024 / 0.933 | 0.027 / 0.959 |
| `M2-i` | 0.024 / 0.935 | 0.027 / 0.960 |
| `M2-n` | 0.026 / 0.935 | 0.027 / 0.960 |
| `M3-i` | 0.028 / 0.935 | 0.035 / 0.964 |
| `W-h1-i` | 0.023 / 0.933 | 0.029 / 0.960 |
| `PP-i` | 0.025 / 0.932 | 0.030 / 0.962 |
| `P-i` | 0.029 / 0.930 | 0.038 / 0.959 |
| `FTh90d-i` | 0.025 / 0.934 | 0.027 / 0.959 |
| `FTh90d-n` | 0.028 / 0.934 | 0.029 / 0.958 |
| `M1h90-i` | 0.024 / 0.932 | 0.028 / 0.956 |

**No arm ended its final stage at the epoch limit in more than a quarter of its fits, so no arm is
labelled "not converged".** The share at the limit is at most 0.02. That label says nothing about
the fine-tunes, for the reason in caveat 3 below.

| Arm | Fits | Mean steps of the final stage | Mean seconds per fit | Share at the limit | Effective rows per batch |
|---|---|---|---|---|---|
| `M1-i` | 68 | 25,156 | 147 | 0.00 | 10,542 |
| `M2-i` | 68 | 19,528 | 109 | 0.00 | 10,542 |
| `M2-n` | 8 | 16,193 | 50 | 0.00 | 10,571 |
| `W-h1-i` | 44 | 18,579 | 101 | 0.02 | 10,232 |
| `PP-i` | 44 | 19,878 | 106 | 0.02 | 11,899 |
| `FTh90d-i` | 68 | 2,109 | 65 | 0.00 | 5,994 |
| `FTh90d-n` | 36 | 2,123 | 89 | 0.00 | 5,993 |
| `M1h90-i` | 36 | 2,123 | 93 | 0.00 | 5,993 |
| `P-i` | 36 | 17,917 | 113 | 0.00 | 9,757 |

**The median forecast is below the mean forecast at every Primary, by 0.08 to 1.8 points of
capacity.** The table gives the mass-weighted mean of the 11 quantiles minus the median, per
Primary, for seeds 0 to 2 of each arm.

| Arm | D5 | D9 | D10 | D11 | D14 | D16 | D2 | D3 | D4 | D8 | D12 | D19 | D21 | D22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `M1-i` | -0.44 | -0.19 | -0.71 | -0.07 | -0.14 | -0.23 | -0.17 | -0.14 | -0.22 | -0.89 | -1.83 | -0.27 | -0.12 | -0.28 |
| `M2-i` | -0.45 | -0.19 | -0.68 | -0.08 | -0.13 | -0.19 | -0.16 | -0.13 | -0.23 | -1.01 | -1.69 | -0.28 | -0.12 | -0.19 |
| `FTh90d-i` | -0.43 | -0.19 | -0.62 | -0.08 | -0.13 | -0.19 | -0.16 | -0.12 | -0.24 | -1.10 | -1.83 | -0.28 | -0.12 | -0.26 |

### Tuning folds

**The four tuning folds (2024-10, 2025-01, 2025-03, 2025-06) chose `B*`, `O*`, and `h_T`.** `B*` is
`FTh90d`, the ENS-only recipe with the lowest CRPS on folds 2025-03 and 2025-06, `O*` is `PP` (the
best one-stage recipe), and `h_T` is 1 year. The table gives the mean CRPS on those two folds,
interim, from two seeds. Recipes such as `S`, `HW730`, `W-h0.5`, and `W-h2` have a tuning score only
and no verdict.

| Arm | Mean CRPS on tuning folds 2025-03 and 2025-06 |
|---|---|
| `M1` | 4.153 |
| `FTh90d` | 4.186 |
| `M2` | 4.221 |
| `PP` | 4.239 |
| `FTh1` | 4.241 |
| `EMA` | 4.254 |
| `RFE` | 4.275 |
| `CYC` | 4.278 |
| `S` | 4.293 |
| `W-h1` | 4.297 |
| `FLR` | 4.305 |
| `W-h2` | 4.313 |
| `M3` | 4.325 |
| `HW730` | 4.346 |
| `W-h0.5` | 4.355 |
| `SMP` | 4.382 |
| `HW365` | 4.575 |

## Checks

- **Median minus mean, per Primary:** the median forecast is below the mean at all 16 Primaries in
  all three arms checked (`M1-i`, `M2-i`, `FTh90d-i`), by 0.08 to 1.8 points of capacity (D15, which
  is report-only, by 3.4 to 3.9). The largest gaps are D12 (-1.7 to -1.8), D8 (-0.9 to -1.1), and
  D10 (-0.6 to -0.7). The mean is a mass-weighted mean of the 11 quantiles, because the forecast
  stores no mean.
- **Error of the median beside each quantile-score contrast:** the MAE of the median points the same
  way as the CRPS contrast in all 20 readings (C1 to C5, E1, S2, and I1 to I3, in both groups), in
  the contrasts table. Audit trigger 3 did not fire. C1's MAE contrasts are -0.050 and -0.017.
- **Audit trigger 1, a control fails: not fired.** Known-answer tests 1 to 7 passed. Both stop
  checks passed (`M2-i` minus `M1-i`: +0.026; `M2-i` minus `M2-n`: +0.065, against a stop at 0.10).
  The placebo passed in the pre-registered sense (C5 below zero), but it is not neutral (see C4 and
  C5 above).
- **Audit trigger 2, a result contradicts its prediction: fired.** C1 lay below both predicted
  ranges and I1 above the predicted bound. The session tested an explanation: E1 shows that the
  90-day half-life accounts for most of the eight's gap, and I1 to I3 and `M2-n` minus `M1-n` show
  that injection hurts the reference more.
- **Audit trigger 4, leaving one Primary out flips a sign: fired.** At the eight, C1 turns positive
  without D8 (+0.012) or D22 (+0.003). At the six no sign flips, and the "beats" wording weakens
  without D5 or D10. The verdict's wording at the eight is `inconclusive` for this reason.
- **Audit trigger 5, a nested arm beats the one that contains it: weakly fired.** `M2-i`, which
  drops ERA5 and CAMS, scores better than `M1-i` at the six (-0.029, interval includes zero).
  Experiment 707's placebo failed at the six, so the six's sign is unattributed to the weather data.
- **Audit triggers 6 and 7: not fired.** C1's half-widths of 0.041 and 0.027 are 1.7 to 1.8 times
  the 0.024 and 0.015 rescaled from experiment 707, below the factor of 2. No design that depends on
  this result was known when this page was written. Dropping ERA5 and CAMS from a production recipe
  would rest on a headline that has one injection world per fold and seed, which the experiment does
  not replicate.
- **No separate audit has run.** The adversarial result review (`result_review_1.md`, in the private
  briefs folder) re-derived the headline contrasts from the stored forecasts with its own code,
  within 0.004 points of the scorer, which is the re-derivation an audit does. Whether that review
  stands in for the audit is for the maintainer to decide.
- **Code reviews 1 to 4,** each triaged in `NOTES_for_reviewer.md`: review 1 (training loop,
  weights, injection, scorer; one blocker on the job order and scorer, fixed); review 2 (the fixes,
  and the launcher, with N1 to N5 fixed before Phase T and Phase S); review 3 (the E1 and E2
  extension: no blocker; the `--stored M2-n` command and the known-answer gate added); review 4 (the
  aggregate-based scorer, which reproduces the earlier scorer's 282 shared numbers to 8e-16, with no
  blocker).
- **Result review 1: seven major findings (M1 to M7) and nine minor ones (m1 to m9),** triaged
  below.

### Triage of the result review

- **M1, injection harms the reference more (accepted).** Reported in the contrasts and in the
  qualifiers, and tested by E1 and E2: I1 is +0.036 at the eight, while I2 and I3 are +0.008 and
  -0.004. The claim "an ENS-only recipe matches the reference" holds only for networks trained with
  injection. The uninjected contrast S1 (`FTh90d-n` minus `M1-n`) is -0.001 at the six and +0.027 at
  the eight, both `inconclusive` (no guards).
- **M2, C1 mixes the half-life with dropping pre-training (accepted).** E1 fitted the reference with
  the 90-day half-life (36 fits). At the eight, C1 (-0.010) is about C2 (+0.014, the effect of
  dropping pre-training) plus S2 (-0.024, the effect of the 90-day half-life).
- **M3, the fine-tune selection set moves against the recency weight (accepted, caveat 3).** The
  unweighted `V_sel` picked epoch 0 in 60 of 60 `M1-i`, 58 of 60 `M2-i`, and 60 of 60 `FTh90d-i`
  fits.
- **M4, leave-one-Primary-out flips the sign at the eight (accepted).** Reported above, with trigger
  4.
- **M5, "beats" at the six rests on three winter months (accepted).** The verdict is reported as the
  rule gives it, followed by the fold-by-fold and without-2026-01 readings.
- **M6, the slump contrasts I4 and I5 were not read (accepted, resolved).** E2 wrote the bank passes
  of `M1-n` and `M2-n`, and fitted `FTh90d-n`, so I4 to I6 are read.
- **M7, injection did not help on the logged event rows at the six (accepted, descriptive).**
  Reported in the scenario tables.
- **m1, the NaN in the slump table's CRPS-change column (accepted, fixed).** The scorer ignores NaN
  actuals (commit `23509a5d0`).
- **m2, the two slump phases hold different event mixes (accepted, stated).**
- **m3, the real-event lines are ambiguous (accepted).** The page labels the December and January
  folds, and does not use the May events.
- **m4, the steps table omits the cap (accepted, partly).** The final-stage steps and the share at
  the limit are in the convergence table, and no one-stage arm reached its cap of 34,040 steps
  (result review). The cap column is not printed. `FTh90d` and `M3` load their `S_ens` state from
  `M2-i`, so the steps of the fine-tune alone appear.
- **m5, the protected sunny guard did not run (accepted, stated).** The plan allows that fallback.
- **m6, the placebo is not neutral (accepted, stated).** Under C4 and C5 above.
- **m7, pre-registered descriptive readings are missing (accepted, partly).** The page adds the
  load-shift scenario "outside events", the readings without fold 2026-01, leave-one-Primary-out at
  both groups, the example forecasts, and the event plots. It does not add per-Primary tables, the
  D15 and D18 scores beyond the median-minus-mean gap, or a `-i` minus `-n` difference for every
  recipe on every metric.
- **m8, `O*` changed between two launches (accepted, stated in Method).** `O*` was `CYC` for two
  unscored `CYC-i` fits and was then rewritten to `PP` before any scored fold read it.
- **m9, housekeeping (accepted).** The known-answer marker and the committed scripts agree, and the
  2026-06 files hold valid times after 2026-06-30, which the scorer drops.

## Caveats

1. **The headline holds for networks trained with injection.** Without injection, the stored
    reference leads the stored ENS-only recipe `M2-n` at the eight by 0.041 points (post hoc). The
    experiment chose the injected state as the headline before the fits, and no `-n` state of `B*`
    was in the minimal plan.
2. **The seeds and the injection worlds are few.** Each (fold, seed) has one injection world, so the
    injection family rests on at most five draws per fold. C1 at the six is -0.024 on seeds 0 to 2
    and -0.045 on seeds 3 and 4, and the result review found that `M1-i` scores worse on seeds 3 and
    4 (2.882 against 2.864 at the six).
3. **A fine-tune is in effect a fixed one-epoch fine-tune.** The unweighted selection set picked
    epoch 0 and then stopped by patience at 7 epochs. For `FTh90d-i`, the chosen state had a worse
    `V_sel` loss than the starting state in 48 of 60 fits, yet the fine-tune lowers test CRPS
    (`M3-i` minus `M2-i` is +0.078 at the six and +0.161 at the eight, post hoc). The half-life
    comparison (`FTh90d` against `FTh1` against `M2`) is a comparison of one AdamW epoch at 2e-4,
    not of selected optima.
4. **The six and the eight rest on few Primaries.** The six has 6 Primaries and the eight 8, and the
    eight's match depends on D8 and D22.
5. **The real-event result is two events.** Only the December pair of D5 and D16 is a clean real
    test, with no interval, and only the injected state was stored.
6. **The event log may miss events.** An "outside events" half-hour may sit inside an unlogged
    event, and the event slices may therefore understate how far the forecasts fail near a switching
    event.
7. **The placebo tests recency and season adjacency together.** The weather-shuffle placebo of
    experiment 707 was not rerun, because no contrast here claims a pre-training gain.
8. **The sunny guard did not run.** The slice `sunny_clear_sky_index` was not built, so the verdict
    has no sunny guard.
9. **The 12-month bootstrap undercovers.** The t tests on the monthly differences sit beside the
    intervals, and C1 at the six has p = 0.077, which does not exclude zero at an uncorrected 95%.

## What was not run

**Several pre-registered arms and contrasts were not run, because the minimal plan and the E1 and E2
subset never included them and the rest of the extension was not released.** The arms `S`, `HW730`,
`W-h0.5`, and `W-h2` have tuning scores only (folds 2025-03 and 2025-06, two seeds), so C6 (`S-i`
minus `W-h0.5-i`) and C7 (`HW730-i` minus `M3-i`) have no reading. `W-h1` was scored in the `-i`
state only. The `-n` states of `M3`, `W-h1`, `PP`, and `P` were not fitted, and `FTh90d-n` has seeds
0 to 2 only, so the secondary contrast `B*-n` minus `M1-n` is read only as S1 on those seeds. The
scorer does not print the scores of the reference rows (climatology, BG2, the manual heuristic, and
`S0M`), so experiment 707's scores stand for them. Per-Primary tables and the D15 and D18 scores
were not produced.

## What to try next

- **Fit the reference with the 90-day half-life in every group and state, and with a second set of
  injection worlds.** `M1h90-i` ran with seeds 0 to 2 only.
- **Select the fine-tune on a recency-weighted held-out set,** or admit the pre-fine-tune state as a
  candidate, and fit the one-epoch fine-tune as such.
- **Replicate the eight without D8 and D22,** or add Primaries, before any claim that a recipe
  matches the reference at the MVA-only Primaries.
- **Test a winning recipe on the Candidate network** only after the points above.

## Stop-rule count

**This verdict adds no `not supported` or `refuted` to the stop rule's count.** The Experiments
table of the theme holds three `not supported` verdicts on recency weighting (experiments 331, 407,
and 466) and one on one-shot ENS training (experiment 519), as the plan counted. The verdict `partly
supported` here leaves both counts as they are. C3 is evidence against one-stage training (+0.052 to
+0.070 against the two-stage recipe), but the decision rule does not read it.

## Method

**The network is the pooled mixture density network of experiment 707, with a widening of 1.15.**
Its input has 46 columns, including 3 origin lags, an 8-wide Primary embedding, 2 GELU layers of
width 64, dropout 0.1, 3 mixture components, and 11 quantile levels. The window F of experiment 707
supplies the ENS rows (2021-07-20 on).

- **Stages:** `M1` (the reference) trains on ERA5 and CAMS, then ENS from 2021-07-20, then 730 days
  of ENS at a 0.5-year half-life. `M2` skips the first stage. `M3` is `M2` without the fine-tune.
  `FTh90d` is `M2` with a 90-day half-life in the fine-tune, `M1h90` is `M1` with the same. `W-h1`
  is one stage from scratch with an exponential loss weight at a 1-year half-life. `PP` is one stage
  with a half-life per Primary. `P` is `W-h1` with the weights' ages permuted among whole calendar
  months.
- **Training:** fresh AdamW at 5e-4 for a stage from scratch and 2e-4 for a stage that follows
  another, batch of 16,384, weight decay 1e-3, gradient clip 1.0, patience 6, and a loss of `(w *
  nll).sum() / w.sum()`. Every stage is selected on an unweighted held-out set `V_sel` of held-out
  weeks in the last 730 days. Fits ran on the workstation's GPU, one fit per job, with 542 jobs and
  13.0 hours of summed wall-clock in the fit logs.
- **Injection (`-i`):** the generator `synthetic_clean` of experiment 581 adds a shift of 3, 7, 15,
  or 30 days, a depth of 0.15, 0.30, or 0.50 of capacity, and a random sign, to the power in the
  three lag inputs only, until 5% of the half-hours are covered. The shift never enters the target.
  One world per (fold, seed) serves every stage and epoch, with seed 7,100,000 + 1,000 times the
  network seed + YYYYMM. The scoring bank is `contaminate` of experiment 581, with passes `k` and
  `k+12` for fold `k`. The weights of an injected event are those of the rows whose lags it shifts.
  The mean share of rows with a shifted lag is 0.064 to 0.065.
- **Scoring:** 12 monthly folds (2025-07 to 2026-06), CRPS (twice the mean pinball loss over 11
  quantiles) at lead days 1 to 10 in points of effective capacity, month-block bootstrap on
  seed-averaged forecasts with 100,000 resamples, Bonferroni levels 99.17% (R, 6 readings), 99.375%
  (W, 8), 97.5% (E, 2), and 99.58% (I, 12), and guards from the decision rule. The seeds are 0 to 4
  for `M1`, `M2`, and `FTh90d`, and 0 to 2 for the others, and `FTh90d` is the `B*` named on the
  tuning folds. All Primaries use the same cleaning and row set `fold_a_daily_gap_v1`.
- **Weather at origin:** every row reads the 00Z ENS run issued 9 hours before its origin, the lags
  are at least 3.5 hours old, and the code asserts this on every row. Experiment 404's voided
  lag-free scores above 24 hours play no part, since the network has no lag-free head.
- **Departures from the plan:** `O*` changed from `CYC` to `PP` on the tuning folds before any
  scored fold read it (two `CYC-i` fits at 2025-07 remain, unscored). The summed fit time is the
  wall-clock sum over the logs, not GPU time. The scorer works on cached sums by slice, month, and
  label, and `OutsideScorer` in `e00710_report.py` re-derives the "outside events" scenario.

## Files

- **Code:** `ml-experiments/00710-ens-only-one-shot-training-recipe/`, with scripts `e00710_arms.py`
  (arms and jobs), `e00710_fit.py` (the fits), `e00710_folds.py` (rows and folds),
  `e00710_inject.py` (the injection), `e00710_known_answer.py` (the seven tests), `e00710_net.py`
  (the network and loop), `e00710_score.py` (the scorer), `e00710_weights.py` (the weight rules),
  `e00710_report.py` (every number on this page, and the four figures), `run_phase.sh`,
  `E1_E2_commands.md`, and `NOTES_for_reviewer.md`. To redraw the figures and reprint the numbers,
  run `run_phase.sh e00710_report.py`.
- **Reused code:** experiments 707, 581, 699, 587, and 425 (imported through `PYTHONPATH`; none
  edited).
- **Figures:** `contrasts.svg`, `slump_share.svg`, `example_forecast.svg`, and
  `load_shift_events.svg` in this folder.
- **Private data:** `data/ml-experiments/00710-ens-only-one-shot-training-recipe/work/` (`models/`,
  `pretrained/`, `seed_preds/`, `bank/`, `score_cache/`, `known_answer/`, `logs_run/`,
  `report_cache/`, `report_tables.md`, and `tuning_choices.json`).
- **Private briefs:** `data/ml-experiments/briefs/ens_one_shot/` (`design.md`, `design_review.md`,
  `findings.md`, `code_review_1.md` to `code_review_4.md`, `result_review_1.md` and its scripts and
  logs).

## Pre-registered plan

**The sections from here to the end are the plan, written and committed before any fit.** The
sections above this one were written after the results.

### What we asked, and what we would decide

**We asked whether the demand network needs the ERA5 and CAMS pre-training stage at all.** The
hypothesis, the prediction, and the decision rule are in the front matter of this page. Experiment
707 had found that the first stage still helped at the eight (-0.041 points) and not clearly at the
six (+0.019). The maintainer approved the attempt on 2026-11-19, through the manager, under the stop
rule: three `not supported` verdicts on recency weighting (experiments 331, 407, and 466) and one on
one-shot ENS training (experiment 519) had already been counted.

### Arms

| Arm | Recipe |
|---|---|
| `M1` | ERA5 and CAMS stage, then `S_ens(F)`, then the 730-day fine-tune at a 0.5-year half-life (the reference; arm `A_F` of experiment 707) |
| `M2` | `S_ens(F)`, then the fine-tune (arm `R_F` of experiment 707) |
| `M3` | `S_ens(F)` only |
| `W-h0.5`, `W-h1`, `W-h2` | One stage from scratch with an exponential loss weight, half-life 0.5, 1, and 2 years |
| `S` | One stage with a half-life that shortens over epochs 1 to 6 |
| `HW730`, `HW365` | Hard window of the last 730 or 365 days |
| `FLR`, `SMP`, `RFE`, `EMA`, `CYC` | Weight floor, sampling probability, recent-only final epoch, weight averaging, warm restarts |
| `PP` | Half-life per Primary |
| `FTh90d`, `FTh1` | The recipe of `M2` with a fine-tune half-life of 90 days or 1 year |
| `M1h90` | The recipe of `M1` with a fine-tune half-life of 90 days (extension E1) |
| `P` | The placebo: `W-h_T` with the weights' ages permuted among whole calendar months |

### Contrasts

**All contrasts are treatment minus reference in the `-i` state, on seed-averaged forecasts, with a
month-block bootstrap of 100,000 resamples, read for the six and for the eight.** Each family has
its own Bonferroni level.

| Family and level | Contrast | Question |
|---|---|---|
| R, 99.17% (6 readings) | C1 (headline): `B*-i` minus `M1-i` | Does the best ENS-only recipe match the 3-stage reference? |
| | C2: `M2-i` minus `M1-i` | Experiment 707's C1 with the sign reversed, refitted with injection |
| | C3: `O*-i` minus `M2-i` | Does one stage match the two-stage ENS-only recipe? |
| W, 99.375% (8 readings) | C4: `W-h_T-i` minus `M3-i` | Does a recency weight beat equal weights in one stage? |
| | C5 (placebo): `W-h_T-i` minus `P-i` | Is that gain a gain from recency? |
| | C6: `S-i` minus `W-h0.5-i` | Does a schedule beat a constant weight with the same final half-life? |
| | C7: `HW730-i` minus `M3-i` | Does discarding rows older than 730 days harm the forecast? |
| E, 97.5% (2 readings) | E1 (pre-registered extension, explanatory): `M1h90-i` minus `FTh90d-i` | Does the 90-day half-life account for the gain of the best ENS-only recipe? |
| I, 99.58% (12 readings) | I1 to I3: `X-i` minus `X-n` for `X` in `M1`, `M2`, `B*`, on CRPS | Does injection harm the forecast on ordinary rows? |
| | I4 to I6: the same three, on the slump share of the injected bank | Does injection reduce the slump? |

**The margin is 0.02 points of CRPS, because ERA5 and CAMS pre-training is a fitted stage with two
weather sources, so an ENS-only recipe needs to match the reference and not to beat it.** The
decision rule in the front matter gives the first test that holds. A group's `supported` also needed
the point estimate of C1 below +0.02 on seeds 0 to 2 and on seeds 3 and 4 separately, the upper
limit of C1 on the eight without D8 below +0.02, and the one-sided 95% bounds of C1 within 0.05 on
the 0.01 and 0.99 pinball tails, each lead band, cold nights, and `sunny_clear_sky_index`. A claim
that a recency weight helps needed the upper limits of C4 and of the placebo contrast C5 both below
zero. Injection "does no harm" for a recipe when the upper limit of its CRPS contrast (I1 to I3) is
below +0.02, and "reduces the slump" when the upper limit of its slump contrast (I4 to I6) is below
0.

### Phases, extension, and stop conditions

**The experiment committed to the minimal plan, about 380 fits and 8 to 10 GPU-hours.** Phase T
fitted 15 ENS-only recipes and `M1-i` on the four tuning folds with seeds 0 and 1 (interim). Phase S
fitted `M1`, `M2`, `W-h_T`, `P`, `B*`, and `O*` on the 12 scored folds with seeds 0 to 2. Phase C
fitted seeds 3 and 4 of `M1-i`, `M2-i`, and `B*-i`. The pre-registered extension subset E1 and E2
ran after the minimal plan's results were reviewed: E1 fitted `M1h90-i`, and E2 fitted `FTh90d-n`
and wrote the bank passes of the stored `M1-n` and `M2-n`. The rest of the extension (the other core
arms, the finalists, and the `-n` states) waited for the manager and did not run.

**The experiment would have stopped on five conditions and would have been `invalid` on five more.**
The stops were a failing known-answer test, a failing origin, cut, or injection guard, a timing
pilot over the approved GPU-hours or 16 GB of host memory, `M2-i` more than 0.10 worse than `M1-i`
or `M2-n` on the tuning folds, and a share of shifted rows outside the pilot's range. It was
`invalid` if any row read a run issued after its origin, a training row passed the refit, a world
changed a half-hour at or after the refit, arms were scored on different rows, or a valid time on or
after 2026-07-01 was scored. None applied.
