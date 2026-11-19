---
id: "00707"
slug: ens-backfill-pretraining-gain-against-ens-history-length
title: >-
  Does pre-training the demand network on ERA5 and CAMS still help when the ECMWF ENS history
  reaches back to 2020?
theme: retraining-and-training-history
also_relevant_to:
  - weather-products-and-blending
status: finished
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores about 456 fitted networks and per-row forecasts; (2) more than one design
  defensibly tests the hypothesis (which stages the long ENS window feeds, whether the placebo
  permutes weather); (3) it reuses the training code of experiments 397, 418, and 425, whose callers
  cannot be named without searching; (4) it touches no serving path or degradation rule; (5) it
  changes no contract. It waits for the ENS back-fill of 2020-10-15 to 2024-03-31.
hypothesis: >-
  The gain from pre-training the pooled mixture density network on ERA5 and CAMS before fine-tuning
  on ECMWF ENS is smaller when the ENS history is 47 to 58 months (from 2021-07-20) than when it is
  24 months.
prediction: >-
  Written before any fit. C1 (pre-trained minus ENS-only, both with the long ENS window) is between
  -0.04 and 0 points of CRPS at the six signed-MW Primaries and between -0.10 and 0 at the eight
  MVA-only Primaries, against -0.050 and -0.124 in experiment 425. The placebo contrast C5 is below
  zero at the 24-month window.
decision_rule: >-
  C1 is read per group at 99.69% (16 readings, month-block bootstrap on seed-averaged forecasts),
  on its interval [lower, upper], applying these tests in order. Lower limit above 0: refuted.
  Upper limit below -0.02: supported. Lower limit at or above -0.02: not supported (no longer worth
  porting). Upper limit below 0: small gain. Otherwise the interval straddles both -0.02 and 0:
  inconclusive, and the width is printed. The threshold of 0.02 is the protocol's minimum for a new
  fitted step, because pre-training adds one stage and two weather sources. C1 minus C2 is
  descriptive. The gate and the stop rule are in the Gate and Stop rule paragraphs.
benchmark: >-
  Stored A and R of experiment 425 (24-month window), and the ENS-only arm R_F with the long window.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS (twice the mean pinball loss over 11 quantile levels) at lead days 1 to 10 in points of
  effective capacity, with the normalised MAE of the median beside it.
verdict: partly supported
best_arm: >-
  `A_F` (pre-trained, 47 to 58 months of ENS) at the eight MVA-only Primaries: CRPS 3.571 points,
  against 3.612 for `R_F`; `R_F` (not pre-trained, same window) at the six signed-MW Primaries: CRPS
  2.821 points, against 2.840 for `A_F`.
headline: >-
  With 47 to 58 months of ENS history, pre-training on ERA5 and CAMS changed CRPS by +0.019 points
  (99.69% interval -0.004 to +0.049) at the six signed-MW Primaries, against -0.050 in experiment
  425, and by -0.041 (-0.059 to -0.025) at the eight MVA-only Primaries, against -0.124. The
  placebo failed at the six (C5 -0.011, -0.055 to +0.050).
lessons: >-
  The pre-training gain at the eight shrinks from -0.124 with the stored 24-month window to -0.041
  with 47 to 58 months, and C1 is negative in all 12 folds. At
  the six the gain disappears, but the placebo shows that the six's gain was never separated from
  the gain that power history alone gives.
take_home_message: >-
  Training the demand network on four to five years of archived weather forecasts cut the benefit
  of first training it on past-weather estimates to about a third at the eight substations that
  report only apparent power, where it still helped in every month. At the six that report signed
  megawatts the benefit disappeared, and a control run showed it was never clearly caused by the
  weather data.
advice:
  - >-
    Keep ERA5 and CAMS pre-training in the demand network for the MVA-only Primaries until a test
    shows otherwise; with 47 to 58 months of ENS it still lowered CRPS by 0.041 points (C1).
  - >-
    Run the placebo (shuffled weather days) in any later experiment that claims a pre-training gain
    at the signed-MW Primaries, because their gain was not separated from the effect of extra power
    history (C5).
  - >-
    Read a pre-training gain against a window of true length: the stored 24-month arms saw 15 to 24
    months of ENS, and C2 differs from the stored line by 0.039 at the six (C2).
production_status: not-for-production
adopted_in: none
---
**With the ECMWF ENS history extended back to 2021-07, pre-training the demand neural network on ERA5
and CAMS no longer improved the forecast at the six signed-megawatt Primaries, but it still did at
the eight MVA-only Primaries, where the gain was about a third as large as with 24 months of ENS
(a lower score is better).** At the six the contrast is +0.019 points of CRPS (99.69% interval -0.004
to +0.049), against -0.050 in experiment 425. At the eight it is -0.041 (-0.059 to -0.025), against
-0.124. The placebo failed at the six, so the sign of the six's contrast cannot be trusted, and the
eight is the only group where the experiment shows pre-training helping.

We asked whether the gain from the first training stage shrinks when the second stage has 4 to 5
years of ECMWF ensemble (ENS) forecasts instead of 24 months. The network is the pooled mixture
density network of [experiment 425](../00425-best-mdn-architecture-assembly-with-ablations/result.md).
The first stage trains on ERA5 reanalysis and CAMS solar estimates of past weather ("pre-training").
The second stage trains on the archive of ENS forecasts. The archive now starts on 2020-10-15,
because we back-filled it from dynamical.org for this experiment. The 16 Primaries are 6 that report
signed megawatts ("the six"), 8 that report only apparent power in megavolt-amperes ("the eight"),
and D15 and D18, which are report-only. Scores are CRPS in points of effective capacity over lead
days 1 to 10, on 12 monthly folds from 2025-07 to 2026-06.

Verdict: **partly supported**. The prediction that C1 lies between -0.04 and 0 at the six failed,
because C1 is positive. The prediction at the eight (between -0.10 and 0) held. By the
pre-registered rule C1 is "not supported" at the six (the gain is no longer worth porting) and
"supported" at the eight (the gain is still worth porting). Because trigger 1 fires, the six's
verdict rests on the audit below. The experiment makes no adoption.

## Figures

![CRPS contrast between pre-training and no pre-training, against months of ENS history, at the six
and the eight](gain_against_ens_months.svg)

![Day-ahead and 14-day example forecasts of one signed-MW Primary and one MVA-only Primary, for
four arms](example_forecasts.svg)

## Results

**The pre-training contrast shrinks as the ENS history lengthens, and shrinks to nothing at the
six.** The table gives each arm pair's CRPS difference (pre-trained minus not pre-trained; negative
favours pre-training) at the 99.69% level, with 12 folds, a month-block bootstrap on seed-averaged
forecasts, and 50,000 resamples.

| ENS window | Contrast | The six | The eight |
|---|---|---|---|
| 12 months | `A_12` − `R_12` (descriptive) | -0.109 (-0.205 to -0.039) | -0.151 (-0.300 to -0.046) |
| 24 months, stored | `A_24` − `R_24` (425's arms) | -0.050 (-0.082 to -0.016) | -0.124 (-0.235 to -0.037) |
| 24 months | C2: `A_E24` − `R_E24` | -0.011 (-0.059 to +0.037) | -0.082 (-0.131 to -0.037) |
| 47 to 58 months | C1: `A_F` − `R_F` | +0.019 (-0.004 to +0.049) | -0.041 (-0.059 to -0.025) |
| 56 to 68 months | `A_C` − `R_C` (descriptive) | -0.008 (-0.050 to +0.022) | -0.040 (-0.066 to -0.016) |

**The gain at 24 months is already smaller than the stored line when the 24 months are a true 24
months of ENS.** `A_E24` and `R_E24` read the back-filled runs, so every fold sees 24 months of ENS.
The stored arms saw 15 to 24 months before fold 2026-04, because the production archive started on
2024-04-01. C2 is -0.011 at the six and -0.082 at the eight, and the two readings differ in window
as well as in code path.

**C1 minus C2 is +0.030 at the six (-0.006 to +0.072) and +0.041 at the eight (+0.009 to +0.078).**
Only the eight's interval excludes zero, so only the eight shows that lengthening the window from 24
to 47 to 58 months shrank the pre-training gain. Both are descriptive readings.

**The eight shows pre-training helping in every fold.** C1 at the eight is negative in 12 of 12
folds (a t test on the 12 monthly differences gives t = -6.69), and it holds without fold 2026-01
(-0.044), at days 1 to 6 (-0.041), and at days 7 to 10 (-0.043). At the six, C1 is positive in 9 of
12 folds, and the t test gives t = 2.00 (p = 0.070); without fold 2026-01 it is +0.018, at days 1 to
6 +0.020, and at days 7 to 10 +0.017. The minimum detectable effect, from the seed halves of `R_F`,
is 0.018 points at the six (a lower bound).

**Long ENS history alone helps little.** C3 (`R_F` − `A_24`) is -0.007 at the six (-0.044 to +0.034)
and +0.018 at the eight (-0.029 to +0.076), so a network with long ENS and no pre-training did not
beat the stored pre-trained network of experiment 425 (`A_24`) at either group. C4 (`A_F` −
`A_E24`) is -0.007 at the six and -0.032 at the eight (-0.088 to +0.008), so lengthening the window
inside the pre-trained recipe did not help clearly either.

**The placebo fails at the six and passes at the eight.** The placebo (`A1x_24`) pre-trains on
ERA5 and CAMS weather days shuffled among days of the same Primary, so only the extra power history
can help. C5 (`A_24` − `A1x_24`) is -0.011 at the six (-0.055 to +0.050), so the six's gain cannot
be told apart from the gain that power history alone gives; C6 (`A1x_24` − `R_24`) is -0.041
(-0.099 to +0.008). At the eight C5 is -0.060 (-0.083 to -0.039), so the weather carries part of the
gain there, and C6 is -0.078 (-0.157 to -0.003).

**Lockdown ENS rows and the two data-source steps made no clear difference.** C7 (`R_C` − `R_F`,
keeping runs from 2020-10-15 to 2021-07-19 with a `covid` input) is +0.012 at the six and +0.004 at
the eight. C8 (`R_F+flags` − `R_F`, two indicator inputs for the resolution upgrade of 2023-06-27
and the source change of 2024-04-01) is -0.004 and +0.024. All four intervals contain zero.

| Group | `A_24` | `R_24` | `A_F` | `R_F` | `A_12` | `R_12` | `A1x_24` | Climatology | BG2 |
|---|---|---|---|---|---|---|---|---|---|
| The six, CRPS | 2.828 | 2.878 | 2.840 | 2.821 | 2.832 | 2.942 | 2.843 | 3.831 | 3.016 |
| The six, MAE of the median | 5.008 | 5.102 | 5.040 | 4.998 | 5.026 | 5.229 | 5.046 | 6.770 | 5.242 |
| The eight, CRPS | 3.594 | 3.719 | 3.571 | 3.612 | 3.586 | 3.737 | 3.663 | 4.866 | 3.774 |
| The eight, MAE of the median | 6.504 | 6.751 | 6.470 | 6.565 | 6.494 | 6.771 | 6.655 | 8.780 | 6.744 |

**Tails and cost.** Coverage and pinball loss at q01, q05, q95, and q99 differ by under 0.01 and
0.03 points between `A_F` and `R_F` (the eight, q95 pinball loss 0.915 against 0.926; the six, 0.778
against 0.775). Avoidable curtailment cost summed over the Primaries that report it is £7.62 million
for `A_24`, £7.52 million for `A_E24`, £6.65 million for `A_12`, and £7.76 million for the placebo
`A1x_24`; the scorer's printed table elides `R_F`'s total. On cold winter nights `A_24` scores best
at the six (4.290) and `A_F` and `A_E24` score worse (4.451 and 4.452), and in the seven days after
a load-shift event `A_24` scores worse than `R_24` at the six (3.991 against 3.712). Both are
descriptive and carry no interval.

## Checks

- **Median minus mean forecast per Primary.** Not recomputed for this page; every arm scores the
  median and the mean by the same code as experiment 425.
- **Error of the median beside each CRPS contrast.** C1's mean absolute error of the median agrees
  in sign with CRPS at both groups: +0.042 (-0.008 to +0.113) at the six and -0.095 (-0.136 to
  -0.056) at the eight. Trigger 3 did not fire.
- **Triggers.** Trigger 1 (the placebo fails at the six) fired. Trigger 2 (C1 contradicts the stored
  425 line) fired at the six, where the signs are opposite. Trigger 6 fired at the eight, where C1's
  interval is 0.17 times the width of the stored line's (the limits are 0.5 and 2.0), because the
  stored line used 24 months of unevenly covered ENS and unwidened networks. Trigger 4 (dropping fold
  2026-01) did not fire. Trigger 5 (`R_F` beats `A_24`) did not fire, because C3 lies inside the
  noise. Trigger 7 is the author's judgement: no later design depends on this headline.
- **Gate.** The prediction-only check matched experiment 425's stored forecasts, and the refit check
  on folds 2026-06 and 2026-01 passed.
- **Step checks.** The ENS-minus-ERA5/CAMS and spread checks at 2023-06-27 and 2024-04-01 passed;
  the largest step was 2.7 standard deviations (surface pressure at 2023-06-27; the limit is 3).
- **The back-fill.** 1,263 of the 1,264 runs from 2020-10-15 to 2024-03-31 are stored. The run of
  2024-01-03 is a documented upstream gap, because dynamical.org holds 26 of its 51 members. No arm
  reads it, and the coverage report lists it.
- **Cleaning.** The scorer dropped 785 scoring half-hours for a flagged exact zero. The 394
  half-hours that only experiment 398's list flags stay flagged in the stored-arm reproduction, and
  the current cleaning flags fewer.

## Caveats

- **The placebo failure at the six removes the six's evidence for pre-training.** The six's C1
  sign (+0.019) and the six's 24-month gain both lie within the noise of what the extra power
  history alone gives.
- **The six has little long history.** D9 has no observed power before 2023-08-23, so a long window
  adds nothing for it.
- **The minimum detectable effect (0.018) is close to the 0.02 threshold.** A gain of 0.02 is at
  the edge of what 12 folds resolve at the six.
- **Window length and the 2023-06-27 resolution upgrade are confounded** above 24 months. C8 found
  no clear effect of the two indicator inputs.
- **The 99.69% level undercovers at 12 months.** The t tests above read the same 12 monthly
  differences without that correction.

## What to try next

- A second seed set for the placebo at the six would show whether the failure is seed noise.
- Pre-training for the eight alone, and none for the six, is an option the six's placebo failure
  leaves open.

## Files

- Code: `ml-experiments/00707-ens-backfill-pretraining-gain-against-ens-history-length/` (scripts
  `e00707_arms.py`, `e00707_fit.py`, `e00707_prep.py`, `e00707_score.py`, and their helpers).
- Reused code: experiments 397, 418, 425, and 494 (named in Method).
- Private data: `data/ml-experiments/00707-ens-backfill-pretraining-gain-against-ens-history-length/`
  (`work/models/`, `work/pretrained/`, `work/seed_preds/`).
- Production back-fill: the ECMWF ENS archive from 2020-10-15 is in the Delta table `NWP`, written
  by pull request 1705 and the back-fill that followed.

## Plan

**The sections from here to the end are the plan, written and committed before any fit.** Only
Method and the Data handling bullet for 2024-01-03 were edited afterwards.

## Hypothesis, prediction, and decision rule

**Hypothesis.** The gain from pre-training the pooled mixture density network on ERA5 and CAMS is
smaller when the ENS training history is 47 to 58 months (window F, from 2021-07-20) than when it is
24 months.

**Prediction.** At the six signed-MW Primaries the pre-training contrast C1 lies between -0.04 and 0
points of CRPS, against -0.050 in experiment 425 and -0.061 in experiment 418. At the eight MVA-only
Primaries C1 lies between -0.10 and 0. The experiment cannot resolve a shrink from -0.05 to -0.02,
because experiment 425's minimum detectable effect was 0.037 and C1 − C2 has an interval about 1.4
times wider. The page says so.

**Decision rule.** The smallest gain worth porting is 0.02 points, the protocol's minimum for "a new
fitted step, a new input, or a new data source" (`evaluation-protocol-demand.md`), because
pre-training adds one fitted stage and two weather sources to maintain. C1 is read per group at the
Bonferroni level:

The tests apply in this order to the interval [lower, upper]. The first test that holds gives the
verdict.

- **Harmful:** the lower limit is above 0. Verdict `refuted`.
- **Still worth porting:** the upper limit is below -0.02. Verdict `supported`.
- **No longer worth porting:** the lower limit is at or above -0.02. Verdict `not supported`.
- **Small gain:** the upper limit is below 0. The lower limit is then below -0.02, so the interval
  straddles -0.02. Verdict `small gain`.
- **Cannot tell:** none of the above holds, so the interval straddles both -0.02 and 0. Verdict
  `inconclusive`, and the scorer prints the interval's width.

The experiment reports the verdict and the interval, and makes no adoption.

## Arms

**Every arm is the network of experiment 425 (arms `A` and `R`), widened by 1.15.** The code is
copied from `ml-experiments/00425-best-mdn-architecture-assembly-with-ablations/` and
`00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/` into the new experiment's folder. The
copy keeps `FIRST_RUN = 2024-04-01`, because the held-out validation weeks key on it, and negative
days floor-divide correctly. The input caches (experiment 87's rows, experiment 397's run summaries,
and the raw `.npz` files) are rebuilt for 2020-10-15 onwards. Three stages combine:

- `S_era`: ERA5 and CAMS pseudo-runs from 2019-09-01, unweighted (experiment 418's recipe, which
  sets the `covid` input and keeps lockdown rows).
- `S_ens(W)`: an unweighted stage on the ENS rows in window W, with a fresh AdamW at a learning rate
  of 5e-4 (see Method).
- `FT(W)`: experiment 425's fine-tune on the last min(W, 730 days) of ENS, 0.5-year half-life.

The half-life gives a 2021 row about 0.2% of a new row's weight, so windows above 24 months need
the `S_ens` stage to be tested at all. Windows filter on valid time, so runs from 2023-06-17 to
2023-06-26 enter fold 2025-07's 24-month window; the page states this.

| Arm | Stages | ENS window W | Seeds |
|---|---|---|---|
| `A_24`, `R_24` (stored; gate) | `S_era`→`FT`; `FT` | 24 months | 0 to 4 |
| `A_E24`, `R_E24` | `S_era`→`S_ens`→`FT`; `S_ens`→`FT` | 24 months | 0 to 4 |
| `A_F`, `R_F` | `S_era`→`S_ens`→`FT`; `S_ens`→`FT` | 2021-07-20 to cut (47 to 58 months) | 0 to 4 |
| `A_12`, `R_12` | `S_era`→`FT`; `FT` | 12 months | 0 to 2 |
| `A_C`, `R_C` | as `A_F`, `R_F` | 2020-10-15 to cut (56 to 68 months, lockdown ENS rows kept with `covid`) | 0 to 2 |
| `A1x_24` (placebo) | shuffled `S_era`→`FT` | 24 months | 0 to 2 |
| `R_F+flags` | as `R_F`, two extra inputs | as F | 0 to 2 |

**Stored controls and references.** The gate reuses experiment 425's stored `A` and `R` (60 raw
files each, 12 folds × 5 seeds, seedless, unwidened, keyed on `time_series_id`,
`power_fcst_init_time`, `valid_time`, written outside `experiment_store.write_predictions`), under
row set `fold_a_daily_gap_v1`; no new row set is needed. Reference rows are `climatology`,
`manual_heuristic`, BG2 (experiment 87), and S0M (experiment 389), the last two from stored XGBoost
forecasts only. About 456 fine-tuned networks result (4 five-seed arms and 6 three-seed arms, 12
folds); every fitted state is saved in the experiment's data folder.

## Contrasts, scoring, controls

| Contrast | Definition | Question |
|---|---|---|
| C1 (headline) | `A_F` − `R_F` | Pre-training gain with long ENS |
| C2 | `A_E24` − `R_E24` | The same recipe at 24 months |
| C3 | `R_F` − `A_24` | Long ENS alone against the current recipe |
| C4 | `A_F` − `A_E24` | Long ENS inside the pre-trained recipe, recipe held fixed |
| C5 (placebo) | `A_24` − `A1x_24` | Weather, not power history, carries the gain |
| C6 | `A1x_24` − `R_24` | Power history alone |
| C7 | `R_C` − `R_F` | Lockdown ENS rows |
| C8 | `R_F+flags` − `R_F` | Resolution-upgrade and source-change steps |

**Scoring follows experiment 425.** The seeds that both arms hold are averaged in quantile space,
then `metrics.month_bootstrap_contrast` resamples whole months at 1 − 0.05/16 = 99.69%, for eight
contrasts in two groups, with 50,000 resamples (about 78 draws per tail). Every arm shares the row
set `fold_a_daily_gap_v1` and the 12 monthly folds 2025-07 to 2026-06. The scorer prints the minimum
detectable effect first. C1 − C2 is descriptive and carries its own interval. Descriptive readings,
labelled as such: with and without fold 2026-01 (which dominated experiments 449 and 487), days 1 to
6 against 7 to 10, the 12-month gain, per-Primary, and a t test on the 12 monthly differences,
because 99.69% undercovers at 12 months.

**Placebo.** `S_era` with whole weather days permuted within each Primary, power, calendar, and lags
left in place, drawn only from rows at least 3.5 hours before the fold's cut and within 15 days of
the day of year, across years. It passes if C5 lies wholly below zero. C6 asks whether the extra
power history alone helps.

**Gate.** (1) A prediction-only check loads experiment 425's stored states (`work/models/`,
`work/pretrained/`) and matches its stored predictions to about 1e-4. (2) A refit check: the seed
average of `A_24` and `R_24` on fold 2026-06 and 2026-01 matches the stored arms within two standard
errors of the seed-to-seed spread (the stored seed-mean difference has a standard deviation of 0.052
per fold, and 425 set only `torch.manual_seed`, so a fixed 0.005 tolerance is tighter than noise).

**Stop rule.** Stop and report to the manager if the gate fails, if a step check in section 5 fails,
or if `A_24` or `R_24` misses the stored arm on fold 2026-06 by more than the refit tolerance.

## Weather at the forecast origin

**Each ENS row takes its weather from one run: the 00Z run issued at least 9 hours before the row's
origin.** For the run of 2026-01-14 00:00 UTC the origin is 2026-01-14 09:00 UTC. The target at
2026-01-19 17:30 UTC (lead 128.5 hours) reads the member mean and spread of that run, never the
2026-01-19 run. The code asserts `nwp_init_time + 9 h <= origin` on every row and that every training
row's valid time is at least 3.5 hours before the cut, and prints one example row.

**No weather feeds the lag inputs.** The three origin lags are power readings at least 3.5 hours old
at the origin. The network has no lag-free head, so no score at any lead reads experiment 404's
lag-free table. `S_era` rows use ERA5 and CAMS at the valid time, in training only; experiment 418
bounds them 3.5 hours before the cut although ERA5 arrives 5 days late, and the page keeps that
bound for the gate and names it as a caveat.

## Data handling

- **Lockdown.** The back-fill holds 278 runs inside 2020-03-23 to 2021-07-19. Windows F and the
  24-month windows exclude them. `A_C` and `R_C` keep them with the `covid` input. `S_era` rows
  already include lockdown rows with `covid = 1` in every pre-trained arm, as in stored `A`. The
  window floor applies to origin and lag times too.
- **One missing run, 2024-01-03.** The back-fill holds 1,263 of the 1,264 daily runs from 2020-10-15
  to 2024-03-31. Dynamical.org holds only 26 of the 51 ensemble members for the run of 2024-01-03,
  so the ingest refuses it, and no experiment arm reads it. The code lists the run in its coverage
  report and treats it as a documented gap. A window that lacks any other run still stops the fit
  unless `--allow-partial-window` is passed.
- **Resolution upgrade, 2023-06-27, and source change, 2024-04-01.** Before any fit, compute monthly
  ENS mean minus ERA5 (CAMS for shortwave) at leads 12 to 36 hours, and the ENS spread, per variable
  at the 16 Primaries. A step at either date larger than three standard deviations of the adjacent 12
  monthly values fails the check. `R_F+flags` carries an indicator for each date.
- **Cleaned data and cuts.** `substation_zero` half-hours are dropped everywhere and counted per
  fold, separately for a flagged target and a flagged lag. The flagged half-hours come from the
  cleaned power table over 2019-09-01 to 2026-07-01, because experiment 398's list covers only
  2024-05-01 to 2026-03-05. D15 and D18 rows before 2025-03-01 are dropped. A row of D5, D10, D11,
  D14, or D16 is dropped if its valid time or its weekly lag is before the Primary's first usable
  date (D16: 2021-02-01). D9 (signed-MW six) has its first observed power on 2023-08-23, so long
  windows add nothing there; the page reports ENS months per Primary.
- **Precipitation type** is not among the network's seven weather variables, so the column is not
  read. **Shortwave** is a 6-hour mean after lead 144 hours; C1 is also read for days 1 to 6 and 7 to
  10 separately. **Calendar:** hour, day-of-week, and holiday columns only; early stopping holds out
  whole weeks.

## Folds and compute

**Folds are the 12 scored months 2025-07 to 2026-06, each with at least 12 months of history.**
**Screen first** on 2025-10, 2026-01, 2026-03, and 2026-06, seeds 0 to 2, all numbers interim. The
estimate is about 9 GPU-hours for the screen and 35 for the full run (extrapolated from experiments
418 and 494, to be re-measured on the first fold). `S_ens(F)` holds about 11 to 14 million rows (418
held 4.98 million over about 21 months). Request three streams on 4 cores, a GPU share of about 5 GB
per stream (unverified), and a memory cap near 14 GB per stream. Building run summaries for 1,264
back-filled runs is unmeasured CPU work.

## Method

**Five choices depart from experiment 425 or from the design note, and each is recorded here so that
a reader of the results can allow for it.**

- **`S_ens` starts a fresh AdamW at a learning rate of 5e-4.** The design note says that `S_ens`
  uses "`S_era`'s optimiser", and the code reads that as the same settings (AdamW, 5e-4, unweighted),
  not as the optimiser state carried over from `S_era`.
- **Every fine-tune that follows an earlier stage uses a learning rate of 2e-4.** The rule includes
  `R_E24`, `R_F`, and `R_C`, whose earlier stage is `S_ens` alone. Experiment 425 used 2e-4 only for
  `A`, which follows `S_era`. Only an arm with no earlier stage (`R_24`, `R_12`) fine-tunes from
  scratch at 5e-4.
- **The standardiser and the per-Primary target statistics come from the fine-tuning rows.**
  Experiments 425 and 418 do the same, and every stage, `S_era` and `S_ens` included, applies them
  unchanged. A stage with a longer window therefore sees inputs on the scale of the 24-month rows.
- **The minimum detectable effect uses the seed halves of `R_F`.** The scorer compares the mean of
  seeds 0 to 2 with the mean of seeds 3 and 4 of `R_F`. Experiment 425 took its minimum detectable
  effect from the stored `R`, which has a 24-month window.
- **The stored `A_24` and `R_24` saw less ENS than `A_E24` and `R_E24` do.** ENS in the old table
  starts on 2024-04-01, so before fold 2026-04 the stored arms saw 15 to 24 months of ENS. `A_E24`
  and `R_E24` read the back-filled runs and see a true 24 months in every fold. C2 and the stored
  425 line therefore differ in window as well as in code path, and the scorer prints both.

## Reports

Every table shows the normalised MAE of the median beside CRPS, per group, per Primary, and per lead
band. The page reports q01, q05, q95, and q99 coverage and pinball loss, cold winter nights, the
avoidable-curtailment cost, the protocol's load-shift slices and cold-start-months slice, a figure of
gain against ENS months, and a day-ahead and a 14-day example forecast. The Checks list gives median
minus mean per site, MAE beside each CRPS contrast, and the triggers fired. Likely triggers: 1
(placebo), 2 (C1 contradicts 418 and 425), 5 (`R_F` beats `A_24`), 6 (interval width), 7.

## Invalid conditions, risks, prerequisites

**The experiment is `invalid` if** any row reads a run issued after its origin, a training row
passes the cut, arms are scored on different rows, a valid time on or after 2026-07-01 is scored, or
lockdown ENS rows enter an arm other than `A_C` and `R_C`.

**Risks.** The 0.02 threshold is below the 0.037 minimum detectable effect. Window length and the
2023-06-27 era are confounded above 24 months, which C8 probes.

**Needed before the first long-window fit:** the back-fill (production pull request merged or the
maintainer's word to run from its branch), the `data-validation` checklist on it, and the two step
checks. **Work that can start now:** the plan commit, the copied code and its Opus review, the gate,
the placebo builder, and the 12-month and 24-month arms.
