---
id: "00422"
slug: mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2
title: >-
  Does training the pooled mixture density network at many origin hours, with the age of the weather
  forecast and of the newest power reading as inputs, remove its loss from stale inputs?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: done
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fitted networks, forecasts, and a score table in this experiment's private folder
  only; experiments 397 and 410 are read and never written. Production serving path: not touched
  (this repository holds none). Degradation rule: not touched, but the result informs one (what the
  forecast does when the weather forecast or the power readings are late, or the readings are
  missing). Designs: more than one defensible (which delays and origin hours to train on, whether
  the older weather run is shown in training), so the ladder of arms registers each. Callers: the
  scripts import experiments 87, 397, and 410 and the `studies` evaluation code, whose callers sit
  in those folders.
hypothesis: >-
  The pooled plain mixture density network loses CRPS when the newest power reading falls at a
  time of day, or lies further back, than it saw in training, and when the weather forecast is
  older than in training. Training with the forecast origin at many hours of the day, the newest
  reading delayed by a random amount, and the previous weather run sometimes shown, with the age of
  the weather forecast and the age of the newest reading as inputs, removes that loss without
  raising the score on fresh inputs.
prediction: >-
  Written before any fit and before any score is read. The arms are T0 (origin 09:00 only), T1
  (origin hour varied), T2 (T1 plus delayed readings and the two age inputs), T2n (T2 without the
  two age inputs), T3 (T2 plus the previous weather run sometimes), and T4 (T3 plus the three origin
  lags sometimes missing). Losses are CRPS at days 1 to 10 in points of effective capacity against
  the same network's own fresh case, at the signed-MW six. P1. In T1 the P12 loss is under 0.10
  points (experiment 410 found 0.197 at the signed six and 0.151 at the MVA-only eight for the
  unmodified network; the 97% rests on one Primary, D5, one month, and one seed), because the audit
  traced 97% of the P12 loss to the time of day of the newest reading; T1's P6 and P24 losses stay
  under 0.05. P2. In T2 every power-delay arm (P6, P12, P24) loses under 0.05 points, W24 loses
  within 0.02 of T0's W24 loss, and W24+P12 and W24+P24 lose under 0.10. P3. In T3 W24, W24+P12, and
  W24+P24 each lose under 0.05 points, and W48, which lies beyond the training range, loses more
  than W24. P4. At a constant hour of day the loss grows with delay beyond the 24 hours trained on:
  in T2 and T3 the loss at P48 exceeds that at P24, at P36 that at P12, and at P30 that at P6, each
  by at least 0.03 points. P2n. T2n and T2 differ by under 0.02 points in every stale arm at both
  groups, because the delay is already visible to the network as the difference between the lag age
  and the lead; a larger difference means the age input is not redundant. P5. The fresh-case CRPS of
  T1 to T4 (and T2n) is within +0.03 points of T0's at both groups. P6. In T4 the DROP arm loses
  under 2 points (T0: 9.7 at the signed six, 6.5 at the MVA-only eight in experiment 410), and its
  q95 coverage falls by under 0.05 (T0: 0.33 at the signed six, 0.27 at the MVA-only eight). P7. In
  T2 to T4, q95 and q99 coverage fall by under 0.02 in every arm except P30, P36, P48, W48, and (for
  T2 and T3) DROP. P8. T3 loses no more than the XGBoost S1M model of experiment 410 in P12 and
  W24+P12 (S1M: 0.306 and 0.338 at the signed six, 0.098 and 0.130 at the MVA-only eight), a test
  that binds at the MVA-only eight, because T0 already beats 0.306 at the signed six. P9 (off the
  training grid). In T2 and T3 the loss at P10, a delay between the 9 and 12 hours on the 3-hour
  training grid, is within 0.02 points of the mean of the losses at P6 and P12 interpolated by
  delay. P4 and P7 to P9 are reported and sit outside the decision rule.
decision_rule: >-
  Each contrast is read per group (signed-MW six; MVA-only eight, with and without D8; D15 and D18
  report-only) on CRPS at days 1 to 10, with the month-block bootstrap of `studies.eval_protocol` at
  95% and 20,000 resamples, on the half-hours that the arm and its reference share. Planned
  contrasts. C1: each stale arm minus the same network's fresh case (the experiment 410 contrast).
  C2: each of T1 to T4 minus T0 in the same stale arm, which measures what the training change
  gained. C3: the fresh case of each of T1 to T4 minus T0's fresh case, which measures what the
  training change cost. The hypothesis is supported if the primary arm, T3 (T2 and T4 are read the
  same way and named if one of them meets the limits when T3 does not), meets every experiment 410
  limit at both groups (CRPS loss under 0.05 for P6 and P12, under 0.10 for W24, P24, W24+P12, and
  W24+P24, q95 and q99 coverage fall under 0.02) and C3 is under +0.03 points at both groups; partly
  supported if the best of T1 to T4 cuts the P12 loss by more than half at both groups (0.098 at the
  signed six, 0.076 at the MVA-only eight) but some limit still fails at either group, or the limits
  hold and C3 exceeds +0.03; not supported otherwise. Every limit is tested on the point estimate,
  and the page also prints the upper 95% bound beside it. The experiment 410 limits were proposed by
  the planning session without evidence, so the page reports every number beside them and the
  maintainer judges. The five screen months give wide intervals, so every number is interim. The
  smallest change worth porting is any that meets the limits, because the added cost is a
  training-row builder and two input columns, with no new data or model. The XGBoost S1M forecasts
  of experiment 410 are the reference for P8 and are not refitted.
benchmark: >-
  The pooled plain network of experiment 397 as re-scored by experiment 410 (stored FRESH and stale
  forecasts), the XGBoost S1M model of experiment 410, and T0, a refit of the pooled plain network
  at origin 09:00 under the cleaned-data and start-cut rules, which is the control for C2 and C3.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, and interval width, in points of effective capacity, from `studies.eval_protocol`;
  leads 0.5 to 336 hours; the headline band is days 1 to 10.
verdict: partly supported
best_arm: >-
  T1 (origin hour varied, no new inputs) is the simplest arm that meets every experiment 410 limit.
  Its P12 loss is 0.039 points at the signed-MW six (T0: 0.183) and 0.020 at the MVA-only eight
  (T0: 0.130). T4 (adds sometimes-missing lags) cuts the DROP loss to 0.30 and 0.32 points.
headline: >-
  Training with the forecast origin at eight hours of the day cuts the loss from readings 12 hours
  late from 0.18 to 0.04 points at the signed-MW six and from 0.13 to 0.02 at the MVA-only eight.
  The two age inputs and the random delays add nothing measurable, and varied-origin training costs
  0.04 to 0.11 points on fresh inputs at the MVA-only eight.
lessons: >-
  The loss from 12-hour-late readings was a time-of-day mismatch, which origin-hour training
  removes. The age inputs are redundant with the lead and the lag ages, and the weather-age input
  worsens the 24-hour-old-weather arm at the signed-MW six.
production_status: candidate
adopted_in: none
take_home_message: >-
  Training the pooled neural network with the forecast made at eight different hours of the day
  removes almost all of its extra error when the newest power reading is 12 hours late (0.18 down to
  0.04 points at the six signed substations). The two extra age inputs did not help, so the simplest
  recipe is enough. Training this way raised the error on fresh inputs by 0.04 to 0.11 points at the
  eight MVA-only substations, which is about the size of the seed-to-seed noise, and training with
  some readings missing cut the missing-readings error from 2.4 to 0.3 points. These are five test
  months, so every number is interim.
advice:
  - >-
    Train the pooled network with the forecast origin at several hours of the day, because readings
    12 hours late then cost 0.04 points instead of 0.18 at the signed-MW six (C2, T1).
  - >-
    Do not add the age of the weather forecast or of the newest reading as inputs, because T2n
    without them matches T2 and T2 loses more at W24 at the signed-MW six (0.054 against 0.025).
  - >-
    Train with the three origin lags sometimes missing before relying on the network when readings
    can be absent, because T4 loses 0.30 points in DROP against 1.7 to 2.4 for T0 to T3.
  - >-
    Do not read an unmodified pooled network's DROP loss as a stable number: the D15 and D18 start
    cut changed it from 9.1 to 2.4 points through the share of all-lags-missing training rows.
  - >-
    Check the fresh-case score at the MVA-only eight when adopting varied-origin training, because
    it rose by 0.04 to 0.11 points there and no five-month screen can separate that from seed noise.
code: ml-experiments/00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2/
---

**Take-home message: the pooled network's loss from stale readings was a time-of-day mismatch, and
training with the forecast origin at many hours of the day removes it.** We asked whether the pooled
mixture density network (one neural network for all substations, from experiment 397) can be trained
to cope with a weather forecast that is a day old and with power readings that arrive late. Each
change to the inputs is an "arm" (keys below). Experiment 410 found that readings 12 hours late
raised the continuous ranked probability score (CRPS, days 1 to 10, points of effective capacity,
where one point is 1% of capacity) by 0.18 points at the six primary substations with signed meters
(the signed six) and by 0.13 at the eight with apparent-power meters (the MVA-only eight). When the
network is trained with the origin at eight hours of the day, those losses fall to 0.04 and 0.02
points (arm T1). **The pre-registered rule gives "partly supported".** Every experiment 410 limit
holds at both groups for the primary arm T3, but T3's score on fresh inputs is 0.078 points worse at
the MVA-only eight, above the +0.03 the rule allows. The age inputs add nothing: T2n, the same
training without them, does as well as T2.

![CRPS loss against the fresh case, per stale arm and training arm](figure_1_loss_by_arm.svg)

**Key to the arms.** Training arms (offsets in hours after the 00:00 UTC run start; draws are made
once per training row and seed):

| Arm | Origin offset h | Reading delay d | Previous weather run | Origin lags missing | Age inputs |
|---|---|---|---|---|---|
| T0 | 9 | 0 | never | as in the data | no |
| T1 | 9, 12, 15, 18, 21, 24, 27, 30 | 0 | never | as in the data | no |
| T2 | as T1 | 0, 3, 6, 9, 12, 15, 18, 21, 24 | never | as in the data | yes |
| T2n | as T1 | as T2 | never | as in the data | no |
| T3 | as T1 | as T2 | probability 0.25 | as in the data | yes |
| T4 | as T1 | as T2 | probability 0.25 | all three, probability 0.10 | yes |

- **Test arms** are experiment 410's: FRESH, W24 and W48 (the weather forecast from the run one or
  two days earlier), P*n* (readings delayed by *n* hours: P6, P12, P24), both at once (W24_P12,
  W24_P24), and DROP (the three origin lags missing). New are P30, P36, and P48, which put the newest
  reading at the same hour of day as P6, P12, and P24 a day further back (23:30, 17:30, 05:30), and
  P10, which lies off the 3-hour training grid.
- **Reference columns.** S397 is experiment 410's stored pooled network (trained at 09:00 on power
  with no cleaning rules). S1M is experiment 410's XGBoost model, which is not refitted.
- **The age inputs** are the hours since the weather run and the hours since the newest reading
  (3.5 plus *d*), each over 48. The reading age is a linear function of two existing inputs (the lag
  ages and the lead), so it is redundant by construction; arm T2n tests that.

## Results

CRPS loss against each network's own fresh case, days 1 to 10 (points; positive is worse), five
screen months, so every number is interim. The "Fresh CRPS" row is the absolute score. The 95%
month-block intervals are in `contrasts_*.parquet` and Figure 1. Experiment 410's limits are 0.05
for P6 and P12 and 0.10 for W24, P24, W24_P12, and W24_P24.

Signed-MW six:

| Arm | S397 | T0 | T1 | T2 | T2n | T3 | T4 | S1M |
|---|---|---|---|---|---|---|---|---|
| Fresh CRPS | 3.270 | 3.324 | 3.319 | 3.293 | 3.328 | 3.318 | 3.305 | 3.525 |
| P6 | 0.024 | 0.020 | 0.018 | 0.013 | 0.015 | 0.014 | 0.012 | 0.020 |
| P10 | n/a | 0.134 | 0.031 | 0.024 | 0.032 | 0.024 | 0.036 | n/a |
| P12 | 0.197 | 0.183 | 0.039 | 0.032 | 0.040 | 0.032 | 0.047 | 0.306 |
| P24 | 0.037 | 0.047 | 0.053 | 0.043 | 0.043 | 0.043 | 0.035 | 0.042 |
| P30 | n/a | 0.067 | 0.072 | 0.055 | 0.058 | 0.057 | 0.047 | n/a |
| P36 | n/a | 0.226 | 0.091 | 0.074 | 0.081 | 0.074 | 0.082 | n/a |
| P48 | n/a | 0.085 | 0.103 | 0.082 | 0.082 | 0.083 | 0.067 | n/a |
| W24 | 0.029 | 0.025 | 0.022 | 0.054 | 0.025 | 0.044 | 0.047 | 0.029 |
| W48 | 0.056 | 0.051 | 0.044 | 0.110 | 0.052 | 0.088 | 0.092 | 0.058 |
| W24_P12 | 0.229 | 0.215 | 0.061 | 0.088 | 0.065 | 0.077 | 0.094 | 0.338 |
| W24_P24 | 0.064 | 0.069 | 0.072 | 0.098 | 0.066 | 0.088 | 0.082 | 0.071 |
| DROP | 9.728 | 2.443 | 1.743 | 2.053 | 1.728 | 2.097 | 0.296 | 4.532 |

MVA-only eight:

| Arm | S397 | T0 | T1 | T2 | T2n | T3 | T4 | S1M |
|---|---|---|---|---|---|---|---|---|
| Fresh CRPS | 3.873 | 3.832 | 3.876 | 3.918 | 3.878 | 3.911 | 3.941 | 3.995 |
| P6 | 0.022 | 0.016 | 0.016 | 0.011 | 0.012 | 0.013 | 0.008 | 0.010 |
| P10 | n/a | 0.085 | 0.016 | 0.013 | 0.014 | 0.016 | 0.013 | n/a |
| P12 | 0.151 | 0.130 | 0.020 | 0.019 | 0.019 | 0.021 | 0.019 | 0.098 |
| P24 | 0.031 | 0.034 | 0.036 | 0.031 | 0.031 | 0.031 | 0.022 | 0.041 |
| P30 | n/a | 0.048 | 0.051 | 0.040 | 0.041 | 0.042 | 0.029 | n/a |
| P36 | n/a | 0.159 | 0.053 | 0.047 | 0.047 | 0.049 | 0.038 | n/a |
| P48 | n/a | 0.060 | 0.065 | 0.055 | 0.054 | 0.055 | 0.038 | n/a |
| W24 | 0.047 | 0.048 | 0.044 | 0.034 | 0.047 | 0.036 | 0.035 | 0.032 |
| W48 | 0.093 | 0.098 | 0.090 | 0.068 | 0.098 | 0.068 | 0.069 | 0.064 |
| W24_P12 | 0.197 | 0.177 | 0.063 | 0.054 | 0.065 | 0.056 | 0.054 | 0.130 |
| W24_P24 | 0.076 | 0.080 | 0.077 | 0.067 | 0.076 | 0.067 | 0.058 | 0.072 |
| DROP | 6.504 | 1.899 | 1.700 | 1.822 | 1.667 | 1.831 | 0.318 | 3.684 |

- **Time of day explains the loss.** P12 falls from 0.183 (T0) to 0.039 (T1) at the signed six
  (T1 minus T0: -0.149, interval -0.233 to -0.072) and from 0.130 to 0.020 at the MVA-only eight
  (-0.066, interval -0.120 to -0.007). P36, which hits the same 17:30 hour, falls from 0.226 to
  0.091.
- **The delay itself costs little where the network was trained.** In T2 and T3 the loss grows by
  0.04 points from P6 to P30, P12 to P36, and P24 to P48 at the signed six, so each day of extra
  delay at a constant hour of day costs about 0.04 points. P10, off the training grid, loses 0.024,
  between P6 (0.013) and P12 (0.032).
- **The age inputs do not help.** T2 and T2n agree at P6, P12, and P24 (within 0.008 at the signed
  six). T2 loses more than T2n at W24 (0.054 against 0.025) and W24_P24 (0.098 against 0.066) at
  the signed six, and less at the MVA-only eight. In absolute terms the two networks score alike
  at W24 (3.347 against 3.353); T2 only gains 0.035 on fresh inputs, and the gain does not carry to
  W24. Feeding T2 a weather age of 9 hours reproduces T2n; clipping the age to 30 hours changes
  nothing (audit).
- **Missing readings.** T4 loses 0.30 (signed six) and 0.32 (MVA-only eight) points in DROP, against
  1.7 to 2.4 and 1.7 to 1.9 for T0 to T3. Its 90% interval is 1.07 and 1.09 times as wide as for
  fresh inputs, and its q95 coverage falls by 0.015 and 0.008 (T3: 0.163 and 0.124).
- **Against the XGBoost S1M model.** T3 loses less than S1M at P12 (0.032 against 0.306) and
  W24_P12 (0.077 against 0.338) at the signed six, and at the MVA-only eight (0.021 and 0.056
  against 0.098 and 0.130).

![Loss against the delay of the newest reading](figure_2_loss_by_delay.svg)

![Fresh-case CRPS of each training arm minus T0](figure_3_fresh_cost.svg)

![Winter day-ahead forecast](figure_4_example_winter_day_ahead.svg)

![Winter 14-day forecast](figure_5_example_winter_14_day.svg)

![Summer day-ahead forecast](figure_6_example_summer_day_ahead.svg)

![Summer 14-day forecast](figure_7_example_summer_14_day.svg)

## Tails and cost on fresh inputs

- **Tail coverage holds.** At both groups, q95 and q99 coverage fall by at most 0.006 and 0.002 in
  every arm except DROP, P30, P36, P48, and W48, for every training arm. Fresh-case q95 coverage
  is 0.892 to 0.896 at the signed six and 0.945 to 0.949 at the MVA-only eight, q99 coverage
  0.960 to 0.962 and 0.988 to 0.990.
- **Cost on fresh inputs.** At the signed six, T1 to T4 score within 0.031 of T0 (T3 minus T0:
  -0.006, interval -0.045 to +0.037). At the MVA-only eight they score worse by +0.044 (T1),
  +0.085 (T2), +0.046 (T2n), +0.078 (T3), and +0.109 (T4); only T3 (interval +0.011 to +0.140) and
  T4 (+0.043 to +0.212) exclude zero. Per-seed fresh means agree (T0 3.905, T2n 3.959, T3 3.989,
  T4 4.047). The q95 and q99 pinball losses rise by under 0.03 and 0.01 at the MVA-only eight.
- **Seed spread.** The standard deviation across the three seeds of one arm's fresh CRPS is 0.01
  to 0.05 points. A contrast of two three-seed averages therefore carries a seed noise of about
  0.03 to 0.04, which the month-block intervals do not include, so a minimum detectable effect on
  a fresh-case contrast is about 0.07 points.

## Design, data, and caveats

- **Recipe.** Experiment 397's pooled plain network (setting `g09`, 3 Gaussian components, Primary
  embedding, ensemble mean and standard deviation of seven weather variables) with its training
  loop, early stopping on every twelfth week, and 1.15 widening; monthly refits on the 24 months
  before the refit; the five screen months and three seeds per arm; 90 fits at 15 to 45 seconds each
  on the GPU. Code reused from experiments 87, 397, and 410 is imported, not copied.
- **Cleaned data.** Flagged `substation_zero` half-hours are missing in lags and dropped from
  targets and scoring actuals: 153, 11, 0, 3, and 0 in the five test months, and about 52,000 to
  53,000 half-hours per fold from training (flagged, D15 and D18 before 2025-03-01, unobserved).
- **Rows.** Held-out weeks purge 15 days before and 9 days after, because a delay of 24 hours moves
  a lag a day back. Draws are once per row and seed, not per epoch.
- **Reproduction gate.** T0's fresh CRPS is 3.324 (signed six) and 3.832 (MVA-only eight) against
  the stored network's 3.270 and 3.873. The audit could not explain the 0.054 gap at the signed six
  (1.5 standard errors of a three-seed difference; the MVA-only eight shows the opposite sign).
  Every contrast here uses T0 and so does not depend on it.
- **Checks.** The median forecast minus the mean forecast is -0.23 to -0.27 points at the signed six
  and -0.39 to -0.47 at the MVA-only eight in every training arm. The error of the median
  forecast moves with the quantile score: T0 P12 raises the median's mean absolute error by 0.33
  points at the signed six (5.81 to 6.14), against 0.05 for T3. Triggers fired: a nested arm beats
  the arm that contains it (T2 against T2n at W24), a control moves more than expected (T0's DROP
  loss is 2.4 points against 9.7 for the stored network), and the results contradict predictions P2,
  P3, and P5. A fresh Sonnet audit re-derived the CRPS of T0, T2, T2n, and T3 in four arms by its own
  code (largest difference 0.0002), found no leakage, and explained the two anomalies.
- **Audit finding on DROP.** The D15 and D18 start cut leaves 9 times as many training rows with
  all three lags missing in T0 (0.20%) as in experiment 397's rows (0.022%). A refit of T0 without
  the cut loses 9.1 and 6.4 points in DROP, like the stored network. T0's DROP loss is therefore
  an accidental result of the start cut, and T4's 0.30 should be read against 2.4 (with the cut)
  and 9.1 (without).
- **Prediction outcomes.** P1 held for P12 (0.039 and 0.020) and P6, and failed for P24 at the signed
  six (0.053 against 0.05). P2 failed at W24 (T2's loss is 0.029 above T0's) and held for the delay
  arms, W24_P12 (0.088), and W24_P24 (0.098). P2n failed at W24, W24_P12, and W24_P24 at the signed
  six. P3 held for W24 (0.044) and W48 above W24, and failed for W24_P12 (0.077) and W24_P24
  (0.088). P4 held at the signed six (0.042, 0.042, 0.040 in T2) and failed at the MVA-only eight
  (0.034, 0.028, 0.016). P5 held at the signed six and failed at the MVA-only eight. P6 held. P7
  held. P8 held. P9 held (P10 within 0.002 of the interpolation).
- **Decision rule.** T3 meets every experiment 410 limit at both groups on point estimates; its
  W24_P24 upper bound at the signed six is 0.109, above 0.10. The fresh-case cost at the MVA-only
  eight (+0.078) exceeds +0.03, so the verdict is "partly supported". T1, T2n, T2, and T4 also meet
  the limits. The experiment 410 limits were proposed without evidence, and the numbers are given
  beside them for the maintainer to judge.
- **Caveats.** Five test months give wide intervals; the audit's mechanism for T2's W24 result
  (age and origin hour are tied in training and decoupled at test) is inferred, not tested. The
  lead range is days 1 to 10 for substations; NGED's horizon is 1 to 10 days. No XGBoost model was
  refitted, and P30, P36, P48, and P10 were scored for the neural networks only.
- **What to try next.** Test T1 with the full scored year, and check the fresh-case score at the
  MVA-only eight with more seeds. Train with the three lags sometimes missing (T4) before the
  network faces absent readings. Decouple the weather age from the origin hour in training before
  using the age input again. Fits and forecasts: `data/ml-experiments/00422-*/work/` (not in git).
