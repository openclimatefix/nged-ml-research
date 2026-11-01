---
id: "00448"
slug: s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool
title: >-
  Does adding pretraining, the B-mix output, and a five-seed pool, one at a time and then together,
  lower the S1 network's CRPS, and does the whole beat its parts and the references?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - lagged-power-and-error-correction
status: done
kind: other
protocol: demand-v2
size: >-
  complex. Stored: 156 new fitted neural networks for the screen and 72 more for the final test
  (table under Compute), their forecasts, and the score tables, in this experiment's private
  folder only; experiments 404, 425, 428, 439, 446 are read and never written. More than one
  design is defensible (which components enter the ladder, which leave-one-out arms get fits, how
  pretraining treats the lag inputs), so this plan fixes each choice before any score is read.
  Production serving path: not touched (this repository holds none). Degradation rule: not
  touched. Callers: the scripts import the folders of experiments 425, 428, 439, 446, and the
  `studies` evaluation code.
hypothesis: >-
  The S1 network (pooled plain mixture density network, Primary embedding, ensemble mean and
  spread once, hand-made calendar columns, monthly refits, three origin lags plus the surprise at
  each lag) improves with each of three components added one at a time (ERA5-and-CAMS pretraining
  then ensemble fine-tuning; the fixed 50/50 B-mix of the S1 and lagless output distributions; a
  five-seed pool), and the network with pretraining and the pool together scores lower CRPS at
  days 1 to 10 than every one of its parts and than the monthly XGBoost recipe.
prediction: >-
  Written before any fit and before any score is read. P1. Pretraining lowers S1's CRPS by 0.03 to
  0.10 points at the signed-MW six and by 0.08 to 0.20 at the MVA-only eight (experiment 425, a
  network with no lag inputs: -0.050 and -0.124). P2. B-mix lowers CRPS on clean-lag rows by under
  0.03 and lowers q99 pinball loss on rows with a lag jump (experiment 404 found B-mix the
  strongest lag-trust design), so it passes only through its harm-row rule. P3. The five-seed pool
  lowers CRPS by 0.04 to 0.07 (experiment 446 T4: -0.056; the three-seed pool of S1 gave -0.042),
  mostly by ensembling. P4. The assembly PT+pool5 beats S1-pool5 by 0.02 to 0.08. P5. Removing the
  Primary embedding from the assembled network raises CRPS by 0.5 to 1.0 points (experiment 425:
  +0.851 and +0.837); removing the surprise inputs raises it by 0.05 to 0.15 (experiment 428's S1
  against R-inj: -0.106); removing pretraining raises it by the size of P1. P6. The assembled
  network beats the monthly XGBoost recipe at both groups.
decision_rule: >-
  Four screen folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 to 4, monthly
  refits, 1.15 widening unless stated, CRPS at days 1 to 10 in points of effective capacity (a
  negative difference is a gain), per metering group (signed-MW six; MVA-only eight; D15 and D18
  report-only) and per Primary. Every screen number is interim and carries no 99% claim. Contrasts
  are paired by fold and seed with the month-block bootstrap of `studies.eval_protocol`, at a
  level Bonferroni-corrected over the contrasts. C1 pretrained S1 minus S1. C2 B-mix minus S1. C3
  pool of 5 seeds minus a single S1 seed (reported beside the pool of 3 seeds, so that ensembling
  is separated from the component). C4 the assembly minus the single best component, where the
  best component is fixed from the C1 to C3 point estimates at the signed-MW six before C4 is
  read, and held for the final test. Pretraining and B-mix each add a stage to the pipeline, so
  the smallest gain worth porting is 0.03 points at the signed-MW six (experiment 425's
  threshold); the pool adds no model and its threshold is zero, with its production cost reported.
  Pretraining advances if its C1 interval lies wholly below -0.03 at the signed-MW six or at the
  MVA-only eight, and is not above +0.02 CRPS at the other group. B-mix advances only through harm
  rows: on real lag-jump rows (event-cluster interval, with and without the largest event) its q99
  pinball loss is lower than S1's with an interval wholly below zero, and its clean-row CRPS is no
  more than +0.03 above S1's; it is a named option, not a default, whatever it scores. The pool
  advances if C3 is wholly below zero. Tail check units: pinball loss at q95 and q99 in points of
  effective capacity, clean rows, no more than +0.02 above S1. The three leave-one-out arms
  (embedding, surprise inputs, pretraining) and LOO-Bmix are read as checks: a component whose
  removal does not raise CRPS is reported as not supported inside the assembly. The minimum
  detectable effect is computed per contrast from the variance of the paired fold-by-seed
  differences (S1 seeds 0 to 5), printed before any contrast is read. The final test (5 folds by 3
  seeds, adding 2025-12) runs once, on the advancing components only, against every stored
  reference, and is not used to tune. The winning configuration of the experiment 447 sweep, if it
  has reported, is confirmed inside the assembled network at the end as one extra arm.
  Pre-registered before S1-w256 is scored (added 2026-11-01 at the neural Discuss session's
  request): the named arm S1-w256 is S1 with hidden width 256 and otherwise the S1 recipe
  (experiment 447's post hoc, interim, unaudited finding: +0.087 clean CRPS in 5 of 5 folds,
  carried by fold 2025-12, with worse real events). It is not the default. It advances to one
  assembled-network variant at width 256 in the final test only if (a) its clean-row CRPS gain
  against S1 holds on the four screen folds, which exclude 2025-12, at both metering groups, and
  (b) on real event rows its CRPS is no worse than +0.03 and its q99 pinball loss no worse than
  +0.02 against S1. The page reports the real-event and cold-winter-night rows beside it.
benchmark: >-
  S1 seeds 0 to 5 from experiments 428 and 439 (raw forecasts, 1.15); R-inj; the monthly XGBoost
  recipe, BG2, `manual_heuristic`, and `climatology` as read by experiment 425. Table P of
  experiment 435 is a last stage run only if the 435 page has merged when the screen finishes; if
  it has not, the page says so and reports raw and 1.15-widened scores only.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles at days 1 to 10; coverage and pinball loss at q01, q05, q95, q99;
  cold winter nights; sunny half-hours; avoidable-curtailment cost; clean rows and rows with a lag
  jump, with and without the largest event; the encroachment sweep of experiment 392 for the final
  arm. Per group and per Primary.
verdict: partly supported
best_arm: >-
  The five-seed pool of S1, an equal-weight average of the mixture distributions of five fits of
  S1 that differ only in their random seed: CRPS 2.802 at the signed-MW six and 3.501 at the MVA-
  only eight (days 1 to 10, four screen folds, clean rows, 1.15 widening), against 2.850 and 3.556
  for one fit of S1 (-0.048 and -0.056 points).
headline: >-
  Of the three components added to S1, only the five-seed pool lowered CRPS. Pretraining raised
  CRPS by 0.053 points at the signed-MW six and 0.047 at the MVA-only eight, and by 0.86 at D15
  and D18. The fixed 50/50 B-mix raised CRPS by 0.060 and 0.087 (0.93 at D15 and D18). Fine-tuning
  at the full learning rate did not remove the harm from pretraining. On the four screen months
  the five-seed pool scored 2.802 and 3.501, level with BG2 (2.845 and 3.543) and ahead of the
  monthly XGBoost recipe (3.017 and 3.615); with the fifth month BG2 is ahead at the signed-MW six
  (2.856 against 2.949).
lessons: >-
  The Primary embedding is the largest component: removing it raised CRPS by 0.65 and 0.74 points
  inside the pretrained five-seed pool. The surprise inputs were worth 0.03 points in that
  network, much less than the 0.106 points they were worth in experiments 428 and 439. Pretraining
  with every recent-power input masked scored worse than S1 after fine-tuning; the cause was not
  tested.
take_home_message: >-
  Averaging five random-seed fits of the S1 neural network lowered its error by about 0.05 points
  at both groups of Primary substations, and no other add-on helped. Pretraining on past-weather
  estimates and the fixed 50/50 mix with the lagless neural network each raised the error slightly
  at the six and eight series and much more at D15 and D18. On four test months the five-seed
  average is level with BG2 and ahead of the monthly XGBoost recipe, but with a fifth month BG2 is
  ahead at the signed-MW six, so all numbers are interim.
advice:
  - >-
    Average five random-seed fits of the S1 network by pooling their mixture distributions,
    and report the production cost of 255 forward passes per series (5 seeds times 51 ensemble
    members) (C3).
  - >-
    Do not pretrain S1 on ERA5 and CAMS rows with the recent-power inputs masked; it raised
    CRPS at all three groups, and the full learning rate did not help (C1, C1b).
  - >-
    Do not use the fixed 50/50 B-mix of S1 and the lagless network as a default; it raised
    clean-row CRPS by 0.06 to 0.09 points at the six and eight series (C2).
  - >-
    Keep the Primary embedding in every demand neural network trained across Primaries;
    removing it raised CRPS by 0.65 to 0.74 points (leave-one-out table).
production_status: candidate
adopted_in: none
code: ml-experiments/00448-s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool/
---
**Take-home message: averaging five random-seed fits of the S1 neural network lowered its error by about
0.05 points at both groups of Primary substations, and no other add-on helped.** Pretraining, the fixed
50/50 B-mix, and a wider hidden layer each failed their pre-registered rule. On four test months the
five-seed average is level with BG2 (the best-guess stack of five demand models) and ahead of the
monthly XGBoost recipe. With a fifth month, BG2 is ahead at the signed-MW six. All numbers are
interim.

## Figures

![CRPS by lead time for the five-seed pool, S1, the leave-one-out arms, and the references](figure_1_crps_by_lead.svg)

*Figure 1. CRPS by lead time, clean rows, four screen folds, 1.15 widening.*

![Each leave-one-out arm minus the pretrained five-seed pool, by lead day](figure_2_loo_by_lead.svg)

*Figure 2. Removing the embedding, the surprise inputs, B-mix, or pretraining, by lead day.*

![Contrasts of every ladder arm against S1, with 95% intervals](figure_3_ladder_contrasts.svg)

*Figure 3. Contrasts against S1 with 95% intervals, by metering group.*

![Tail coverage by arm](figure_4_tail_coverage.svg)

*Figure 4. Coverage at q01, q05, q95, and q99.*

![Day-ahead example forecast](figure_5_example_day_ahead.svg)

*Figure 5. Day-ahead forecasts of the five-seed pool for series D9, winter and summer.*

![Ten-day example forecast](figure_6_example_10_day.svg)

*Figure 6. Ten-day forecasts of the five-seed pool for series D9. Stored forecasts end at 240 hours,
so the page shows 10 days, not 14.*

![Six real lag-jump events](figure_7_real_events.svg)

*Figure 7. Six real events, three per metering group, with the event start and end marked.*

## Definitions

- **S1:** a mixture density neural network trained jointly on all Primary substations, with three
  lagged power readings and the surprise (observed minus forecast power) at each lag.
- **Signed-MW six and MVA-only eight:** the six Primaries metered in signed megawatts and the eight
  metered in megavolt-amperes only. D15 and D18 (short-history Primaries) are reported separately.
- **Clean rows:** half-hours with no logged lag jump (a step in the recent-power readings). **Real
  event rows** are half-hours inside an event that NGED's log confirms.
- **1.15 widening:** each forecast's quantile spread multiplied by 1.15 about the median.
  **CRPS** is in points of effective capacity, days 1 to 10 unless stated.
- **Five-seed pool:** the equal-weight average of the mixture distributions of five fits of one
  arm that differ in random seed. **B-mix** is the fixed 50/50 mixture of S1's distribution and
  that of the lagless neural network L0.

## Results

- **Contrasts against S1** (clean rows, CRPS points, negative is better; 95% intervals from a paired
  bootstrap within folds, which understates fold-to-fold and seed uncertainty; labels follow the
  plan; minimum detectable effect 0.057 at the six and 0.056 at the eight):

  | Contrast | Signed-MW six | MVA-only eight | D15 and D18 |
  |---|---|---|---|
  | C1 pretrained S1 | +0.053 (0.037 to 0.067) | +0.047 (0.014 to 0.078) | +0.859 |
  | C1b pretrained S1, fine-tuned at the full learning rate (added after the scores were seen) | +0.058 (0.042 to 0.073) | +0.048 (0.016 to 0.081) | +0.902 |
  | C2 B-mix | +0.060 (0.050 to 0.070) | +0.087 (0.070 to 0.104) | +0.927 |
  | C3 five-seed pool | -0.048 (-0.052 to -0.045) | -0.056 (-0.060 to -0.052) | -0.213 |
  | C3b three-seed pool | -0.033 (-0.038 to -0.029) | -0.044 (-0.051 to -0.037) | -0.053 |
  | C4 pretrained five-seed pool minus five-seed pool | +0.059 (0.043 to 0.073) | +0.047 (0.015 to 0.080) | +0.843 |
  | C5 hidden width 256 | -0.022 (-0.037 to -0.005) | -0.037 (-0.052 to -0.022) | +0.375 |

- **Leave-one-out against the pretrained five-seed pool:** removing the Primary embedding raises CRPS
  by 0.649 (0.597 to 0.702) at the six and 0.741 (0.686 to 0.794) at the eight; removing the surprise
  inputs, by 0.033 (0.026 to 0.039) and 0.027 (0.017 to 0.036); removing B-mix lowers CRPS by 0.059
  and 0.096; removing pretraining lowers CRPS by 0.059 and 0.047.
- **Pretraining fails.** C1 is above zero, so no interval lies below -0.03. Plan predictions P1
  (pretraining helps) and P4 (the assembly beats the five-seed pool) failed.
- **B-mix fails.** Its clean-row CRPS exceeds the +0.03 cap at both groups. On real harm rows its q99
  pinball loss is lower than S1's at the six (0.255 against 0.278) and higher at the eight (0.367
  against 0.347), without intervals. At D15 and D18 it lowers real-event CRPS. It stays a named option.
- **The five-seed pool advances.** C3 is below zero and the pool adds no extra fitted network, only
  255 forward passes per series (5 seeds times 51 ensemble members).
- **Width 256 fails rule (b).** Rule (a) asks for the clean gain to hold on the four screen folds
  at both groups; it held in 3 of 4 folds at the six and 4 of 4 at the eight, and is under the
  minimum detectable effect. Rule (b) caps real-event CRPS at +0.03 and q99 pinball loss at +0.02
  against S1. At the eight, real-event CRPS is +0.323 (0.217 to 0.420), so the rule fails. At the six
  it is -0.200, and the q99 pinball loss is unchanged at the eight (0.449) and 0.025 lower at the six.
- **Five-seed pool against the references** (days 1 to 10, the 579,044 half-hours that every arm and
  reference shares; XGBoost monthly is unwidened):

  | Slice | S1 | Five-seed pool | BG2 | XGBoost monthly | `manual_heuristic` | `climatology` |
  |---|---|---|---|---|---|---|
  | Signed-MW six | 2.850 | 2.802 | 2.845 | 3.017 | 3.554 | 3.946 |
  | MVA-only eight | 3.556 | 3.501 | 3.543 | 3.615 | 4.171 | 4.599 |
  | D15 and D18 | 9.160 | 8.952 | 7.687 | 8.879 | 12.088 | 22.740 |
  | Cold nights, six | 2.801 | 2.722 | 2.405 | 2.814 | 4.820 | 5.159 |
  | Cold nights, eight | 2.917 | 2.821 | 2.848 | 3.015 | 4.450 | 4.799 |

  The differences between the pool and BG2 (0.043 at both groups) are under the minimum detectable
  effect and have no interval. BG2 is better on cold nights at the six and at D15 and D18.
- **With the fifth month (2025-12)** (S1, its pool, and the references on the same 713,323 rows):
  CRPS at the six is 3.038 for S1, 2.949 for the five-seed pool, 2.856 for BG2, and 3.046 for
  XGBoost monthly; at the eight it is 3.602, 3.523, 3.563, and 3.654. On cold nights at the six the pool
  scores 3.237, XGBoost monthly 3.045, and BG2 2.568.
- **Tails** (five folds, clean rows, days 1 to 10): the five-seed pool has q95 coverage 0.932 and
  q99 coverage 0.989 at the six, and 0.949 and 0.990 at the eight; S1 has 0.925 and 0.987 at the six.
  On the four folds, the cold-night q99 pinball loss at the six is 0.155 for the pool, against 0.26
  for BG2 and 0.39 for XGBoost monthly (five folds: 0.161 and 0.242).
- **Avoidable-curtailment cost** (the cost of curtailing more than needed, signed-MW six, lead 12 to
  36 hours, four folds, shared clean half-hours, seed-mean forecast for the S1 arms): with the
  export limit set at 0.5 and 0.75 of its reference value, the pool costs £18.0M and £5.82M against
  £19.3M and £6.40M for BG2. At 0.9, BG2 is lower (£1.28M against £1.76M).
- **Real events:** the pool lowers real-event CRPS at the eight (7.049 against 7.142 for S1) and at the
  six (5.569 against 5.689). Its q99 coverage at the six is 0.871 against 0.860 for S1, and the
  target is 0.99. NGED's logs may miss events, so clean rows may hold unlogged events and real-event
  rows undercount.

## Design

- **Arms.** S1; the pretrained S1; the pretrained S1 fine-tuned at the full learning rate; S1 with B-mix;
  the three-seed and five-seed pools; the pretrained five-seed pool, with and without B-mix; the
  leave-one-out arms without the embedding and without the surprise inputs; and S1 with hidden
  width 256. S1 seeds 0 to 2 (experiment 428) and seeds 3 and 4 (experiment 439) were already stored.
- **Fits.** 20 pretrained S1, 20 full-learning-rate, 40 leave-one-out, 16 lagless, 20 width-256, and
  five S1 fits at the fifth fold. Each took about a minute.
- **Pretraining.** ERA5 and CAMS rows (reanalysis estimates of past weather) from experiment 418
  pass through the input scaling fitted on the fold's training data. The ensemble-spread inputs are
  set to the median spread at each lead-time band, since past-weather estimates have no
  ensemble. Every recent-power input is masked (lag values 0, missing flags 1, surprise 0). Fine-tuning
  then switches the lags on, so the network meets live lag values for the first time there.
- **Pooling.** The pool averages the members' cumulative distribution functions on the capacity
  scale. Single-seed arms report tails from the seed-mean quantile forecast, as in experiment 428,
  so their tail coverage is not strictly comparable with the pool's.
- **Reused code.** Experiments 428, 439, 425, 418, 404, and 446, imported and not edited.
- **Cadence and cleaning.** Monthly refits; cleaned data; D15 and D18 cut at 2025-03-01.
- **Not run.** The export gate is not a ladder arm, because it is scored on signed output (see
  experiments 420 and 424). The recalibration table of experiment 435 was not available when the
  screen finished, so scores are raw and 1.15-widened only.
- **Reference rows.** The reference forecasts of experiment 87 cover only 25 of 31 forecast
  origins per month. The reference comparison therefore loses about 20% of the S1 rows. All other
  tables use every row.

## Caveats

- Four screen folds plus a fifth fold for S1 and its pool; every number is interim with no 99% claim.
  The pool's gain (0.048) is under the minimum detectable effect of 0.057, which comes from seed-to-seed
  spread. The paired interval nonetheless excludes zero.
- D5 has no clean rows in 2026-01, so the signed-MW six holds five series that month.
- The 14-day horizon is not available; stored forecasts end at 240 hours.
- Pretraining with masked lags is one design. Pretraining that includes lagged power from the
  ERA5 period, with a lagless cross-fit, was not tried.

## What to try next

- Use the five-seed pool of S1 in the next demand experiments, and look at why BG2 is better on cold
  nights at the signed-MW six and in the fifth fold.
- Try pretraining with lagged power in the pretraining rows, because the masked-lag design may be why
  pretraining failed.

## Checks

- **Audit triggers.** Trigger 2 (the result contradicts its predictions and earlier experiments)
  fired: P1, P2, P4, and the surprise-input range failed, and experiments 425 (pretraining helped)
  and 428 and 439 (surprise inputs worth 0.106) point the other way. Trigger 3 is not cleared,
  because the median minus mean forecast per site and the mean-forecast error beside each quantile
  contrast were not computed. Trigger 4 (leave one site out) was not run. Trigger 7 may apply,
  because the pool is recommended as the default and is not replicated. An audit is needed before
  merge.
- **Reviews.** The plan review, code review, numbers review, and prose review ran; their findings are
  in the pull request.
