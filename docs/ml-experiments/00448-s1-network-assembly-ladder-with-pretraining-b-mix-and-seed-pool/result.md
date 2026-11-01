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
  S1-pool5, an equal-weight pool of five seeds of S1: CRPS 2.802 at the signed-MW six and 3.501 at
  the MVA-only eight (days 1 to 10, four screen folds, clean rows, 1.15 widening), against 2.850
  and 3.556 for one seed of S1 (-0.048 and -0.056 points).
headline: >-
  Of the three components added to S1, only the five-seed pool lowered CRPS. Pretraining raised it
  by 0.053 points at the signed-MW six and 0.047 at the MVA-only eight, the fixed 50/50 B-mix
  raised it by 0.060 and 0.087, and fine-tuning at the full learning rate did not rescue
  pretraining. The pooled S1 scored 2.802 and 3.501, against 2.845 and 3.543 for BG2 and 3.017 and
  3.615 for the monthly XGBoost recipe.
lessons: >-
  The Primary embedding is the largest single component (removing it raises CRPS by 0.65 and 0.74
  points). The surprise inputs are worth 0.03 points once the pool is in place, much less than
  their 0.106 points in experiments 428 and 439. A network pretrained with every recent-power
  input masked does not learn to use the lags afterwards.
take_home_message: >-
  Averaging five seeds of the S1 network is the only add-on that helped: it lowered CRPS by 0.05
  points at both groups of Primaries and put the neural network just ahead of BG2 and the monthly
  XGBoost recipe on days 1 to 10. Pretraining on weather estimates and the fixed 50/50 mix with
  the lagless network each made the forecast slightly worse, so the best architecture for now is
  S1 pooled over five seeds. The result comes from four test months and is interim; BG2 is still
  better on cold nights at the signed-MW six and at D15 and D18.
advice:
  - >-
    Pool five seeds of the S1 network by averaging their mixture distributions, and report the
    cost of 255 forward passes per series for 51 ensemble members (C3).
  - >-
    Do not pretrain S1 on ERA5 and CAMS rows with the lag inputs masked; it raised CRPS at
    both groups, and the full learning rate did not help (C1).
  - >-
    Do not use the fixed 50/50 B-mix of S1 and the lagless network as a default; it raised
    clean-row CRPS by 0.06 to 0.09 points (C2).
  - >-
    Keep the Primary embedding in every pooled demand network; removing it raised CRPS by 0.65
    to 0.74 points (leave-one-out table).
production_status: candidate
adopted_in: none
code: ml-experiments/00448-s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool/
---
**Take-home message: averaging five seeds of the S1 network is the only add-on that helped, so the
best demand network for now is S1 pooled over five seeds.** Pretraining on weather estimates, the
fixed 50/50 mix with the lagless network, and a wider hidden layer each failed their pre-registered
rule. The pooled network scores just ahead of BG2 and the monthly XGBoost recipe on days 1 to 10,
and behind BG2 on cold nights at the signed-MW six and at D15 and D18. All numbers are interim, from
four test months.

## Question and answer

**The question was whether three cheap add-ons to the S1 network, added one at a time and then
together, lower its error.** S1 is the pooled mixture density network with three lagged power
readings and the surprise at each. The add-ons were pretraining on ERA5-and-CAMS rows then
fine-tuning on ensemble rows, the fixed 50/50 mix of the S1 and lagless distributions (B-mix), and
a pool of five seeds. Only the pool helped. The fixed rules in the plan decided each outcome before
any score was read, and the plan's prediction P1 (pretraining helps) and P2 (B-mix passes through
the harm rows) failed.

## Figures

![CRPS by lead time for the pooled S1, S1, the ablations, and the references](figure_1_crps_by_lead.svg)

*Figure 1. CRPS by lead time, clean rows, four screen folds, 1.15 widening.*

![Each leave-one-out arm minus the pooled S1, by lead day](figure_2_loo_by_lead.svg)

*Figure 2. Removing the embedding, the surprise inputs, or pretraining, by lead day.*

![Paired contrasts of every ladder arm against S1](figure_3_ladder_contrasts.svg)

*Figure 3. Contrasts against S1 with intervals, by metering group.*

![Tail coverage by arm](figure_4_tail_coverage.svg)

*Figure 4. Coverage at q01, q05, q95, and q99.*

![Day-ahead example forecast](figure_5_example_day_ahead.svg)

*Figure 5. A day-ahead forecast of the pooled S1 at D9, winter and summer.*

![Ten-day example forecast](figure_6_example_10_day.svg)

*Figure 6. The longest stored horizon is 240 hours, so the page shows 10 days, not 14.*

![Six real lag-jump events](figure_7_real_events.svg)

*Figure 7. Six real events, three per metering group, with the event start and end marked.*

## Results

- **Contrasts against S1** (clean rows, CRPS points, negative is better, 95% month-block bootstrap
  intervals; minimum detectable effect 0.057 at the signed-MW six and 0.056 at the MVA-only eight):

  | Contrast | Signed-MW six | MVA-only eight | D15 and D18 |
  |---|---|---|---|
  | C1 pretrained S1 | +0.053 (0.037 to 0.067) | +0.047 (0.015 to 0.079) | +0.859 |
  | C1b pretrained, full learning rate (post hoc) | +0.058 (0.042 to 0.073) | +0.048 (0.016 to 0.081) | +0.902 |
  | C2 B-mix | +0.060 (0.050 to 0.070) | +0.087 (0.071 to 0.104) | +0.927 |
  | C3 five-seed pool | -0.048 (-0.052 to -0.045) | -0.056 (-0.060 to -0.052) | -0.213 |
  | C3b three-seed pool | -0.033 | -0.044 | n/a |
  | C5 hidden width 256 | -0.022 (-0.037 to -0.005) | -0.037 (-0.052 to -0.022) | +0.375 |

- **Leave-one-out against the pretrained, pooled network:** removing the Primary embedding raises
  CRPS by 0.649 (0.597 to 0.702) at the signed-MW six and 0.741 (0.688 to 0.799) at the MVA-only
  eight; removing the surprise inputs, by 0.033 (0.026 to 0.039) and 0.027 (0.017 to 0.036);
  removing B-mix lowers it by 0.059 and 0.096; removing pretraining lowers it by 0.059 and 0.047.
- **Advance rules.** Pretraining fails (C1 is above zero). B-mix fails: its clean-row CRPS is above
  the +0.03 cap at both groups. The pool advances (C3 is below zero; it adds no model, and costs
  255 forward passes per series for 51 ensemble members). Width 256 fails rule (b): on real event
  rows at the MVA-only eight its CRPS is +0.323 (0.217 to 0.420) against S1, above the +0.03 cap,
  although it is better at the signed-MW six (-0.200). Its q99 pinball loss on real event rows is
  unchanged at the eight (0.449) and 0.025 lower at the six. Its clean gain (-0.022, -0.037) is
  under the minimum detectable effect, and it is 0.375 worse at D15 and D18.
- **Pooled S1 against the references** (days 1 to 10, the 579,044 half-hours that every arm and
  reference shares, 1.15 widening of the S1 family and XGBoost-w115 only):

  | Slice | S1 | S1 pooled | BG2 | XGBoost monthly | `manual_heuristic` | `climatology` |
  |---|---|---|---|---|---|---|
  | Signed-MW six | 2.850 | 2.802 | 2.845 | 3.017 | 3.554 | 3.946 |
  | MVA-only eight | 3.556 | 3.501 | 3.543 | 3.615 | 4.171 | 4.599 |
  | D15 and D18 | 9.16 | 8.952 | 7.687 | 8.879 | 12.088 | 22.74 |
  | Cold nights, six | 2.801 | 2.722 | 2.405 | 2.814 | 4.82 | 5.159 |
  | Cold nights, eight | 2.917 | 2.821 | 2.848 | 3.015 | 4.45 | 4.799 |

- **Fifth fold.** With 2025-12 added for S1 and its pool, the pool lowers CRPS from 3.038 to 2.949 at
  the signed-MW six and from 3.602 to 3.523 at the MVA-only eight (all clean rows, five folds).
- **Tails** (five folds, clean rows, days 1 to 10): the pooled S1 has q95 coverage 0.932 and q99
  coverage 0.989 at the signed-MW six, and 0.949 and 0.990 at the MVA-only eight. S1 alone has 0.925
  and 0.987 at the six. The cold-night q99 pinball loss at the six is 0.155 for the pooled S1,
  against 0.26 for BG2 and 0.39 for the monthly XGBoost recipe.
- **Avoidable curtailment cost** (signed-MW six, lead 12 to 36 hours, four folds): at limit
  fractions 0.5 and 0.75 the pooled S1 costs 18.0M and 5.82M pounds, against 19.3M and 6.40M for
  BG2. At 0.9 BG2 is lower (1.28M against 1.76M). The cost is computed on shared clean half-hours,
  with the seed-mean quantile forecast for the S1 arms.
- **Real events.** The pooled S1 lowers real-event CRPS at the MVA-only eight (7.049 against 7.142
  for S1) and at the signed-MW six (5.569 against 5.689), and q99 coverage at the six is 0.871
  against 0.860. The unlogged-event caveat applies: NGED's logs may miss events, so clean-lag rows
  may hold events no label covers, and real-event slices undercount.

## Design

- **Arms, data, and recipe.** Arms are in the plan table. S1 seeds 0 to 2 (experiment 428) and 3 and
  4 (experiment 439) are stored. New here: 20 pretrained S1 fits, 20 full-learning-rate fits that
  reuse the same pretraining, 40 leave-one-out fits (without the embedding, and without the
  surprise inputs), 16 lagless fits for seeds 1 to 4, 20 width-256 fits, and 5 S1 fits at the fifth
  fold. Each fit took about a minute.
- **Pretraining design.** ERA5-and-CAMS rows from experiment 418 pass through the fold's
  standardiser, with the `std_*` columns set to the median ensemble spread by lead band. Every
  recent-power input is masked (lag values 0, missing flags 1, surprise 0). Fine-tuning then
  switches the lags on, so the network meets live lag values for the first time there.
- **Pooling.** The pool averages the members' cumulative distribution functions on the capacity
  scale. The tail columns of single-seed arms are computed on the seed-mean quantile forecast, as
  in experiment 428, whereas the pooled arms are mixture pools; compare their tails with that in
  mind.
- **Reused code.** Experiments 428, 439, 425, 418, 404, and 446 (the folders are imported, not
  edited). The reference forecasts of experiment 87 cover only 25 of 31 origins per month, so the
  reference comparison loses about 20% of the S1 rows; the plan's other tables use all rows.
- **Cadence and cleaning.** Monthly refits; cleaned data; D15 and D18 cut at 2025-03-01.
- **Export gate.** Not a ladder arm (it is scored on signed output); see experiments 420 and 424.
  Table P (experiment 435) had not merged, so no recalibration stage was run.
- **Hyperparameter sweep.** Experiment 447 found no setting better than the S1 recipe except width
  256 post hoc; this experiment tested width 256 as a named arm and it failed rule (b).

## Caveats

- Four screen folds plus one fifth fold for the pool; every number is interim and carries no 99%
  claim. The pool's gain is under the minimum detectable effect of 0.057 by the seed-spread
  convention, although the paired bootstrap interval excludes zero.
- B-mix's harm-row rule was not scored with an interval: its q99 pinball loss on real harm rows is
  lower than S1's at the six (0.255 against 0.278) and higher at the eight (0.367 against 0.347),
  but the clean-row cap already fails it.
- D5 has no clean rows in 2026-01, so the signed-MW six holds five series that month.
- The 14-day horizon is not available: stored forecasts end at 240 hours.
- Pretraining with masked lags is one design; pretraining that includes lagged power from ERA5-era
  power and a lagless cross-fit was not tried.

## What to try next

- Pool the S1 network over five seeds in the next demand experiments, and compare against BG2 on
  cold nights at the signed-MW six, where BG2 still wins (2.405 against 2.722).
- Try pretraining with lagged power in the pretraining rows, because masking every lag may be why
  pretraining failed.

## Checks

- Results review and write-up review: to be recorded in the pull request.
- Median minus mean forecast per site and the mean-forecast error beside each quantile contrast were
  not computed for this screen.
