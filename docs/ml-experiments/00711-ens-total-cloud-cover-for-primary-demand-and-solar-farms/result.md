---
id: "00711"
slug: ens-total-cloud-cover-for-primary-demand-and-solar-farms
title: >-
  Does the ECMWF ensemble's forecast of total cloud cover improve the Primary demand forecast or the
  forecast of NGED's metered solar farms?
theme: weather-products-and-blending
also_relevant_to:
  - demand-inputs
  - metered-solar-farms
status: finished
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It planned about 950 fitted neural networks and ran 282, storing their per-row
  forecasts, and it builds
  new cloud summary tables; (2) more than one design defensibly tests the hypothesis (the cloud
  summary, the placebo, and the handling of a 19-month gap in the cloud history); (3) it copies
  training code from experiments 707 and 666, whose callers cannot be named without searching; (4)
  it touches no serving path or degradation rule; (5) it changes no contract. Full set of reviews:
  plan, code before each script's first run, results, write-up, prose, and the audit triggers.
hypothesis: >-
  Giving the neural network the mean and the spread, across the 51 ECMWF ENS members, of total
  cloud cover from the run issued at or before the forecast origin lowers the CRPS of (a) the
  Primary demand forecast and (b) the metered solar farm forecast by more than a placebo cloud
  series with the same missing-value pattern does.
prediction: >-
  Written before any fit, and all numbers interim. For Primary demand, the real-minus-placebo
  contrast is between -0.03 and +0.01 points of CRPS at both groups of Primaries, so the 0.02
  threshold is not cleared; experiment 559's arm A11c found +0.014 (-0.023 to +0.050) at the eight
  MVA-only Primaries and +0.033 (-0.025 to +0.099) at the six signed-MW Primaries. For the solar
  farms, the contrast is between -0.03 and 0 points, about 1% of the 2.91 points of daylight CRPS
  of experiment 666's PLP network, because ENS cloud comes from the same weather model as the ENS
  shortwave radiation that PLP already reads.
decision_rule: >-
  Each planned reading (C1, C2, C3, S1, S2, S3) is read at 1 - 0.05/9 = 99.44% on its interval
  [lower, upper], and the decisive interval is the wider of the month-block bootstrap interval and
  the t interval on the monthly differences (the interval with the larger upper limit minus lower
  limit, never the union of the two). The first test that holds decides. Lower above 0:
  refuted. Upper below -0.02: supported. Upper below 0: small gain. Lower at or above -0.02 and
  upper at or above 0: not supported (a gain of 0.02 is excluded). Otherwise inconclusive. The
  threshold of 0.02 points is the minimum gain that both protocols set for a new input, because
  the column adds one variable to ingest, validate, and serve, plus a missing-value rule for the
  19-month gap. A target reads as the better of its two summary readings, ranked supported, small
  gain, inconclusive, not supported, refuted. Partly supported: the two summary readings at a
  group are inconclusive or not supported and C3 is supported there. A failed positive control
  makes both Primary demand groups inconclusive. Stage A (section 10 of the appendix, compute) reads
  C1 at the six signed-MW Primaries and at the eight MVA-only Primaries, and S1, and its verdict is
  decided by those three readings alone; C2, C3, S2, S3, the
  positive control, and every other contrast are Stage B, which starts only on the manager's word
  after Stage A is scored and reviewed.
benchmark: >-
  The matching placebo arm of each real arm; stored `R_F` and `A_F` of experiment 707, stored PLP of
  experiment 666, climatology, the manual heuristic, and the stored XGBoost-based forecasts BG2
  (experiment 87) and S0M (experiment 389) as reference rows only.
row_set_id: >-
  fold_a_daily_gap_v1 (Primary demand); backfill_daily_2023_04_to_2024_03_v1 (Primary demand
  replicate); solar_daily_1200_2025_12_to_2026_06_v1 (solar farms)
metric: >-
  Continuous ranked probability score (CRPS; twice the mean pinball loss over 11 quantile levels) in
  points of effective capacity at lead days 1 to 10, with the normalised mean absolute error (MAE)
  of the median beside it; q01, q05, q95, and q99 coverage and
  pinball loss; cold winter nights; avoidable-curtailment cost.
verdict: inconclusive
best_arm: >-
  No arm wins. At the six signed-MW Primaries the real-cloud arm minus its placebo is -0.029 points
  of CRPS (99.44% interval -0.112 to +0.056), and the best arm by absolute score is not decided by
  this experiment.
headline: >-
  Final for the reduced design (Stage A and Stage B1; 282 of 946 planned fits). Real cloud minus
  placebo cloud, at 99.44%, days 1 to 10, 7 folds, seeds 0 to 2: the six Primaries -0.029 (-0.112 to
  +0.056), the eight Primaries -0.001 (-0.065 to +0.087), the six solar farms pooled +0.022 (-0.056
  to +0.103). In the gap-free 12-month replicate: the five Primaries -0.006 (-0.045 to +0.031), the
  eight +0.017 (-0.033 to +0.059). The C1, S1, and C3 readings exclude gains above 0.033 to 0.11
  points; the G1 readings exclude only gains above 0.11 and 0.21.
lessons: >-
  A missing-value flag, probably acting as a date proxy (an untested hypothesis), made the Primary
  demand forecast 0.122 points worse at the six and 0.147 worse at the eight than the stored
  reference, so the gap-masking recipe is a weaker base than the plain neural network. The 19-month
  gap in the cloud history, the 7 months of recent cloud, and the 0.02-point threshold together
  leave the intervals too wide to show a gain that small. The solar farms' gain over stored PLP is
  not a gain from cloud; the arm that would separate the refit from the training window (`P0L`) was
  not run.
take_home_message: >-
  We found no detectable gain from the ECMWF ensemble's total cloud cover in the Primary demand
  forecast or the solar farm forecast, at the statistical precision these data allow. The intervals
  rule out a gain larger than about 0.03 to 0.11 points of CRPS, and they cannot rule out a gain of
  0.02 points, so the result is inconclusive and does not show that cloud is useless. A future test
  needs a longer history of recent cloud before its intervals can show a gain that small.
advice:
  - >-
    Do not add ENS total cloud cover to the Primary demand or solar farm forecast on this evidence
    (C1, S1, C3).
  - >-
    Test a missing-value flag against a flag-only arm before reading any cloud gain, because the
    flag alone raised CRPS by 0.12 to 0.15 points (section "The missing-value flag raises the CRPS
    of Primary demand"), and add a placebo-flag arm to test the date-proxy hypothesis.
  - >-
    Plan a cloud test so that its interval half-width is below 0.02 points; 7 months of recent cloud
    gave 0.076 to 0.084 and a gap-free 12 months gave 0.038 to 0.046 (C1, S1, C3).
production_status: not-for-production
adopted_in: none
---

**Result: `inconclusive`. The experiment found no detectable gain from the ECMWF ensemble's forecast
of total cloud cover, at the statistical precision these data allow.** We asked whether a forecast
of cloud cover helps two forecasts that already use the weather. The first forecast is the power
that Primary substations draw, and the second is the output of six metered solar farms. The European
Centre for Medium-Range Weather Forecasts (ECMWF) runs 51 versions of each weather forecast (an
ensemble, ENS) from slightly different starting conditions. We gave each neural network the mean and
the spread of the 51 cloud-cover forecasts. We compared that neural network with a twin that
received the cloud forecast of a different day, with exactly the same gaps. A gain over the twin is
a gain from the cloud forecast itself.

**Primary demand reads `inconclusive` at both groups of Primaries, and the solar farms read
`inconclusive`.** The continuous ranked probability score (CRPS) is lower for a better forecast.
Real cloud minus placebo cloud, in points of CRPS at 99.44%, is -0.029 (-0.112 to +0.056) at the six
Primaries that report signed megawatts, -0.001 (-0.065 to +0.087) at the eight Primaries that report
apparent power in megavolt-amperes only, and +0.022 (-0.056 to +0.103) at the six solar farms
pooled. The intervals exclude gains larger than 0.056 to 0.11 points and cannot exclude a gain of
0.02 points, so the data neither show a gain nor show that cloud is useless.

**Every reading is final for the reduced design, which ran 282 of the 946 planned fits.** The
maintainer asked for the experiment by name on 2026-11-19. The reduced design is Stage A (132 fits)
and Stage B1 (150 fits), both on seeds 0 to 2. The other 664 fits of Stage B were not run, among
them seeds 3 to 5 of the replicate arms `R1` and `R1p`. Section 13 of the appendix (differences
between the plan and the code) records the amendments.

## Figures

**Each difference is the real-cloud arm minus its placebo arm in points of CRPS, so a negative
difference favours the real cloud.** CRPS is twice the mean pinball loss over 11 quantile levels, in
points of effective capacity, at lead days 1 to 10. Each interval is the wider of the month-block
bootstrap interval and the t interval on the monthly differences, at 99.44% (section 8 of the
appendix, the interval). The last column is the gain that the lower limit excludes.

| Reading | Folds | Difference | Interval at 99.44% | Gain excluded | Verdict |
|---|---|---|---|---|---|
| C1, the six signed-MW Primaries | 7, 2025-12 to 2026-06 | -0.029 | [-0.112, +0.056] | larger than 0.11 | `inconclusive` |
| C1, the eight MVA-only Primaries | 7, 2025-12 to 2026-06 | -0.001 | [-0.065, +0.087] | larger than 0.065 | `inconclusive` |
| S1, the six solar farms, pooled | 7, 2025-12 to 2026-06 | +0.022 | [-0.056, +0.103] | larger than 0.056 | `inconclusive` |
| C3, the five Primaries, replicate with no gap and no flag | 12, 2023-04 to 2024-03 | -0.006 | [-0.045, +0.031] | larger than 0.045 | `inconclusive` |
| C3, the eight Primaries, replicate with no gap and no flag | 12, 2023-04 to 2024-03 | +0.017 | [-0.033, +0.059] | larger than 0.033 | `inconclusive` |
| G1, real cloud minus no cloud, no flag, the six | 7, 2025-12 to 2026-06 | -0.044 | [-0.209, +0.078] | larger than 0.21 | `inconclusive` |
| G1, real cloud minus no cloud, no flag, the eight | 7, 2025-12 to 2026-06 | +0.001 | [-0.109, +0.113] | larger than 0.11 | `inconclusive` |

**The half-widths of the intervals are 0.038 to 0.14 points, 2 to 7 times the 0.02-point threshold,
so the experiment cannot say whether a gain of 0.02 points exists.** The decision rule needs an
upper limit below -0.02 for `supported`, and no interval comes near. The replicate has the narrowest
intervals, because it has 12 months of cloud in training and in scoring and no gap. Even there, the
half-widths are 0.038 and 0.046. The two C1 estimates fall inside the range of -0.03 to +0.01 that
the plan predicted. The solar estimate lies outside the predicted range of -0.03 to 0, on the
placebo's side and within noise.

**The C1 reading at the six is carried by one fold.** In fold 2025-12 the difference is -0.118, and
the other six folds average about -0.013 (audit). The six's estimate therefore rests on the fold
with the least recent cloud in its fine-tune window.

**The verdict wording is "no detectable gain at this resolution", never "cloud does not help".**
Here "resolution" means the statistical precision of the intervals, and has nothing to do with the
spatial resolution of the weather forecast. A reading of `not supported` needs an interval that
excludes a gain of 0.02 points, and none of the seven intervals does.

**Two further results qualify the figures.**

**The flag that marks the cloud gap probably acts as a date proxy, and it makes the Primary demand
forecast worse.** The arm `DF` has the missing-value flag and no cloud values. Against stored `D0`
(the pooled neural network of experiment 707 without the flag), `DF` is 0.122 points worse at the
six (95% interval +0.034 to +0.217) and 0.147 worse at the eight (+0.012 to +0.269). These readings
are exploratory. The date-proxy explanation is a hypothesis that no arm tested. Stage B1 removed the
flag and the gap in two further tests (C3 and G1), and both stayed `inconclusive`. The section "The
missing-value flag raises the CRPS of Primary demand" gives the detail.

**The solar farms' gain over the stored PLP forecast is not a gain from cloud, and its cause is not
separable without `P0L`.** The arm `P1` scores 0.163 points better than stored `P0` (95% interval
-0.349 to -0.019), but `P1` is no better than its own placebo `P1p` (S1 above). The monthly refit,
the longer training window, or both may explain the gain. The arm `P0L` would separate them, and
`P0L` was not run.

**An exploratory split of the replicate by the ECMWF resolution change of 2023-06 detected no
difference in cloud effect.** The five Primaries scored +0.006 (95% interval -0.140 to +0.153) on
the three folds before the change and -0.010 (-0.034 to +0.013) on the nine folds after it. The
eight scored +0.004 (-0.082 to +0.092) before and +0.021 (-0.018 to +0.054) after. The fold 2023-06
straddles the change of 2023-06-27. The plan pre-registered the split as exploratory, so these four
readings carry no verdict word and are not corrected for the many exploratory lines the scorer
prints.

## Checks

- **The stop rule allowed the run, and the counts stand.** The stop rule requires the maintainer's
  approval for a fourth experiment on one idea after three `not supported` or `refuted` verdicts.
  The number of earlier experiments on ENS total cloud cover that count is 0 (1 if experiment 568 is
  counted). The number on irradiance-like inputs to Primary demand is 3, and the maintainer's
  request of 2026-11-19 is the approval for this fourth experiment. The number on irradiance-like
  inputs to the solar farm forecast is 2 on the strict reading and 5 on the broad reading. A verdict
  of `inconclusive` adds nothing to any number.
- **The placebo checks passed with little power.** The placebo arm minus the no-cloud arm is +0.001
  (-0.094 to +0.186) at the six, -0.003 (-0.053 to +0.050) at the eight, and +0.003 (-0.068 to
  +0.065) at the six farms, at 99.375%. No placebo shift was detected, and the interval at the six
  permits a placebo shift from -0.09 to +0.19 points. In Stage B1 the placebo and real arms against
  no cloud (`R1p` minus `R0`, `R1` minus `R0`) are all `inconclusive`: the five +0.014 (-0.048 to
  +0.104) and +0.008 (-0.064 to +0.105), the eight -0.001 (-0.052 to +0.049) and +0.015 (-0.037 to
  +0.056).
- **Leakage and anonymisation passed.** Every prediction row has its origin at the planned hour, and
  no donor run was published after its fold's cut. The scorer prints solar farm labels A to F with
  no identifier, coordinate, or cloud value per farm.
- **The gate passed.** The fit gate holds `D0n` within 2 standard errors of the seed spread of
  stored `D0`, with pooled differences from -0.025 to +0.008 points against 2 standard errors of
  0.030 to 0.053. The audit found that `D0n` differs from `D0` by between -2e-6 and +3e-7 points per
  fold and seed on the two gate folds (2026-01 and 2026-06).
- **Audit trigger 5 fired, an audit ran, and the verdict stands.** The trigger (a nested arm beats
  its container) fired because `DF`, which is `D0` plus a flag, is worse than `D0`. The audit
  (`data/ml-experiments/briefs/total_cloud/audit.md`) recomputed `DF` minus `D0` at +0.122 and
  +0.147 from the raw predictions. It found the refit code is not the cause, and every scored row
  has the flag set to 0. The harm is positive in 7 of 7 folds at the six and 6 of 7 at the eight,
  and in all three seeds. The median offset changes sign by fold, so the flag does not add a
  constant shift. "Date proxy" is therefore a likely but untested hypothesis, and a placebo-flag arm
  would test it. The arm `D0g`, which has no flag and drops the gap rows, is worse than `D0` by
  about +0.30 at the six and +0.33 at the eight, so any recipe that masks or drops the gap rows is a
  weaker base. C1 is unaffected except in power, and the audit reproduced C1, G1, and C3.
- **Audit trigger 2 is borderline and triaged.** The solar estimate lies outside its predicted
  range, though its interval contains the range. The flag harm contradicts experiment 707, where the
  arm `R_F+flags` scored 2.820 against 2.821 for `R_F`. The audit's findings above are the recorded
  explanation.
- **Triggers 1, 4, 6, and 7 did not fire.** No planned reading changes its verdict when any one
  Primary or farm is left out. Trigger 3 could not be assessed (next bullet).
- **Not run.** The median forecast minus the mean forecast per site, the error of the median beside
  each CRPS contrast, the cold winter nights, the avoidable-curtailment cost, the coverage and
  pinball loss of the extreme quantiles, the reference rows (climatology, manual heuristic, `BG2`,
  `S0M`, `A_F`), and the unrun Stage B arms (`C2`, `S2`, `S3`, the positive control, `D1t`, `D2`,
  `D3`, `P0L`, `P2`) are in the front matter's plan and were not produced.
- **No positive control was run.** A positive control is in the unrun Stage B, so a null cannot yet
  be told from a masked-gap recipe that cannot detect a known signal.

## What a future experiment needs

**A future experiment needs narrower intervals, which means more months of recent cloud and a base
neural network without a date proxy.** The recent cloud history holds 7 months (2025-12 to 2026-06),
and the back-filled history ends in 2024-03, with a 19-month gap between them. Seven months give
half-widths of 0.076 to 0.084 points, and the gap-free replicate with 12 months gives 0.038 to
0.046. The threshold is 0.02. The month-to-month spread sets the width, so more months would narrow
the intervals, and adding seeds would barely do so. The experiment did not test whether more sites
would help.

**Nothing from this experiment is ready to adopt in the production forecast.** The result is
`inconclusive`, so no change follows, and `production_status` is `not-for-production`.

## Review points

**An independent review of the Stage A scorer output accepted the three Stage A readings and found
one scorer bug, which is fixed.**

- **The per-fold solar rows ignored their fold.** The scorer's exploratory section built the solar
  filter with all seven folds, so the seven per-fold rows printed the S1 figure. The filter now
  takes the fold, a test covers it, and the rerun prints the correct rows. The seven per-fold
  differences of `P1` minus `P1p` range from -0.050 to +0.096 points, and six of the seven favour
  the placebo. They are descriptive only. The planned readings, the checks, and the bounds were
  never affected.
- **Exploratory rows carry no verdict word.** With one month, the interval of a row holds only the
  range of the three per-seed contrasts. The scorer prints "range of 3 seeds" and "exploratory" on
  every such line.
- **The exploratory section has a multiplicity problem.** The Stage A report prints 94 exploratory
  lines at 95% with no correction, so about 5 are expected to exclude zero by chance.
- **S3 is no longer blind.** The exploratory "six farms, day 1" band equals the S3 band read on the
  Stage A arms (+0.001, [-0.074, +0.070] at 95%), so any later S3 reading is labelled "read after
  the exploratory band was seen".
- **Stage B1 was reviewed before it ran, and its findings were applied.** Section 13 of the appendix
  lists the amendment.
- **Three reviews of this page ran after the scores were final, and their findings were verified and
  applied.** The reviews were of the numbers, the prose, and the audit of trigger 5 (above).

**Findings not applied.** Each finding below was checked and left out, for the reason given.

- **Replace "at this resolution" everywhere.** The coordinator fixed the verdict wording, so the
  phrase stays and the page glosses it once.
- **Delete the appendix passages that repeat the Checks list.** The duplicate stop-rule counts,
  placebo check, and review-findings passages are removed, and the longer plan text in sections 1 to
  13 stays as the pre-registered record.
- **Fix the stop-rule dates to the git commit dates.** The numbers review could not verify them, and
  a bulk commit probably explains the difference, so the dates are left as written.
- **Reconcile "only the month and year" (section 11) with the full dates in the solar example
  (section 7).** Neither names a farm, an identifier, or a cloud value, so the rule is not breached,
  and both sections are plan text.
- **Rename the plan section leads that are labels (for example "Level of G1").** The reviewer rated
  the finding low priority, and the labelled leads are the appendix's convention.
- **Print the median-minus-mean forecast and the median-error contrasts.** Neither is in either
  score report, and no fit or scoring was allowed, so the Checks list says "not run".
- **Cite experiment 559's confidence level for trigger 6.** The level is not stated on the page and
  was not looked up, and the trigger did not fire on the unadjusted ratios.

## Method

**The experiment compares each real-cloud neural network with a placebo twin, and the plan in the
appendix, with its amendments, is the Method.** The architecture, training regime, and measurement
are in sections 3 to 8 of the appendix, in the arms (section 5), the placebos (section 6), and the
scoring (section 8). The bullets here summarise them.

- **Architecture:** the Primary demand neural network is the pooled mixture density network of
  experiment 707 in its arm `R_F`
  ([write-up](../00707-ens-backfill-pretraining-gain-against-ens-history-length/result.md)), with up
  to three extra inputs: the mean and the spread of `total_cloud_cover_atmosphere` across the 51 ENS
  members, from the run issued at or before the forecast origin, and the missing-value flag. The
  arms of Stage B1 (`R0`, `R1`, `R1p`, `D0g`, `D1g`) leave out the flag. The solar neural network is
  the pooled network with a physics-lite power input (PLP) of experiments 659 and 666, with the same
  three inputs. Section 7 of the appendix describes how the forecast is kept from using weather
  after the origin.
- **Training regime:** the Primary demand arms train as in experiment 707 on the six signed-MW and
  eight MVA-only Primaries, with seeds 0 to 2 and a refit at each monthly fold. The solar arms train
  on six metered solar farms, refit monthly. The placebo arm receives the cloud of a donor run from
  a different day, with the same missing-value pattern (sections 4 and 6). The 19-month gap from
  2024-04-01 to 2025-11-20 is masked with a flag in the headline arms and removed in the replicate.
- **Measurement:** CRPS at lead days 1 to 10 in points of effective capacity, on 7 cloud-scored
  monthly folds from 2025-12 to 2026-06 (7 solar folds) and 12 replicate folds from 2023-04 to
  2024-03. The interval is the wider of a month-block bootstrap (50,000 resamples) and a t interval
  on the monthly differences, at 99.44% (1 - 0.05/9). The decision rule and the 0.02-point threshold
  are in the front matter and section 8 of the appendix.

## Appendix A: the pre-registered plan and Stage A detail

**The sections of this appendix are the plan committed before any fit, with the Stage B1 amendments
of 2026-11-20 and the Stage A detail added.** Section 13 lists where the code departs from the plan.
Numbers in sections 1 to 12 come from earlier experiments, from a log, or from an estimate, except
the era-check, dry-run, and Stage A figures, which are results of this experiment.

### Stage A detail (scoring of 2026-11-20)

#### Stage A placebo checks

**All three placebo checks passed, but each has little power.** The numbers are in the Checks list
above. In fold 2025-12 alone, the real cloud minus the placebo at the six is -0.118 points, and the
audit attributes the six's C1 estimate to that fold.

#### The missing-value flag raises the CRPS of Primary demand

**The flag alone makes the Primary demand forecast 0.12 to 0.15 points worse than stored `R_F`, so
C1 is measured on a weaker base neural network than `R_F`.** `DF` has one extra input column, the
`cloud_missing` flag, and no other change from `D0n`. The flag-only arm `DF` minus stored `D0`
(experiment 707's `R_F`) is +0.122 points at the six and +0.147 at the eight. The audit found the
difference positive in all 7 folds at the six and in 6 of 7 at the eight. `D1` minus `D0` is +0.094
at the six and +0.142 at the eight, so the masked-gap recipe as a whole is worse than `R_F`.

**The flag probably acts as a date proxy, but no arm tested the explanation.** In training, the flag
marks the 19-month gap, which carries 93% of the weight of the fine-tune window in fold 2025-12 and
45% in fold 2026-06. Every scored row has the flag set to 0, so the flag=0 mode that scoring uses
rests mostly on older rows. The harm is largest in the fold with the most gap weight, but the
correlation across 7 folds is only 0.41 at the six and 0.59 at the eight. The mean median forecast
of `DF` minus `D0` changes sign by fold at the six (-1.26 to +1.30 points of capacity), so the flag
does not add a constant shift. The arm `D0g`, which drops the gap rows instead, is worse than `D0`
by about +0.30 and +0.33, so the gap rows carry information. A placebo-flag arm would separate a
date proxy from an optimisation change. Experiment 568 found a similar effect.

**The refit code is not the cause.** The fit gate holds `D0n` within 2 standard errors of the seed
spread of stored `D0`, and the audit found `D0n` equal to `D0` within -2e-6 and +3e-7 points per
fold and seed on the two gate folds.

**C1 stays valid, and a null on this base is weaker evidence than a null on `R_F`.** `D1` and `D1p`
hold the identical flag and the identical missing pattern, and each seed starts from the same
initialisation in both arms, so C1 measures the information in the cloud values. The flag's level
shift could absorb part of what cloud would otherwise add. A gain found on this base would need a
re-test without the flag, and the `D1g` minus `D0g` pair of Stage B1 does that.

#### Solar farms: the gain over stored PLP is not a gain from cloud

**`P1` minus `P0` is -0.163 points, and the gain does not belong to cloud, because `P1` is no better
than its placebo `P1p` (S1 above).** The refit schedule, the training window, and the number of
seeds differ between `P1` and `P0`, and `P0L` (Stage B, not run) would separate the window from the
refit. The audit output does not contain the over-forecast figures, so the page does not quote them.

#### Review findings

**The review findings of the Stage A scorer output are in the Review points section above.** Two
extra facts: the scorer's solar example of section 7 carries no farm label and no cloud value, and
the leakage and anonymisation checks passed.

#### What Stage A does not show

**Stage A does not show that cloud is useless, only that its gain, if any, is smaller than the
intervals above.** It has no positive control, so a null cannot yet be told from a masked-gap recipe
that cannot pass a known signal. The base neural network carries a flag penalty, and the
cloud-scored
folds cover December to June only. A reading of `not supported` needs an interval that excludes 0.02
points, which Stage A cannot give.

### Summary of arms and contrasts

**The cloud variable is `total_cloud_cover_atmosphere`, from 0% to 100%.** The experiment adds it to
two neural networks. The first is the pooled mixture density network of experiment 707 for Primary
demand, in the arm that experiment calls `R_F`. The second is the solar network PLP of experiments
659 and 666 for the six metered solar farms. The deciding contrast subtracts a placebo arm that
holds another day's cloud with the same missing-value pattern, so a gain cannot come from the
missing-value flag acting as a date.

**The 19-month gap in the cloud history gets three answers, and each answer has its own arm.** The
cloud history has a gap from 2024-04-01 to 2025-11-20. The headline arms keep every training row
and mask the gap with a flag. A same-row pair drops the gap rows from both the cloud arm and its
reference. A gap-free replicate for Primary demand scores 2023-04 to 2024-03, when every training
and scoring run carries cloud.

**The experiment is `kind: other` and adopts nothing.** Neither base network is a logged research
default. The Primary demand research default 3 is the GAM stack, and Metered solar farms research
default 1 is a three-way blend. A supported result is a candidate for a later `change` experiment
on the port target (the Candidate `P7+D50` for Primary demand, and PLP for the solar farms).

**No new XGBoost model is fitted.** The stored XGBoost-based forecasts BG2 and S0M appear as
reference rows only, as the maintainer's decision of 2026-11-02 requires.

### 1. Hypothesis, prediction, and the smallest gain worth porting

**The hypothesis and the prediction are in the front matter and are written before any fit.** The
prediction rests on three earlier results. Experiment 22 found that replacing every ENS weather
input with ERA5 at serving gained only 0.10 points. Experiment 559's arm A11c found small positive
gains with intervals that contain zero. For the solar farms, UKV irradiance and cloud, from a
different weather model, gained 1.1% in experiment 248.

**The smallest gain worth porting is 0.02 points of CRPS for both targets.** Both evaluation
protocols set 0.02 points as the minimum gain for a new input. A threshold of zero would be too
low, because the variable adds one more column to ingest, validate, and serve.

**The cloud-scored folds cannot be expected to resolve 0.02 points at the six signed-MW Primaries.**
The minimum detectable effect, estimated in section 8, is larger than 0.02 there. A reading at the
six is therefore expected to come out `inconclusive` or `small gain`, which is why the gap-free
replicate contrast C3 is planned.

### 2. Stop rule and the maintainer's approval

**The rule says the fourth experiment on one idea after three `not supported` or `refuted` verdicts
needs the maintainer's approval, and is scored at every Primary with no screening arm.** The text is
in `evaluation-protocol-demand.md` ("Stop rule for blind alleys"). A `supported` or `small gain`
verdict on the idea resets the count. The dates below are the commit dates of each `result.md`.

**Count on ENS total cloud cover itself: 0.** The direct cloud tests are experiment 559
(`partly supported`), experiment 57 (`inconclusive`), and experiments 229, 246, and 248
(`small gain`, UKV cloud). Experiment 568 is `not supported`, but its verdict is on the attention
mixer; if the manager counts experiment 568, the count is 1.

**Count on irradiance-like inputs to the Primary demand forecast: 3 after the reset.** The
experiments 187, 193, and 237 were `supported` on 2026-10-22 and reset the count to 0. Three
experiments then returned `not supported`.

| Order | Experiment | Theme | Verdict | Date | Running count |
|---|---|---|---|---|---|
| 1 | 9 | weather-products-and-blending | `refuted` | 2026-10-14 | 1 |
| 2 | 22 | production-reference-and-best-guess | `refuted` | 2026-10-15 | 2 |
| 3 | 50 | weather-products-and-blending | `refuted` | 2026-10-16 | 3 |
| 4 | 187, 193, 237 | differentiable-physics themes | `supported` | 2026-10-22 | reset to 0 |
| 5 | 419 | production-reference-and-best-guess | `not supported` | 2026-10-31 | 1 |
| 6 | 437 | demand-inputs | `not supported` | 2026-10-31 | 2 |
| 7 | 569 | demand-inputs | `not supported` | 2026-11-05 | 3 |

**Experiments 703 and 704 do not reset the count.** Both were `supported` on 2026-11-18, and each
supports removing two irradiance columns, not adding one. If the manager rules that the GAM stack's
solar summary (experiments 187, 193, and 237) is a different idea from an input to a neural network,
no reset applies, and the count is 3 or more on either reading.

**Count on irradiance-like inputs to the solar farm forecast: 2 strict (experiments 272 and 292),
5 broad.** The `small gain` verdicts of experiments 229, 246, and 248 reset the count on 2026-10-22.
Experiment 272 (`not supported`, extra ensemble variables) and experiment 292 (`not supported`,
neighbour-cell irradiance) followed on 2026-10-23. The broad reading adds experiments 275, 430, and
443, which change the training data or the design and add no input.

**Consequence.** The rule applies to Primary demand, and the plan applies it to the solar farms too.
Every Primary and every farm is scored on every fold from the first run, and no four-month screen
is run. The standing practice "screen on a few folds first" is therefore not followed.

**Stop conditions inside the experiment.** Stop and report to the manager if the gate fails, if the
era check fails and the manager does not approve an era indicator, if any origin assertion fails,
or if the first fit's memory exceeds the cap.

### 3. Base networks, rows, and folds

#### Primary demand

- **Network.** Experiment 707's arm `R_F`: the pooled mixture density network of experiment 425,
  widened by 1.15, with no ERA5 and CAMS pre-training, an `S_ens` stage on every ENS row from
  2021-07-20 to the fold's cut, and a fine-tune on the last 730 days.
- **Why `R_F`.** The arm scores 2.821 points at the six, against 2.820 for `R_F+flags` and 2.840 for
  the pre-trained `A_F`, and is 0.041 points behind `A_F` at the eight. Choosing `R_F` keeps ERA5
  cloud, a fraction from a different product, out of the experiment.
- **Rows.** Row set `fold_a_daily_gap_v1`: every daily 00 UTC run, origin 09:00 UTC, lead days 1 to
  10, all 16 Primaries. The six, the eight (with and without D8), and D15 and D18 (report-only) are
  scored separately and never pooled.
- **Folds.** The 12 monthly folds 2025-07 to 2026-06 of experiment 707. Fold M trains on rows known
  3.5 hours before the first origin of month M and scores the runs of month M. No valid time on or
  after 2026-07-01 is scored, because `experiment_store.write_predictions` refuses it.
- **Cloud-scored folds.** The seven folds 2025-12 to 2026-06, whose scored runs all carry cloud once
  the re-write of the Delta table lands. The planned contrasts C1 and C2 read these seven folds
  only.
- **Gap-scored folds.** The four folds 2025-07 to 2025-10, whose scored runs carry no cloud. The
  cloud arms see the flag set on every scored row there, so real minus placebo should be zero, and
  the folds are a built-in negative control. Fold 2025-11 mixes both kinds of run (cloud from
  2025-11-21), is reported on its own, and enters no contrast.
- **Gap-free replicate.** A second row set, `backfill_daily_2023_04_to_2024_03_v1`: the same
  recipe, 12 monthly folds 2023-04 to 2024-03, training from 2021-07-20. Every training and scoring
  run carries cloud, so the replicate needs no flag and no mask. D9's first reading in the cleaned
  power table is 2023-11-21, so the replicate scores the six without D9 and labels the group "five
  signed-MW Primaries". D15 and D18 have no replicate.

#### Solar farms

- **Network.** PLP of experiments 659 and 666: 18 inputs, two hidden layers of 64 units, a
  three-component Gaussian mixture centred on the mean physics-lite power, 20 epochs, no farm
  column.
- **Two departures from experiment 666, applied to every arm.** The refit is monthly, as the
  standing practice asks, and the training rows start at the run of 2020-10-29 (the first
  back-filled run is 2020-10-15, and its 14 earliest runs serve as placebo donors only) instead of
  2024-04-01. The stored quarterly PLP forecasts of experiment 666 stay as a reference
  row.
- **Rows.** A new row set, `solar_daily_1200_2025_12_to_2026_06_v1`: every daily 00 UTC run from
  2025-12-01, origin 12:00 UTC, daylight half-hours, leads 12 hours to under 348 hours from the run,
  valid times before 2026-07-01.
- **Headline band.** The headline covers days 1 to 10 (12 to under 252 hours). The solar protocol
  sets 12 to 36 hours as its headline band, so the plan departs from the protocol, because NGED
  care about days 1 to 10. The 12-to-36-hour reading (S3) is reported beside S1 in the headline
  caveat.
- **Folds.** Seven monthly folds 2025-12 to 2026-06. A refit at the first of month M trains on
  daylight half-hours whose run is before the refit and whose valid time is at least 1 day before
  it, and forecasts the origins of month M.
- **Farms with short histories.** The farms have readings from different dates. The training window
  of each farm starts at its first usable run, and the page publishes no per-label start date,
  first-data date, or commissioning date.

**Every arm of a contrast scores exactly the same rows.** Rows are dropped only by rules that read
the target or the calendar: a flagged `substation_zero` half-hour in the target or a lag, a D15 or
D18 row before 2025-03-01, a Primary's first usable date, and the run-date windows of section 4. No
rule reads a cloud value. The scorer asserts equal row keys across arms before any contrast.

### 4. Cloud summaries and the handling of the gap

**Three summaries of the 51 members are built per row, all from the run named in section 7.**

- **T1, mean and spread:** the ensemble mean and the population standard deviation of total cloud
  cover, divided by 100 (two columns).
- **T2, clear and overcast shares:** T1 plus the share of members at or below 20% cloud and the
  share at or above 80% (four columns). Cloud cover piles up near 0% and 100%, so a mean of 50% can
  describe 51 half-cloudy members or a split between clear and overcast members, and the two shares
  separate those cases.
- **T3, daylight mean (exploratory, Primary demand only):** T1 plus the mean of the ensemble-mean
  cloud over the daylight half-hours of the row's local valid day (three columns). T3 is in no
  planned contrast.

**Headline handling: train on every row, and mask the gap.** A row whose run falls in 2024-04-01 to
2025-11-20, or is the upstream gap of 2024-01-03, gets 0 in every standardised cloud column and 1 in
one `cloud_missing` column. Standardisation uses the fold's training rows with `cloud_missing = 0`.
The rule reads the run date, never the cloud value.

**The flag-only arm measures what the flag alone does.** The arm holds `cloud_missing` and no cloud
column. Flag-only minus the no-cloud reference is the date-proxy effect that experiment 568 warned
about, and is reported with its interval.

**Same-row sensitivity: drop the gap from both arms.** A cloud arm and its reference train only on
rows whose run is outside 2024-04-01 to 2025-11-20, with no flag. Both arms lose 19 months of recent
power history, so only the pair's difference is read, never its absolute scores.

**An era check runs before any fit.** The back-filled runs and the recent runs come from two
ingests, and ECMWF changed the ENS resolution on 2023-06-27. For each of the 16 Primaries and 6 farm
cells, compute the monthly mean of ENS cloud minus ERA5 `total_cloud_cover` (times 100) at leads 12
to 36 hours, and the monthly ENS spread. The ERA5 source is the column `total_cloud_cover` of
`ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31/work/era5_wind_components.parquet`,
a fraction for 22 series from 2019-09-01 to 2026-10-31. Compare the same calendar months across
eras (December to March of 2021-22 to 2023-24 against December 2025 to March 2026), because cloud
bias is seasonal. A step at 2023-06-27, or between those two groups of months, larger than 3
standard deviations of the adjacent 12 monthly values fails the check. On a failure, every arm gains
an era indicator if the manager approves, and the page says so.

### 5. Arms

**Primary demand.** "Stored" means no fit. The seed count is per fold.

| Arm | Inputs beyond `R_F` | Folds | Seeds |
|---|---|---|---|
| `D0` (stored `R_F`) | none | 12 | 0 to 4, stored |
| `D0n` (gate) | none, refit by the new code | 2026-01, 2026-06 | 3 |
| `DF` | `cloud_missing` only | 12 | 6 |
| `D1` | T1 real, masked | 12 | 6 |
| `D1p` | T1 shuffled-run placebo, same mask | 12 | 6 |
| `D1t` | T1 time-shifted placebo, same mask | 12 | 3 |
| `D2` | T2 real, masked | 12 | 6 |
| `D2p` | T2 shuffled-run placebo, same mask | 12 | 6 |
| `D3` (exploratory) | T3 real, masked | 7 cloud-scored | 3 |
| `D0g`, `D1g` | none; T1 real. Gap rows dropped | 7 cloud-scored | 3 each |
| `K1`, `K1p` (positive control) | section 6 | 7 cloud-scored | 3 each |
| `R0`, `R1`, `R1p` (replicate) | none; T1 real; T1 placebo. No flag | 12 replicate | 3, 6, 6 |

**Solar farms.**

| Arm | Inputs beyond PLP's 18 | Folds | Seeds |
|---|---|---|---|
| `P0` (stored PLP, experiment 666) | none; quarterly refit, window from 2024-04-01 | overlap only | 0 to 2, stored |
| `P0L` | none; monthly refit, rows from the run of 2020-10-29 | 7 | 3 |
| `PF` | `cloud_missing` only | 7 | 5 |
| `P1`, `P1p` | T1 real; T1 shuffled-run placebo | 7 | 5 each |
| `P1t` | T1 time-shifted placebo | 7 | 3 |
| `P2`, `P2p` | T2 real; T2 shuffled-run placebo | 7 | 5 each |
| `P0g`, `P1g` | none; T1 real. Gap rows dropped | 7 | 3 each |

**Seed sets are pre-registered for every contrast, because `losses_for_bootstrap` raises when two
arms hold different seeds.** Each contrast is read on the seeds that the two arms share. `D1t` minus
`DF` is read on seeds 0 to 2. `P1t` minus `PF` is read on seeds 0 to 2. `D1` minus `D0` is read on
seeds 0 to 4. `P1` minus `P0L` is read on seeds 0 to 2. In the minimum-detectable-effect null, the
second half of the seeds is relabelled before pairing.

**Reference rows come from stored forecasts or seedless recipes only:** `climatology` (the
month-by-half-hour `ClimatologyForecaster`, trained on each fold's window and repeated once per
seed), `manual_heuristic`, BG2 and S0M for Primary demand, and stored `R_F` and `A_F` of experiment
707. No reference row comes from experiment 404's lag-free head or from any forecast that the memory
file `weather-look-ahead-404.md` lists as voided.

**The scorer prints every arm's input column list**, so a reviewer can check each arm against these
tables. A real arm and its placebo have the same columns in the same order.

### 6. Placebos and controls

**Both placebos copy the missing pattern exactly.** A placebo row is missing if and only if the real
row is missing, and the code asserts that the two `cloud_missing` columns are equal on every row
before any fit. With a different pattern, the placebo and the real arm would differ in what the
flag says about the date.

- **Shuffled-run placebo (in the planned contrasts).** For each series, the cloud columns of run
  `r`, at every lead, are replaced by the cloud columns of a donor run at the same leads. The donor
  is drawn among the runs that carry cloud, lie in another calendar year, fall within 15 days of
  `r`'s day of the year, and were published before the fold's cut. The donor map is seeded by
  series, fold, and fitting seed, and is stored under `work/placebo/` so an audit can read it.
  Swapping a whole run keeps the daily cycle, the growth of spread with lead, and the season. The
  rule is experiment 707's donor rule applied to forecast runs. A recent run (2025-11-21 on) takes
  its donor from the back-filled years, and the page states that the donor then comes from the
  earlier ingest.
- **Time-shifted placebo (second control).** The cloud columns of run `r` are replaced by those of
  the run issued 14 days earlier, at the same leads. Where that run does not exist (the first 14
  runs of a segment, or the upstream gap of 2024-01-03), the run issued 21 days earlier is used, and
  then the run 7 days earlier. Every donor is issued before the row's own run.

**The checks use a common correction.** Eight checks are read at 1 - 0.05/8 = 99.375%: `D1p` minus
`DF`, `D1t` minus `DF`, and the gap-scored contrast, each at the six and at the eight, and `P1p`
minus `PF` and `P1t` minus `PF`. The six and the eight are never pooled in a check, because pooling
signed-MW and MVA-only Primaries would let a placebo effect at one group hide behind the other. A
failing check triggers an audit. It changes no verdict, with one exception that Stage A needs: if
`D1p` minus `DF` fails at a group (its interval does not contain zero, in either direction), the
group's C1 verdict reads `inconclusive` until the audit reports, and if the check is incomplete the
verdict is withheld. The same holds for S1 and `P1p` minus `PF`. The reason is that a shuffled-run
donor that makes the placebo worse than no cloud at all would make C1 look like a gain. The scorer
prints `D1` minus `DF` (and `P1` minus `PF`) beside C1 (and S1) as the bound, so that a reader can
see how much of C1 the placebo's own shift explains.

- **Placebo carries nothing:** `D1p` minus `DF`, `D1t` minus `DF`, `P1p` minus `PF`, and `P1t` minus
  `PF` each have an interval that contains zero. A failure fires audit trigger 5.
- **Gap-scored folds:** `D1` minus `D1p` on folds 2025-07 to 2025-10 has an interval that contains
  zero. The interval is a t interval over the four folds and the seeds, because four months give
  only 35 distinct bootstrap resamples. A failure means that the training-time difference between
  real and placebo cloud moves forecasts that receive no cloud, and fires audit trigger 5.
- **Positive control (Primary demand):** temperature goes through the cloud's exact handling. `K1`
  removes the mean and spread of 2 m temperature and of 2 m dew point from the base inputs and adds
  them back as four masked columns with the cloud's missing pattern and the flag. `K1p` is its
  shuffled-run placebo. The control passes if the 99.375% interval of `K1` minus `K1p` is wholly
  below zero at both groups.
  A failure means the masked-gap handling cannot pass a known signal across the gap, and the Primary
  demand reading of C1 and C2 is then `inconclusive` whatever the intervals say. The control is a
  Stage B arm, so the Stage A verdict does not use it.
- **Solar farms have no positive control of this kind.** Every strong PLP input feeds the
  physics-lite power at the centre of the mixture, so masking one changes the network. A solar null
  is read only through an interval that bounds the effect, and the same-row pair checks the gap
  handling.
- **Gate.** `D0n` matches stored `R_F` on folds 2026-01 and 2026-06 within 2 standard errors of the
  seed-to-seed spread, and a prediction-only check reloads stored `R_F` states and reproduces the
  stored forecasts to 1e-4. `P0L` is compared with stored `P0` descriptively, because the refit
  schedule and the window both differ.

### 7. Weather at the forecast origin

**Every row takes all of its weather, cloud included, from one run: the 00 UTC run issued at or
before the forecast origin and published by then.** A run counts as published 9 hours after its
initialisation. For Primary demand the origin is the run time plus 9 hours (09:00 UTC). For the
solar farms the origin is 12:00 UTC on the run's day. No weather value, cloud included, comes from a
run issued after the origin.

**The code asserts the rule on every row of every arm, training and scoring, real and placebo.** The
Primary demand code calls experiment 707's `assert_origin_after_run` (`nwp_init_time + 9 h <=
origin`) on the cloud summaries as well as on the seven existing variables. The solar input builder
has no row-level assertion, so the solar code adds one: `init_time + 9 h <= origin`, and `init_time`
equal to 00 UTC of the origin's date. A failed assertion makes the experiment `invalid`.

**The leakage condition for a shuffled-run donor is its publication before the fold's cut.** Every
donor run satisfies `donor_init + 9 h <= cut`. The donor need not precede the row's own run, because
a donor issued before the cut has no bearing on the row's origin. The time-shifted donor precedes
the row's run by construction.

**The code prints one example row per target.** For Primary demand: run 2026-01-14 00:00 UTC, origin
2026-01-14 09:00 UTC, target 2026-01-19 17:30 UTC (lead 128.5 hours), with cloud mean and spread
from the run of 2026-01-14 and never from the run of 2026-01-19. For the solar farms: run
2026-04-10 00:00 UTC, origin 12:00 UTC, target 2026-04-13 11:30 UTC, with cloud from the run of
2026-04-10.

**No score above 24 hours comes from experiment 404's lag-free head.** Neither network is that
head: `R_F` reads three origin lags of power at least 3.5 hours old, and PLP reads no power.
Experiment 404's head took the run nearest each valid time, which is weather issued after the
origin. No arm, reference row, or comparison here reads a forecast or a score from that lineage.

**Cloud feeds no lag.** T3's daylight mean covers the valid day's forecast cloud from the row's own
run, never a later run's.

### 8. Planned contrasts, scoring, and the decision rule

**Nine planned readings, from six contrasts, are each read at 1 - 0.05/9 = 99.44%.** Negative
favours
real cloud. Every other number is exploratory and labelled so.

| Contrast | Definition | Rows | Readings |
|---|---|---|---|
| C1 | `D1` minus `D1p` | cloud-scored folds, days 1 to 10 | the six; the eight |
| C2 | `D2` minus `D2p` | cloud-scored folds, days 1 to 10 | the six; the eight |
| C3 | `R1` minus `R1p` | replicate folds, days 1 to 10 | the five; the eight |
| S1 | `P1` minus `P1p` | daylight, days 1 to 10, six farms pooled | one |
| S2 | `P2` minus `P2p` | daylight, days 1 to 10, six farms pooled | one |
| S3 | `P1` minus `P1p` | daylight, 12 to 36 hours (the solar protocol's band) | one |

**Stage A can find a gain or a loss; under a null it will most likely read `inconclusive`.** The
estimated minimum detectable effect is about 0.05 points at the six, 0.03 at the eight, and 0.02 at
the farms. A verdict of `not supported` needs a lower limit at or above -0.02, which at a half-width
of 0.05 needs a point estimate of +0.03 or more at the six and about +0.01 at the eight. Stage A can
therefore return `supported`, `small gain`, or `refuted`, and a true null at the six is expected to
read `inconclusive`. Stage B does not narrow C1, because adding seeds does not narrow an interval
that draws one fitting seed per resample (section 12, point 1). Stage A prints an interim minimum
detectable effect before any contrast, from a split of seed 0 against seed 1 of `D1p` (at the six
and at the eight) and of `P1p`. That split has the noise structure of the 3-against-3 split, so a
Stage A `inconclusive` can be told from an underpowered contrast by comparing the interval's
half-width with the interim figure.

**Stage A reads C1 at the six and at the eight, and S1; Stage B would read the other contrasts.**
Each of the three readings is read at 99.44% so that Stage B keeps the family of nine readings, with
seeds 0 to 2. Its
verdict for each Primary demand group is the C1 verdict, its verdict for the solar farms is the S1
verdict, and the front-matter rule combines the three. The Stage A verdict does not use C2, C3, S2,
S3, the positive control, the 3-against-3 minimum detectable effect (Stage A prints only the
interim one), or any other Stage B contrast. C2, C3, S2,
S3, `D1t`, `P1t`, the gap-scored contrast, the positive control, the same-row pairs, the replicate,
and every exploratory slice that needs a Stage B arm are Stage B. A Stage B reading can change the
verdict only through the rules above (the modifiers and audit trigger 2), after the manager has
approved Stage B.

**Stage B1 reads two contrasts, each at 99.44%, and gives no front-matter verdict.** The scorer
(`e00711_score.py score --stage B1`) reads only the arms of Stage B1, with seeds 0 to 2.

| Contrast | Definition | Rows | Readings |
|---|---|---|---|
| C3 | `R1` minus `R1p`, with no gap rows and no flag | the 12 replicate folds, days 1 to 10 | the five (the six without D9, whose power history starts after the replicate begins); the eight |
| G1 | `D1g` minus `D0g` (real cloud minus no cloud, not a placebo contrast), with no flag and no gap rows | the 7 cloud-scored folds, days 1 to 10 | the six; the eight |

The scorer also prints `R1p` minus `R0` and `R1` minus `R0` at 99.375% as descriptive bounds beside
C3, because `R0` has no cloud column and no flag. The verdict of each reading is the decision rule
above applied to the decisive interval. A Stage B1 reading changes the front-matter verdict through
the `partly supported` and audit trigger 2 rules only if the manager later runs Stage B, because the
front-matter rule needs C2 and the positive control. Until then, Stage B1 answers the date-proxy
question of the Stage A results and nothing else.

**What each Stage B1 outcome would mean for the flag and date-proxy question.** C3 and G1 contain
no missing-value flag, so neither can contain the date-proxy penalty of `DF`.

- **C3 `supported` or `small gain`, or G1 the same.** Cloud carries information that the flag-based
  Stage A recipe could not show, and the flag's date proxy hid it. The Stage A `inconclusive` would
  then read as a limit of the base network, and a flag-free re-test on the recent folds would be
  worth proposing to the manager.
- **C3 and G1 both `inconclusive`.** The flag was not hiding a gain that these intervals can
  resolve, and the question stays open at the width of the intervals.
- **C3 or G1 `not supported`.** A gain of 0.02 points is excluded on a base without the flag, and
  that is the only Stage B1 outcome that supports a statement of "no gain" at that level.
- **C3 or G1 `refuted`.** The real cloud is worse than the placebo with no flag present, so the
  harm does not come from the flag.
- **`R1p` minus `R0` far from 0 while C3 is near 0.** The cloud columns, real or shuffled, change
  the forecast without adding information, which is a column effect and not a date-proxy effect.

**G1 is the total effect of the two cloud columns on a flag-free base, and Stage B1 cannot split
it into information and a column effect,** because Stage B1 holds no `D1g` placebo. A G1 gain is a
gain that a service could deliver.

**Level of G1.** G1 was added to the planned readings by the amendment of 2026-11-20, after the
Stage A results showed the harm that the flag does. With C2, S2, and S3 unread, the family holds
seven
readings (C1 at two groups, S1, C3 at two groups, and G1 at two groups), and 99.44% holds. If Stage
B runs, the family becomes 11 readings and every reading is read at 1 - 0.05/11 = 99.55%.

**Stage B1 also prints exploratory lines, at 95% and with no verdict word.** They are C3 and G1 per
fold, C3 on the folds 2023-04 to 2023-06 against the folds 2023-07 to 2024-03 (fold 2023-06 counts
as before the change), and C3 and G1 with each Primary left out.

**Metric.** CRPS as twice the mean pinball loss over the 11 quantile levels, in points of effective
capacity, with the normalised MAE of the median beside every CRPS number. Each farm's error is
divided by its own capacity before any mean.

**Interval.** Per-row losses of each seed go through `experiment_store.losses_for_bootstrap` and
`studies.bootstrap.bootstrap_difference_at_level`, which resamples whole calendar months paired
across arms and draws a fitting seed, with 50,000 resamples (about 140 draws per tail at 99.44%).
Seed noise is therefore inside the interval. The scorer totals the per-row differences by seed and
month and resamples those totals with the same random stream, which gives the same interval at a
cost of one number per seed and month (a test checks the equality). The decisive interval is the
wider of this interval and a t interval on the monthly differences, where wider means the larger
upper minus lower, not the union of the two. A month bootstrap on 7 months covers less often than
its nominal level, which is why the t interval is read beside it. The contrast on seed-averaged
quantiles (experiment 707's convention) is printed beside both as a descriptive reading.

**Minimum detectable effect.** The scorer prints it before any contrast is read. The minimum
detectable effect is the half-width, at 99.44%, of a null contrast between two halves of the
placebo arm's seeds (seeds 0 to 2 against 3 to 5 for Primary demand; 0 and 1 against 2 and 3 for
solar). Because the bootstrap draws one fitting seed per resample, a half-split has the same noise
structure as the real contrast. Experiment 707's intervals left seed noise out, and its caveat says
the seed noise was about as large as its C1 at the six. The minimum detectable effect is therefore
estimated (interim, not measured) at about 0.05 points at the six and 0.03 at the eight on the
cloud-scored folds, and about 0.03 and 0.02 on the replicate. For the solar farms the only guide is
experiment 292's half-width of 0.33% at 12 to 36 hours, about 0.02 points. **C1 and C2 at the six
are not expected to resolve a gain of 0.02 points.**

**Verdict per target.** The target's verdict is the better of its C1 and C2 readings (S1 and S2 for
the solar farms), per group, ranked `supported`, `small gain`, `inconclusive`, `not supported`,
`refuted`, with two modifiers for Primary demand. If C1 and C2 are `inconclusive` or `not
supported` at a group and C3 is `supported` there, the group reads `partly supported`: cloud
carries information that did not cross the gap. If C1 or C2 is `supported` and C3's lower limit is
above 0, audit trigger 2 fires and the group reads `inconclusive` until the audit reports.

**Front-matter verdict.** `supported` if the six, the eight, and the solar farms are all
`supported`. `partly supported` if any is `supported`, `partly supported`, or `small gain`.
`refuted` if any is `refuted` and none is `supported`. `not supported` if all three are
`not supported`. Otherwise `inconclusive`. The first paragraph of the finished page states each
target's verdict separately, because the question has two parts.

**Slices, all exploratory, each with its half-hour count and a 95% interval:**

- **By lead time.** Primary demand: day 1, days 2 to 3, days 4 to 6, days 7 to 10. Solar farms: the
  same bands plus days 11 to 14. ENS stamps are 6-hourly after 144 hours, so cloud past day 6 is a
  linear interpolation between 6-hourly values, and the page says so.
- **Sunny and overcast half-hours.** The slices are defined from what happened, never from an arm's
  input: the realised clear-sky index (CAMS global horizontal irradiance over the Haurwitz clear-sky
  irradiance) at the valid time, on half-hours with the sun at least 10 degrees above the horizon.
  Sunny is an index of 0.8 or more, overcast 0.4 or less
  (`studies.era5_ladder.CLEAR_SKY_INDEX_THRESHOLDS`),
  and the half-hours between are a third row. Night half-hours are a fourth row for Primary demand.
- **Overall**, per Primary, per farm label (A to F), per fold, and against the number of months of
  recent cloud in the training window (1 to 7), which shows whether a gain grows as cloud history
  returns.
- **Protocol reports:** the eight with and without D8; q01, q05, q95, and q99 coverage and pinball
  loss; cold winter nights; sunny April to June; the avoidable-curtailment cost; a day-ahead and a
  14-day example forecast per target.
- **Real minus reference** (`D1` minus `D0`, `P1` minus `P0L`) and **flag-only minus reference** are
  reported with intervals as the total effect and the date-proxy effect, at the six and at the
  eight separately.

**Checks list on the finished page:** the median minus the mean forecast per site; the MAE contrast
beside each CRPS contrast; which audit triggers fired. A leave-one-Primary-out and a
leave-one-farm-out reading of each planned contrast is printed, because experiment 292's solar gain
vanished when one farm was left out.

**The four load-shift scenarios do not apply**, because the experiment does not concern load-shift
events.

### 9. Rebuilding the ENS summary caches

**The rebuild writes only inside this experiment's data folder and needs no change outside the
experiment's code folder.** `pyproject.toml` pins production revision `f9dcd22`, whose `Nwp` schema
holds the column, and the production half-hourly resampler treats cloud as a continuous variable.
Nothing under `src/`, `packages/`, or a contract changes. The main research checkout pins an older
revision with no cloud field, so the session runs `uv sync --group gpu` in this worktree, and every
build script asserts `"total_cloud_cover_atmosphere" in Nwp.model_fields` at its start.

**Primary demand.** A new script writes `work/ens_cloud/init=YYYY-MM-DD.parquet`, one file per run,
keyed like the existing summaries (`time_series_id`, `power_fcst_init_time`, `valid_time`). The
script reads `Nwp.scan_delta` for one run and one column, resamples the 51 members with
`ml_core.features._nwp._upsample_nwp_to_half_hourly`, and writes the T1 and T2 columns. The existing
seven-variable summaries of experiments 397, 607, and 707 are read as they are and joined on the
three keys. A second script builds the raw matrices with the cloud columns appended. The runs needed
are 2021-07-20 to 2024-03-31 (985 runs, because 2024-01-03 is absent) and 2025-11-21 to 2026-06-30
(222 runs). No file is written for a gap run.

**Solar farms.** A copy of `e00666_solar_inputs.py` in this experiment's folder reads the six farm
cells from 2020-10-15, adds the 51-member cloud to `read_members`, and writes its tables under this
experiment's `solar_inputs/`. The copy reads through `Nwp.scan_delta`, or first checks on one
partition from 2023 and one from 2025 that `pl.scan_delta` returns the cloud column (values for the
2023 run, nulls for the 2025 run) rather than raising. Experiment 666's tables are not overwritten.

**Experiment 561's 222 cloud files are a cross-check, not an input.** They were built from the
Dynamical.org store with a different area weighting. The build prints the correlation and the mean
absolute difference between the two builds for the overlapping runs.

**Three conditions come before any fit or any build of a recent run, in order.**

1. The manager's message that the re-write worker has reported its validation. No fit and no build
   of a recent run starts before that message. The back-filled runs (to 2024-03-31) are not being
   re-written, and their cloud summaries can be built first if the manager agrees that reading the
   table during the re-write is safe.
2. The experiment's own check of the table at one pinned Delta version, recorded in the lineage note
   and passed to every later read: every run from 2025-11-21 to 2026-06-30 holds 51 members,
   7,243,785 rows, and no null cloud; cloud lies in 0 to 100; recomputed means and spreads of the
   seven older variables match the stored summary files for 20 sampled runs to float tolerance; and
   the era check of section 4 passes.
3. The `data-validation` checklist on the cloud summaries, and a fresh code review of the build
   script before its first run.

**The build is CPU work of unmeasured length.** Time one run first, then run 4 single-threaded
workers, each writing one run's file atomically and skipping a file that exists.

### 10. Compute

**The experiment runs on 2 concurrent fit streams.** The manager's instruction is that the caps of
concurrent `systemd-run` scopes must sum below the free memory, which is about 43 GB available now.
Each stream's cap is set from a peak that one measuring fit has measured, and the two caps together
stay below the free memory at launch. The GPU share is for the manager to give when the first fit is
requested.

**No fit starts until the manager confirms that the ENS re-run is validated.** Before the first fit
(CPU only), the experiment runs the cloud builds (4 workers, 1 thread each), the era check, the
placebo donor maps, the raw matrices, and a code review of every script.

**Launch rules.** Before each launch, `free -g` shows memory available for the sum of the caps and
`uptime` shows a load below 24. Every Python command runs with `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`, under `systemd-run --user --scope -p
MemorySwapMax=0 -p OOMPolicy=continue` with a `MemoryMax` set as above (12G for the measuring fit),
and with `uv run --no-sync` after `uv sync --group gpu`. Each stream runs one (arm, fold, seed) job
at a time, with a fixed seed, its own output file, and an atomic write, so a crashed job reruns
without touching the others. The fourth core is for loading and scoring.

**Job count (estimate).**

| Target | Fits | Basis |
|---|---|---|
| Primary demand, 12 recent folds | 423 | `DF`, `D1`, `D1p`, `D2`, and `D2p` at 72 each; `D1t` 36; `D0n` 6; `D3` 21; no fit for `D0` |
| Primary demand, 7-fold pairs | 84 | `D0g`, `D1g`, `K1`, and `K1p` at 21 each |
| Primary demand, replicate | 180 | `R0` 36; `R1` and `R1p` at 72 each |
| Solar farms | 259 | `PF`, `P1`, `P1p`, `P2`, and `P2p` at 35 each; `P0L`, `P1t`, `P0g`, and `P1g` at 21 each |
| Total | 946 | |

**Time (estimate).** A Primary demand fit takes 2 min 4 s (124 s, measured), so 687 fits take about
23.7 stream-hours. The solar fit time rests on
experiment 666's log, which prints the height of the forecast table and not the training rows, so
the training-row count is unknown. The window from 2020-10-29 is about 3 times longer than the
window of experiment 666, so a solar fit is estimated at 125 to 250 s, and 259 fits at 9 to 18
stream-hours. The whole plan is about 33 to 42 stream-hours, so on 2 streams it takes about 16 to 21
hours of wall time. The run does not start as a whole: Stage A comes first. The first fit
measures wall time, peak resident memory (the scope's `MemoryPeak`), and peak GPU memory
(`torch.cuda.max_memory_allocated`), and the page reports the corrected figures.

#### Stage A and Stage B

**Stage A is 132 fits and about 2.3 to 3.4 hours on 2 streams; the whole plan is 946 fits.** The
manager asked for the smallest subset that decides the question within about 5 hours on 2 streams.
Stage A fits seeds 0 to 2 of the arms below, on the folds shown, and nothing else. The scorer
(`e00711_score.py score --stage A`) reads only these arms, and `e00711_jobs.py --stage A` lists the
jobs and prints the count and the estimate.

| Arms | Folds | Seeds | Fits | Reason |
|---|---|---|---|---|
| `D0n` | 2026-01, 2026-06 | 0 to 2 | 6 | the gate |
| `DF`, `D1`, `D1p` | the 7 cloud-scored folds | 0 to 2 | 63 | no-cloud reference, real cloud, placebo (C1; `D1p` minus `DF`) |
| `PF`, `P1`, `P1p` | the 7 solar folds | 0 to 2 | 63 | no-cloud reference, real cloud, placebo (S1; `P1p` minus `PF`) |
| Total | | | 132 | 69 Primary demand fits at 124 s, 63 solar fits at 125 to 250 s |

**The solar inputs are built before the first solar fit.** `e00711_solar_inputs.py` writes one table
per farm. A dry run of 30 runs and of 120 runs for one farm, into a scratch folder, took 7 s and 24
s
with a peak resident memory of 0.56 and 0.60 GB. That scales to about 6.5 minutes and 1.5 GB per
farm for the 2,086 runs, so the six farms take about 40 minutes in sequence. The dry runs found no
null cloud at any stamped half-hour of the runs that serve as donors, and every row passed the
origin
assertion.

**Stage A needs the gate first.** The fit refuses every arm but `D0n` until `work/gate_passed.json`
holds the fit gate and the prediction gate at both gate months. The one override is `--measure` for
the measuring fit, which writes under the name `measure-<arm>`.

**Stage B1 is the reduced Stage B that the manager approved on 2026-11-20: 150 fits on seeds 0 to
2.** Stage B1 fits `R0`, `R1`, and `R1p` on the 12 replicate folds (36 fits each), and `D0g` and
`D1g` on the 7 cloud-scored folds (21 fits each). `e00711_jobs.py --stage B1` lists the jobs, checks
the count of 150 (`STAGE_B1_COUNT`), and takes `--pending` to skip finished fits. The scorer reads
Stage B1 with `e00711_score.py score --stage B1` (section 8).

**Stage B is the other 664 of the 946 fits, and it is not run.** It holds seeds 3 to 5 of every arm
that the plan gives 6 seeds (including `R1` and `R1p`), the folds 2025-07 to 2025-11 of the Stage A
arms, and the arms `D1t`, `D2`, `D2p`, `D3`, `K1`, `K1p`, `P0L`, `P0g`, `P1g`, `P1t`, `P2`, and
`P2p`. `e00711_jobs.py --stage B` lists them and checks the count of 664. They would answer the
questions that Stage A and Stage B1 leave open: the 3-against-3 minimum detectable effect, the
summary T2, the gap-handling checks, the positive control, and the solar S3 band. Running any of
them needs a new word from the manager.

**Amendment of 2026-11-20: the 6-seed rows of the plan for `R1` and `R1p` are not run.** The
manager approved the reduced Stage B on 2026-11-20. Stage A excludes only gains above 0.11 points at
the six, so the replicate answers the date-proxy question and not a power question, and seeds 3 to
5 would not change that answer. Section 12, point 1 records the same decision.

**First-fit request to the manager.** One stream runs one fit of `D1` on fold 2026-06, seed 0. The
request asks for a GPU share for 2 streams (experiment 707 planned 5 GB per stream and never
verified it; the cloud columns add under 5% to the input matrix) and for confirmation that the
memory cap is per command and not for the slot. The experiment reports one line when the fits start
and one when the last fit ends and the slot is free.

### 11. Data hygiene, anonymisation, and standing practices

**Farms are labelled A to F only, and no farm name, identifier, coordinate, or megawatt series
appears in a chart, a report, a commit message, a brief, or a pull-request body.** The labels come
from `studies.anonymise.site_labels_for`. The page publishes no start date, first-data date, or
commissioning date for any label, because such a date would let a reader link a label to a public
record. The solar example forecast uses a day-count axis with no calendar dates, states only the
month and year in the text, and sets `aria=False` on the mark.

**Standing practices followed:** monthly refits; cleaned data and the D15 and D18 cut; every fitted
state saved under this experiment's data folder; stored predictions reused as references; tails and
curtailment cost reported; example forecasts plotted; a draft pull request. **Departed from:** the
four-month screen (section 2).

**Earlier advice followed:** a placebo with the same missing pattern as the real arm (experiments
559 and 568), a cloud-missing flag tested on its own (experiment 568), and a paired contrast on the
same rows with the seed noise inside the interval (experiment 707).

### 12. Review points to settle at implementation

**The design review raised six points. Each is either settled with a recorded decision or marked
"to be settled before Stage B".** Stage A depended on points 1 and 2 only. Point 3 is
settled for Stage B1.

- **Fewer seeds.** The bootstrap draws one fitting seed per resample, so the interval does not
  narrow as seeds are added. Cutting the planned arms to 3 seeds would save about 180 Primary demand
  fits and 70 solar fits, with no loss of interval width. The alternative is to keep 6 seeds and
  read a seed-averaged contrast with a bootstrap that resamples seeds inside each month draw, which
  needs a new helper whose move into `src/studies/` would need its own reviewed pull request.
  **Settled for Stage A (the manager's Stage A instruction, 2026-11-20):** Stage A uses seeds 0 to
  2. **Settled for Stage B1 (the manager's approval, 2026-11-20):** Stage B1 keeps seeds 0 to 2, and
  seeds 3 to 5 of `R1` and `R1p` are not run. They are part of the 664 unrun fits, with seeds 3 to 5
  of the other arms, where the question of the 3-against-3 minimum detectable effect stays open.
- **The era of the placebo cloud.** For a scored run in 2025-12 to 2026-06, the real cloud comes
  from ECMWF's open data and every shuffled-run donor comes from the back-filled runs before
  2024-04-01, many of them before the 2023-06-27 resolution change. C1 and C2 are then an era
  contrast as well as an information contrast. The options are to draw donors from the recent
  segment, at least 14 days from the row's run, and to use the time-shifted rule where fewer than
  about 20 eligible recent runs exist (folds 2025-12 and 2026-01), or to make the time-shifted
  arms `D1t` and `P1t` the planned comparator with 6 and 5 seeds. **Decided by the worker, reported
  to the manager:** keep the shuffled-run donors as they are. The era check passed (64 tests, 0
  failed, Delta version 2589; `work/checks/era_check.json`), which is evidence that the back-filled
  and recent cloud summaries do not differ by more than the check's thresholds. The `D1p` minus
  `DF` check (section 6) is the safeguard: if an old-era donor makes the placebo worse than no
  cloud, the check fails and C1 reads `inconclusive` until the audit reports. The time-shifted
  arms stay in Stage B as the second comparator. The era check's 64 tests run on 16 series that
  take 8 distinct values, because several Primaries share a weather cell, so the tests are not 64
  independent passes.
- **The replicate and the resolution change.** Fold 2023-06 straddles 2023-06-27. The `study` skill
  asks to drop the straddling part-month on both sides, add an era indicator to all three replicate
  arms, and restrict donors to the row's era. **Decided by the worker, reported to the manager
  2026-11-20:** accept the replicate as it is, with the caveat below and with the split of the
  replicate folds before and after 2023-06-27 pre-registered as exploratory (section 8). The
  evidence, from the Stage B1 code and design review:
    - The era check passed on all 16 series. The step in the monthly ENS-minus-ERA5 cloud bias is a
      fall of 2.8 to 3.5 points on a 0 to 100 scale, and the deseasonalised step size is at most
      0.93
      against a fail limit of 3.
    - The fall lies inside the year-to-year range of the same bias before the change, which was
      -5.1 to +8.3 points. The step is nevertheless consistent in sign, so a real shift of about 2
      to 3 points is plausible.
    - Every `R1p` donor for a post-change run is an old-resolution run, because the donor rule needs
      another calendar year and no same-era donor exists before the last replicate cut. `R1p`
      therefore sees old-resolution cloud throughout, and `R1` meets new-resolution cloud in the
      folds 2023-07 to 2024-03 after training mostly on old-resolution cloud. The change can only
      push C3 against real cloud, never towards `supported`.
    - The remedies fail. A same-era donor is not available, an era indicator is a step function of
      the date and would reintroduce the date-proxy confound that Stage A measured, and dropping the
      4-run part-month changes the reading by a fraction of its interval.

  **Caveat for the finished page.** Folds 2023-07 to 2024-03 score new-resolution ENS cloud (from
  2023-06-27). A C3 `supported` or `small gain` stands. A C3 `refuted` or `inconclusive` is read
  beside the exploratory split and the per-fold lines. If the harm concentrates in folds 2023-07 to
  2023-09, where `R1` had trained on at most two months of new-resolution cloud, the page attributes
  the harm to the era and not to cloud.
- **A calibrated positive control.** The `K1` control is large and trivial, because `K1p` has no
  temperature. A calibrated known-answer control would add a synthetic column equal to the realised
  clear-sky index plus Gaussian noise, tuned on one fold so that the unmasked column gains about
  0.04 points, then masked with the cloud's pattern and contrasted with its placebo. The control
  passes if the masked gain is at least half the unmasked gain. The column reads outcome data, so
  the column is a control only. PLP's `longwave`, `relative_humidity`, `precipitation`, and
  `pressure_surface` inputs do not enter the physics-lite power, so a solar version is possible.
  Reporting `K1` minus `DF` on shared seeds would measure how much masking reduces the gain of a
  strong signal.
  **To be settled before Stage B**, because the positive control is Stage B.
- **A solar replicate.** A back-fill replicate for the farms with enough early history is possible.
  **Settled:** the plan leaves it out, to hold the experiment to one screen of results.
- **The solar holdout.** `evaluation-protocol-solar.md` would allow July and August 2026, two
  high-sun months, but the store refuses valid times from 2026-07-01. **Settled:** the experiment
  reads no valid time from 2026-07-01 on, and a labelled exploratory reading of those months is
  not part of this plan.

**One further limit applies to every result.** `R_F` is not the port target, so a gain on `R_F` may
not carry to the Candidate, which reads 14 thermal-lag and solar-gain columns that `R_F` lacks. The
cloud-scored folds cover December to June only, and fold 2025-12 holds only 10 recent runs with
cloud in its fine-tune window.

### 13. Differences between the plan and the code

#### Amendment of 2026-11-20: Stage B1

**The manager approved a reduced Stage B (Stage B1) on 2026-11-20, and the plan changes in three
places.**

- **Stage B1 is 150 fits on seeds 0 to 2:** `R0`, `R1`, and `R1p` at 36 each, and `D0g` and `D1g` at
  21 each. The other 664 fits are Stage B and are not run.
- **The 6-seed rows of the plan for `R1` and `R1p` are among the 664 fits not run.** Stage A
  excludes only gains above
  0.11 points at the six, so the replicate answers the date-proxy question and not a power question.
- **`e00711_jobs.py` and `e00711_score.py` take `--stage B1`,** and every scorer run saves its full
  output to `work/scores/<command>[-stage]_<UTC time>.txt` under the experiment's data folder.

#### Stage A scoring code

**The scorer's per-fold solar rows were wrong in the first Stage A run, and the code now fixes
them.** The exploratory filter ignored the fold, and the rows now carry no verdict word (section
"Stage A detail"). The planned readings, checks, and bounds did not change.

#### Stage 2 code

**The stage 2 code departs from the plan above in the places listed here, and the code review
(2026-11-20) accepted the first three.**

- **Solar rows start at the run of 2020-10-29, not 2020-10-15.** The farm tables keep the 14
  earliest runs as donors for the time-shifted placebo, and `daylight_rows` starts the rows at
  2020-10-29. The 14 dropped runs are about 0.7% of the training runs and are dropped from every
  arm.
- **The half-hour labelled at the origin is not forecast.** Its period ends at the origin, so each
  solar run holds 671 half-hours, not 672. Stored `P0` holds that half-hour, and the `P0` reader
  drops it.
- **The replicate's scored rows come from the back-filled raw rows.** They are the rows the
  replicate trains on, and scoring drops the flagged exact-zero half-hours as on the recent folds.
  The replicate's fine-tune window keeps no row, origin, or weekly lag before 2021-07-20.
- **The checks are eight, at 99.375%, and are read at the six and at the eight separately.** The
  positive control is read at the same level.
- **Bootstrap pooling.** The pooled mean weights each Primary or farm by its row count, which the
  scorer prints beside every planned reading.
- **Not implemented, and not run:** the avoidable-curtailment cost, cold winter nights, the example
  forecasts, and the reference rows climatology, manual heuristic, BG2, S0M, and `A_F`. The
  Checks list records them as not run.

#### Stage 1 code

**The stage 1 code departs from the plan above in six places, and each departure is deliberate.**
The code reviewer accepted the first four. The era screens await the maintainer's agreement before
`era-check` runs, so that the definition is not chosen after its results are seen.

- **The cloud cache starts at the first back-filled run, 2020-10-15, not at 2021-07-20.** Rows
  still read only runs from 2021-07-20 (985 back-filled runs). The 278 earlier runs serve as donors
  and lag sources, because the time-shifted donor of run 2021-07-20 is the run of 2021-07-06. The
  cache therefore holds 1,485 files (1,263 back-filled runs and 222 recent runs). The earlier runs
  also enter the shuffled-run donor pool, so a scored run can draw a donor from 2020-10 to
  2021-07, which widens the era mix that section 12 discusses.
- **The time-shifted donor has a fourth fallback of 744 days.** After 14, 21, and 7 days earlier,
  the rule takes the run issued 744 days earlier. The runs from 2025-11-21 to 2025-11-27 have no
  donor at 7, 14, or 21 days, because each of those days lies in the gap. The 744 days map
  2025-11-21 to 2023-11-08 (two years and 14 days of the calendar, less 29 February 2024).
- **The origin assertion is a copy.** The Primary demand code calls a copy of experiment 707's
  `assert_origin_after_run`, because `e00707_prep.py` imports `torch` and the tests run without it.
  The copy behaves identically, and `assert_primary_rows` adds two checks: the origin is exactly the
  run plus 9 hours, and the run is a 00:00 UTC run.
- **7,243,785 is the number of rows of each run, not of all runs.** Each of the 222 runs from
  2025-11-21 to 2026-06-30 holds 7,243,785 rows at Delta version 2589, and the 222 runs together
  hold 1,608,120,270 rows. The table check tests the per-run figure.
- **The era screens use two scales, and a test fails if either reading exceeds 3.** The literal
  reading divides the step by the standard deviation of the adjacent 12 monthly values, which
  includes the seasonal cycle. The deseasonalised reading removes the cycle. For the winters test
  (December to March of 2021-22, 2022-23, and 2023-24 against December 2025 to March 2026), the
  scale is the standard deviation of the earlier months about their calendar-month means. For the
  resolution-change test (each month from 2023-07 to 2024-03 against the same month a year
  earlier), the scale is the standard deviation of the year-over-year differences in the 12 months
  before 2023-06. The code also reports the step divided by its standard error.
- **The build refuses to run unless three checks have passed at the pinned Delta version.** The
  checks are `table-check`, `older-columns`, and `era-check`, each of which records its result
  under `work/checks/`. Every cache file holds its Delta version, `pin --force` deletes or refuses
  to keep a file from another version, and the lineage note records the code commit, the package
  versions, and the SHA-256 of each script. The data check also fails on cloud stored as a
  fraction, on duplicate keys, on a stuck series, on two runs with identical cloud, and on a
  correlation below 0.95 with the cloud files of experiment 561. Delta-rs keeps old table versions
  for 30 days by default, so the cache can be rebuilt from the pinned version for a limited time.

## Files

- `ml-experiments/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms/e00711_cloud.py`
  (wrote the per-run cloud summaries): builds the mean and the spread of total cloud cover across
  the 51 members for each ENS run from the pinned Delta table, and the checks that guard the cache.
- `.../e00711_placebo.py`, `e00711_origin.py`, `e00711_rowcloud.py` (wrote the donor maps and
  per-row cloud): the donor rule for the placebo, the origin assertion, and the join of cloud onto
  the training rows.
- `.../e00711_solar_inputs.py`, `e00711_solar.py` (wrote the solar tables, read the PLP code): the
  per-farm input tables and the solar fit, copied from experiment 666.
- `.../e00711_arms.py`, `e00711_folds.py`, `e00711_common.py`, `e00711_fit.py` (fit the networks):
  the arm definitions, the monthly folds, shared helpers, and the training code copied from
  experiment 707.
- `.../e00711_gate.py` (ran the gate): reproduces stored `D0` with the arm `D0n` before any other
  fit.
- `.../e00711_jobs.py` (listed the jobs): lists the fits of each stage and checks the counts of 132
  (Stage A), 150 (Stage B1), and 664 (Stage B).
- `.../e00711_score.py` (scored the predictions): the planned readings, the checks, the exploratory
  rows, and the verdicts.
- `ml-experiments/00707-ens-backfill-pretraining-gain-against-ens-history-length/` (read,
  [write-up](../00707-ens-backfill-pretraining-gain-against-ens-history-length/result.md)): the
  `R_F` training code, `assert_origin_after_run`, the donor rule, and the stored `D0` forecasts.
- `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_solar_inputs.py`
  and `e00666_solar_fit.py` (read, then copied): the PLP inputs and fit and the stored PLP
  forecasts.
- `data/ml-experiments/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms/predictions/`
  (written): the per-row forecasts of every arm, read through `experiment_store`.
- `data/ml-experiments/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms/work/`
  (written): `ens_cloud/` (per-run cloud summaries), `placebo/` (donor maps), `raw_cloud/`,
  `checks/`, `logs/`, `gate_passed.json`, and `scores/` (full scorer output,
  `score-A_20261120T045753Z.txt` and `score-B1_20261120T064304Z.txt`, the sources of every number on
  this page).
- `data/ml-experiments/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms/solar_inputs/`
  (written): one input table per solar farm.
- `data/ml-experiments/briefs/total_cloud/` (read): the private findings, design, and design review.
