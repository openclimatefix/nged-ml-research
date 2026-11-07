---
id: "00640"
slug: demand-fit-age-refit-k1-k3
title: >-
  How much worse does the demand network score the longer ago it was fitted: 0, 4, 8, 16, or 26
  weeks before the forecast month?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: other
protocol: demand-v2
size: >-
  Medium. Answers to the five triggers: (1) it stores new network weights and forecasts in the
  private data folder only; (2) one design is defensible, because the refit schedule is the only
  thing varied; (3) it spans code from experiments 510 to 620 whose callers cannot be named without
  searching, so it reuses that code by import and patches only the cutoffs; (4) it touches no
  serving path and no degradation rule; (5) it changes no contract.
hypothesis: >-
  A demand network fitted on data that ends A weeks before the forecast month scores worse the
  larger
  A is, and the loss is under 1% of the continuous ranked probability score (CRPS) for every 4 weeks
  of fit age, for K1 (the lag-using network with the gate) and for K3 (the lag-free network, the
  control).
prediction: >-
  Written before any fit. P1: the mean CRPS rises by less than 1% per 4 weeks of fit age, fitted as
  a straight line through the five fit ages, for K1 and for K3 (percent of the age-0 CRPS, 0 to 240
  h). P2: K1 loses more per 4 weeks than K3 at 24 to 120 h, because K1's lag inputs and fine-tune
  rows go stale with the fit. P3: the Primaries with a labelled event in the test month lose more
  per 4 weeks than the others. P4: the age-0 fit scores within 1% of experiment 613's roster on the
  first 14 days of each month (the roster's fit cutoff of day 0 applies to those days).
decision_rule: >-
  The slope is the CRPS change per 4 weeks of fit age, as a percentage of the age-0 CRPS, with a
  month-block bootstrap interval (resampling the 5 development months, with Primaries paired).
  Weekly refits in production are worth running if the interval's lower end exceeds 1% per 4 weeks,
  monthly refits if the CRPS at 4 weeks of age exceeds the age-0 CRPS by more than 1% with an
  interval excluding zero, and neither if the upper end of the slope is under 1%. There is no
  Bonferroni correction because the slope per network is the one planned contrast per network (2 in
  total, so the printed level is 97.5%).
benchmark: >-
  Experiment 613's roster scores for K1 and K3 (day 0 and day 14 refits), climatology, and the
  manual heuristic, on the same rows.
row_set_id: development_months_2025_10_to_2026_06_16_primaries
verdict: refuted
best_arm: >-
  Not applicable: the experiment compares fit ages, and the newest fit (age 0) scores best at
  every age comparison.
headline: >-
  K1's CRPS rises 5.4% per 4 weeks of fit age at all 16 Primaries (97.5% interval +2.9 to +7.6) and
  K3's 6.2% (+4.3 to +7.4), against a pre-registered limit of 1%. Without D15 and D18 the slope is
  1.0% (-0.2 to +1.8) for K1, but it is 1.6% (+0.7 to +3.2) once the history-limited 2025-10 fits
  are left out, and the first 4 weeks cost K1 3.0% (+0.9 to +5.3).
take_home_message: >-
  A demand network fitted 4 weeks before the forecast month scores about 3% worse (K1) than one
  fitted at the month's start, and the loss then levels off. The loss at D15 and D18 is several
  times larger, but this experiment cannot say whether fit age, their short training history, or
  level shifts in their power cause it. A refit every 4 to 8 weeks is worth running for K1, and
  weekly refits are not shown to be worth more.
lessons: >-
  Fit age matters mostly at D15 and D18. The loss at the 14 long-history Primaries is
  small and levels off, and K1 and K3 lose the same amount, so the lag inputs do not make K1 age
  faster.
production_status: candidate
adopted_in: none
advice:
  - >-
    Refit the demand networks at least every 4 to 8 weeks, because K1 loses 2.9% of CRPS after 4
    weeks and 6.3% after 8 weeks at the 14 Primaries with a long history (see the contrast table).
  - >-
    Test whether refitting D15 and D18 more often helps them, because their CRPS rises 17% in the
    first 4 weeks (see the group figure) but fit age, training history, and level shifts in their power
    are confounded here.
  - >-
    Do not expect K1 to age faster than K3 on account of its lag inputs; the slopes match at every
    lead band (see the slope figure).
metric: >-
  CRPS and MAE by lead band (0-24, 24-120, 120-240 h) and by weeks since the fit; q05-q95 coverage.
---

**A demand network that was fitted longer ago scores worse, but mostly at the two Primaries that
have
the least training history.** We refitted two demand networks for each of the 5 development months
(October 2025, December 2025, January 2026, March 2026, and June 2026), with the end of the training
data 0, 4, 8, 16, and 26 weeks before the month started. We then scored each fit on the whole month
and compared the five fit ages. At the 14 Primaries with a long history, the continuous ranked
probability score (CRPS, where lower is better) of K1, the network that reads recent power, rose by
3.0% after 4 weeks and by 6.3% after 8 weeks, then levelled off. At D15 and D18 it rose by 17% after
4 weeks (pooled over the months) and doubled by 26 weeks. K3, the network that reads no recent
power, lost the same amount as K1.

**The pre-registered limit of 1% per 4 weeks fails for both networks at all 16 Primaries, and is not
safely met at the other 14.** At the 14 long-history Primaries K1's slope is 1.0% (97.5% interval
-0.2 to +1.8) and K3's is 1.0% (-0.3 to +2.3), but the slope rises to 1.6% for K1 when the
history-limited 2025-10 fits are left out, and the loss in the first 4 weeks is about 3%. The loss
is
not a straight line: it flattens after the first months.

## Figures

![CRPS above the age-0 fit, against fit age, for all 16 Primaries and for the 14 without D15 and
D18](crps_vs_fit_age.svg)

*Caption: at the 14 Primaries without D15 and D18, K1 scores 3.0% worse at 4 weeks and 10.0% worse
at
16 weeks than at age 0, with a flat or falling curve after that. With D15 and D18 the loss reaches
35% (K1) at 26 weeks. Error bars are 97.5% month-block bootstrap intervals; the intervals are wide
because they rest on 5 months.*

![CRPS against fit age, for D15 and D18 beside the other 14 Primaries](crps_by_primary_group.svg)

*Caption: at the 14 other Primaries the CRPS moves from 3.0 to 3.3 (K1) and from 3.4 to 3.5 (K3)
percent of capacity. At D15 and D18 it doubles, from 9.3 to 19.0 (K1) and from 8.8 to 20.2 (K3).*

![CRPS change per 4 weeks of fit age, by lead band](slope_by_lead_band.svg)

*Caption: the slope is similar at every lead band, and K1 and K3 do not differ. The 14-Primary
intervals contain 1% at every band.*

![Event-free q05-q95 coverage against fit age](coverage_vs_fit_age.svg)

*Caption: the 90% band covers 0.86 of the readings at age 0 and 0.79 (K1) or 0.77 (K3) at 26
weeks. Both networks are over-confident at every age.*

## Results

- **Scored system.** Each network is scored on its own quantiles, without the gate that
  [experiment 620](../00620-bakeoff-final-test-demand-networks-at-primaries/result.md) added to K1,
  because the gate's code is tied to month names. The scored rows are the rows that
  [experiment 613](../00613-nra-roster-contrasts-with-intervals-five-seeds/result.md) scored: 16
  Primaries, event-free rows unless stated, leads 0 to 240 h, seeds 0 to 2.
- **P4, the reproduction check, holds.** The age-0 K1 fit scores 0.03643 on the first 14 days of the
  five months, against 0.03677 for the roster's ungated K1 (a ratio of 0.991, inside the 1%
  tolerance). The roster has a refit on day 14 and five seeds, so only days 0 to 13 compare.
- **P1 fails at 16 Primaries and is not safely met at 14.** Slope per 4 weeks, percent of the
  age-0
  CRPS at 0 to 240 h:

  | Primaries | K1 | K3 |
  |---|---|---|
  | All 16 | 5.4 (2.9 to 7.6) | 6.2 (4.3 to 7.4) |
  | 14, without D15 and D18 | 1.0 (-0.2 to 1.8) | 1.0 (-0.3 to 2.3) |

- **CRPS above the age-0 fit at 14 Primaries (percent, 97.5% interval).** K1: 4 weeks +2.95 (0.92
  to 5.25), 8 weeks +6.25 (1.78 to 11.96), 16 weeks +9.97 (3.43 to 16.98), 26 weeks +6.08 (-0.77 to
  11.69). K3: +1.14 (-0.15 to 2.68), +5.77 (2.61 to 9.90), +11.32 (4.43 to 20.42), +4.62 (-1.65 to
  12.76). The lower end of K1's 4-week interval is under 1%.
- **P2 fails.** K1 does not lose more than K3 at 24 to 120 h: the slopes are 1.1 and 1.1 at the 14
  Primaries (5.9 and 6.3 at 16).
- **P3 is not separable.** Primary-months with an event lose 7.7% (K1; 2.2 to 16.0) per 4 weeks on
  event-free rows, against 4.8% (2.0 to 6.9) without an event, but the intervals overlap and D15
  and D18 contribute to both.
- **Absolute scores (percent of capacity, 16 Primaries, event-free rows).**

  | Fit age (weeks) | K1 CRPS | K1 MAE | K1 q05-q95 coverage | K3 CRPS | K3 MAE | K3 coverage |
  |---|---|---|---|---|---|---|
  | 0 | 3.76 | 6.74 | 0.864 | 3.99 | 7.15 | 0.855 |
  | 4 | 4.03 | 7.17 | 0.850 | 4.20 | 7.48 | 0.853 |
  | 8 | 4.18 | 7.41 | 0.836 | 4.45 | 7.86 | 0.821 |
  | 16 | 4.67 | 8.54 | 0.803 | 5.24 | 9.19 | 0.770 |
  | 26 | 5.08 | 8.94 | 0.788 | 5.50 | 9.44 | 0.768 |

- **Tails.** The q95 and q99 pinball losses grow the most with age (K1 q99: 0.0029 at age 0, 0.0064
  at 26 weeks; q01 barely moves), so an older fit mainly misses upward excursions.
- **Not shown.** Climatology and the manual heuristic are not rescored at each fit age, because
  their fits do not depend on the network's fit age; their roster scores are on experiment 613's
  page.

## What would change production

- **Monthly refits are worth running for K1.** Its CRPS at 4 weeks of age exceeds the age-0 CRPS by
  more than 1% with an interval above zero (2.95%, 0.92 to 5.25) at the 14 long-history Primaries.
- **Weekly refits are not shown to be worth more than monthly refits.** The loss is about 3% in the
  first 4 weeks and flattens later, and this experiment did not run a weekly refit, so the extra
  gain of a weekly schedule over a monthly one is not measured here.
- **The short-history Primaries may need frequent refits.** D15 and D18 lose 17% (K1, pooled) within
  4 weeks. This experiment cannot separate that loss
  from their small number of training months or from level shifts in their power.

## Checks

- **Median minus mean forecast per site:** not computed, because the stored forecasts hold quantiles
  only. The absolute error of the median rises with fit age at the same rate as CRPS (K1 +33% from
  age 0 to 26 weeks in both), so the two scores agree.
- **Triggers fired:** trigger 2 (a result contradicts its prediction) and trigger 4 (leaving D15
  and D18 out changes the verdict of P1; leaving out any one of three other Primaries moves the
  14-Primary slope to 1.1 to 1.2). A fresh Sonnet audit re-derived every number from the stored
  forecasts and confirmed them. Its findings and their triage:
    - **The 14-Primary "at the margin" verdict is fragile (accepted).** The page now says P1 is not
      safely met at 14 Primaries.
    - **The history bullet contradicted its own number (accepted).** Corrected above: the
sensitivity
      analysis raises the slope.
    - **The D15 and D18 loss is not shown to be fit age or history length (accepted).** The page
      states the confound and no longer attributes the loss to history.
    - **No leakage found, but two loaders bypass the train-phase patch (accepted as a caveat).** The
      audit found by grep that the 640 chain never calls `e00428_data.load_stack` or
      `prep422.cleaned_arrays`, and did not trace a run.
    - **Scorer bias is negligible (accepted).** Averaging quantiles over seeds inflates the
16-Primary
      slope by about 0.2 points and changes no verdict.
    - **P4 reproduces for K1 only (accepted).** By month the ratio spans 0.972 to 1.019, and the
roster
      has no K3 reference.
- **History confound.** The oldest cutoffs of the earliest months have fewer months of fine-tune
  rows, and D15 and D18 start on 2025-03-01, so they have 1.0 month of cleaned power at the oldest
  cutoff of October 2025 and 3.0 months at that of December 2025 (never under 5.8 months for March
  and June 2026). Leaving the 2025-10 fits older than 16 weeks out raises K1's 14-Primary slope from
  1.0% to 1.6% (0.7 to 3.2), so the shorter history does not explain the loss at the 14 Primaries.
  The audit found that D15's and D18's CRPS depends more on the test month than on fit age or
  history length (K1 age-0 CRPS at the two is 16.4% of capacity in March 2026, 4.8% in January
  2026), and that the CRPS steps line up with level shifts in their raw power. Fit age, history
  length, and those shifts are confounded, so the page does not attribute the D15 and D18 loss to
any
  one of them.
- **Lab conditions.** Each fit has no refit inside the month, so the oldest origins are about 30
  weeks after the fit. The gate is not in the scored K1. Synthetic load-shift events are seeded from
  the development month, so every age of one month shares them.

## What to try next

- **Refit D15 and D18 at a monthly or shorter cadence in an experiment of their own**, with a
  control that holds the number of training months fixed while the fit age varies, and a check of
the
  level shifts in their power.
- **Add climatology and the manual heuristic to the headline chart** after a run that scores the
  gated K1 at each fit age.

## Registered plan

- **Question.** The production service will refit on some schedule. Experiments
  [494](00494-simplify-demand-network-refit-cadence-and-pretrain-once/result.md) and
  [511](00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/result.md) measured
  the price of an old fit on the lag-free network only, and the July to October window of
  [experiment 620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) used one
  fit with no refit, so its fit age rose from 0 to 17 weeks. This experiment measures the loss
  per 4 weeks of fit age on the development months, where an answer cannot touch the closed window.
- **Design.** For each development test month (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), fit
  K1 and K3 once with the cutoff A weeks before the month starts, for A = 0, 4, 8, 16, and 26. Each
  fit has no refit during the month, so the fit age of a forecast origin is A plus its days into the
  month, and the score by weeks since fit covers 0 to about 30 weeks. The recipe, seeds, 16
  Primaries, and pretraining rules are experiment 613's. Pretraining reruns at each cutoff, with
  its newest row older than the cutoff by the usual refit age, as experiment 620's final fold does.
  Each fit's cleaned power is hidden after its cutoff, as in experiment 620's training phase, so no
  code path can read a later reading.
- **Cheapest recipe.** One cutoff per fit and no day-14 refit, with seeds 0 to 2 in place of
  experiment 613's five (the recipe is otherwise experiment 613's, and the K1 seeds 0 and 1 are not
  bit-identical to 613's, which came from experiment 591's stored chain). From experiment 620's
  logs, K1 takes about 7.5 minutes per cutoff at 3 seeds and K3 about 4 to 5 minutes, so the 25
  cutoffs cost about 5.5 to 6.5 GPU-hours with the prediction processes. That reaches the 6
  GPU-hour ceiling, so the first cutoff of each network is timed before the chain starts, and the
  manager is asked before crossing the ceiling. Prediction P4 checks that the recipe reproduces
  the roster within 1%; if it fails, the page quotes only the contrast between ages.
- **History confound.** The oldest cutoffs of the earliest months have less fine-tune history
  (about 12 months at 2025-10 with age 26, against about 18 at age 0), and D15 and D18 have under
  3 months of cleaned power there. The slope is therefore also reported without D15 and D18, and
  with the 2025-10 fits older than 16 weeks left out. The synthetic load-shift events are seeded
  from the development month, so every age of one month shares them.
- **Scoring.** Reuse experiment 613's scorer (`e00613_score.py`) for the cells by Primary, month,
  and lead band, with K1 assembled with the gate from the K3 fit of the same cutoff, and group the
  rows by weeks since the fit. Report separately the Primaries with a labelled event in the
  test month (NGED event labels, a lower bound because labels are incomplete) and the others.
  Quote the age-0 roster rows, climatology, and the manual heuristic beside the networks, and the
  tail table (q01, q05, q95, q99 coverage) beside the CRPS.
- **Standing practices followed.** Cleaned data everywhere; the D15 and D18 cut before 2025-03-01;
  models saved as `.ubj` or state files in the data folder; stored 613 predictions reused as the
  reference; the closed window (valid time from 2026-07-01) is never opened; the run is a screen on
  the development months, not a protocol change, so `kind: other`.
- **Compute.** Ceiling 6 GPU-hours (process hours). One GPU process at a time, 8 GB host cap and
  one OMP thread, as experiment 620's `gpu.sh` does, and `systemd-run` with `-p MemoryMax=8G -p
  OOMPolicy=continue`.
- **Reporting.** The headline chart is CRPS against weeks since the fit for K1, K3, climatology,
  and the manual heuristic, with month-block intervals. Further figures: slope per lead band, the
  event and non-event split, and per-month curves.

## What the plan said would change production

- If the slope interval's upper end is under 1% per 4 weeks, a refit every 4 to 8 weeks is
  enough, and the production service should not spend effort on weekly refits.
- If the lower end exceeds 1% per 4 weeks, weekly or monthly refits are worth running, and the
  experiment names which by the age at which the loss passes 1%.
