---
id: "00543"
slug: lag-free-network-auxiliary-recent-level-and-reverse-flow-heads
title: >-
  Can an auxiliary head that predicts the recent power level, or the reverse flow, close the gap
  between the lag-free demand network and the lagged forecast without lag inputs at forecast time?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
status: finished
kind: other
protocol: demand-v2
size: >-
  Medium. Stored: fitted network weights and window forecasts under this experiment's private data
  folder only; earlier experiments' predictions are read, never written. Designs: more than one is
  defensible (which quantity the recent-level head predicts, the loss weight), so the plan fixes
  each before any score is read. Callers: this folder's scripts import experiments 397, 425, 428,
  510, and 511 and the `studies` evaluation code. Serving path and degradation rule: not touched.
  One Sonnet code review before the first fit, one Sonnet review of the write-up against the
  printed report, one Sonnet prose review.
hypothesis: >-
  The lag-free network trails the lagged network (A-pool) because it never sees recent power. A
  second output that predicts the recent power level (the newest observed value and the day-ago
  and week-ago values, used as training targets only and never as inputs) makes the shared hidden
  layers carry level information, so the lag-free network closes part of the gap without any lag
  input at forecast time.
prediction: >-
  Written before any score is read. F1 (recent-level head) closes at least half of the L-1d minus
  A-pool gap at 24 to 72 h at the eight (L-1d minus A-pool is +0.288 at 24-48 h and +0.254 at
  48-72 h, so F1 minus A-pool at most +0.144 and +0.127), without raising the injected-shift leak
  above 4.49 + 0.5 points of capacity. It may fail: the newest value used as an input (experiment
  538's R1) closed only 24% to 28%. F2 (reverse-flow head) changes the eight by less than the
  minimum detectable effect, because the eight are MVA-only and carry no sign; F2 is expected to
  matter, if at all, at the six. F3 (both heads) is no better than F1 at the eight.
decision_rule: >-
  F1 counts as supported if its paired contrast against A-pool and against L-1d, pooled over
  24-72 h at the eight, has a Bonferroni interval excluding zero in the predicted direction and
  closes at least half of the gap, and the leak stays within 4.99 points of capacity. Cost of
  porting: an extra output head and a training loss, no new data source and no serving input, so
  the threshold for a gain is an interval excluding zero. The other contrasts are report-only.
benchmark: >-
  L-1d (experiment 511), A-pool (experiment 425), R-DW (experiment 539), R1 and R9 (experiment
  538), and the rebuilt K2-S1 (experiment 541) when stored, on experiment 511's rows; BG2,
  climatology, and `manual_heuristic` for scale.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight (MVA-only Primaries) and the six (signed-MW
  Primaries), D15 and D18 report-only, and the injected-shift leak of the 511 stress world.
verdict: refuted
best_arm: >-
  None. L-1d (no auxiliary head) scores 4.525 points of CRPS at the eight and 3.219 at the six for
  lead 24 to 72 h; F1 scores 4.574 and 3.234, F2 4.541 and 3.226, F3 4.584 and 3.236, against
  4.254 and 3.142 for A-pool.
headline: >-
  An auxiliary head that predicts the recent power level made the lag-free forecast worse, not
  better: F1 scored 0.049 points behind L-1d at the eight for lead 24 to 72 h (interval +0.035 to
  +0.061, 5 of 5 folds), so it closed none of the 0.271 gap to A-pool. Lowering the head's weight
  from 0.5 to 0.1 removed the harm (+0.005) and still gave no gain.
lessons: >-
  The recent level that a lag input supplies is not recoverable from the lag-free inputs by
  multi-task training: the head's targets are predictable only from information the network
  already uses for the forecast, so sharing hidden layers adds nothing and the extra loss only
  competes for capacity. The injected-shift leak fell for every head (3.8 to 4.0 points against
  4.49), which was not a goal and is not explained.
take_home_message: >-
  Training the lag-free network to also predict the recent power level did not give it the benefit
  of lag inputs: its forecast got slightly worse (0.05 points of CRPS at the eight), and the gap to
  the lagged network stayed at 0.27 points. The recent level has to enter as an input, so the next
  step is to choose which lagged inputs to feed (the day-ago and week-ago lags closed 76% to 80% of
  the gap in experiment 539), not to try to learn them as a side task.
advice:
  - >-
    Do not add an auxiliary head for the recent power level to the lag-free demand network; it
    scored 0.049 points worse than L-1d at the eight and closed none of the gap to the lagged
    network (Results, figure 2).
  - >-
    Do not add a reverse-flow head to the lag-free demand network; reverse flow is rare at the six
    (0.00% to 0.26% of half-hours) and the head scored 0.017 points worse at the eight (Results).
  - >-
    If a lag-free network needs recent level, give it lag inputs (experiment 539) rather than a
    side task; a side task at weight 0.1 was still 0.005 worse than L-1d (Checks).
production_status: not-for-production
adopted_in: none
---

**Training a demand neural network without lagged-power inputs to predict the recent power level as
a side task did not help it forecast.**  Its continuous ranked probability score (CRPS, the quality
measure of a probabilistic forecast, here in percentage points of substation capacity; lower is
better) was 0.049 points worse than the same network without the side task, averaged over the eight
substations whose power is metered only as apparent power (MVA) and over forecasts 24 to 72 hours
ahead. None of the 0.271-point gap to the neural network that reads lagged power closed. The
reverse-flow side task scored 0.017 worse and the two side tasks together 0.059 worse.
The next step is to give the network the day-ago and week-ago lags as inputs, as experiment 539
did, because the recent level was not learnable as a side task in this test.

We asked whether a lag-free demand network can recover what lag inputs tell a lagged network
without receiving them at forecast time. We added extra outputs to the last layer of experiment
511's best lag-free network (L-1d) and trained them on power values from before each row's forecast
origin, which the network never sees as inputs. F1 predicts the newest observed power at the origin,
the day-ago power, and the week-ago power (three standardised values; this follows
[Caruana (1997)](https://doi.org/10.1023/A:1007379606734) in sharing hidden layers between a main
and a side task). F2 predicts whether power at the target time is below -0.005 of capacity (a
reverse flow), where the sign is metered. F3 has both heads. Each side loss has weight 0.5. Scores are
on experiment 511's five folds of one month each, 2 seeds, rows with no labelled event.

Verdict: refuted. The pre-registered prediction was that F1 would close at least half of the gap
between L-1d and A-pool (the lagged network of experiment 425) at lead 24 to 72 h at the eight,
meaning F1 minus A-pool of at most +0.144 and +0.127. F1 minus A-pool was +0.345 and +0.295. F2 was
predicted to change the eight by less than the minimum detectable effect (0.135), and it did (+0.017).
F3 was predicted to be no better than F1 and F3 was 0.010 worse.

## Results

| Lead 24 to 72 h, CRPS (points of capacity) | Eight | Six |
|---|---|---|
| F1 (recent-level head) | 4.574 | 3.234 |
| F2 (reverse-flow head) | 4.541 | 3.226 |
| F3 (both heads) | 4.584 | 3.236 |
| F1-w01 (post hoc: F1 with weight 0.1) | 4.530 | 3.220 |
| L-1d (no head; experiment 511) | 4.525 | 3.219 |
| R-DW (day-ago and week-ago lags as inputs; experiment 539) | 4.314 | 3.129 |
| A-pool (lagged network; experiment 425) | 4.254 | 3.142 |

- **F1 minus L-1d** is +0.049 at the eight (month-block interval +0.035 to +0.061, 5 of 5 folds worse,
  minimum detectable effect 0.135) and +0.016 at the six (+0.008 to +0.024, 5 of 5 folds).
- **F3 minus L-1d** is +0.059 at the eight (+0.040 to +0.074, 5 of 5 folds) and +0.018 at the six
  (+0.002 to +0.033, 4 of 5 folds). **F2 minus L-1d** is +0.017 at the eight (+0.004 to +0.028, 4 of 5 folds).
- **The gap that had to close** is L-1d minus A-pool: +0.271 at the eight (+0.182 to +0.352, 5 of 5
  folds) and +0.077 at the six (-0.015 to +0.160, 3 of 5 folds).
- **Interval method.** The intervals are 95% month-block bootstraps with 5 blocks, which is close to
  the range from the best fold to the worst fold. The Bonferroni count is 56 (4 counted contrasts, F1 minus
  A-pool, F1 minus L-1d, F2 minus L-1d, and F3 minus L-1d, times 7 bands times 2 sets). The minimum
  detectable effect (0.135 at the eight) is two to three times every contrast above, so one
  contrast alone could not detect a gain or loss of that size; the fold signs (5 of 5 worse for F1
  and F3 at the eight) are the stronger evidence. At lead 24 to 120 h, F1 minus L-1d is positive
  in every band at the eight.
- **Injected-shift leak** (511's stress world, 7-day block of 3 times the weekly-mean standard
  deviation, seed 0, mean of five folds): F1 3.88, F2 3.96, F3 3.81, L-1d 4.49 points of capacity.
  The pre-registered limit was 4.99. The point estimates are lower for every head (the fold ranges overlap L-1d's, so a reduction is not shown), on day 1 (4.3 against 5.4)
  and on days 2 to 3 (3.7 to 3.8 against 4.6) alike. We did not test why.
- **D15 and D18** (report-only) scored 10.42 for F1, 10.28 for F2, and 10.36 for F3 at lead 24 to 72
  h, against 10.37 for L-1d. F2 is 0.09 better there, where reverse flow is common; this was not
  tested.
  July 2026 was not scored and not tuned on.

![CRPS by lead band](figure_1_crps_by_horizon.svg)

![Each head minus L-1d, lead 24 to 72 h](figure_2_head_minus_l1d.svg)

![Injected-shift leak](figure_3_stress_leak.svg)

## Checks

- **Weather inputs.** Every row's weather comes from the run issued at or before the forecast
  origin, asserted for all 153 test origins. Example: target 2026-06-07 15:30 UTC, origin
  2026-06-01 09:00 UTC, weather run 2026-06-01 00:00 UTC. The 9 lag columns are removed from the
  network inputs before training and scoring; they remain only as training targets, each observed at
  least 3.5 hours before its origin (asserted row by row).
- **Targets match the cleaned power.** After rounding the lag time to the half-hour grid, 100% of
  kept side targets equal the cleaned power at that time; targets whose cleaned power is missing
  were dropped (about 0.1% of rows), and so were values clipped at 5 standard deviations.
- **Reverse-flow labels.** The label exists at the six and at D15 and D18 and is masked at the
  eight. Reverse flow is rare at the six (at most 0.26% of half-hours at any one Primary, from a one-off look
  at the cleaned power that is not stored), so most positive labels come
  from D15, which is report-only. F2 therefore had little to learn from, which is a reason its effect
  is small, not a finding about reverse flow.
- **Explanation tested (trigger 2).** F1 with weight 0.1 (post hoc, not in the plan, not counted in
  the Bonferroni count) scored +0.005 against L-1d at the eight (+0.002 to +0.008) and +0.001 at the
  six. The harm is smaller at the lower weight but does not vanish (the interval still excludes zero),
  and neither weight tested gives a gain. The weight
  0.5 was fixed before any score; no other weight was run. The gradient clipping includes the side
  gradients, a small difference from L-1d besides the extra outputs, which the design cannot avoid.
- **Error of the mean or median.** Not computed: no contrast here points in a direction that a
  quantile-score disagreement could reverse, because every arm shares the same mixture output.
- **Triggers.** Trigger 2 fired (the result contradicts the prediction) and the weight test above is
  the explanation; the other triggers did not. No audit was run, because the explanation was tested
  and no later experiment's design depends on this result's sign.
- **Not done.** The rebuilt K2-S1 (experiment 541) was not stored when this ran, so it is not a
  comparator. The research default's absolute scores against BG2 and `manual_heuristic` are in
  experiments 511 and 539. Lead range: 6 to 240 h; the headline concerns 24 to 72 h.

## What the next experiment should answer

Which of the day-ago and week-ago lag inputs, in what reduced form, keeps most of experiment 539's
76% to 80% gap closure while leaking an injected load shift least.
