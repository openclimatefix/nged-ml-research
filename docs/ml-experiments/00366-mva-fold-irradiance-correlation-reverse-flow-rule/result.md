---
id: "00366"
slug: mva-fold-irradiance-correlation-reverse-flow-rule
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
kind: other
size: >-
  simple. Stored: small result files in the private folder only. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one statistic on the readings would test
  the hypothesis; the Pearson correlation with irradiance is the one experiment 365 proposed, and the
  low-percentile test is kept as the second condition. Callers: reads the stored results of
  experiment 358 and the feature cache, so no new model fit is needed.
hypothesis: >-
  Reverse flow can be told apart from invented reverse flow using the meter readings and the
  irradiance forecast alone: where a substation has reverse flow, its sunny-midday meter readings
  rise with irradiance (the magnitude of a negative flow grows), and where it has none, they fall.
prediction: >-
  At the scored cut with reactive power zero, the rule picks the naive fit at D5 with no injection
  (plain direction accuracy within 0.02 of the signed fit's), picks the fit through the meter (L0) at
  every injected case (levels 1.0 and 2.0), and picks L0 at Stickney (D12) in the census of the 8
  MVA-only Primaries. The correlation at D12 is about +0.75 to +0.83.
decision_rule: >-
  The rule uses L0 if the Pearson correlation of the training sunny-midday readings with irradiance
  exceeds +0.3, or the 1st percentile of those readings is below 1.0 percentage point of capacity;
  otherwise it uses the naive fit (N). Passes when, at the scored cut with reactive power zero, the
  chosen arm is within 0.02 of the signed fit's plain sunny-half-hour direction accuracy at D5 with
  no injection (its own criterion), and at 4 of 5 substations in each injected case at both cuts,
  and the rule picks L0 at D12. Fails otherwise, and the sequence stops for a decision. The 1.0
  threshold on the 1st percentile is informed by the numbers of experiment 365, so it is not a clean
  out-of-sample test; the correlation threshold of +0.3 comes from the design brief's measurements.
benchmark: >-
  The signed fit, the ungated fit through the meter, and the naive fit, all from experiment 358
row_set_id: >-
  dp_mw_v1 sites D5, D10, D11, D14, D16 (scored); the 8 MVA-only Primaries (census only); cuts
  2024-07-01 and 2025-07-01
metric: >-
  Plain share of sunny half-hours with the flow direction right, for the arm the rule chooses; the
  rule's two statistics
verdict: partly supported
production_status: not-for-production
adopted_in: none
date: 2026-10-28
best_arm: >-
  the rule on its own: plain direction accuracy within 0.02 of the signed fit at D5 and at 4 or 5 of
  5 substations in every injected case at both cuts, with reactive power zero
headline: >-
  The rule passes its registered test and picks L0 only at Stickney (D12) in the census, but with
  reactive power at power factor 0.97 the rule fails at injection level 1.0 (0 of 5 substations at
  the 2025 cut).
take_home_message: >-
  A test on how the midday meter readings move with sunshine, together with a test for readings
  near zero, picks the right fit at the substation with no reverse flow and at every substation
  where solar output was added, and it picks reverse flow at Stickney (D12) and at no other
  MVA-only Primary. The rule only works when the meter reading has no reactive power in it. At power
  factor 0.97 the readings no longer show the reverse flow, so the rule stays with the naive fit
  and loses badly at moderate injected solar. The 1.0 threshold on the low readings was chosen after
  seeing experiment 365's numbers, and it carries the moderate-injection cases.
lessons:
  - >-
    The correlation with irradiance picks out heavy reverse flow (+0.54 to +0.59 at injection level
    2.0, +0.55 at Stickney) but not moderate reverse flow (-0.31 to +0.47 at level 1.0).
  - >-
    The threshold of 1.0 on the lowest readings carries the moderate-injection cases, and it was set
    after seeing experiment 365's numbers, so those cases are not an out-of-sample test.
  - >-
    Any rule that reads the magnitude of apparent power cannot see reverse flow once the reactive
    power is realistic, because the reading then stays well above zero and stops following
    irradiance.
advice:
  - >-
    Test any reverse-flow rule on readings that include reactive power at power factor 0.97, not
    only on readings with none (the Power factor 0.97 bullet).
  - >-
    Do not tune a threshold on the substations that also score the rule, and say so where an earlier
    experiment's numbers informed the threshold (the Caveats section).
  - >-
    State the substation with no reverse flow as its own pass criterion (D5 in the results).
---

# MVA fold: reverse-flow rule from the correlation with irradiance

**A rule that checks two things on the training sunny-midday meter readings picks the right fit at
every case the registered test scores, but it breaks as soon as the readings include reactive
power.** We asked whether the 8 MVA-only Primaries (the eight substations that report only apparent
power, in MVA) can be told apart from substations with no reverse flow, using the meter readings
and the irradiance alone. The rule picks the fit through the meter (L0, which can see reverse flow)
if the readings rise with irradiance (Pearson correlation above +0.3) or the lowest readings come
close to zero (1st percentile below 1.0 percentage point of capacity). Otherwise the rule picks the
naive fit (N), which reads the magnitude as signed power. With reactive power zero, the rule is
right at D5 (the substation with no reverse flow), at the five injected cases, and at Stickney
(D12). It picks the naive fit at the other seven MVA-only Primaries. With reactive power at power
factor 0.97, the rule fails at moderate injected solar. Verdict: partly supported, because the
registered pass test is met but the test does not cover the realistic reactive setting, and one
threshold was set after seeing experiment 365's numbers.

This follows [experiment 365](../00365-mva-fold-readings-only-reverse-flow-rule/result.md), whose
rule on the lowest readings alone failed at D5 and at Stickney, and which proposed testing how the
readings move with irradiance. No model was refitted: the score of each case is the stored score of
the fit the rule chooses, from
[experiment 358](../00358-mva-fold-known-answer-at-folded-mw-primaries/result.md), which defines the
injection levels and the signed fit (S) used for comparison. The code reuses the way experiment 365
builds the readings. The rule and both thresholds were written into `result.md` before any
statistic was computed. **The threshold of 1.0 on the 1st percentile was chosen after seeing the
numbers of experiment 365** (D5's 1st percentile was 3.79 and the injected cases' were 0.2 to 1.6),
so it is not an out-of-sample test. The correlation threshold of +0.3 comes from the design brief's
measurements.

![Figure 1: correlation of training sunny-midday readings with irradiance, by substation](figure_1_correlation_by_substation.svg)

![Figure 2: lowest sunny-midday readings in training, by substation](figure_2_lowest_readings_by_substation.svg)

## Results (reactive power zero, plain direction accuracy on sunny half-hours)

- **D5, no injection: passes.** The statistics are a correlation of -0.28 and a 1st percentile of
  3.79 (2025 cut), so the rule chooses N. Accuracy is 0.980, the same as the signed fit's 0.980. At
  the 2024 cut the chosen accuracy is 0.963 against 0.961. Experiment 365's threshold of 5 chose L0
  here, with accuracy 0.35.
- **Injected cases: pass.** At the 2025 cut the chosen fit is within 0.02 of the signed fit at 5 of
  5 substations at both levels. At the 2024 cut the counts are 4 of 5 at level 1.0 (D11, gap 0.026)
  and 5 of 5 at level 2.0. The rule chooses L0 at all 5 substations in every injected case. At level
  2.0, where 92 to 99% of sunny half-hours have reverse flow, the chosen accuracy equals the signed
  fit's at every substation (0.94 to 0.99).
- **Which test carries which case.** At level 2.0 the correlation exceeds +0.3 at 5 of 5 substations
  at both cuts. At level 1.0 the correlation exceeds +0.3 at 1 of 5 (D5), so the 1st percentile
  below 1.0 chooses L0 at the other four. The correlation alone would miss four of the five level 1.0
  cases.
- **No injection, other substations.** D10 has genuine reverse flow in 14% of sunny half-hours,
  and the rule chooses L0 (1st percentile 0.28) with the same accuracy as the signed fit. D11, D14,
  and D16 have none, and the rule chooses N.
- **Census, not scored.** The rule chooses L0 at Stickney (D12), with correlation +0.55 and 1st
  percentile 26.7, as predicted. It chooses N at D2, D3, D4, D8, D19, D21, and D22, whose
  correlations are -0.54 to -0.04 and whose 1st percentiles are 5.9 to 40.2. Marsh Lane (D8) and
  Langrick (D22) are the closest calls on the 1st percentile (5.9 and 7.6) and stay with N. D12 is
  the only one of the eight with a positive correlation.
- **Power factor 0.97: fails.** With reactive power at power factor 0.97 in the readings, the
  readings stay well above zero (1st percentile 10.7 to 37.6 with no injection, and 11.0 to 17.4 at
  level 1.0). At level 1.0 the rule picks N at 5 of 5 substations at the 2025 cut and 4 of 5 at the
  2024 cut. The mean chosen accuracy is 0.31 (2025 cut) and 0.52 (2024 cut), against the signed
  fit's 0.86 and 0.81. The number of substations within 0.02 is 0 of 5 (2025 cut) and 1 of 5 (2024
  cut) at level 1.0, and 5 of 5 at level 2.0. With no injection, D10 at the 2025 cut is 0.042 below
  the signed fit. This repeats experiment 358's finding for the fit through the meter.

## Checks

- **Triggers.** Trigger 7 of the `ml-experiment` skill may apply, because the planned next steps
  of the MVA sequence depend on this result. No audit was run, because the experiment refits nothing
  and reads two stored result sets, and this session started no sub-agents. The maintainer decides
  whether to order one. Trigger 1 (a failed control) did not fire at reactive power zero, and fired
  at power factor 0.97, where the rule fails at level 1.0 as the earlier experiments did.
- **Median against mean forecast.** Not applicable: no forecasts were made.

## Caveats and what to try next

- **The pass test covers only reactive power zero.** The earlier experiments of the sequence found
  that real readings contain reactive power. If the sequence continues, the next step is a rule that
  still works at power factor 0.97, for example one that corrects each reading for an estimated
  reactive power before computing the statistics.
- The 1.0 threshold was informed by experiment 365, as stated above.
- The injected cases add the same recorded solar-farm output (farm A) to every substation, so the
  five injected cases are not independent evidence of how the rule behaves on other farms.
- The census statistic uses the metered reading at the MVA-only Primaries, so a reading changed by
  generation added to the magnitude (the Marsh Lane change in April 2025) also moves it.
- The census has no ground truth: the claim that D12 has reverse flow and the other seven do not
  comes from [experiment 110][e110] and the design brief, not from a signed measurement.

[e110]: ../00110-differentiable-physics-5-embedded-generation-and-capacity/result.md
